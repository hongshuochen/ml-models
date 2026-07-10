"""Annotate a golf video with the HIT-detection STATUS timeline + live count.

Uses the shared ego-invariant hit detector (golf/hit_detector.py, picked by an adversarial
multi-algorithm eval on real first-person clips). It counts BOTH putts (ball rolls away) and full
swings (ball vanishes from a stable address) — the on-device twin is android/.../PuttCounter.kt.

4 user-facing statuses:
    IDLE     (grey)   - nothing being addressed
    PREPARE  (yellow) - a ball is at address with a club nearby (getting ready to hit)
    HIT      (red)    - contact confirmed (ball left the address & didn't come back) -> count++  (flashed)
    FOLLOW   (green)  - just after a hit (ball rolling / flying away)

Two-pass so the HIT flash lands on the real contact frame: pass 1 runs YOLO once and collects
detections; the detector resolves hit frames; pass 2 re-reads the video (no YOLO) and renders.

Out: <name>_status.mp4   Run: uv run python golf/annotate_status.py <video> [--imgsz 1280]
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hit_detector import detect_hits, track_balls

BALL_COL, CLUB_COL = (0, 0, 255), (0, 200, 0)
STATUS_COL = {"IDLE": (150, 150, 150), "PREPARE": (0, 200, 255), "HIT": (0, 0, 255), "FOLLOW": (0, 200, 0)}


def nearest_dist(balls, clubs):
    """Nearest ball<->club_head center distance / club_head diagonal (for the context curve)."""
    if not clubs:
        return None
    cb = max(clubs, key=lambda b: b[4]); cx = (cb[0]+cb[2])/2; cy = (cb[1]+cb[3])/2
    cd = max(math.hypot(cb[2]-cb[0], cb[3]-cb[1]), 1e-4)
    best = None
    for b in balls:
        bx = (b[0]+b[2])/2; by = (b[1]+b[3])/2
        d = math.hypot(cx-bx, cy-by)/cd
        best = d if best is None else min(best, d)
    return 99.0 if best is None else best


class CurvePlot:
    """A distance-vs-time strip that GROWS in sync with the video: each frame extends the curve by
    one segment coloured by that frame's status, over dLow/dHigh reference lines, with a live cursor."""
    def __init__(self, width, total, fps, d_low, d_high, ymax=6.0, height=340):
        self.W, self.H, self.N, self.fps, self.ymax = width, height, max(total, 2), fps, ymax
        self.L, self.R, self.T, self.B = 78, 16, 18, 34
        self.pw = width - self.L - self.R; self.ph = height - self.T - self.B
        self.base = np.full((height, width, 3), 22, np.uint8)
        cv2.rectangle(self.base, (self.L, self.T), (width - self.R, height - self.B), (60, 60, 60), 1)
        for val, col, tag in [(d_low, (90, 190, 90), f"dLow {d_low:g}"), (d_high, (90, 90, 210), f"dHigh {d_high:g}")]:
            y = self._y(val)
            for x in range(self.L, width - self.R, 16):
                cv2.line(self.base, (x, y), (x + 8, y), col, 1)
            cv2.putText(self.base, tag, (width - self.R - 170, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        for val in (0, int(self.ymax)):
            cv2.putText(self.base, str(val), (10, self._y(val) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(self.base, "ball-clubhead distance", (self.L, self.T - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
        self.prev = None

    def _x(self, i): return int(self.L + i / (self.N - 1) * self.pw)
    def _y(self, d): return int(self.T + (1 - min(max(d, 0), self.ymax) / self.ymax) * self.ph)

    def add(self, i, d, col, is_hit):
        if d is None:                       # no club -> break the line (a real gap in the signal)
            self.prev = None; return
        pt = (self._x(i), self._y(d))       # d>=ymax (ball gone / BIG) pegs the curve to the top
        if self.prev is not None:
            cv2.line(self.base, self.prev, pt, col, 2, cv2.LINE_AA)
        self.prev = pt
        if is_hit:
            cv2.circle(self.base, pt, 10, (0, 0, 255), 3)

    def render(self, i):
        c = self.base.copy(); x = self._x(i)
        cv2.line(c, (x, self.T), (x, self.H - self.B), (255, 255, 255), 1)
        cv2.putText(c, f"{i / self.fps:4.1f}s", (min(self.W - 70, max(4, x - 28)), self.H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="runs/detect/golf_ego_v2_1280/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--d-low", type=float, default=1.6)
    ap.add_argument("--d-high", type=float, default=3.0)
    args = ap.parse_args()

    src = Path(args.video)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_status.mp4")
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 3000
    cap.release()

    # ---- pass 1: YOLO once -> per-frame boxes (memory-light) ----
    m = YOLO(args.model)
    dets = []   # [{'b': [[x1,y1,x2,y2],...], 'c': [[x1,y1,x2,y2,conf],...]}]
    for r in m.predict(source=str(src), stream=True, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False):
        b = [box.xyxy[0].tolist() for box in r.boxes if int(box.cls) == 0]
        c = [box.xyxy[0].tolist() + [float(box.conf)] for box in r.boxes if int(box.cls) == 1]
        dets.append({"b": b, "c": c})
    n = len(dets)

    # ---- detect hits (offline: aligns the flash to real contact) ----
    hits, armed = detect_hits(dets, fps)
    hit_set = set(hits)
    ball_track = track_balls(dets)          # active ball centroid per frame (for the trajectory trail)
    hit_flash = int(0.4 * fps)
    follow = int(0.8 * fps)

    def status_at(i):
        for hf in hits:
            if hf <= i <= hf + hit_flash:
                return "HIT"
            if hf + hit_flash < i <= hf + hit_flash + follow:
                return "FOLLOW"
        return "PREPARE" if (i < len(armed) and armed[i]) else "IDLE"

    def count_at(i):
        return sum(1 for hf in hits if hf <= i)

    # ---- pass 2: re-read frames (no YOLO) and render ----
    cap = cv2.VideoCapture(str(src))
    writer = None; plot = None
    trail = []; trailing_until = -1
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        balls = dets[i]["b"]; clubs = dets[i]["c"]
        d = nearest_dist(balls, clubs)
        status = status_at(i); is_hit = i in hit_set
        for b in balls:
            x1, y1, x2, y2 = map(int, b[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COL, 3)
        for c in clubs:
            x1, y1, x2, y2 = map(int, c[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), CLUB_COL, 3)
        # rolling-ball TRAJECTORY after a hit (image-space comet tail)
        if is_hit:
            trail = []; trailing_until = i + int(2.5 * fps)
        cur = ball_track[i]
        if i <= trailing_until and cur is not None:
            trail.append((cur[0], cur[1]))
        if i <= trailing_until and len(trail) >= 2:
            npts = len(trail)
            for k in range(1, npts):
                a = k / npts
                p0 = tuple(map(int, trail[k-1])); p1 = tuple(map(int, trail[k]))
                cv2.line(frame, p0, p1, (0, int(90 + 120*a), 255), max(2, int(2 + 6*a)), cv2.LINE_AA)
                cv2.circle(frame, p1, max(2, int(3 + 4*a)), (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(map(int, trail[-1])), 12, (0, 220, 255), 3, cv2.LINE_AA)
        # status banner + live count
        col = STATUS_COL[status]
        cv2.rectangle(frame, (0, 0), (W, 130), (0, 0, 0), -1)
        cv2.putText(frame, status, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.6, col, 6, cv2.LINE_AA)
        cv2.putText(frame, f"HITS: {count_at(i)}", (W-430, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6, cv2.LINE_AA)
        ds = "--" if d is None else f"{d:.2f}"
        cv2.putText(frame, f"d={ds}", (30, H-30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        if is_hit:
            cv2.rectangle(frame, (0, 0), (W, H), (0, 0, 255), 30)
        # growing distance strip below the video
        if plot is None:
            plot = CurvePlot(W, total, fps, args.d_low, args.d_high)
        plot.add(i, d, col, is_hit)
        out_frame = np.vstack([frame, plot.render(i)])
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + plot.H))
        writer.write(out_frame)
    cap.release()
    if writer:
        writer.release()
    print(f"{src.name}: {n} frames -> {out} | HITS counted: {len(hits)} at {[round(h/fps,1) for h in hits]}")


if __name__ == "__main__":
    main()
