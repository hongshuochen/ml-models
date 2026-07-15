"""Annotate a golf video with the HIT-detection STATUS timeline + live count (+ hole/cup boxes).

Defaults to the 3-class model (ball/club_head/hole); draws hole boxes in cyan. Pass a 2-class model
and hole is simply absent (h=[] per frame).

Uses the shared EGO-COMPENSATED hit detector (golf/hit_detector.py v3): pass 1 runs YOLO once and
simultaneously estimates per-frame CAMERA affines from background optical flow (half-res LK,
golf/cam_affine.py), so the detector separates true ball motion from head motion. Counts BOTH
putts (ball rolls away) and full swings (ball vanishes from a stable address).

4 user-facing statuses:
    IDLE     (grey)   - nothing being addressed
    PREPARE  (yellow) - a ball is at address with a club nearby (getting ready to hit)
    HIT      (red)    - contact confirmed (ball left the address & didn't come back) -> count++  (flashed)
    FOLLOW   (green)  - just after a hit (ball rolling / flying away)

Two-pass so the HIT flash lands on the real contact frame: pass 1 YOLO+flow collects detections;
the detector resolves hit frames; pass 2 re-reads the video (no YOLO) and renders.

Out: <name>_status.mp4   Run: uv run python golf/annotate_status.py <video> [--imgsz 1280]
     (--dets-cache/--cams-cache reuse golf/cache_dets.py + golf/cam_affine.py JSONs, skipping YOLO)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cam_affine import pair_affine
from hit_detector import detect_hits, track_balls

BALL_COL, CLUB_COL, HOLE_COL = (0, 0, 255), (0, 200, 0), (238, 210, 34)   # ball red, club green, hole cyan
STATUS_COL = {"IDLE": (150, 150, 150), "PREPARE": (0, 200, 255), "HIT": (0, 0, 255),
              "FOLLOW": (0, 200, 0), "MADE PUTT": (60, 220, 255)}   # made = gold
BALL_MM = 42.7          # golf ball diameter (mm) — scales pixel motion -> real ball speed
SPEED_DEADBAND = 0.45   # m/s — below this is ball-box jitter / camera-comp residual (stationary ball) -> show 0


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
    ap.add_argument("--model", default="runs/detect/golf_ego_v3_hole/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--d-low", type=float, default=1.6)
    ap.add_argument("--d-high", type=float, default=3.0)
    ap.add_argument("--dets-cache", default=None, help="cache_dets.py JSON: skip YOLO in pass 1")
    ap.add_argument("--cams-cache", default=None, help="cam_affine.py JSON: skip flow in pass 1")
    ap.add_argument("--no-trail", action="store_true", help="don't draw the post-hit ball trajectory tail")
    ap.add_argument("--trail-secs", type=float, default=5.0, help="how long the post-hit ball trail lingers [5.0s]")
    args = ap.parse_args()

    src = Path(args.video)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_status.mp4")
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 3000
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 2048
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1536
    cap.release()

    # ---- pass 1: YOLO + camera affines in one sweep (or load caches) ----
    if args.dets_cache:
        dets = json.load(open(args.dets_cache))["frames"]
        cams = json.load(open(args.cams_cache)) if args.cams_cache else None
    else:
        from ultralytics import YOLO
        m = YOLO(args.model)
        dets = []   # [{'b': [[x1,y1,x2,y2],...], 'c': [[x1,y1,x2,y2,conf],...]}]
        cams = []   # per-frame camera affine i -> i+1 (background LK; detections masked)
        SCALE = 0.5
        pg = None
        for r in m.predict(source=str(src), stream=True, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False):
            b = [box.xyxy[0].tolist() for box in r.boxes if int(box.cls) == 0]
            c = [box.xyxy[0].tolist() + [float(box.conf)] for box in r.boxes if int(box.cls) == 1]
            h = [box.xyxy[0].tolist() for box in r.boxes if int(box.cls) == 2]      # hole (cup); [] if a 2-class model
            dets.append({"b": b, "c": c, "h": h})
            cg = cv2.cvtColor(cv2.resize(r.orig_img, None, fx=SCALE, fy=SCALE), cv2.COLOR_BGR2GRAY)
            if pg is not None:
                cams.append(pair_affine(pg, cg, b + [cc[:4] for cc in c], SCALE))
            pg = cg
    n = len(dets)

    # ---- detect hits (offline: aligns the flash to real contact) ----
    hits, armed = detect_hits(dets, fps, cams=cams, size=(W0, H0))
    hit_set = set(hits)
    ball_track = track_balls(dets, fps, cams, size=(W0, H0))   # ball centroid per frame (trajectory trail)
    hit_flash = int(0.4 * fps)
    follow = int(0.8 * fps)

    # --- ball speed (m/s), ego-compensated: remove camera motion between frames, px->m via ball size ---
    all_diam = [bb[2] - bb[0] for f in dets for bb in f["b"] if bb[2] - bb[0] > 0]
    med_diam = sorted(all_diam)[len(all_diam) // 2] if all_diam else 14.0

    def diam_at(i):
        return max((bb[2] - bb[0] for bb in dets[i]["b"]), default=med_diam) or med_diam

    def speed_mps(i):
        if i < 1:
            return 0.0
        p0, p1 = ball_track[i - 1], ball_track[i]
        if p0 is None or p1 is None or not (cams and i - 1 < len(cams) and cams[i - 1] is not None):
            return 0.0
        M = cams[i - 1]                                    # map prev ball into this frame (subtract camera motion)
        px = M[0] * p0[0] + M[1] * p0[1] + M[2]; py = M[3] * p0[0] + M[4] * p0[1] + M[5]
        s = math.hypot(p1[0] - px, p1[1] - py) * fps * (BALL_MM / 1000.0) / diam_at(i)
        return 0.0 if s > 8.0 else s      # >8 m/s in one frame = a mis-detection (a DIFFERENT ball), not a real putt

    # --- MADE PUTT: ball track reaches inside a hole box within 2.5s after a hit ---
    made = set()
    for hf in hits:
        for i in range(hf, min(hf + int(2.5 * fps), n)):
            cur = ball_track[i]
            if cur is None:
                continue
            for hb in dets[i].get("h", []):
                ex, ey = (hb[2] - hb[0]) * 0.35, (hb[3] - hb[1]) * 0.35     # cup-lip / detection slack
                if hb[0] - ex <= cur[0] <= hb[2] + ex and hb[1] - ey <= cur[1] <= hb[3] + ey:
                    made.add(hf); break
            if hf in made:
                break

    # per-hit "putt speed" = peak ego-comp ball speed in the first 0.6s after contact
    hit_speed = {hf: max((speed_mps(i) for i in range(hf, min(hf + int(0.6 * fps), n))), default=0.0)
                 for hf in hits}

    def status_at(i):
        for hf in hits:
            if hf <= i <= hf + hit_flash + follow:
                if hf in made:
                    return "MADE PUTT"
                return "HIT" if i <= hf + hit_flash else "FOLLOW"
        return "PREPARE" if (i < len(armed) and armed[i]) else "IDLE"

    def count_at(i):
        return sum(1 for hf in hits if hf <= i)

    def made_at(i):
        return sum(1 for hf in hits if hf <= i and hf in made)

    # ---- pass 2: re-read frames (no YOLO) and render ----
    cap = cv2.VideoCapture(str(src))
    writer = None; plot = None
    trail = []; trailing_until = -1; spd_smooth = 0.0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        balls = dets[i]["b"]; clubs = dets[i]["c"]; holes = dets[i].get("h", [])
        d = nearest_dist(balls, clubs)
        status = status_at(i); is_hit = i in hit_set
        for hb in holes:      # cup first so ball/club boxes draw over it
            x1, y1, x2, y2 = map(int, hb[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), HOLE_COL, 3)
            cv2.putText(frame, "hole", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, HOLE_COL, 2, cv2.LINE_AA)
        for b in balls:
            x1, y1, x2, y2 = map(int, b[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COL, 3)
        for c in clubs:
            x1, y1, x2, y2 = map(int, c[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), CLUB_COL, 3)
        # rolling-ball TRAJECTORY after a hit — GROUND-ANCHORED: past points are propagated
        # through each frame's camera affine so the tail stays glued to the grass (not the screen)
        if trail and cams and 0 <= i - 1 < len(cams) and cams[i - 1] is not None:
            M = cams[i - 1]
            trail = [(M[0] * x + M[1] * y + M[2], M[3] * x + M[4] * y + M[5]) for x, y in trail]
        if args.no_trail:
            trailing_until = -1
        elif is_hit:
            trail = []; trailing_until = i + int(args.trail_secs * fps)
        cur = ball_track[i]
        if i <= trailing_until and cur is not None:
            # reject an implausibly large single-frame jump (mis-detection onto a DIFFERENT ball, e.g. after
            # the putt drops) — freeze the trail at the last good point instead of leaping across the green
            if not trail or math.hypot(cur[0] - trail[-1][0], cur[1] - trail[-1][1]) < max(160.0, 10 * diam_at(i)):
                trail.append((cur[0], cur[1]))
        if i <= trailing_until and len(trail) >= 2:
            npts = len(trail)
            for k in range(1, npts):
                a = k / npts
                p0 = tuple(map(int, trail[k-1])); p1 = tuple(map(int, trail[k]))
                cv2.line(frame, p0, p1, (0, int(90 + 120*a), 255), max(2, int(2 + 6*a)), cv2.LINE_AA)
                cv2.circle(frame, p1, max(2, int(3 + 4*a)), (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(map(int, trail[-1])), 12, (0, 220, 255), 3, cv2.LINE_AA)
        # status banner + live count + made count
        col = STATUS_COL[status]
        cv2.rectangle(frame, (0, 0), (W, 130), (0, 0, 0), -1)
        cv2.putText(frame, status, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.4, col, 6, cv2.LINE_AA)
        cv2.putText(frame, f"HITS: {count_at(i)}", (W-470, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, f"MADE: {made_at(i)}", (W-470, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.6, STATUS_COL["MADE PUTT"], 4, cv2.LINE_AA)
        # top-left speed panel (below the banner): live ego-compensated ball speed + the hit's peak putt speed
        spd_smooth = 0.55 * spd_smooth + 0.45 * speed_mps(i)
        # the ball only moves after contact: within the post-hit roll window (trail-secs) show the real
        # speed (deadband trims the stopped tail); 0 at address/IDLE/PREPARE. Peak "putt speed" lingers
        # for the whole roll window too (not just the ~1.2s flash).
        roll_win = int(args.trail_secs * fps)
        rh = next((hf for hf in reversed(hits) if hf <= i <= hf + roll_win), None)   # most recent hit still rolling
        spd_show = spd_smooth if (rh is not None and spd_smooth >= SPEED_DEADBAND) else 0.0
        cv2.putText(frame, f"ball  {spd_show:4.1f} m/s", (30, 200),      # yellow, always shown (top-left)
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4, cv2.LINE_AA)
        if rh is not None and hit_speed[rh] > 0:
            cv2.putText(frame, f"putt  {hit_speed[rh]:.1f} m/s peak", (30, 262),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 165, 255), 4, cv2.LINE_AA)   # orange, lingers the roll window
        ds = "--" if d is None else f"{d:.2f}"
        cv2.putText(frame, f"d={ds}", (30, H-30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
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
