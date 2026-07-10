"""Annotate a golf video with the putt-detection STATUS timeline + live count.

Mirrors the on-device PuttCounter (android/.../PuttCounter.kt): the signal is the nearest
ball<->club_head distance normalized by club-head size (ego-motion invariant), and a state machine
emits 4 user-facing statuses:
    IDLE     (grey)   - nothing detected (no ball AND no club_head in frame)
    PREPARE  (yellow) - a ball and/or club_head is present (getting ready to hit)
    HIT      (red)    - contact confirmed (ball separated & stayed) -> count++  (flashed ~0.4s)
    FOLLOW   (green)  - just after a hit (ball rolling / follow-through)

Out: <name>_status.mp4   Run: uv run python golf/annotate_status.py <video> [--imgsz 1280]
Tune the thresholds to match reality (this is the offline twin of the app's field calibration).
"""
import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BALL_COL, CLUB_COL = (0, 0, 255), (0, 200, 0)
STATUS_COL = {"IDLE": (150, 150, 150), "PREPARE": (0, 200, 255), "HIT": (0, 0, 255), "FOLLOW": (0, 200, 0)}


def nearest_dist(balls, clubs):
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


class PuttSM:
    """Same logic as PuttCounter.kt, plus a 4-status display resolver."""
    def __init__(self, fps, d_low=1.6, d_high=3.0, addr=4, confirm=8, cool=40, hit_flash=0.4, follow=0.8):
        self.fps = fps; self.dLow = d_low; self.dHigh = d_high
        self.addr = addr; self.confirm = confirm; self.cool = cool
        self.hit_flash = int(hit_flash*fps); self.follow = int(follow*fps)
        self.count = 0; self.state = "W"; self.low = 0; self.sep = 0; self.last = -10000
        self.frame = 0; self.hit_until = -1; self.follow_until = -1
        # false-alarm guards:
        self.saw_ball = False                            # a real ball was seen AT the club this cycle
        self.last_addr = -10000                          # last frame a ball was genuinely AT the club (d<=dLow)
        self.max_addr_age = int(2.0 * fps)               # address older than this = stale (walking) -> drop it
        self.max_sep = int(1.5 * fps)                    # a separation must confirm within this -> else stale

    def update(self, d, has_any):
        self.frame += 1; f = self.frame; hit = False
        if d is not None and d <= self.dLow:
            self.last_addr = f                              # a ball is genuinely AT the club right now
        # stale address = no ball at the club for a while (wearer walking / looking away) -> drop it,
        # so the club reappearing without a ball can't fire a false HIT. Skip mid-separation, where the
        # ball has legitimately left the club (rolling putt / a struck ball flying away).
        if self.state != "S" and f - self.last_addr > self.max_addr_age:
            self.state = "W"; self.low = 0; self.saw_ball = False
        # a separation that never confirms within max_sep is stale (a long gap / walking after entering
        # S kept it alive) -> discard. Real strokes confirm within ~1s (killed the 7 s "separation" at
        # golf_049 t=14.9 while keeping short roll-gaps like golf_049 t=35.3).
        if self.state == "S" and f - self.sep > self.max_sep:
            self.state = "W"; self.low = 0; self.saw_ball = False
        if d is not None:
            if self.state == "W":
                if d <= self.dLow:
                    self.low += 1
                    if self.low >= self.addr: self.state = "A"; self.saw_ball = True
                else: self.low = 0
            elif self.state == "A":
                if d <= self.dLow: self.saw_ball = True     # a real ball is at the putter (d small, not BIG)
                if d >= self.dHigh: self.state = "S"; self.sep = f
            elif self.state == "S":
                if d <= self.dLow: self.state = "A"; self.saw_ball = True
                elif f - self.sep >= self.confirm:
                    # Fix 2: only count if a real ball was actually seen at the putter this cycle
                    # (guards against "putter visible, no ball" being read as a strike).
                    if self.saw_ball and f - self.last >= self.cool:
                        self.count += 1; self.last = f; hit = True
                        self.hit_until = f + self.hit_flash; self.follow_until = f + self.hit_flash + self.follow
                    self.state = "W"; self.low = 0; self.saw_ball = False
        # display = status1 (state-based): PREPARE while addressing the ball, IDLE otherwise.
        if f <= self.hit_until: status = "HIT"
        elif f <= self.follow_until: status = "FOLLOW"
        elif self.state in ("A", "S"): status = "PREPARE"
        else: status = "IDLE"
        return status, hit


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
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()
    cap = cv2.VideoCapture(str(src)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 3000; cap.release()
    m = YOLO(args.model)
    sm = PuttSM(fps, d_low=args.d_low, d_high=args.d_high)
    writer = None; plot = None; n = 0
    trail = []; prev_ball = None; trailing_until = -1   # ball rolling-trajectory after a putt
    for i, r in enumerate(m.predict(source=str(src), stream=True, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False)):
        frame = r.orig_img.copy(); H, W = frame.shape[:2]
        balls = [b.xyxy[0].tolist() for b in r.boxes if int(b.cls) == 0]
        clubs = [b.xyxy[0].tolist()+[float(b.conf)] for b in r.boxes if int(b.cls) == 1]
        d = nearest_dist([b for b in balls], clubs)
        status, hit = sm.update(d, bool(balls) or bool(clubs))
        for b in balls:
            x1, y1, x2, y2 = map(int, b[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COL, 3)
        for c in clubs:
            x1, y1, x2, y2 = map(int, c[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), CLUB_COL, 3)
        # --- rolling-ball TRAJECTORY after a putt (image-space; clean while the head is still) ---
        cur_ball = None
        if balls:
            cents = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in balls]
            if prev_ball is not None:
                cb = min(cents, key=lambda p: (p[0]-prev_ball[0])**2 + (p[1]-prev_ball[1])**2)
                cur_ball = cb if math.hypot(cb[0]-prev_ball[0], cb[1]-prev_ball[1]) < 0.4*W else None
            if cur_ball is None and clubs:
                cc = ((clubs[0][0]+clubs[0][2])/2, (clubs[0][1]+clubs[0][3])/2)
                cur_ball = min(cents, key=lambda p: (p[0]-cc[0])**2 + (p[1]-cc[1])**2)
            if cur_ball is None:
                cur_ball = cents[0]
        if cur_ball is not None:
            prev_ball = cur_ball
        if hit:
            trail = []; trailing_until = i + int(2.5 * fps)          # fresh trail from the strike
        if i <= trailing_until and cur_ball is not None:
            trail.append(cur_ball)
        if i <= trailing_until and len(trail) >= 2:                  # draw the comet tail
            npts = len(trail)
            for k in range(1, npts):
                a = k / npts                                         # older = fainter/thinner
                p0 = tuple(map(int, trail[k-1])); p1 = tuple(map(int, trail[k]))
                cv2.line(frame, p0, p1, (0, int(90 + 120*a), 255), max(2, int(2 + 6*a)), cv2.LINE_AA)
                cv2.circle(frame, p1, max(2, int(3 + 4*a)), (0, 200, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(map(int, trail[-1])), 12, (0, 220, 255), 3, cv2.LINE_AA)
        # status banner
        col = STATUS_COL[status]
        cv2.rectangle(frame, (0, 0), (W, 130), (0, 0, 0), -1)
        cv2.putText(frame, status, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.6, col, 6, cv2.LINE_AA)
        cv2.putText(frame, f"PUTTS: {sm.count}", (W-460, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6, cv2.LINE_AA)
        ds = "--" if d is None else f"{d:.2f}"
        cv2.putText(frame, f"d={ds}", (30, H-30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        if hit:  # extra flash frame
            cv2.rectangle(frame, (0, 0), (W, H), (0, 0, 255), 30)
        # distance curve strip (grows in sync), coloured by status, below the video
        if plot is None:
            plot = CurvePlot(W, total, fps, args.d_low, args.d_high)
        plot.add(i, d, col, hit)
        out_frame = np.vstack([frame, plot.render(i)])
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + plot.H))
        writer.write(out_frame); n += 1
    if writer: writer.release()
    print(f"{src.name}: {n} frames -> {out} | PUTTS counted: {sm.count}")


if __name__ == "__main__":
    main()
