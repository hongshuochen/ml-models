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
        self.gap = 0; self.reset_gap = int(0.25 * fps)  # no-club gap > this -> drop a stale ADDRESS
        self.saw_ball = False                            # a real ball was seen AT the putter this cycle

    def update(self, d, has_any):
        self.frame += 1; f = self.frame; hit = False
        if d is None:
            # Fix 1: a sustained no-club gap (wearer walking / looking away) drops a stale ADDRESS,
            # so the putter reappearing WITHOUT a ball can't fire a false HIT.
            self.gap += 1
            if self.gap > self.reset_gap:
                self.state = "W"; self.low = 0; self.saw_ball = False
        else:
            self.gap = 0
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
    m = YOLO(args.model)
    sm = PuttSM(fps, d_low=args.d_low, d_high=args.d_high)
    writer = None; n = 0
    for r in m.predict(source=str(src), stream=True, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False):
        frame = r.orig_img.copy(); H, W = frame.shape[:2]
        balls = [b.xyxy[0].tolist() for b in r.boxes if int(b.cls) == 0]
        clubs = [b.xyxy[0].tolist()+[float(b.conf)] for b in r.boxes if int(b.cls) == 1]
        d = nearest_dist([b for b in balls], clubs)
        status, hit = sm.update(d, bool(balls) or bool(clubs))
        for b in balls:
            x1, y1, x2, y2 = map(int, b[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COL, 3)
        for c in clubs:
            x1, y1, x2, y2 = map(int, c[:4]); cv2.rectangle(frame, (x1, y1), (x2, y2), CLUB_COL, 3)
        # status banner
        col = STATUS_COL[status]
        cv2.rectangle(frame, (0, 0), (W, 130), (0, 0, 0), -1)
        cv2.putText(frame, status, (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 2.6, col, 6, cv2.LINE_AA)
        cv2.putText(frame, f"PUTTS: {sm.count}", (W-460, 92), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6, cv2.LINE_AA)
        ds = "--" if d is None else f"{d:.2f}"
        cv2.putText(frame, f"d={ds}", (30, H-30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        if hit:  # extra flash frame
            cv2.rectangle(frame, (0, 0), (W, H), (0, 0, 255), 30)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        writer.write(frame); n += 1
    if writer: writer.release()
    print(f"{src.name}: {n} frames -> {out} | PUTTS counted: {sm.count}")


if __name__ == "__main__":
    main()
