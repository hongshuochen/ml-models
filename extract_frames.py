#!/usr/bin/env python3
"""Sample frames from the egocentric golf videos into a labeling set.

Reads videos sequentially (fast), keeps ~--fps frames/sec, drops near-duplicate frames (many are
static address holds), caps per-video so one long clip can't dominate. Output = jpgs ready to
auto-prelabel (pretrained golf detector) + hand-correct.

    uv run python extract_frames.py --fps 1.5 --dedup 7 --max-per-video 40
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="datasets/golf_videos")
    ap.add_argument("--out", default="datasets/golf_frames")
    ap.add_argument("--fps", type=float, default=1.5, help="target frames sampled per second")
    ap.add_argument("--dedup", type=float, default=7.0, help="skip if 64x64 mean-abs-diff from last kept < this (0=off)")
    ap.add_argument("--max-per-video", type=int, default=40)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = sorted(glob.glob(args.videos + "/golf_*.mp4"))
    total = 0
    for v in vids:
        c = cv2.VideoCapture(v)
        fps = c.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / args.fps)))
        stem = Path(v).stem
        last = None
        fi = -1
        kept = 0
        while kept < args.max_per_video:
            ok, fr = c.read()
            if not ok:
                break
            fi += 1
            if fi % step != 0:
                continue
            if args.dedup > 0:
                small = cv2.resize(fr, (64, 64)).astype(np.int16)
                if last is not None and np.abs(small - last).mean() < args.dedup:
                    continue
                last = small
            cv2.imwrite(str(out / f"{stem}_f{fi:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            kept += 1
            total += 1
        c.release()
        print(f"{stem}: {kept}", flush=True)
    print(f"TOTAL: {total} frames -> {out}")


if __name__ == "__main__":
    main()
