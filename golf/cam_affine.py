"""Per-frame CAMERA-motion affines for a video, from background optical flow (detections masked out).

For ego-motion compensation in the hit detector: a world-static point at p in frame i should appear
at  p' = M_i @ [x, y, 1]  in frame i+1 under head/camera motion alone; the ball's TRUE (ground-
relative) motion is its observed position minus that prediction.

Method: half-res gray -> goodFeaturesToTrack (ball + club_head boxes masked, padded) -> pyramidal LK
-> RANSAC estimateAffinePartial2D. Validated on real clips: at walk-away false hits the ball's TRUE
speed is ~0.03-0.08 D/frame (static ball, moving head) vs RAW 0.4-0.7; a real putt is TRUE ~1.05.

Output JSON: list of length n_frames-1; entry i = [a,b,tx,c,d,ty] (row-major 2x3, FULL-res pixels)
or null where estimation failed (too few features / RANSAC fail) — consumers must handle null.

Run: uv run python golf/cam_affine.py <video> <dets.json> <out.json> [--scale 0.5]
"""
import argparse
import json

import cv2
import numpy as np


def pair_affine(prev_gray, cur_gray, boxes, scale):
    """Camera affine between two HALF-RES (scale) gray frames -> [a,b,tx,c,d,ty] in FULL-res
    pixels, or None. boxes = full-res [x1,y1,x2,y2] detections to mask out of the background."""
    mask = np.full(prev_gray.shape, 255, np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(v * scale) for v in box[:4]]
        cv2.rectangle(mask, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), 0, -1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask)
    if p0 is None or len(p0) < 12:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None)
    g0 = p0[st == 1]
    g1 = p1[st == 1]
    if len(g0) < 12:
        return None
    A, _ = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if A is None:
        return None
    # half-res -> full-res: linear part is conjugation-invariant, translation scales
    return [float(A[0, 0]), float(A[0, 1]), float(A[0, 2] / scale),
            float(A[1, 0]), float(A[1, 1]), float(A[1, 2] / scale)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("dets", help="cache_dets.py JSON (boxes to mask out of the background)")
    ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    frames = json.load(open(args.dets))["frames"]
    s = args.scale
    cap = cv2.VideoCapture(args.video)
    ok, img = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    pg = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    Ms = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        cg = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        boxes = []
        if i < len(frames):
            boxes = list(frames[i]["b"]) + [c[:4] for c in frames[i]["c"]]
        Ms.append(pair_affine(pg, cg, boxes, s))
        pg = cg
        i += 1
    with open(args.out, "w") as f:
        json.dump(Ms, f)
    print(f"{args.video}: {len(Ms)} affines ({sum(m is None for m in Ms)} failed) -> {args.out}")


if __name__ == "__main__":
    main()
