#!/usr/bin/env python3
"""Label image pools face / no-face with InsightFace -> a classification ImageFolder.

Whole images only (no crops): InsightFace SCRFD (buffalo_l) decides whether each full image contains
a face that is large enough (min-side >= --min-face-frac * image-min-dim — small/distant faces the
classifier can't see at its input become "noface"). Splits follow the SOURCE dirs you pass (official
splits): --train-dirs go to train/, --val-dirs to val/. Used to train a frozen-backbone binary
"is there a (usable) face?" classifier (see train_face_cls.py).
"""
import argparse
import os
import random
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis


def label_dirs(app, dirs, out, split, min_score, min_face_frac, limit, seed):
    for lab in ("face", "noface"):
        (Path(out) / split / lab).mkdir(parents=True, exist_ok=True)
    imgs = []
    for d in dirs:
        imgs += list(Path(d).rglob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    if limit:
        imgs = imgs[:limit]
    counts = {"face": 0, "noface": 0}
    for i, img in enumerate(imgs):
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        thr = min_face_frac * min(h, w)
        has_face = any(
            f.det_score >= min_score
            and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= thr
            for f in app.get(im)
        )
        lab = "face" if has_face else "noface"
        dst = Path(out) / split / lab / f"{i:07d}.jpg"  # index-unique within the split
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        counts[lab] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {split} {i + 1}/{len(imgs)} ...", flush=True)
    print(f"{split}: {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dirs", nargs="+",
                    default=["datasets/coco_dl/train2017", "datasets/widerface/images/train"])
    ap.add_argument("--val-dirs", nargs="+",
                    default=["datasets/coco_dl/val2017", "datasets/widerface/images/val"])
    ap.add_argument("--out", default="datasets/face_cls")
    ap.add_argument("--min-score", type=float, default=0.5, help="face det_score to count as a face")
    ap.add_argument("--min-face-frac", type=float, default=0.12,
                    help="face counts only if min-side / image-min-dim >= this (small -> noface)")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    label_dirs(app, args.train_dirs, args.out, "train", args.min_score, args.min_face_frac, args.limit_train, args.seed)
    label_dirs(app, args.val_dirs, args.out, "val", args.min_score, args.min_face_frac, args.limit_val, args.seed)


if __name__ == "__main__":
    main()
