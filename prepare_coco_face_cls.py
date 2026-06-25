#!/usr/bin/env python3
"""Label a diverse image pool (COCO) face / no-face with InsightFace -> a classification ImageFolder.

Whole images only (no crops): InsightFace SCRFD (buffalo_l) decides whether each full image contains
a face; that becomes the label. Output: <out>/{train,val}/{face,noface}/<name>.jpg (symlinks).
Used to train a frozen-backbone binary "is there a face?" classifier (see train_face_cls.py).
"""
import argparse
import os
import random
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="datasets/coco_dl/val2017")
    ap.add_argument("--out", default="datasets/coco_face_cls")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-score", type=float, default=0.5, help="face det_score to count as a face")
    ap.add_argument("--min-side", type=int, default=20, help="min face box side (px) to count")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    imgs = sorted(Path(args.img_dir).glob("*.jpg"))
    random.Random(args.seed).shuffle(imgs)
    if args.limit:
        imgs = imgs[: args.limit]

    for split in ("train", "val"):
        for lab in ("face", "noface"):
            (Path(args.out) / split / lab).mkdir(parents=True, exist_ok=True)

    n_val = int(len(imgs) * args.val_frac)
    counts = {"train_face": 0, "train_noface": 0, "val_face": 0, "val_noface": 0}
    for i, img in enumerate(imgs):
        im = cv2.imread(str(img))
        if im is None:
            continue
        faces = app.get(im)
        has_face = any(
            f.det_score >= args.min_score
            and min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= args.min_side
            for f in faces
        )
        lab = "face" if has_face else "noface"
        split = "val" if i < n_val else "train"
        dst = Path(args.out) / split / lab / img.name
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        counts[f"{split}_{lab}"] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(imgs)} ...", flush=True)

    print(f"done: {counts}  -> {args.out}")


if __name__ == "__main__":
    main()
