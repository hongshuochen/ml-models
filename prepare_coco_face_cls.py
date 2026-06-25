#!/usr/bin/env python3
"""Label image pools face / no-face with InsightFace -> a classification ImageFolder.

Whole images only (no crops). InsightFace SCRFD (buffalo_l) finds the largest face per image; the
box is CACHED (datasets/face_cls_cache.json) so re-thresholding never re-runs detection. An image is
"face" iff its largest face min-side >= --min-face-frac * image-min-dim (small/distant faces the
classifier can't see become "noface"). Splits follow the source dirs (official): --train-dirs ->
train/, --val-dirs -> val/. Feeds train_face_cls.py (frozen-backbone "is there a face?" head).
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2


def sample(dirs, limit, seed):
    imgs = []
    for d in dirs:
        imgs += list(Path(d).rglob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    return imgs[:limit] if limit else imgs


def frac_of(entry, img):
    e = entry.get(str(img.resolve()))
    if not e or e["box"] is None:
        return 0.0
    x1, y1, x2, y2 = e["box"]
    w, h = e["wh"]
    return min(x2 - x1, y2 - y1) / min(w, h)


def build(imgs, cache, out, split, thr):
    base = Path(out) / split
    for lab in ("face", "noface"):
        if (base / lab).exists():
            shutil.rmtree(base / lab)
        (base / lab).mkdir(parents=True, exist_ok=True)
    c = {"face": 0, "noface": 0}
    for i, img in enumerate(imgs):
        lab = "face" if frac_of(cache, img) >= thr else "noface"
        dst = base / lab / f"{i:07d}.jpg"
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        c[lab] += 1
    print(f"{split} @frac>={thr}: {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dirs", nargs="+",
                    default=["datasets/coco_dl/train2017", "datasets/widerface/images/train"])
    ap.add_argument("--val-dirs", nargs="+",
                    default=["datasets/coco_dl/val2017", "datasets/widerface/images/val"])
    ap.add_argument("--out", default="datasets/face_cls")
    ap.add_argument("--cache", default="datasets/face_cls_cache.json")
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--min-face-frac", type=float, default=0.08)
    ap.add_argument("--limit-train", type=int, default=80000)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = json.load(open(args.cache)) if Path(args.cache).exists() else {}
    train_imgs = sample(args.train_dirs, args.limit_train, args.seed)
    val_imgs = sample(args.val_dirs, args.limit_val, args.seed)

    # detect only the images not already cached
    todo = [im for im in train_imgs + val_imgs if str(im.resolve()) not in cache]
    print(f"cache has {len(cache)}; {len(todo)} new to detect")
    if todo:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        for n, img in enumerate(todo):
            k = str(img.resolve())
            im = cv2.imread(k)
            if im is None:
                cache[k] = {"wh": [0, 0], "box": None}
                continue
            h, w = im.shape[:2]
            fc = [f for f in app.get(im) if f.det_score >= args.min_score]
            if fc:
                b = max(fc, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                cache[k] = {"wh": [w, h], "box": [float(v) for v in b.bbox]}
            else:
                cache[k] = {"wh": [w, h], "box": None}
            if (n + 1) % 2000 == 0:
                print(f"  detected {n + 1}/{len(todo)} ...", flush=True)
        json.dump(cache, open(args.cache, "w"))
        print(f"cache saved: {len(cache)} imgs")

    build(train_imgs, cache, args.out, "train", args.min_face_frac)
    build(val_imgs, cache, args.out, "val", args.min_face_frac)


if __name__ == "__main__":
    main()
