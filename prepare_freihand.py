#!/usr/bin/env python3
"""FreiHAND -> YOLO-pose 21-kpt hand-landmark dataset (adds back-of-hand / dorsal views).

Egocentric (glasses) cameras mostly see the BACK of the hand, which HaGRID (palm-toward-camera
gestures) under-covers. FreiHAND has lots of dorsal / varied hand orientations with real 21-joint
labels. We project its 3D joints to 2D with the provided intrinsics, build a hand bbox from the
joints, and write <out>/{images,labels}/{train,val} in the same format as our other hand-landmark
datasets. FreiHAND's joint order already matches MediaPipe (wrist, thumb, index, middle, ring,
pinky), so no reordering — but run `--viz N` first to eyeball it.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
         [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]]


def project(xyz, K):
    uv = (K @ xyz.T).T          # 21x3
    return uv[:, :2] / uv[:, 2:3]   # 21x2 pixel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/freihand_raw/FreiHAND_pub_v2")
    ap.add_argument("--out", default="datasets/freihand")
    ap.add_argument("--versions", type=int, default=2, help="background versions 1..4 to use (x32560 each)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--pad", type=float, default=0.20, help="bbox padding fraction around the joints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz", type=int, default=0, help="save N skeleton-overlay images then exit (sanity check)")
    args = ap.parse_args()

    root = Path(args.root)
    K = np.array(json.load(open(root / "training_K.json")), np.float32)        # 32560 x 3 x 3
    xyz = np.array(json.load(open(root / "training_xyz.json")), np.float32)    # 32560 x 21 x 3
    n_uniq = len(K)
    rgb = root / "training" / "rgb"

    if args.viz:
        Path("freihand_check").mkdir(exist_ok=True)
        for i in range(args.viz):
            im = Image.open(rgb / f"{i:08d}.jpg").convert("RGB")
            pts = project(xyz[i], K[i])
            dr = ImageDraw.Draw(im)
            for a, b in EDGES:
                dr.line([pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1]], fill=(0, 255, 255), width=2)
            for x, y in pts:
                dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 180, 0))
            im.save(f"freihand_check/{i:08d}.png")
        print(f"saved {args.viz} viz to freihand_check/")
        return

    total = n_uniq * max(1, min(4, args.versions))
    if args.limit:
        total = min(total, args.limit)
    idx = list(range(total))
    random.Random(args.seed).shuffle(idx)
    n_val = int(total * args.val_frac)
    val_set = set(idx[:n_val])

    for split in ("train", "val"):
        (Path(args.out) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = 0
    for i in idx:
        ann = i % n_uniq
        img_path = rgb / f"{i:08d}.jpg"
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        pts = project(xyz[ann], K[ann])
        x, y = pts[:, 0], pts[:, 1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        side = max(x.max() - x.min(), y.max() - y.min()) * (1 + 2 * args.pad)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        kp = np.stack([x / W, y / H, np.full(21, 2.0, np.float32)], 1)  # vis = 2
        line = (f"0 {cx / W:.6f} {cy / H:.6f} {side / W:.6f} {side / H:.6f} "
                + " ".join(f"{v:.6f}" for v in kp.reshape(-1)))
        split = "val" if i in val_set else "train"
        name = f"frei{i:08d}"
        dst_img = Path(args.out) / "images" / split / f"{name}.jpg"
        if not dst_img.exists():
            os.symlink(img_path.resolve(), dst_img)
        (Path(args.out) / "labels" / split / f"{name}.txt").write_text(line + "\n")
        kept += 1
    print(f"wrote {kept} samples ({n_val} val) -> {args.out}")


if __name__ == "__main__":
    main()
