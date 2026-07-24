#!/usr/bin/env python3
"""Draw YOLO labels on their images so you can eyeball an auto-labeled set (e.g. SAM's out_sam/).

Reads <dir>/images/*.jpg + <dir>/labels/*.txt (+ classes.txt), draws each box colored by class,
and writes annotated copies to <dir>/_viz/. --seg also overlays the mask polygons from labels_seg/.

    python golf/viz_labels.py out_sam --n 20          # 20 random labeled frames
    python golf/viz_labels.py out_sam --seg           # also draw masks
"""
import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np

COL = [(255, 255, 0), (11, 158, 245), (94, 197, 34), (200, 120, 255)]  # cyan, amber, green, violet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="dir with images/ + labels/ (+ classes.txt, labels_seg/)")
    ap.add_argument("--n", type=int, default=20, help="how many random labeled frames to draw")
    ap.add_argument("--seg", action="store_true", help="also overlay mask polygons from labels_seg/")
    ap.add_argument("--out", default="_viz", help="output subdir under <dir>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D = Path(args.dir)
    names = (D / "classes.txt").read_text().split() if (D / "classes.txt").is_file() else []
    outdir = D / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(glob.glob(str(D / "images" / "*.jpg")))
    random.Random(args.seed).shuffle(imgs)
    done = 0
    for p in imgs[: args.n]:
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        stem = Path(p).stem
        seg = D / "labels_seg" / f"{stem}.txt"
        if args.seg and seg.is_file():
            overlay = im.copy()
            for ln in seg.read_text().splitlines():
                t = ln.split()
                c = int(t[0]); pts = np.array([(float(t[k]) * w, float(t[k + 1]) * h)
                                               for k in range(1, len(t) - 1, 2)], np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(overlay, [pts], COL[c % len(COL)])
            im = cv2.addWeighted(overlay, 0.4, im, 0.6, 0)
        lp = D / "labels" / f"{stem}.txt"
        n = 0
        if lp.is_file():
            for ln in lp.read_text().splitlines():
                t = ln.split()
                if len(t) < 5:
                    continue
                c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                col = COL[c % len(COL)]
                cv2.rectangle(im, (x1, y1), (x2, y2), col, 3)
                cv2.putText(im, names[c] if c < len(names) else str(c), (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                n += 1
        cv2.putText(im, f"{stem}  ({n} boxes)", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(outdir / Path(p).name), im)
        done += 1
    print(f"wrote {done} annotated images to {outdir}/  — open them to check the labels")


if __name__ == "__main__":
    main()
