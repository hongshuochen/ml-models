#!/usr/bin/env python3
"""Build a HOLE-ONLY dataset from the public golf ball+hole Roboflow sets, for the bootstrap
hole-teacher (GOLF_HOLE_PLAN.md Phase 1).

The teacher exists ONLY to pre-label the `hole` class on our egocentric putting footage — its ball
and club_head are never used (ball comes from golf_ego_v2, which is far better on our domain; the
public sets have no club_head at all). So we train it as a literal 1-class {hole} detector:
  * keep only `golf-hole` boxes (public class 1), rewrite them to class 0 = hole;
  * DROP the `golf-ball` boxes (public class 0) — a hole-only model doesn't predict ball, so an
    unlabelled ball is just correct background;
  * keep ball-only / hole-less images as NEGATIVES (they teach "a ball / bright speck is not a
    hole", cutting the teacher's false positives).
Uses each source's own train/valid split (no leakage). Images are symlinked; labels are rewritten.

Sources (download first via golf/preview_datasets.py or the Phase-0 step; both CC BY 4.0):
  datasets/golf_preview/golf-ball-and-hole-detection-v6         (bosharluke, 415 imgs)
  datasets/golf_preview/golf-ball-and-hole-detection-1k7-v6     (sai-gon, 1171 imgs, 2x2-tiled)

Out: datasets/golf_hole/{images,labels}/{train,val} + golf_hole_teacher.yaml
Run: uv run python golf/prepare_hole_teacher.py
"""
import os
from pathlib import Path

ROOT = Path("/home/max/2026/ml-models")
SOURCES = [
    ROOT / "datasets/golf_preview/golf-ball-and-hole-detection-v6",
    ROOT / "datasets/golf_preview/golf-ball-and-hole-detection-1k7-v6",
]
HOLE_SRC_CLASS = 1          # public label index for golf-hole
OUT = ROOT / "datasets/golf_hole"
SPLIT_MAP = {"train": "train", "valid": "val"}   # ignore each set's tiny 'test' split


def main():
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_img = {"train": 0, "val": 0}
    n_hole = {"train": 0, "val": 0}
    n_neg = {"train": 0, "val": 0}
    for si, src in enumerate(SOURCES):
        if not src.exists():
            raise SystemExit(f"missing {src} — run Phase 0 download first")
        tag = f"s{si}"          # deterministic, collision-safe filename prefix per source
        for sub, split in SPLIT_MAP.items():
            idir = src / sub / "images"
            if not idir.is_dir():
                continue
            for img in sorted(idir.iterdir()):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl = src / sub / "labels" / (img.stem + ".txt")
                holes = []
                if lbl.exists():
                    for ln in lbl.read_text().splitlines():
                        p = ln.split()
                        if len(p) >= 5 and int(float(p[0])) == HOLE_SRC_CLASS:
                            holes.append("0 " + " ".join(p[1:5]))   # rewrite hole -> class 0
                name = f"{tag}_{img.stem}{img.suffix}"
                link = OUT / "images" / split / name
                if not link.exists():
                    os.symlink(img.resolve(), link)
                (OUT / "labels" / split / f"{tag}_{img.stem}.txt").write_text(
                    "\n".join(holes) + ("\n" if holes else ""))
                n_img[split] += 1
                n_hole[split] += len(holes)
                n_neg[split] += (len(holes) == 0)

    yaml = OUT / "golf_hole_teacher.yaml"
    yaml.write_text(
        "# hole-only bootstrap teacher (public golf ball+hole sets, ball dropped)\n"
        f"path: {OUT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: hole\n")
    print(f"train: {n_img['train']} imgs, {n_hole['train']} holes, {n_neg['train']} negatives")
    print(f"val:   {n_img['val']} imgs, {n_hole['val']} holes, {n_neg['val']} negatives")
    print(f"-> {yaml}")


if __name__ == "__main__":
    main()
