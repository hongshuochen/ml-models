"""Build the golf egocentric v1 YOLO dataset from the Label Studio export.

Splits by VIDEO/SESSION (never random per-frame: consecutive frames of one swing are near-dup ->
random split leaks; same lesson as the Roboflow golf sets). Whole videos go to train or val.

In:  <export>/labels/*.txt (corrected human labels) + datasets/golf_frames/*.jpg
Out: datasets/golf_ego_v1/{images,labels}/{train,val}  (+ golf_ego_v1.yaml at repo root)
Run: uv run python golf/build_golf_v1_dataset.py [export_labels_dir]
"""
import glob
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

EXPORT_LABELS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/tmp/claude-1000/-home-max-2026-ml-models/ef3aba0d-9d1b-4678-8a09-02d72865ef6e/scratchpad/export_v2/labels")
FRAMES = Path("datasets/golf_frames")
OUT = Path("datasets/golf_ego_v1")
YAML = Path("golf_ego_v1.yaml")
VAL_FRAC = 0.15
SEED = 7


def main():
    labels = sorted(glob.glob(str(EXPORT_LABELS / "*.txt")))
    byvid = defaultdict(list)
    for lp in labels:
        stem = Path(lp).stem
        vid = re.sub(r"_f\d+$", "", stem)
        byvid[vid].append(stem)
    vids = sorted(byvid)
    # deterministic shuffle (no Math.random needed): sort by a hash of name+SEED
    vids.sort(key=lambda v: hash((v, SEED)))
    total = sum(len(byvid[v]) for v in vids)
    # accumulate videos into val until ~VAL_FRAC of frames
    val_vids, n = set(), 0
    for v in vids:
        if n < VAL_FRAC * total:
            val_vids.add(v); n += len(byvid[v])
    # build dirs
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = OUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    cnt = {"train": 0, "val": 0}
    for v in vids:
        split = "val" if v in val_vids else "train"
        for stem in byvid[v]:
            img = FRAMES / f"{stem}.jpg"
            if not img.exists():
                continue
            link = OUT / f"images/{split}/{stem}.jpg"
            if not link.exists():
                os.symlink(img.resolve(), link)
            shutil.copy(EXPORT_LABELS / f"{stem}.txt", OUT / f"labels/{split}/{stem}.txt")
            cnt[split] += 1

    YAML.write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n  1: club_head\n")
    print(f"golf_ego_v1 dataset: train {cnt['train']} / val {cnt['val']}  "
          f"({len(vids)-len(val_vids)} train vids, {len(val_vids)} val vids, split by video)")
    print(f"  yaml -> {YAML.resolve()}")


if __name__ == "__main__":
    main()
