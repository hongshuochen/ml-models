#!/usr/bin/env python3
"""Build a 2-class detection dataset (0=face, 1=hand) by combining:
  - WIDER FACE boxes      -> class 0  (labels already '0 cx cy w h')
  - hand-keypoints boxes  -> class 1  (take box, drop keypoints, relabel)
Images are symlinked (no copying); labels are generated. Filenames are
prefixed (face__ / hand__) and event subdirs flattened to avoid collisions.
"""
import shutil
from pathlib import Path

ROOT = Path("/home/max/2026/ml-models")
FACE = ROOT / "datasets/widerface"
HAND = ROOT / "datasets/hand-keypoints"
OUT = ROOT / "datasets/face-hand"


def reset():
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def add_faces(split):
    n = 0
    for lbl in (FACE / "labels" / split).rglob("*.txt"):
        rel = lbl.relative_to(FACE / "labels" / split)
        img = FACE / "images" / split / rel.with_suffix(".jpg")
        if not img.exists():
            continue
        stem = "face__" + "__".join(rel.with_suffix("").parts)
        (OUT / "images" / split / f"{stem}.jpg").symlink_to(img.resolve())
        # WIDER labels are already class 0 -> copy verbatim
        (OUT / "labels" / split / f"{stem}.txt").write_text(lbl.read_text())
        n += 1
    return n


def add_hands(split):
    n = 0
    for lbl in (HAND / "labels" / split).glob("*.txt"):
        img = HAND / "images" / split / f"{lbl.stem}.jpg"
        if not img.exists():
            continue
        out_lines = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            # class -> 1 (hand), keep box, drop keypoints
            out_lines.append("1 " + " ".join(p[1:5]))
        stem = f"hand__{lbl.stem}"
        (OUT / "images" / split / f"{stem}.jpg").symlink_to(img.resolve())
        (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        n += 1
    return n


reset()
for split in ("train", "val"):
    f = add_faces(split)
    h = add_hands(split)
    print(f"[{split}] faces={f} hands={h} total={f + h}")
print("done ->", OUT)
