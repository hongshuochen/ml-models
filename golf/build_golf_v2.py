"""Build a de-duplicated, LEAKAGE-FREE 2-class golf dataset from golf-driver-tracker.

Problem: the Roboflow split is video-sourced -> ~48% of val/test frames have a near-duplicate
in train (pHash), so metrics were inflated. Fix here:
  1. Pool train+valid+test, remap to {ball(0), club_head(1)} (drop club-handle).
  2. pHash every image, union-find cluster near-duplicate frames (hamming <= THRESH = same swing).
  3. Assign each CLUSTER wholly to one split (deterministic) -> no near-dup crosses train/val/test.
  4. Thin redundancy: cap frames/cluster (train CAP_TRAIN; val/test 1 each = diverse eval).
  5. Verify: report 0 cross-split near-dups + class counts.

Output: datasets/golf_v2/{train,valid,test}/{images,labels} + golf_v2.yaml
Run: uv run python golf/build_golf_v2.py
"""
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "datasets/golf_preview/golf-driver-tracker-v2"
DST = REPO / "datasets/golf_v2"
REMAP = {0: 0, 2: 1}          # driver-tracker cls -> ours; drop 1 (handle)
NAMES = {0: "ball", 1: "club_head"}
THRESH = 5                     # hamming <= THRESH => near-duplicate (same cluster)
CAP_TRAIN = 6                  # max frames kept per cluster in train
POP = np.array([bin(i).count("1") for i in range(256)], np.uint16)


def phash(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    d = cv2.dct(cv2.resize(im, (32, 32)).astype(np.float32))[:8, :8]
    return np.packbits((d.flatten() > np.median(d.flatten()[1:])).astype(np.uint8))


def remap_label(text):
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5 and int(float(p[0])) in REMAP:
            out.append(" ".join([str(REMAP[int(float(p[0]))])] + p[1:]))
    return out


def main():
    # 1) gather all images + hashes
    imgs = [p for p in SRC.rglob("*.jpg") if "/images/" in str(p).replace("\\", "/")]
    print(f"pooled images: {len(imgs)}")
    H, keep = [], []
    for p in imgs:
        h = phash(p)
        if h is not None:
            H.append(h); keep.append(p)
    H = np.array(H)
    n = len(keep)

    # 2) union-find cluster by hamming <= THRESH
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        d = POP[np.bitwise_xor(H[i + 1:], H[i])].sum(1)  # compare i to all j>i
        for off in np.where(d <= THRESH)[0]:
            union(i, i + 1 + int(off))
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    print(f"clusters (unique swings/scenes): {len(clusters)}  (from {n} frames)")

    # 3) assign each cluster to a split deterministically (~85/10/5 by cluster)
    def split_of(root):
        b = int(H[root].sum()) % 100          # deterministic bucket from the hash
        return "train" if b < 85 else ("valid" if b < 95 else "test")

    for s in ("train", "valid", "test"):
        (DST / s / "images").mkdir(parents=True, exist_ok=True)
        (DST / s / "labels").mkdir(parents=True, exist_ok=True)

    stats = {s: {"imgs": 0, "cls": defaultdict(int)} for s in ("train", "valid", "test")}
    split_hashes = {s: [] for s in ("train", "valid", "test")}
    for root, members in clusters.items():
        s = split_of(root)
        cap = CAP_TRAIN if s == "train" else 1     # eval splits: 1 frame/cluster
        for idx in members[:cap]:
            img = keep[idx]
            lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            lines = remap_label(lbl.read_text()) if lbl.exists() else []
            link = DST / s / "images" / img.name
            if link.exists() or link.is_symlink(): link.unlink()
            os.symlink(img.resolve(), link)
            (DST / s / "labels" / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[s]["imgs"] += 1
            for ln in lines: stats[s]["cls"][int(ln.split()[0])] += 1
            split_hashes[s].append(H[idx])

    for s in ("train", "valid", "test"):
        c = stats[s]["cls"]
        print(f"[{s}] {stats[s]['imgs']} imgs | ball {c[0]}  club_head {c[1]}")

    # 5) verify: cross-split near-dup leakage should be ~0
    tr = np.array(split_hashes["train"])
    for s in ("valid", "test"):
        arr = np.array(split_hashes[s]); leak = 0
        for h in arr:
            if POP[np.bitwise_xor(tr, h)].sum(1).min() <= THRESH: leak += 1
        print(f"LEAKAGE {s}->train (hamming<={THRESH}): {leak}/{len(arr)} ({100*leak/max(1,len(arr)):.1f}%)")

    (REPO / "golf_v2.yaml").write_text(
        "path: /home/max/2026/ml-models/datasets\ntrain: golf_v2/train/images\n"
        "val: golf_v2/valid/images\ntest: golf_v2/test/images\nnames:\n  0: ball\n  1: club_head\n"
    )
    print(f"wrote {DST} + golf_v2.yaml")


if __name__ == "__main__":
    main()
