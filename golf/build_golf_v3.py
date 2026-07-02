"""Build golf_v3: merge multiple vetted golf sources, de-dup ACROSS all of them, leakage-free split.

Sources (each with a per-source class remap into ours {ball:0, club_head:1}):
  - golf-driver-tracker : 0->ball, 2->club_head   (drop 1=handle)   [full-scene, both classes]
  - Tidbury golf-club-head : 0->club_head          [clean full-scene club_head, varied settings]
  - uentu golf-ball-uentu : 0->ball                [full-scene balls incl. distant/small; 416px, some noise]
    -> uentu is capped (UENTU_CAP) before hashing to keep clustering tractable + avoid ball domination.

Same pipeline as build_golf_v2: pHash -> union-find near-dup clusters (hamming<=THRESH) across ALL sources
-> assign each cluster wholly to one split -> cap frames/cluster -> verify 0 cross-split leakage.

Run: uv run python golf/build_golf_v3.py
"""
import os, random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PRE = REPO / "datasets/golf_preview"
DST = REPO / "datasets/golf_v3"
random.seed(0)

# (glob-dir, {src_cls: our_cls})
SOURCES = [
    (next(PRE.glob("golf-driver-tracker*")), {0: 0, 2: 1}),
    (next(PRE.glob("golf-club-head*")),      {0: 1}),          # Tidbury: club head
    (next(PRE.glob("golf-ball-uentu*")),     {0: 0}),          # uentu: golfball
]
UENTU_CAP = 8000
NAMES = {0: "ball", 1: "club_head"}
THRESH = 5
CAP_TRAIN = 6
POP = np.array([bin(i).count("1") for i in range(256)], np.uint16)


def phash(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    d = cv2.dct(cv2.resize(im, (32, 32)).astype(np.float32))[:8, :8]
    return np.packbits((d.flatten() > np.median(d.flatten()[1:])).astype(np.uint8))


def remap(text, m):
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5 and int(float(p[0])) in m:
            out.append(" ".join([str(m[int(float(p[0]))])] + p[1:]))
    return out


def main():
    # gather (image_path, remap) pairs
    items = []
    for root, m in SOURCES:
        imgs = [p for p in root.rglob("*.jpg") if "/images/" in str(p).replace("\\", "/")]
        if "uentu" in root.name and len(imgs) > UENTU_CAP:
            random.shuffle(imgs); imgs = imgs[:UENTU_CAP]
        print(f"{root.name}: {len(imgs)} imgs")
        for p in imgs:
            items.append((p, m))

    H, keep, maps = [], [], []
    for p, m in items:
        h = phash(p)
        if h is not None:
            H.append(h); keep.append(p); maps.append(m)
    H = np.array(H); n = len(keep)
    print(f"hashed {n} images; clustering (hamming<={THRESH})...")

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        d = POP[np.bitwise_xor(H[i + 1:], H[i])].sum(1)
        for off in np.where(d <= THRESH)[0]:
            a, b = find(i), find(i + 1 + int(off))
            if a != b: parent[a] = b
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    print(f"clusters (unique scenes): {len(clusters)} from {n} frames")

    for s in ("train", "valid", "test"):
        (DST / s / "images").mkdir(parents=True, exist_ok=True)
        (DST / s / "labels").mkdir(parents=True, exist_ok=True)
    def split_of(root):
        b = int(H[root].sum()) % 100
        return "train" if b < 85 else ("valid" if b < 95 else "test")

    stats = {s: defaultdict(int) for s in ("train", "valid", "test")}
    nimg = {s: 0 for s in stats}
    shashes = {s: [] for s in stats}
    for root, members in clusters.items():
        s = split_of(root); cap = CAP_TRAIN if s == "train" else 1
        for idx in members[:cap]:
            img = keep[idx]; m = maps[idx]
            lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            lines = remap(lbl.read_text(), m) if lbl.exists() else []
            # unique name (source prefix avoids collisions across datasets)
            name = f"{img.parent.parent.parent.name[:6]}_{img.name}"
            link = DST / s / "images" / name
            if link.exists() or link.is_symlink(): link.unlink()
            os.symlink(img.resolve(), link)
            (DST / s / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            nimg[s] += 1
            for ln in lines: stats[s][int(ln.split()[0])] += 1
            shashes[s].append(H[idx])

    for s in ("train", "valid", "test"):
        print(f"[{s}] {nimg[s]} imgs | ball {stats[s][0]}  club_head {stats[s][1]}")
    tr = np.array(shashes["train"])
    for s in ("valid", "test"):
        arr = np.array(shashes[s]); leak = sum(1 for h in arr if POP[np.bitwise_xor(tr, h)].sum(1).min() <= THRESH)
        print(f"LEAKAGE {s}->train: {leak}/{len(arr)} ({100*leak/max(1,len(arr)):.1f}%)")

    (REPO / "golf_v3.yaml").write_text(
        "path: /home/max/2026/ml-models/datasets\ntrain: golf_v3/train/images\n"
        "val: golf_v3/valid/images\ntest: golf_v3/test/images\nnames:\n  0: ball\n  1: club_head\n"
    )
    print(f"wrote {DST} + golf_v3.yaml")


if __name__ == "__main__":
    main()
