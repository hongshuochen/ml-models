"""De-duplicate extracted golf frames (keep 1 per near-duplicate cluster).

2fps extraction produces many near-identical frames (a held address = several copies).
pHash + union-find cluster (hamming <= THRESH) -> keep ONE representative per cluster.
No detection-based filtering (that would bias the set toward what the model already knows).

In:  datasets/golf_frames/*.jpg
Out: datasets/golf_frames_dedup/  (symlinks to the kept representatives)
Run: uv run python golf/dedup_frames.py [THRESH]   (default THRESH=5)
"""
import glob, os, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SRC = Path("datasets/golf_frames")
DST = Path("datasets/golf_frames_dedup")
THRESH = int(sys.argv[1]) if len(sys.argv) > 1 else 5
POP = np.array([bin(i).count("1") for i in range(256)], np.uint16)


def phash(p):
    # fast 1/8 JPEG decode is plenty for a 32x32 DCT hash
    im = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if im is None:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    d = cv2.dct(cv2.resize(im, (32, 32)).astype(np.float32))[:8, :8]
    return np.packbits((d.flatten() > np.median(d.flatten()[1:])).astype(np.uint8))


def main():
    files = sorted(glob.glob(str(SRC / "*.jpg")))
    print(f"frames: {len(files)}  | hashing (fast decode)...")
    H, keep = [], []
    for i, p in enumerate(files):
        h = phash(p)
        if h is not None:
            H.append(h); keep.append(p)
        if (i + 1) % 5000 == 0:
            print(f"  hashed {i+1}/{len(files)}")
    n = len(keep)
    # group by video (filename prefix date_golf_NNN), keep chronological order
    import re
    byvid = defaultdict(list)
    for idx, p in enumerate(keep):
        vid = re.sub(r"_f\d+\.jpg$", "", Path(p).name)
        byvid[vid].append(idx)

    # PER-VIDEO consecutive dedup: keep a frame only if it differs from the last KEPT
    # frame in the same video by > THRESH (removes held-pose runs, keeps real changes).
    if DST.exists():
        for f in DST.glob("*.jpg"):
            f.unlink()
    DST.mkdir(parents=True, exist_ok=True)
    kept_idx = []
    for vid, idxs in byvid.items():
        idxs.sort(key=lambda i: keep[i])     # chronological by filename
        last = None
        for i in idxs:
            if last is None or int(POP[np.bitwise_xor(H[i], H[last])].sum()) > THRESH:
                kept_idx.append(i); last = i
    for i in kept_idx:
        p = Path(keep[i]); link = DST / p.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(p.resolve(), link)

    kept = len(kept_idx)
    print(f"\n=== PER-VIDEO consecutive dedup (hamming<={THRESH}) ===")
    print(f"  input frames : {n}  across {len(byvid)} videos")
    print(f"  unique kept  : {kept}  ({100*kept/n:.0f}%  -> {n-kept} held-pose dups removed)")
    print(f"  kept/video   : median {int(np.median([sum(1 for i in idxs if i in set(kept_idx)) for idxs in byvid.values()]))}")
    print(f"  output: {DST}/")


if __name__ == "__main__":
    main()
