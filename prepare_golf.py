"""Build the 2-class {ball, club_head} golf detection dataset.

Source: golf-driver-tracker (Roboflow salo-levy-nlqrn v2, CC BY 4.0) downloaded under
        datasets/golf_preview/golf-driver-tracker-v2/{train,valid,test}/{images,labels}.
Remap:  0 golf-ball        -> 0 ball
        2 golf club-head   -> 1 club_head
        1 golf club-handle -> DROPPED
Output: datasets/golf/{train,valid,test}/{images,labels}  (images symlinked, labels rewritten).

Run:  uv run python prepare_golf.py
"""
import os
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "datasets/golf_preview/golf-driver-tracker-v2"
DST = REPO / "datasets/golf"
REMAP = {0: 0, 2: 1}  # src class -> dst class; anything not here is dropped
NAMES = {0: "ball", 1: "club_head"}
SPLITS = ["train", "valid", "test"]

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def remap_label(text):
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        if c in REMAP:
            out.append(" ".join([str(REMAP[c])] + p[1:]))
    return out


def main():
    if not SRC.exists():
        raise SystemExit(f"source not found: {SRC}\nDownload it first (golf/preview_datasets.py).")
    grand = {}
    for split in SPLITS:
        s_img, s_lbl = SRC / split / "images", SRC / split / "labels"
        if not s_img.exists():
            print(f"[{split}] missing, skip")
            continue
        d_img, d_lbl = DST / split / "images", DST / split / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)
        cls_count = Counter()
        n_img = n_bg = 0
        for img in sorted(s_img.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = s_lbl / (img.stem + ".txt")
            lines = remap_label(lbl.read_text()) if lbl.exists() else []
            # symlink image (absolute target so it resolves from any cwd)
            link = d_img / img.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(img.resolve(), link)
            (d_lbl / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            if not lines:
                n_bg += 1
            for ln in lines:
                cls_count[int(ln.split()[0])] += 1
        grand[split] = (n_img, n_bg, cls_count)
        pretty = {NAMES[k]: v for k, v in sorted(cls_count.items())}
        print(f"[{split}] {n_img} images ({n_bg} background/no-object) | instances {pretty}")
    tot_img = sum(v[0] for v in grand.values())
    tot_cls = Counter()
    for _, _, c in grand.values():
        tot_cls.update(c)
    print(f"\nTOTAL {tot_img} images | instances { {NAMES[k]: v for k, v in sorted(tot_cls.items())} }")
    print(f"Output: {DST}  (config: golf.yaml)")


if __name__ == "__main__":
    main()
