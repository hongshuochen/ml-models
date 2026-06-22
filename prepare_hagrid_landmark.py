#!/usr/bin/env python3
"""Build a HaGRID hand-landmark dataset for the stage-2 regressor.

Converts HaGRID's bundled MediaPipe hand_landmarks (21 [x,y], image-normalized) +
hand bboxes into YOLO-pose labels (class cx cy w h + 21*(x,y,v)), the same format
train_hand_landmark.py's HandCropDataset reads. Images symlinked. Uses the official
subject-disjoint splits (train/val). HaGRID landmarks are MediaPipe pseudo-labels,
so the regressor is bounded by MediaPipe quality (but gains webcam-domain diversity).

  uv run python prepare_hagrid_landmark.py --ann-dir datasets/hagrid_raw/annotations/train \
      --img-root datasets/hagrid_raw/HaGRIDv2_dataset_512 --per-gesture-limit 1500 --shuffle
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path("/home/max/2026/ml-models")
NK = 21


def clip01(v):
    return max(0.0, min(1.0, float(v)))


def build_index(img_root):
    return {p.stem: p for p in Path(img_root).rglob("*.jpg")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", default="datasets/hagrid-landmark")
    ap.add_argument("--target-split", default="train")
    ap.add_argument("--per-gesture-limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_img = ROOT / args.out / "images" / args.target_split
    out_lbl = ROOT / args.out / "labels" / args.target_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    print(f"indexing {args.img_root} ...", flush=True)
    index = build_index(args.img_root)
    print(f"  {len(index)} images", flush=True)

    n_img = n_hands = 0
    for gi, jf in enumerate(sorted(Path(args.ann_dir).glob("*.json"))):
        g = jf.stem
        items = list(json.load(open(jf)).items())
        if args.shuffle:
            random.Random(args.seed + gi).shuffle(items)
        gc = 0
        for img_id, e in items:
            if args.per_gesture_limit and gc >= args.per_gesture_limit:
                break
            ip = index.get(img_id)
            if ip is None:
                continue
            boxes = e.get("bboxes", [])
            lms = e.get("hand_landmarks", [])
            lines = []
            for i, box in enumerate(boxes):
                if i >= len(lms) or not lms[i] or len(lms[i]) != NK:
                    continue
                x, y, w, h = box
                cx, cy, ww, hh = clip01(x + w / 2), clip01(y + h / 2), clip01(w), clip01(h)
                if ww <= 0 or hh <= 0:
                    continue
                kpts = " ".join(f"{clip01(px):.6f} {clip01(py):.6f} 2" for px, py in lms[i])
                lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f} {kpts}")
            if not lines:
                continue
            stem = f"hagrid__{g}__{img_id}"
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            dst = out_img / f"{stem}.jpg"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(ip.resolve())
            n_img += 1
            n_hands += len(lines)
            gc += 1
        print(f"  [{g}] +{gc} (total imgs {n_img})", flush=True)

    print(f"done: images={n_img} hands={n_hands}")


if __name__ == "__main__":
    main()
