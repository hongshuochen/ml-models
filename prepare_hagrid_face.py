#!/usr/bin/env python3
"""Build a HaGRID 5-point FACE-landmark dataset for face alignment.

Faces are detected by InsightFace SCRFD-10G; per image we keep the LARGEST face and its
5 keypoints — [left_eye, right_eye, nose, left_mouth, right_mouth] (the ArcFace alignment
points). Output is YOLO-pose-style labels (class 0 + box + 5*(x,y,v)) so
train_hand_landmark.py --num-kpts 5 can consume it directly.

Subject-disjoint train/val: reuse HaGRID's official hand-annotation splits (they list the
image UUIDs per split, split by user_id). Images symlinked.

  uv run python prepare_hagrid_face.py --device cpu --shuffle
"""
import argparse
import json
import random
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

ROOT = Path("/home/max/2026/ml-models")
ANN = ROOT / "datasets/hagrid_raw/annotations"  # official hand annots -> UUIDs per split


def clip01(v):
    return max(0.0, min(1.0, float(v)))


def build_index(img_root):
    return {p.stem: p for p in Path(img_root).rglob("*.jpg")}


def load_app(device):
    prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"])
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=prov)
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", default="datasets/hagrid_raw/HaGRIDv2_dataset_512")
    ap.add_argument("--out", default="datasets/hagrid-face")
    ap.add_argument("--per-gesture-limit", type=int, default=1500)
    ap.add_argument("--val-per-gesture", type=int, default=200)
    ap.add_argument("--det-score", type=float, default=0.5)
    ap.add_argument("--min-face-area", type=float, default=0.004)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    app = load_app(args.device)
    print(f"indexing {args.img_root} ...", flush=True)
    index = build_index(args.img_root)
    print(f"  {len(index)} images", flush=True)

    for split, limit in (("train", args.per_gesture_limit), ("val", args.val_per_gesture)):
        out_img = ROOT / args.out / "images" / split
        out_lbl = ROOT / args.out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        n_img = 0
        for gi, jf in enumerate(sorted((ANN / split).glob("*.json"))):
            g = jf.stem
            uuids = list(json.load(open(jf)).keys())
            if args.shuffle:
                random.Random(args.seed + gi).shuffle(uuids)
            gc = 0
            for uuid in uuids:
                if limit and gc >= limit:
                    break
                ip = index.get(uuid)
                if ip is None:
                    continue
                img = cv2.imread(str(ip))
                if img is None:
                    continue
                H, W = img.shape[:2]
                # largest face passing det_score + min-area, with valid 5 kps
                best, best_area = None, 0.0
                for f in app.get(img):
                    if f.det_score < args.det_score or f.kps is None:
                        continue
                    x1, y1, x2, y2 = f.bbox
                    w, h = (x2 - x1) / W, (y2 - y1) / H
                    if w <= 0 or h <= 0 or w * h < args.min_face_area:
                        continue
                    if w * h > best_area:
                        best_area, best = w * h, f
                if best is None:
                    continue
                x1, y1, x2, y2 = best.bbox
                cx, cy = clip01((x1 + x2) / 2 / W), clip01((y1 + y2) / 2 / H)
                bw, bh = clip01((x2 - x1) / W), clip01((y2 - y1) / H)
                kp = " ".join(f"{clip01(px / W):.6f} {clip01(py / H):.6f} 2" for px, py in best.kps)
                stem = f"hagrid__{g}__{uuid}"
                (out_lbl / f"{stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp}\n")
                dst = out_img / f"{stem}.jpg"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(ip.resolve())
                n_img += 1
                gc += 1
            print(f"  [{split}/{g}] +{gc} (total {n_img})", flush=True)
        print(f"{split}: faces={n_img}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
