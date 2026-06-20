#!/usr/bin/env python3
"""Pseudo-label HaGRID for 2-class (0=face, 1=hand) YOLO detection.

  - HAND boxes (class 1): from HaGRID annotations (normalized top-left xywh).
  - FACE boxes (class 0): pseudo-labeled with InsightFace SCRFD-10G (buffalo_l),
    kept when det_score >= --det-score and area >= --min-face-area.

RESEARCH USE ONLY: InsightFace pretrained models are non-commercial; a model
trained on these labels inherits that restriction. For commercial use swap the
teacher for YuNet (MIT) or MediaPipe BlazeFace (Apache-2.0).

Works with either layout (annotations are per-gesture <gesture>.json keyed by
image UUID; images located by a global UUID->path index under --img-root):
  official:  --ann-dir datasets/hagrid_raw/annotations/train  --img-root datasets/hagrid_raw/images
  cj-mills:  --ann-dir .../ann_train_val  --img-root .../hagrid_250k

Writes hagrid__<gesture>__<id>.{jpg-symlink,txt} into <out>/images|labels/<split>.
Usage:
  uv run python prepare_hagrid.py --ann-dir <d> --img-root <d> --per-gesture-limit 4000
"""
import argparse
import json
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis

ROOT = Path("/home/max/2026/ml-models")


def clip01(v):
    return max(0.0, min(1.0, float(v)))


def build_index(img_root):
    """Map image-UUID (filename stem) -> path, for all jpgs under img_root."""
    idx = {}
    for p in Path(img_root).rglob("*.jpg"):
        idx[p.stem] = p
    return idx


def load_app(device):
    prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"])
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=prov)
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True, help="dir of per-gesture <gesture>.json (UUID-keyed)")
    ap.add_argument("--img-root", required=True, help="root searched recursively for <UUID>.jpg")
    ap.add_argument("--gestures", nargs="+", default=None, help="subset of gestures (default: all jsons)")
    ap.add_argument("--out", default="datasets/hagrid_det")
    ap.add_argument("--target-split", default="train")
    ap.add_argument("--per-gesture-limit", type=int, default=0, help="cap images per gesture (0=all)")
    ap.add_argument("--det-score", type=float, default=0.5)
    ap.add_argument("--min-face-area", type=float, default=0.004,
                    help="drop faces smaller than this fraction of image area (kills tiny photo faces)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--viz", type=int, default=0)
    args = ap.parse_args()

    app = load_app(args.device)
    out_img = ROOT / args.out / "images" / args.target_split
    out_lbl = ROOT / args.out / "labels" / args.target_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    viz_dir = ROOT / "hagrid_viz"
    if args.viz:
        viz_dir.mkdir(exist_ok=True)

    print(f"indexing images under {args.img_root} ...")
    index = build_index(args.img_root)
    print(f"  {len(index)} images indexed")

    ann_dir = Path(args.ann_dir)
    jsons = sorted(ann_dir.glob("*.json"))
    if args.gestures:
        jsons = [j for j in jsons if j.stem in args.gestures]

    n_img = n_hand = n_face = n_noface = n_viz = 0
    for jf in jsons:
        gesture = jf.stem
        data = json.load(open(jf))
        g_count = 0
        for img_id, e in data.items():
            if args.per_gesture_limit and g_count >= args.per_gesture_limit:
                break
            img_path = index.get(img_id)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            lines = []
            for box in e.get("bboxes", []):
                x, y, w, h = box
                cx, cy, ww, hh = clip01(x + w / 2), clip01(y + h / 2), clip01(w), clip01(h)
                if ww > 0 and hh > 0:
                    lines.append(f"1 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            n_hand += len(lines)
            nf = 0
            for f in app.get(img):
                if f.det_score < args.det_score:
                    continue
                x1, y1, x2, y2 = f.bbox
                cx, cy = clip01((x1 + x2) / 2 / W), clip01((y1 + y2) / 2 / H)
                ww, hh = clip01((x2 - x1) / W), clip01((y2 - y1) / H)
                if ww > 0 and hh > 0 and ww * hh >= args.min_face_area:
                    lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
                    nf += 1
            n_face += nf
            n_noface += (nf == 0)

            stem = f"hagrid__{gesture}__{img_id}"
            (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            dst = out_img / f"{stem}.jpg"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(img_path.resolve())
            n_img += 1
            g_count += 1

            if args.viz and n_viz < args.viz:
                for ln in lines:
                    c, cx, cy, ww, hh = ln.split()
                    cx, cy, ww, hh = float(cx) * W, float(cy) * H, float(ww) * W, float(hh) * H
                    p1 = (int(cx - ww / 2), int(cy - hh / 2))
                    p2 = (int(cx + ww / 2), int(cy + hh / 2))
                    col = (0, 0, 255) if c == "0" else (0, 255, 0)
                    cv2.rectangle(img, p1, p2, col, 2)
                cv2.imwrite(str(viz_dir / f"{stem}.jpg"), img)
                n_viz += 1
        print(f"  [{gesture}] +{g_count}  (running total {n_img})", flush=True)

    print(f"done: images={n_img}  hand_boxes={n_hand}  face_boxes={n_face}  "
          f"imgs_without_face={n_noface} ({100*n_noface/max(n_img,1):.1f}%)")


if __name__ == "__main__":
    main()
