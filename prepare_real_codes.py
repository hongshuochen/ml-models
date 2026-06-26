#!/usr/bin/env python3
"""Convert real-world QR / barcode detection datasets into our 4-class YOLO layout (qr=2, barcode=3)
so we can eval/fine-tune the face/hand/qr/barcode detector on REAL photos (not just synthetic).

Sources:
  qr   : NHMNguyen/QR_CODE (YOLO, class 0=qr_code). Real phone photos are the ds1_* files; robo_* are
         mostly clean rendered QRs -> we split the val into val_qr_real (ds1_) vs val_qr_synth (robo_)
         to measure the synthetic->real gap directly. All train goes to `train`.
  bc   : benjamintli/barcode-object-detection (HF parquet, COCO bbox [x,y,w,h], single class=barcode).
  mipt : mipt-modern-cv/barcodes (per-image *.jpg.json, objects[].data = 4 quad corners, tags qr/...).

Images are symlinked when already on disk (qr), materialized when embedded (bc parquet).
"""
import argparse
import io
import json
import os
from pathlib import Path

from PIL import Image


def _dirs(out: Path, split: str):
    img = out / "images" / split
    lab = out / "labels" / split
    img.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)
    return img, lab


def convert_qr(src: Path, out: Path, cls: int):
    """NHM QR YOLO -> remap class to `cls`; split val into real (ds1_) vs synth (robo_)."""
    n = {"train": 0, "val_qr_real": 0, "val_qr_synth": 0}
    for split in ("train", "val"):
        for lab_file in sorted((src / "labels" / split).glob("*.txt")):
            stem = lab_file.stem
            # find the matching image
            imgs = list((src / "images" / split).glob(stem + ".*"))
            if not imgs:
                continue
            img_path = imgs[0]
            if split == "train":
                dst = "train"
            else:
                dst = "val_qr_real" if stem.startswith("ds") else "val_qr_synth"
            img_dir, lab_dir = _dirs(out, dst)
            # remap every row's class id to `cls`
            rows = []
            for line in lab_file.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                rows.append(f"{cls} " + " ".join(p[1:5]))
            if not rows:
                continue
            link = img_dir / img_path.name
            if not link.exists():
                os.symlink(img_path.resolve(), link)
            (lab_dir / (stem + ".txt")).write_text("\n".join(rows))
            n[dst] += 1
    print("qr:", n)


def convert_bc(parquets, out: Path, split: str, cls: int, limit: int = 0):
    """benjamintli parquet (COCO bbox [x,y,w,h] absolute) -> YOLO class `cls`."""
    import pyarrow.parquet as pq

    img_dir, lab_dir = _dirs(out, split)
    made = 0
    for pq_path in parquets:
        t = pq.read_table(pq_path)
        rows = t.to_pylist()
        for i, row in enumerate(rows):
            img = row["image"]
            b = img["bytes"] if isinstance(img, dict) else img
            try:
                im = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                continue
            W, H = im.size
            obj = row.get("objects") or {}
            bboxes = obj.get("bbox") or obj.get("bboxes") or []
            lines = []
            for box in bboxes:
                x, y, w, h = box[:4]
                if w <= 0 or h <= 0:
                    continue
                xc, yc = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}")
            if not lines:
                continue
            name = f"bc_{split}_{Path(pq_path).stem}_{i:06d}"
            im.save(img_dir / f"{name}.jpg", quality=92)
            (lab_dir / f"{name}.txt").write_text("\n".join(lines))
            made += 1
            if limit and made >= limit:
                break
        if limit and made >= limit:
            break
    print(f"bc[{split}]: wrote {made}")


def convert_mipt(src: Path, out: Path, qr_cls: int, bc_cls: int):
    """mipt per-image quad JSON -> axis-aligned YOLO boxes; tag 'qr' -> qr_cls else barcode."""
    img_dir, lab_dir = _dirs(out, "val")
    made = 0
    for js in sorted(src.glob("*.jpg.json")):
        img_path = src / js.name[:-5]  # strip .json
        if not img_path.exists():
            continue
        d = json.loads(js.read_text())
        W, H = d.get("size", [None, None])
        if not W:
            with Image.open(img_path) as im:
                W, H = im.size
        lines = []
        for o in d.get("objects", []):
            pts = o.get("data") or []
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            tags = [str(t).lower() for t in o.get("tags", [])]
            cls = qr_cls if any("qr" in t for t in tags) else bc_cls
            xc, yc = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        if not lines:
            continue
        link = img_dir / img_path.name
        if not link.exists():
            os.symlink(img_path.resolve(), link)
        (lab_dir / (img_path.stem + ".txt")).write_text("\n".join(lines))
        made += 1
    print(f"mipt: wrote {made}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["qr", "bc", "mipt"])
    ap.add_argument("--src")
    ap.add_argument("--parquet", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val_bc")
    ap.add_argument("--qr-class", type=int, default=2)
    ap.add_argument("--bc-class", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == "qr":
        convert_qr(Path(args.src), out, args.qr_class)
    elif args.mode == "bc":
        convert_bc(args.parquet, out, args.split, args.bc_class, args.limit)
    else:
        convert_mipt(Path(args.src), out, args.qr_class, args.bc_class)


if __name__ == "__main__":
    main()
