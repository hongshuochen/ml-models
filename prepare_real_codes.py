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


def convert_boof(src: Path, out: Path, cls: int, val_frac: float, seed: int):
    """BoofCV qrcodes_v3: detection/<cat>/imageNNN.txt = per-QR 4-corner polygons -> YOLO bbox.
    Split per-image into train vs a held-out val_qr_boof (real, hard, 16 difficulty categories)."""
    import random
    random.seed(seed)
    det = src / "qrcodes" / "detection"
    items = []
    for txt in sorted(det.rglob("*.txt")):
        if "note" in txt.name.lower():
            continue
        ip = next((txt.with_suffix(e) for e in (".jpg", ".jpeg", ".png", ".JPG", ".PNG") if txt.with_suffix(e).exists()), None)
        if ip:
            items.append((ip, txt))
    random.shuffle(items)
    nval = int(len(items) * val_frac)
    nt = nv = 0
    for i, (ip, txt) in enumerate(items):
        split = "val_qr_boof" if i < nval else "train"
        with Image.open(ip) as im:
            W, H = im.size
        # Two GT variants exist: "SETS" + 8-floats-per-line (one polygon/line), and no-header with one
        # corner (2 floats) per line. Universal: gather every float token (skip #/SETS), chunk by 8 ->
        # one 4-corner polygon per chunk.
        floats = []
        for ln in txt.read_text().splitlines():
            if ln.strip().startswith("#"):
                continue
            for tok in ln.split():
                try:
                    floats.append(float(tok))
                except ValueError:
                    pass  # "SETS" and other non-numeric tokens
        lines = []
        for j in range(0, len(floats) - 7, 8):
            poly = floats[j:j + 8]
            xs, ys = poly[0::2], poly[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        if not lines:
            continue
        img_dir, lab_dir = _dirs(out, split)
        name = f"boof_{ip.parent.name}_{ip.stem}"
        link = img_dir / (name + ip.suffix)
        if not link.exists():
            os.symlink(ip.resolve(), link)
        (lab_dir / (name + ".txt")).write_text("\n".join(lines))
        nv += split == "val_qr_boof"
        nt += split == "train"
    print(f"boof: train {nt} / val_qr_boof {nv}")


def convert_kolabit(src: Path, out: Path, cls: int):
    """kolabit/qr-codes (YOLO, class 0=qr, whole-QR boxes, natural scenes) -> train, remapped to cls."""
    n = 0
    img_dir, lab_dir = _dirs(out, "train")
    for split in ("train", "test"):
        for txt in sorted((src / "labels" / split).glob("*.txt")):
            imgs = list((src / "images" / split).glob(txt.stem + ".*"))
            if not imgs:
                continue
            rows = [f"{cls} " + " ".join(l.split()[1:5]) for l in txt.read_text().splitlines() if len(l.split()) >= 5]
            if not rows:
                continue
            name = f"kola_{split}_{txt.stem}"
            link = img_dir / (name + imgs[0].suffix)
            if not link.exists():
                os.symlink(imgs[0].resolve(), link)
            (lab_dir / (name + ".txt")).write_text("\n".join(rows))
            n += 1
    print(f"kolabit: {n}")


def oversample(out: Path, prefixes, k: int):
    """Duplicate (symlink) train images whose name starts with any of `prefixes` to k total copies,
    so scarce real QR isn't drowned out by the abundant rendered QR. Idempotent (_ov tag)."""
    img_dir, lab_dir = out / "images" / "train", out / "labels" / "train"
    targets = [p for p in img_dir.iterdir() if "_ov" not in p.name and any(p.name.startswith(pre) for pre in prefixes)]
    made = 0
    for p in targets:
        lab = lab_dir / (p.stem + ".txt")
        if not lab.exists():
            continue
        real = os.readlink(p) if p.is_symlink() else str(p.resolve())
        for c in range(1, k):
            np_ = img_dir / f"{p.stem}_ov{c}{p.suffix}"
            if not np_.exists():
                os.symlink(real, np_)
            (lab_dir / f"{p.stem}_ov{c}.txt").write_text(lab.read_text())
            made += 1
    print(f"oversample x{k} of {len(targets)} real-QR imgs -> +{made}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["qr", "bc", "mipt", "boof", "kolabit", "oversample"])
    ap.add_argument("--src")
    ap.add_argument("--parquet", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val_bc")
    ap.add_argument("--qr-class", type=int, default=2)
    ap.add_argument("--bc-class", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--prefixes", nargs="*", default=["ds", "boof_", "kola_"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == "qr":
        convert_qr(Path(args.src), out, args.qr_class)
    elif args.mode == "bc":
        convert_bc(args.parquet, out, args.split, args.bc_class, args.limit)
    elif args.mode == "mipt":
        convert_mipt(Path(args.src), out, args.qr_class, args.bc_class)
    elif args.mode == "boof":
        convert_boof(Path(args.src), out, args.qr_class, args.val_frac, args.seed)
    elif args.mode == "kolabit":
        convert_kolabit(Path(args.src), out, args.qr_class)
    else:
        oversample(out, args.prefixes, args.k)


if __name__ == "__main__":
    main()
