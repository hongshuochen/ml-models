#!/usr/bin/env python3
"""Synthesize QR + 1D-barcode DETECTION data: paste generated codes onto no-face backgrounds.

Extends the face(0)/hand(1) detector with qr(2)/barcode(3). Codes are generated with lots of
variety (size, error-correction, colours, centre logos, rotation) and pasted at random scale /
position / rotation onto COCO images that have NO face (from the InsightFace cache, so we don't
inject unlabelled faces). Output is YOLO-detect format -> datasets/qrbar/{images,labels}/{train,val}.
"""
import argparse
import io
import json
import random
from pathlib import Path

import barcode
import qrcode
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw

DARK = [(0, 0, 0), (20, 20, 60), (10, 60, 20), (80, 10, 10), (40, 40, 40), (30, 10, 60)]
LIGHT = [(255, 255, 255), (245, 245, 245), (255, 250, 230), (230, 245, 255)]


def rand_text():
    r = random.random()
    if r < 0.4:
        return "https://" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(4, 11))) + ".com"
    if r < 0.7:
        return "".join(random.choices("0123456789", k=random.randint(6, 18)))
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", k=random.randint(8, 30)))


def make_qr():
    qr = qrcode.QRCode(error_correction=random.choice([
        qrcode.constants.ERROR_CORRECT_L, qrcode.constants.ERROR_CORRECT_M,
        qrcode.constants.ERROR_CORRECT_Q, qrcode.constants.ERROR_CORRECT_H]),
        box_size=10, border=random.choice([2, 3, 4]))
    qr.add_data(rand_text())
    qr.make(fit=True)
    fg, bg = random.choice(DARK), random.choice(LIGHT)
    if random.random() < 0.1:
        fg, bg = bg, fg
    img = qr.make_image(fill_color=fg, back_color=bg).convert("RGB")
    if random.random() < 0.3:  # branded QR: a logo blob in the centre
        w, h = img.size
        s = int(min(w, h) * random.uniform(0.12, 0.2))
        cx, cy = w // 2, h // 2
        d = ImageDraw.Draw(img)
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=(255, 255, 255))
        d.ellipse([cx - s + 4, cy - s + 4, cx + s - 4, cy + s - 4],
                  fill=tuple(random.randint(0, 200) for _ in range(3)))
    return img


def make_barcode():
    kind = random.choice(["code128", "code39", "ean13"])
    if kind == "ean13":
        data = "".join(random.choices("0123456789", k=12))
    elif kind == "code39":
        data = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=random.randint(6, 12)))
    else:
        data = "".join(random.choices("ABCabc0123456789-", k=random.randint(6, 14)))
    buf = io.BytesIO()
    barcode.get_barcode_class(kind)(data, writer=ImageWriter()).write(
        buf, options={"write_text": random.random() < 0.5, "module_height": random.uniform(8, 18)})
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def paste_code(bg, code, cls, labels):
    bw, bh = bg.size
    target = int(min(bw, bh) * random.uniform(0.12, 0.55))
    cw, ch = code.size
    s = target / max(cw, ch)
    code = code.resize((max(8, int(cw * s)), max(8, int(ch * s))), Image.LANCZOS).convert("RGBA")
    ang = random.uniform(-45, 45) if cls == 3 else random.uniform(0, 360)  # barcodes near-axis, QR any
    code = code.rotate(ang, expand=True, fillcolor=(0, 0, 0, 0))
    rw, rh = code.size
    if rw >= bw or rh >= bh:
        return
    x, y = random.randint(0, bw - rw), random.randint(0, bh - rh)
    bg.paste(code, (x, y), code)
    labels.append((cls, (x + rw / 2) / bw, (y + rh / 2) / bh, rw / bw, rh / bh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="datasets/face_cls_cache.json")
    ap.add_argument("--out", default="datasets/qrbar")
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--qr-class", type=int, default=2)
    ap.add_argument("--bar-class", type=int, default=3)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    cache = json.load(open(args.cache))
    bgs = [k for k, v in cache.items() if v and v.get("box") is None and "train2017" in k]
    random.shuffle(bgs)
    bi = 0

    for split, n in [("train", args.n_train), ("val", args.n_val)]:
        (Path(args.out) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "labels" / split).mkdir(parents=True, exist_ok=True)
        made = 0
        for i in range(n):
            bg = None
            while bg is None:
                try:
                    bg = Image.open(bgs[bi % len(bgs)]).convert("RGB").resize((args.size, args.size))
                except Exception:
                    bg = None
                bi += 1
            labels = []
            for _ in range(random.randint(1, 2)):  # 1-2 codes per image
                if random.random() < 0.6:
                    paste_code(bg, make_qr(), args.qr_class, labels)
                else:
                    paste_code(bg, make_barcode(), args.bar_class, labels)
            if not labels:
                continue
            name = f"qrbar_{split}_{i:06d}"
            bg.save(Path(args.out) / "images" / split / f"{name}.jpg", quality=90)
            (Path(args.out) / "labels" / split / f"{name}.txt").write_text(
                "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels))
            made += 1
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i + 1}/{n} ...", flush=True)
        print(f"{split}: wrote {made}")


if __name__ == "__main__":
    main()
