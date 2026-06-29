#!/usr/bin/env python3
"""Synthesize QR + 1D-barcode DETECTION data: paste generated codes onto no-face backgrounds.

Extends the face(0)/hand(1) detector with qr(2)/barcode(3). Codes are generated with lots of
variety (size, error-correction, colours, centre logos) and pasted onto COCO images that have NO
face (from the InsightFace cache, so we don't inject unlabelled faces). Output is YOLO-detect
format -> datasets/qrbar/{images,labels}/{train,val}.

v2 (real-world hardening): a synthetic-only detector scores ~0.95 on clean rendered codes but
~0.15 on real photos (perspective, blur, lighting, small/distant, jpeg noise). To close that gap
each code now gets a random PERSPECTIVE warp + rotation, a fraction are small/distant, and the
final composite gets brightness/contrast/colour jitter, gaussian blur and variable JPEG quality.
The bounding box is recomputed from the warped alpha so labels stay tight after any transform.
"""
import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import barcode
import qrcode
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

DARK = [(0, 0, 0), (20, 20, 60), (10, 60, 20), (80, 10, 10), (40, 40, 40), (30, 10, 60)]
LIGHT = [(255, 255, 255), (245, 245, 245), (255, 250, 230), (230, 245, 255)]


def rand_text():
    r = random.random()
    if r < 0.35:
        return "https://" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(4, 11))) + ".com"
    if r < 0.6:
        return "".join(random.choices("0123456789", k=random.randint(6, 18)))
    if r < 0.82:
        return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", k=random.randint(8, 30)))
    # long payload -> high-version / DENSE QR (many small modules) so the model also learns dense codes
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ", k=random.randint(45, 130)))


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


def _find_coeffs(dst, src):
    """PIL PERSPECTIVE coeffs mapping output corners `dst` from input corners `src`."""
    m = []
    for (xd, yd), (xs, ys) in zip(dst, src):
        m.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
        m.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
    A = np.array(m, dtype=np.float64)
    B = np.array(dst, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B).tolist()


def warp(code, strength):
    """Random perspective warp of an RGBA code on an expanded transparent canvas."""
    w, h = code.size
    pad = int(max(w, h) * 0.6)
    canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    canvas.paste(code, (pad, pad), code)
    W, H = canvas.size
    corners = [(pad, pad), (pad + w, pad), (pad + w, pad + h), (pad, pad + h)]

    def jit(p):
        return (p[0] + w * random.uniform(-strength, strength),
                p[1] + h * random.uniform(-strength, strength))

    dst = [jit(c) for c in corners]
    try:
        coeffs = _find_coeffs(dst, corners)
    except np.linalg.LinAlgError:
        return code
    return canvas.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def paste_code(bg, code, cls, labels):
    bw, bh = bg.size
    # scale: 40% small/distant codes, else normal
    frac = random.uniform(0.05, 0.18) if random.random() < 0.4 else random.uniform(0.15, 0.55)
    target = int(min(bw, bh) * frac)
    cw, ch = code.size
    s = target / max(cw, ch)
    code = code.resize((max(8, int(cw * s)), max(8, int(ch * s))), Image.LANCZOS).convert("RGBA")
    if random.random() < 0.75:  # perspective (milder for barcodes)
        code = warp(code, strength=random.uniform(0.05, 0.32 if cls == 2 else 0.2))
    ang = random.uniform(-35, 35) if cls == 3 else random.uniform(0, 360)  # barcodes near-axis, QR any
    code = code.rotate(ang, expand=True, fillcolor=(0, 0, 0, 0))
    bbox = code.getbbox()  # tight box of the visible (non-transparent) content after all transforms
    if bbox is None:
        return
    code = code.crop(bbox)
    rw, rh = code.size
    if rw >= bw or rh >= bh or rw < 6 or rh < 6:
        return
    x, y = random.randint(0, bw - rw), random.randint(0, bh - rh)
    bg.paste(code, (x, y), code)
    labels.append((cls, (x + rw / 2) / bw, (y + rh / 2) / bh, rw / bw, rh / bh))


def _fit(code, target_long):
    cw, ch = code.size
    s = target_long / max(cw, ch)
    return code.resize((max(8, int(cw * s)), max(8, int(ch * s))), Image.LANCZOS).convert("RGBA")


def paste_pair(bg, qr_img, qr_cls, bar_img, bar_cls, labels):
    """Paste a QR + a barcode RIGHT NEXT TO each other at a small scale (the sticker-pair case:
    two adjacent codes, both small). Each gets a mild perspective warp; boxes from the warped alpha."""
    bw, bh = bg.size
    base = int(min(bw, bh) * random.uniform(0.09, 0.30))  # QR side length (small)
    qr = _fit(qr_img, base)
    bar = _fit(bar_img, int(base * random.uniform(1.0, 1.7)))  # barcodes a bit longer
    if random.random() < 0.5:
        qr = warp(qr, random.uniform(0.04, 0.18))
    if random.random() < 0.5:
        bar = warp(bar, random.uniform(0.04, 0.12))
    qb, bb = qr.getbbox(), bar.getbbox()
    if not qb or not bb:
        return
    qr, bar = qr.crop(qb), bar.crop(bb)
    gap = int(base * random.uniform(0.02, 0.22))
    horiz = random.random() < 0.7
    if horiz:
        wp, hp = qr.width + gap + bar.width, max(qr.height, bar.height)
    else:
        wp, hp = max(qr.width, bar.width), qr.height + gap + bar.height
    if wp >= bw or hp >= bh:
        return
    x0, y0 = random.randint(0, bw - wp), random.randint(0, bh - hp)
    if horiz:
        qx, qy = x0, y0 + (hp - qr.height) // 2
        bx, by = x0 + qr.width + gap, y0 + (hp - bar.height) // 2
    else:
        qx, qy = x0 + (wp - qr.width) // 2, y0
        bx, by = x0 + (wp - bar.width) // 2, y0 + qr.height + gap
    bg.paste(qr, (qx, qy), qr)
    bg.paste(bar, (bx, by), bar)
    labels.append((qr_cls, (qx + qr.width / 2) / bw, (qy + qr.height / 2) / bh, qr.width / bw, qr.height / bh))
    labels.append((bar_cls, (bx + bar.width / 2) / bw, (by + bar.height / 2) / bh, bar.width / bw, bar.height / bh))


def degrade(img):
    """Whole-image capture-realism: lighting jitter + blur (jpeg noise is applied at save)."""
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.55, 1.35))
    if random.random() < 0.8:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.3))
    if random.random() < 0.4:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.4, 1.2))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 2.4)))
    return img


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
    ap.add_argument("--real-patches", default=None, help="dir with qr/ and bar/ PNG crops; paste these instead of generated codes")
    ap.add_argument("--pair-prob", type=float, default=0.35, help="probability an image is an adjacent small QR+barcode pair")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cache = json.load(open(args.cache))
    bgs = [k for k, v in cache.items() if v and v.get("box") is None and "train2017" in k]
    random.shuffle(bgs)
    bi = 0

    # Optionally paste REAL code crops (real texture) instead of freshly-generated codes.
    real_qr = real_bar = None
    if args.real_patches:
        from glob import glob
        real_qr = glob(args.real_patches + "/qr/*.png")
        real_bar = glob(args.real_patches + "/bar/*.png")
        print(f"real patches: qr={len(real_qr)} bar={len(real_bar)}")

    def get_qr():
        return Image.open(random.choice(real_qr)).convert("RGB") if real_qr else make_qr()

    def get_bar():
        return Image.open(random.choice(real_bar)).convert("RGB") if real_bar else make_barcode()

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
            if random.random() < args.pair_prob:  # adjacent small QR+barcode pair (the sticker case)
                paste_pair(bg, get_qr(), args.qr_class, get_bar(), args.bar_class, labels)
                if random.random() < 0.3:  # plus an occasional stray code elsewhere
                    (paste_code(bg, get_qr(), args.qr_class, labels) if random.random() < 0.5
                     else paste_code(bg, get_bar(), args.bar_class, labels))
            else:
                for _ in range(random.randint(1, 2)):  # 1-2 separate codes per image
                    if random.random() < 0.55:
                        paste_code(bg, get_qr(), args.qr_class, labels)
                    else:
                        paste_code(bg, get_bar(), args.bar_class, labels)
            if not labels:
                continue
            bg = degrade(bg)
            name = f"qrbar_{split}_{i:06d}"
            bg.save(Path(args.out) / "images" / split / f"{name}.jpg", quality=random.randint(45, 92))
            (Path(args.out) / "labels" / split / f"{name}.txt").write_text(
                "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels))
            made += 1
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i + 1}/{n} ...", flush=True)
        print(f"{split}: wrote {made}")


if __name__ == "__main__":
    main()
