"""Inspect the 2-class {ball, club_head} data from golf-driver-tracker before training.

Renders, from the already-downloaded golf-driver-tracker preview:
  1) full images with ONLY ball(red) + club_head(green) drawn (handle dropped)
  2) zoomed padded crops of ball instances     (is the tiny-ball box tight?)
  3) zoomed padded crops of club_head instances
Usage: uv run python golf/preview_2class.py
Outputs into golf/sample_sheets/.
"""
import random
from pathlib import Path

import cv2
import numpy as np

random.seed(1)
ROOT = next(Path("datasets/golf_preview").glob("golf-driver-tracker*"))
OUT = Path("golf/sample_sheets"); OUT.mkdir(parents=True, exist_ok=True)

# driver-tracker class ids -> our 2 classes (drop 1=handle)
KEEP = {0: ("ball", (0, 0, 255)), 2: ("club_head", (0, 200, 0))}

imgs = [p for p in ROOT.rglob("*.jpg") if "/images/" in str(p).replace("\\", "/")]
print(f"{ROOT.name}: {len(imgs)} images")


def labels_for(img):
    return Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")


def read_boxes(img):
    out = []
    lp = labels_for(img)
    if not lp.exists():
        return out
    for line in lp.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        if c in KEEP:
            out.append((c, *[float(v) for v in p[1:5]]))
    return out


def cell(im, size):
    h, w = im.shape[:2]
    s = (size - 8) / max(h, w)
    r = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.full((size, size, 3), 30, np.uint8)
    canvas[: r.shape[0], : r.shape[1]] = r
    return canvas


def grid(cells, cols, title, cellpx):
    while len(cells) % cols:
        cells.append(np.full((cellpx, cellpx, 3), 30, np.uint8))
    rows = [np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)]
    g = np.vstack(rows)
    bar = np.full((38, g.shape[1], 3), 15, np.uint8)
    cv2.putText(bar, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, g])


# ---- 1) full images, 2-class only ----
random.shuffle(imgs)
full = []
for p in imgs:
    bxs = read_boxes(p)
    if not any(c == 0 for c, *_ in bxs):  # prefer frames that actually contain a ball
        continue
    im = cv2.imread(str(p))
    if im is None:
        continue
    h, w = im.shape[:2]
    for c, cx, cy, bw, bh in bxs:
        nm, col = KEEP[c]
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        cv2.putText(im, nm, (x1, max(13, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    full.append(cell(im, 420))
    if len(full) == 20:
        break
cv2.imwrite(str(OUT / "driver_2class_full.jpg"), grid(full, 5, "golf-driver-tracker  ball(red)+club_head(green)  handle dropped", 420))
print("wrote driver_2class_full.jpg")


# ---- 2/3) zoomed crops per class ----
def crop_sheet(target_c, name):
    cells = []
    order = imgs[:]
    random.shuffle(order)
    for p in order:
        boxes = [b for b in read_boxes(p) if b[0] == target_c]
        if not boxes:
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        c, cx, cy, bw, bh = boxes[0]
        bwp, bhp = bw * w, bh * h
        pad = max(bwp, bhp) * 1.6 + 30
        x1, y1 = int(cx * w - bwp / 2 - pad), int(cy * h - bhp / 2 - pad)
        x2, y2 = int(cx * w + bwp / 2 + pad), int(cy * h + bhp / 2 + pad)
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
        crop = im[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        # box coords within crop
        col = KEEP[target_c][1]
        cv2.rectangle(crop, (int(cx * w - bwp / 2) - x1, int(cy * h - bhp / 2) - y1),
                      (int(cx * w + bwp / 2) - x1, int(cy * h + bhp / 2) - y1), col, 2)
        cells.append(cell(crop, 240))
        if len(cells) == 16:
            break
    cv2.imwrite(str(OUT / f"driver_{name}_crops.jpg"), grid(cells, 4, f"{name} crops (zoomed, box drawn)  n_instances-shown={len(cells)}", 240))
    print(f"wrote driver_{name}_crops.jpg")


crop_sheet(0, "ball")
crop_sheet(2, "club_head")
