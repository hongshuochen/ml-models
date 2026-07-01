"""Preview public golf datasets BEFORE training.

Downloads each Roboflow dataset (needs ROBOFLOW_API_KEY) and renders an annotated
contact sheet per dataset — the dataset's ORIGINAL classes drawn as boxes — so we can
eyeball label quality + relevance before committing to any training / remapping.

Usage:
    export ROBOFLOW_API_KEY=...          # Roboflow -> Settings -> API Keys
    uv run python golf/preview_datasets.py

Outputs: golf/sample_sheets/<dataset>.jpg  (+ a combined all.jpg)
Slugs come from GOLF_PLAN.md dataset URLs; edit DATASETS to add/remove.
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

random.seed(0)

_KEYFILE = Path(__file__).with_name(".roboflow_key")
KEY = os.environ.get("ROBOFLOW_API_KEY") or (_KEYFILE.read_text().strip() if _KEYFILE.exists() else None)
if not KEY:
    raise SystemExit(
        "No Roboflow key. Either `export ROBOFLOW_API_KEY=...` or write it to golf/.roboflow_key "
        "(Roboflow -> Settings -> API Keys). The .roboflow_key file is git-ignored."
    )

DL = Path("datasets/golf_preview")
OUT = Path("golf/sample_sheets")
OUT.mkdir(parents=True, exist_ok=True)

# (workspace, project, version|None -> latest). From GOLF_PLAN.md §C URLs.
# Small/most-relevant first; the 31k `golfball` set is skipped here (too big just to preview).
DATASETS = [
    ("salo-levy-nlqrn", "golf-driver-tracker", None),   # classes: ball / club-handle / club-head  (closest to our 3-class)
    ("golfswing-e1qwd", "golf_club_pose", None),        # club keypoints (shaft/grip/head)
    ("club-head-tracking", "golf-club-tracking", 2),    # club-head boxes
    ("golf-balls", "golf-ball-tracker-sksye", None),    # ball boxes
]

N_SAMPLES, COLS, CELL = 12, 4, 380
COLORS = [(0, 0, 255), (0, 200, 0), (255, 90, 0), (0, 180, 255), (200, 0, 200), (40, 220, 220)]


def resolve_and_download(ws, proj_slug, version):
    from roboflow import Roboflow
    rf = Roboflow(api_key=KEY)
    project = rf.workspace(ws).project(proj_slug)
    if version is None:
        vers = [v.version if hasattr(v, "version") else int(str(v).split("/")[-1]) for v in project.versions()]
        version = max(int(v) for v in vers)
    dest = DL / f"{proj_slug}-v{version}"
    if not dest.exists():
        project.version(int(version)).download("yolov8", location=str(dest))
    return dest


def draw_sample(img_path, names):
    im = cv2.imread(str(img_path))
    if im is None:
        return None
    h, w = im.shape[:2]
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            col = COLORS[c % len(COLORS)]
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
            nm = names[c] if c < len(names) else str(c)
            cv2.putText(im, nm, (x1, max(13, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return im


def sheet(images, title):
    cells = []
    for im in images:
        h, w = im.shape[:2]
        s = (CELL - 8) / max(h, w)
        r = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.full((CELL, CELL, 3), 30, np.uint8)
        yh, xw = r.shape[:2]
        canvas[:yh, :xw] = r
        cells.append(canvas)
    while len(cells) % COLS:
        cells.append(np.full((CELL, CELL, 3), 30, np.uint8))
    rows = [np.hstack(cells[i:i + COLS]) for i in range(0, len(cells), COLS)]
    grid = np.vstack(rows)
    bar = np.full((40, grid.shape[1], 3), 15, np.uint8)
    cv2.putText(bar, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, grid])


def main():
    made = []
    for ws, proj, ver in DATASETS:
        try:
            print(f"\n=== {proj} ===")
            root = resolve_and_download(ws, proj, ver)
            data_yaml = next(root.rglob("data.yaml"), None)
            names = []
            if data_yaml:
                nd = yaml.safe_load(data_yaml.read_text())["names"]
                names = list(nd.values()) if isinstance(nd, dict) else list(nd)
            imgs = [p for p in root.rglob("*.jpg") if "/images/" in str(p).replace("\\", "/")]
            imgs += [p for p in root.rglob("*.png") if "/images/" in str(p).replace("\\", "/")]
            random.shuffle(imgs)
            drawn = [d for p in imgs[:N_SAMPLES * 2] if (d := draw_sample(p, names)) is not None][:N_SAMPLES]
            if not drawn:
                print("  no images found")
                continue
            out = OUT / f"{proj}.jpg"
            cv2.imwrite(str(out), sheet(drawn, f"{proj}   classes={names}   (n={len(imgs)})"))
            print(f"  wrote {out}  classes={names}  images={len(imgs)}")
            made.append(str(out))
        except Exception as e:
            print(f"  FAILED {proj}: {type(e).__name__}: {e}")
    print("\nDONE. Sheets:", made)


if __name__ == "__main__":
    main()
