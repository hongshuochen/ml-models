"""Classify golf frames egocentric(first-person) vs third-person via COCO person detection.

Rule: third-person = a LARGE, FULL person is the subject -> a person box with height > TH_H of
frame AND its top above TH_TOP (i.e. head/torso visible high up = a standing person at distance).
Kept as first-person: no person, only the wearer's own hands/arms (enter from bottom, no head high),
or small distant background people (fine at a range while wearing the cam).

In:  a frame-list file (default datasets/golf_prelabels/v1_frames.txt)
Out: <list>.fpclass.json  {path: {"third": bool, "maxh": float, "top": float, "n": int}}
     + two QA sheets (third-person sample, first-person sample) with person boxes drawn.
Run: uv run python golf/detect_firstperson.py [list.txt] [--th-h 0.45] [--th-top 0.55]
"""
import argparse, json, os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

TMP = Path("/tmp/claude-1000/-home-max-2026-ml-models/ef3aba0d-9d1b-4678-8a09-02d72865ef6e/scratchpad/fp_frames")


def make_sheet(cells, path, cols=6, cell_h=340):
    if not cells:
        return
    rows, row = [], []
    for im in cells:
        h, w = im.shape[:2]
        row.append(cv2.resize(im, (int(w * cell_h / h), cell_h)))
        if len(row) == cols:
            rows.append(row); row = []
    if row:
        rows.append(row)
    maxw = max(sum(im.shape[1] for im in r) for r in rows)
    canvas = []
    for r in rows:
        strip = np.zeros((cell_h, maxw, 3), np.uint8); x = 0
        for im in r:
            strip[:, x:x + im.shape[1]] = im; x += im.shape[1]
        canvas.append(strip)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, np.vstack(canvas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("list", nargs="?", default="datasets/golf_prelabels/v1_frames.txt")
    ap.add_argument("--th-h", type=float, default=0.45, help="person height frac -> third-person")
    ap.add_argument("--th-top", type=float, default=0.55, help="person box top frac (head high up)")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--sheet-n", type=int, default=24)
    args = ap.parse_args()

    frames = [l for l in Path(args.list).read_text().splitlines() if l.strip()]
    # symlink into a temp dir so we can use the fast dir-stream predict (a list source OOMs)
    if TMP.exists():
        for f in TMP.glob("*"):
            f.unlink()
    TMP.mkdir(parents=True, exist_ok=True)
    for p in frames:
        (TMP / Path(p).name).symlink_to(Path(p).resolve())

    m = YOLO("yolo26s.pt")
    cls = {}
    for r in m.predict(source=str(TMP), imgsz=640, conf=args.conf, classes=[0], device=0,
                       stream=True, verbose=False):
        H, W = r.orig_shape
        maxh, top_at_maxh = 0.0, 1.0
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            hf = (y2 - y1) / H
            if hf > maxh:
                maxh, top_at_maxh = hf, y1 / H
        third = maxh > args.th_h and top_at_maxh < args.th_top
        cls[r.path] = {"third": third, "maxh": round(maxh, 3), "top": round(top_at_maxh, 3), "n": len(r.boxes)}

    n3 = sum(v["third"] for v in cls.values())
    print(f"frames: {len(cls)}  | third-person: {n3} ({100*n3/len(cls):.0f}%)  | first-person kept: {len(cls)-n3}")
    hist = {}
    for v in cls.values():
        k = f"{int(v['maxh']*10)*10}-{int(v['maxh']*10)*10+10}%"
        hist[k] = hist.get(k, 0) + 1
    print("  max person-height distribution:", dict(sorted(hist.items())))

    out = Path(args.list).with_suffix(".fpclass.json")
    out.write_text(json.dumps(cls))
    print(f"  classes -> {out}")

    # QA sheets: draw person boxes + label maxh/top
    third_cells, first_cells = [], []
    for p, v in cls.items():
        want_third = v["third"] and len(third_cells) < args.sheet_n
        want_first = (not v["third"]) and len(first_cells) < args.sheet_n and v["n"] > 0  # first-person WITH a (small/partial) person, to check false-keeps
        if not (want_third or want_first):
            continue
        r = m.predict(p, imgsz=640, conf=args.conf, classes=[0], device=0, verbose=False)[0]
        img = cv2.imread(p)
        H, W = img.shape[:2]
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            col = (0, 0, 255) if v["third"] else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        cv2.putText(img, f"maxh={v['maxh']} top={v['top']}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
        (third_cells if v["third"] else first_cells).append(img)

    make_sheet(third_cells, "golf/sample_sheets/fp_third_person.jpg")
    make_sheet(first_cells, "golf/sample_sheets/fp_first_person_withppl.jpg")
    print("  sheets -> golf/sample_sheets/fp_third_person.jpg , fp_first_person_withppl.jpg")


if __name__ == "__main__":
    main()
