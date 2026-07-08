"""Build the v1 hand-label set: ~0.2fps even subset of golf_frames (every 10th of the ~2fps
extraction, cap 40/video) reusing the existing golf_v3_1280 pre-labels.

Outputs:
  - datasets/golf_prelabels/v1_frames.txt   (the chosen frame paths; the training subset)
  - datasets/golf_prelabels/ls_tasks_v1.json (Label Studio tasks + predictions for a fresh project)

Run: uv run python golf/build_v1_labelset.py
Then create an LS project and POST this JSON to /api/projects/<id>/import.
"""
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

FRAMES = Path("datasets/golf_frames")
LABELS = Path("datasets/golf_prelabels/labels")
OUT_LIST = Path("datasets/golf_prelabels/v1_frames.txt")
OUT_JSON = Path("datasets/golf_prelabels/ls_tasks_v1.json")
FPCLASS = Path("datasets/golf_prelabels/all_frames.fpclass.json")  # first/third-person map (optional)
TARGET = 2000                  # aim for ~this many first-person frames
CAP = 40                       # max frames per video (avoid long-clip dominance)
TH_H, TH_TOP = 0.15, 0.60      # third-person = person box height>TH_H AND top<TH_TOP (recomputed from fpclass)
STRIP_CLUB = True              # drop model club_head prelabels (mostly on shaft/FP) -> human draws heads fresh
NAMES = {0: "ball", 1: "club_head"}
MODEL_VERSION = "golf_v3_1280"


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def yolo_to_results(txt, W, H):
    out = []
    for j, line in enumerate(txt.strip().splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        if STRIP_CLUB and c == 1:   # human labels club_head fresh; model's are mostly shaft/FP
            continue
        out.append({
            "id": f"p{j}", "type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": W, "original_height": H, "image_rotation": 0,
            "value": {"x": clamp((cx - bw / 2) * 100), "y": clamp((cy - bh / 2) * 100),
                      "width": clamp(bw * 100), "height": clamp(bh * 100), "rotation": 0,
                      "rectanglelabels": [NAMES.get(c, str(c))]},
        })
    return out


def select_subset():
    # optional first-person filter (drop third-person = someone else's swing filmed at distance)
    keep = None
    if FPCLASS.exists():
        cls = json.loads(FPCLASS.read_text())
        # recompute third-person from stored maxh/top with the tighter TH_H (no GPU re-run)
        keep = {os.path.basename(p) for p, v in cls.items()
                if not (v["maxh"] > TH_H and v["top"] < TH_TOP)}
    byvid = defaultdict(list)
    pool = 0
    for p in sorted(glob.glob(str(FRAMES / "*.jpg"))):
        if keep is not None and os.path.basename(p) not in keep:
            continue
        m = re.match(r"(.+)_f(\d+)\.jpg$", os.path.basename(p))
        byvid[m.group(1)].append((int(m.group(2)), p))
        pool += 1
    stride = max(1, round(pool / TARGET))   # even temporal sampling to hit ~TARGET
    sel = []
    for lst in byvid.values():
        lst.sort()
        sel += [p for _, p in lst[stride // 2::stride][:CAP]]
    print(f"  first-person pool: {pool} ({'filtered' if keep is not None else 'NO fpclass -> all frames'}) | stride {stride}")
    return sorted(sel), len(byvid)


def main():
    sel, nvid = select_subset()
    tasks, empty = [], 0
    for i, p in enumerate(sel):
        name = Path(p).name
        with Image.open(p) as im:
            W, H = im.size
        txt_path = LABELS / (Path(p).stem + ".txt")
        txt = txt_path.read_text() if txt_path.exists() else ""
        results = yolo_to_results(txt, W, H)
        empty += not results
        tasks.append({
            "data": {"image": f"/data/local-files/?d=golf_frames/{name}"},
            "predictions": [{"model_version": MODEL_VERSION, "result": results}],
        })
    OUT_LIST.write_text("\n".join(sel))
    OUT_JSON.write_text(json.dumps(tasks))
    print(f"v1 subset: {len(sel)} frames across {nvid} videos  (first-person only, target ~{TARGET}, cap {CAP})")
    print(f"  with boxes: {len(sel)-empty}  | empty (negatives): {empty} ({100*empty/len(sel):.0f}%)")
    print(f"  frame list -> {OUT_LIST}")
    print(f"  LS tasks   -> {OUT_JSON}  ({OUT_JSON.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
