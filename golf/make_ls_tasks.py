"""Build a Label Studio tasks JSON (images + YOLO pre-labels as predictions) for golf annotation.

Each task = one frame with:
  - data.image  -> a local-files URL Label Studio serves (needs LOCAL_FILES_SERVING_ENABLED +
    LOCAL_FILES_DOCUMENT_ROOT=<datasets dir>), i.e. /data/local-files/?d=golf_frames/<name>.jpg
  - predictions -> the YOLO boxes (ball=0, club_head=1) as RectangleLabels, so the human just
    corrects (delete sun->ball FPs, add misses) instead of drawing from scratch.

In:  datasets/golf_frames/*.jpg  +  datasets/golf_prelabels/labels/*.txt
Out: datasets/golf_prelabels/ls_tasks.json
Run: uv run python golf/make_ls_tasks.py
Labeling config: golf/ls_golf_config.xml  (control name "label", image name "image" -> matched below)
"""
import glob
import json
from pathlib import Path

from PIL import Image

FRAMES = Path("datasets/golf_frames")
LABELS = Path("datasets/golf_prelabels/labels")
OUT = Path("datasets/golf_prelabels/ls_tasks.json")
DOC_ROOT_SUBDIR = "golf_frames"          # relative to LOCAL_FILES_DOCUMENT_ROOT (= datasets/)
NAMES = {0: "ball", 1: "club_head"}
MODEL_VERSION = "golf_v3_1280"


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def yolo_to_results(txt, W, H):
    results = []
    if not txt.strip():
        return results
    for j, line in enumerate(txt.strip().splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
        results.append({
            "id": f"p{j}",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": W,
            "original_height": H,
            "image_rotation": 0,
            "value": {
                "x": clamp((cx - bw / 2) * 100),
                "y": clamp((cy - bh / 2) * 100),
                "width": clamp(bw * 100),
                "height": clamp(bh * 100),
                "rotation": 0,
                "rectanglelabels": [NAMES.get(c, str(c))],
            },
        })
    return results


def main():
    frames = sorted(glob.glob(str(FRAMES / "*.jpg")))
    tasks = []
    empty = 0
    for i, p in enumerate(frames):
        name = Path(p).name
        with Image.open(p) as im:
            W, H = im.size
        txt_path = LABELS / (Path(p).stem + ".txt")
        txt = txt_path.read_text() if txt_path.exists() else ""
        results = yolo_to_results(txt, W, H)
        empty += not results
        tasks.append({
            "data": {"image": f"/data/local-files/?d={DOC_ROOT_SUBDIR}/{name}"},
            "predictions": [{"model_version": MODEL_VERSION, "result": results}],
        })
        if (i + 1) % 5000 == 0:
            print(f"  built {i+1}/{len(frames)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tasks))
    mb = OUT.stat().st_size / 1e6
    print(f"\n{len(tasks)} tasks -> {OUT}  ({mb:.1f} MB)")
    print(f"  with boxes: {len(tasks)-empty}  | empty (negatives): {empty}")
    print(f"  image URLs: /data/local-files/?d={DOC_ROOT_SUBDIR}/<name>.jpg")


if __name__ == "__main__":
    main()
