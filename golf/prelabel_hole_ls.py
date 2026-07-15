#!/usr/bin/env python3
"""Build + import a HOLE-labeling Label Studio project (GOLF_HOLE_PLAN.md Phase 2).

Two-model pre-annotation over the extracted egocentric frames:
  * hole-teacher (golf_hole_teacher)  -> hole boxes  (finds the putting/cup frames worth labeling)
  * golf_ego_v2                       -> ball + club_head boxes
Keeps only frames where the teacher sees a hole (those are the putting-green frames), capped per
video/session for variety, and creates a 3-class Label Studio project with the boxes pre-filled so
the human only CORRECTS the hole class (add missed cups, fix ball/club). See [[golf-label-studio]].

Run: uv run python golf/prelabel_hole_ls.py [--cap-session 25] [--hole-conf 0.30] [--dry-run]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import requests
from ultralytics import YOLO

FRAMES_DIR = "/home/max/2026/ml-models/datasets/golf_frames"
DOC_ROOT_REL = "golf_frames"                       # image URL is /data/local-files/?d=golf_frames/<name>
LS = "http://localhost:8080"
TOKEN = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
HOLE_W = "runs/detect/golf_hole_teacher/weights/best.pt"
BC_W = "runs/detect/golf_ego_v2_1280/weights/best.pt"
LABELS = {0: "ball", 1: "club_head", 2: "hole"}
CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="ball" hotkey="1" background="#e63946"/>
    <Label value="club_head" hotkey="2" background="#2a9d8f"/>
    <Label value="hole" hotkey="3" background="#22d3ee"/>
  </RectangleLabels>
</View>"""


def session(path):
    return os.path.basename(path).rsplit("_f", 1)[0]


def chunked(seq, n):
    """Yield seq in lists of <=n. Ultralytics classifies a LIST source as in-memory images
    (`from_img`) and eagerly loads EVERY element via autocast_list — feeding all 19,700 paths
    at once decodes them into RAM (~169 GiB -> OOM). Chunking caps peak memory to n frames.
    (In-memory sources also get synthetic `image{i}.jpg` r.path values, so callers must track
    the real path via zip(group, predict(group)), NOT r.path.)"""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def region(cls, box, W, H):
    x1, y1, x2, y2 = box
    return {"type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": W, "original_height": H, "image_rotation": 0,
            "value": {"x": 100 * x1 / W, "y": 100 * y1 / H,
                      "width": 100 * (x2 - x1) / W, "height": 100 * (y2 - y1) / H,
                      "rotation": 0, "rectanglelabels": [LABELS[cls]]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-session", type=int, default=25, help="max frames kept per video/session")
    ap.add_argument("--hole-conf", type=float, default=0.30)
    ap.add_argument("--hole-imgsz", type=int, default=960)
    ap.add_argument("--bc-imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16,
                    help="frames per predict call — caps RAM/VRAM (a list source is one batch)")
    ap.add_argument("--title", default="Golf HOLE labels (3-class)")
    ap.add_argument("--dry-run", action="store_true", help="build tasks, don't create the LS project")
    args = ap.parse_args()

    frames = sorted(glob.glob(f"{FRAMES_DIR}/*.jpg"))
    print(f"{len(frames)} frames; pass 1 = hole-teacher @ {args.hole_imgsz} (finds cup frames)...")

    # ---- pass 1: hole-teacher -> keep hole-positive frames, capped per session ----
    hole_m = YOLO(HOLE_W)
    kept, per_sess, done, last = {}, defaultdict(int), 0, 0
    for group in chunked(frames, args.batch):           # chunk bounds autocast_list's eager load
        # zip real path <- result: a list source yields synthetic image{i}.jpg r.path (see chunked)
        for path, r in zip(group, hole_m.predict(group, imgsz=args.hole_imgsz, conf=args.hole_conf,
                                                 stream=True, device=0, verbose=False)):
            hs = [b.xyxy[0].tolist() for b in r.boxes]  # 1-class hole
            if not hs:
                continue
            s = session(path)
            if per_sess[s] >= args.cap_session:
                continue
            kept[path] = hs
            per_sess[s] += 1
        done += len(group)
        if done - last >= 2000:                         # heartbeat so a long scan visibly progresses
            print(f"  ...{done}/{len(frames)} scanned, {len(kept)} kept", flush=True)
            last = done
    print(f"  {len(kept)} hole-positive frames across {len(per_sess)} sessions")
    if not kept:
        raise SystemExit("no holes found — lower --hole-conf")

    # ---- pass 2: golf_ego_v2 -> ball + club on the kept frames ----
    print(f"pass 2 = golf_ego_v2 @ {args.bc_imgsz} (ball+club on kept frames)...")
    bc_m = YOLO(BC_W)
    tasks = []
    paths = sorted(kept)
    bc_batch = max(1, args.batch // 2)                  # 1280 is heavier than 960 -> smaller groups
    for group in chunked(paths, bc_batch):
        for path, r in zip(group, bc_m.predict(group, imgsz=args.bc_imgsz, conf=0.25, stream=True,
                                               device=0, verbose=False)):
            H, W = r.orig_shape
            results = [region(2, hb, W, H) for hb in kept[path]]                      # hole
            for b in r.boxes:                                                         # ball / club
                results.append(region(int(b.cls), b.xyxy[0].tolist(), W, H))
            name = os.path.basename(path)
            tasks.append({"data": {"image": f"/data/local-files/?d={DOC_ROOT_REL}/{name}"},
                          "predictions": [{"model_version": "hole_prelabel", "result": results}]})
    print(f"built {len(tasks)} tasks "
          f"({sum(len(kept[p]) for p in paths)} hole + ball/club pre-annotations)")

    if args.dry_run:
        json.dump(tasks, open("/tmp/hole_tasks.json", "w"))
        print("dry-run: wrote /tmp/hole_tasks.json, not creating the project")
        return

    # ---- create project + local storage + import ----
    h = {"Authorization": f"Token {TOKEN}"}
    pr = requests.post(f"{LS}/api/projects", headers=h,
                       json={"title": args.title, "label_config": CONFIG}).json()
    pid = pr["id"]
    print(f"created project {pid}: {args.title}")
    requests.post(f"{LS}/api/storages/localfiles", headers=h,
                  json={"project": pid, "path": FRAMES_DIR, "use_blob_urls": True,
                        "title": "golf_frames"})     # authorizes /data/local-files serving
    imp = requests.post(f"{LS}/api/projects/{pid}/import", headers=h, json=tasks)
    print(f"import status {imp.status_code}: {imp.json() if imp.status_code < 300 else imp.text[:200]}")
    print(f"\n-> open {LS}/projects/{pid}/data to start labeling (3 labels: ball/club_head/hole)")


if __name__ == "__main__":
    main()
