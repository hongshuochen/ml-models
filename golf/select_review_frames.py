#!/usr/bin/env python3
"""Active-learning frame selection for human review — STANDALONE (run on the video machine).

The auto-miner (mine_golf_videos.py) can only recover balls it can still find in a zoom crop; it is
BLIND to genuinely new conditions the model misses entirely (a new person, ball, lighting, or scene).
Those blind spots are exactly what "more people & scenes" introduces — so the highest-ROI human
effort is a SMALL set of frames chosen to cover the NEW diversity and the model's UNCERTAINTY, not
thousands of random frames.

This script recursively scans a video folder, coarse-samples frames, and greedily selects a budgeted
set that is simultaneously:
  * UNCERTAIN / likely-wrong — scored from the detector, weighted toward the ball-recall weakness:
      club_head present but NO ball   (a ball is almost surely there and was missed)  -> highest
      green golf scene but NO ball     (probable total miss)                           -> high
      mid-confidence detections (0.2-0.5)                                              -> medium
  * DIVERSE — a cheap appearance descriptor (16x16 gray + hue histogram) with a farthest-point
      penalty so near-duplicate frames (and one long video) can't dominate the budget.

Output (set up Label Studio local storage on OUT/images, or just use the YOLO pre-labels):
  OUT/images/<video-id>_f######.jpg          selected full-res frames
  OUT/labels/<video-id>_f######.txt          YOLO pre-labels (the detector's guess, for correction)
  OUT/ls_tasks.json                          Label Studio import with pre-annotations
  OUT/manifest.csv                           per-frame score breakdown (why it was picked)

For the 3-class extension (adding `hole`): pass `--hole-model` (the bootstrap hole teacher). Its
hole boxes are merged in as class 2 while ball/club_head come from `--model` (golf_ego_v2) —
"two-model pre-annotation". The selection then also favours hole sightings and puttable scenes
where the (weak) teacher missed a hole, so the human labels the new class where it actually occurs.

Setup:  pip install ultralytics opencv-python ; copy best.pt (+ the hole teacher) over
Run:    python select_review_frames.py /videos out_review --model golf_ego_v2.pt \
            [--hole-model golf_hole_teacher.pt] --total 500
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("pip install ultralytics opencv-python")

BALL, CLUB, HOLE = 0, 1, 2
LABELS = ["ball", "club_head", "hole"]


def vid_id(root: Path, video: Path) -> str:
    rel = video.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rel))


def descriptor(bgr) -> np.ndarray:
    """Cheap appearance signature: 16x16 gray (256) + 12-bin hue histogram, L2-normalized."""
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (16, 16)).astype(np.float32).ravel() / 255.0
    hsv = cv2.cvtColor(cv2.resize(bgr, (64, 64)), cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0], None, [12], [0, 180]).ravel()
    hh = hh / (hh.sum() + 1e-6)
    v = np.concatenate([g, hh]).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def green_frac(bgr) -> float:
    hsv = cv2.cvtColor(cv2.resize(bgr, (64, 64)), cv2.COLOR_BGR2HSV)
    m = (hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 60)
    return float(m.mean())


def uncertainty(dets, green) -> tuple:
    """(score, reason) from one frame's [(box, conf, cls)] + greenness. Weighted for the two
    scarce/weak classes we most want more labels of: ball (recall) and hole (new class)."""
    n_ball = sum(1 for _b, _c, cl in dets if cl == BALL)
    n_club = sum(1 for _b, _c, cl in dets if cl == CLUB)
    hole_conf = max((c for _b, c, cl in dets if cl == HOLE), default=0.0)
    midconf = sum(1 for _b, c, cl in dets if cl in (BALL, CLUB) and 0.20 <= c < 0.50)
    s, why = 0.0, []
    # ball-recall signals
    if n_club > 0 and n_ball == 0:
        s += 3.0; why.append("club_no_ball")           # ball almost surely present & missed
    if green > 0.45 and n_ball == 0 and n_club == 0 and hole_conf == 0:
        s += 1.5; why.append("green_no_det")           # probable total miss in a golf scene
    if midconf:
        s += 0.6 * midconf; why.append(f"midconf{midconf}")
    # hole signals (new class — every hole example is valuable; the teacher is weak, so BOTH a
    # hole sighting to verify AND a puttable scene the teacher missed a hole in are worth labeling)
    if hole_conf > 0:
        s += 2.0 if hole_conf >= 0.5 else 1.2; why.append("hole_seen")
    elif green > 0.45 and (n_ball > 0 or n_club > 0):
        s += 1.0; why.append("putt_no_hole")           # puttable scene, hole likely present & missed
    return s, "+".join(why) if why else "diverse"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="best.pt", help="ball+club_head detector (golf_ego_v2)")
    ap.add_argument("--hole-model", default=None,
                    help="hole-only teacher weights; if given, its hole boxes are merged in as "
                         "class 2 (two-model pre-annotation). Omit for the ball+club-only pass.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15, help="detection floor (low, to see the misses)")
    ap.add_argument("--hole-conf", type=float, default=0.15, help="hole-teacher detection floor")
    ap.add_argument("--coarse-sec", type=float, default=0.5, help="s between analysed frames")
    ap.add_argument("--total", type=int, default=500, help="frames to select for review")
    ap.add_argument("--per-video-cap", type=int, default=40)
    ap.add_argument("--novelty", type=float, default=1.0, help="diversity weight (higher = spread wider)")
    ap.add_argument("--min-bytes", type=int, default=1_000_000)
    ap.add_argument("--device", default="")
    ap.add_argument("--ls-prefix", default="/data/local-files/?d=images/",
                    help="Label Studio local-storage URL prefix for the images/ folder")
    args = ap.parse_args()

    root, out = Path(args.videos_dir), Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    hole_model = YOLO(args.hole_model) if args.hole_model else None
    videos = sorted(p for p in root.rglob("*.mp4") if p.stat().st_size >= args.min_bytes)
    if not videos:
        sys.exit(f"no .mp4 >= {args.min_bytes} bytes under {root}")
    print(f"{len(videos)} videos; coarse-scanning at {args.coarse_sec}s spacing"
          + (" (+hole teacher)" if hole_model else "") + "...")

    # ---- coarse scan: one descriptor + score + dets per sampled frame (frames not held in RAM) ----
    cand = []   # dict(vid, video, fidx, desc, score, reason, dets, wh)
    for vi, v in enumerate(videos):
        vid = vid_id(root, v)
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(args.coarse_sec * fps)))
        cap.release()
        dev = args.device or None
        gen = model.predict(source=str(v), stream=True, imgsz=args.imgsz, conf=args.conf,
                            vid_stride=step, device=dev, verbose=False)
        # two-model pre-annotation: zip a second stream (same video+stride -> aligned frames) for
        # the hole class; ball/club_head come from `model`, hole from `hole_model`
        if hole_model is not None:
            genh = hole_model.predict(source=str(v), stream=True, imgsz=args.imgsz, conf=args.hole_conf,
                                      vid_stride=step, device=dev, verbose=False)
            pairs = zip(gen, genh)
        else:
            pairs = ((r, None) for r in gen)
        for k, (r, rh) in enumerate(pairs):
            img = r.orig_img
            dets = [(b.xyxy[0].tolist(), float(b.conf), int(b.cls)) for b in r.boxes]  # ball/club
            if rh is not None:
                for b in rh.boxes:                       # hole teacher is 1-class -> map to HOLE(2)
                    dets.append((b.xyxy[0].tolist(), float(b.conf), HOLE))
            g = green_frac(img)
            s, why = uncertainty(dets, g)
            cand.append({"vid": vid, "video": str(v), "fidx": k * step, "desc": descriptor(img),
                         "score": s, "reason": why, "dets": dets, "wh": (img.shape[1], img.shape[0])})
        print(f"  [{vi + 1}/{len(videos)}] {vid}: {sum(1 for c in cand if c['vid'] == vid)} frames scanned")

    if not cand:
        sys.exit("no frames scanned")
    D = np.stack([c["desc"] for c in cand])
    u = np.array([c["score"] for c in cand], np.float32)
    u = (u - u.min()) / (u.max() - u.min() + 1e-6) + 0.05     # keep a floor so pure-diverse frames can win

    # ---- greedy selection: maximize uncertainty x novelty(min-dist to already-picked) ----
    n = len(cand)
    picked, per_vid = [], {}
    mind = np.full(n, 1e9, np.float32)                       # min descriptor distance to the picked set
    avail = np.ones(n, bool)
    budget = min(args.total, n)
    while len(picked) < budget:
        nov = np.where(np.isfinite(mind) & (mind < 1e9), mind, 1.0)
        gain = u * (nov ** args.novelty if picked else 1.0)
        gain[~avail] = -1
        i = int(np.argmax(gain))
        if gain[i] < 0:
            break
        picked.append(i)
        avail[i] = False
        if per_vid.get(cand[i]["vid"], 0) + 1 >= args.per_video_cap:
            for j in range(n):
                if cand[j]["vid"] == cand[i]["vid"]:
                    avail[j] = False
        per_vid[cand[i]["vid"]] = per_vid.get(cand[i]["vid"], 0) + 1
        d = np.linalg.norm(D - D[i], axis=1)                 # update novelty distances
        mind = np.minimum(mind, d)

    # ---- write selected frames, YOLO pre-labels, and a Label Studio import ----
    caps = {}
    tasks, rows = [], []
    for rank, i in enumerate(sorted(picked, key=lambda j: (cand[j]["video"], cand[j]["fidx"]))):
        c = cand[i]
        cap = caps.get(c["video"]) or cv2.VideoCapture(c["video"])
        caps[c["video"]] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["fidx"])
        ok, img = cap.read()
        if not ok:
            continue
        W, H = c["wh"]
        name = f'{c["vid"]}_f{c["fidx"]:06d}'
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines, results = [], []
        for (box, conf, cls) in c["dets"]:
            x1, y1, x2, y2 = box
            lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            results.append({"type": "rectanglelabels", "from_name": "label", "to_name": "image",
                            "original_width": W, "original_height": H, "image_rotation": 0,
                            "value": {"x": 100 * x1 / W, "y": 100 * y1 / H,
                                      "width": 100 * (x2 - x1) / W, "height": 100 * (y2 - y1) / H,
                                      "rotation": 0, "rectanglelabels": [LABELS[cls]]}})
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        tasks.append({"data": {"image": f"{args.ls_prefix}{name}.jpg"},
                      "predictions": [{"model_version": "auto", "result": results}]})
        rows.append([name, c["vid"], c["fidx"], round(c["score"], 2), c["reason"]])
    for cap in caps.values():
        cap.release()

    (out / "ls_tasks.json").write_text(json.dumps(tasks, indent=1))
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "video", "src_frame", "score", "reason"]); w.writerows(rows)

    from collections import Counter
    reasons = Counter(r[4] for r in rows)
    labels_hint = "3 labels: ball, club_head, hole" if hole_model else "2 labels: ball, club_head"
    print(f"\nSelected {len(rows)} frames for review across {len(per_vid)} videos.")
    print("reasons:", dict(reasons))
    print(f"-> import {out/'ls_tasks.json'} into Label Studio ({labels_hint}), correct, "
          f"export YOLO. Or hand-fix the YOLO {out/'labels'}/ directly.")


if __name__ == "__main__":
    main()
