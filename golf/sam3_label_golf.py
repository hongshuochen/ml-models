#!/usr/bin/env python3
"""SAM 3.1 auto-labeler for golf videos — STANDALONE (run it on the H100 server).

For each .mp4 under VIDEOS_DIR, prompt SAM 3.1 with the class CONCEPTS (text) and let it detect +
TRACK them across every frame, then export YOLO-format labels + frames — the same images/ + labels/
layout the training builder eats. SAM 3.1's Object Multiplex tracks all classes in one pass.

Why SAM here: it needs no golf-specific training — a text prompt ("golf ball" / "golf club head" /
"golf hole") finds + tracks the object, so one clip is auto-labelled end to end. Strongest on the
NEW / hard classes (hole) and on propagating a label across a whole clip; verify small-ball recall
on your first clip (tiny fast objects are where a concept model can trail a trained detector).

Setup on the H100 box (needs internet once):
    pip install -U ultralytics opencv-python
    # sam3.pt (or the SAM 3.1 checkpoint) is GATED — request access + download from
    #   https://huggingface.co/facebook/sam3   then put the .pt next to this script (or pass --model)
    # if CLIP errors:  pip uninstall -y clip && pip install git+https://github.com/ultralytics/CLIP.git

Run (one process PER GPU — SAM 3.1 fits on a single H100 with room to spare, so use the 2nd GPU for
throughput, NOT model-parallel):
    CUDA_VISIBLE_DEVICES=0 python sam3_label_golf.py /videos out_sam --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 python sam3_label_golf.py /videos out_sam --shard 1/2 &

Output (feeds golf/build_and_train_golf.py as a --reviewed / --mined source after a human spot-check):
    out_sam/images/<video-id>_f######.jpg
    out_sam/labels/<video-id>_f######.txt      (YOLO: cls cx cy w h, normalized)
    out_sam/classes.txt                        (class order = the YOLO class ids)
"""
import argparse
import sys
from pathlib import Path

import cv2

try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError:
    sys.exit("pip install -U ultralytics opencv-python  (and download sam3.pt from HF)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="sam3.pt", help="SAM 3 / 3.1 checkpoint (.pt)")
    ap.add_argument("--prompts",
                    default="a small white golf ball on the grass:ball,a golf club head:club_head,"
                            "a golf hole cup on the green:hole",
                    help="comma list of 'text prompt:class_name'; ORDER = YOLO class id (keep "
                         "ball,club_head,hole). BE SPECIFIC — bare 'golf ball' over-fires on any round "
                         "object (verified: device icons -> many FPs); the specific phrasing is clean.")
    ap.add_argument("--conf", type=float, default=0.4, help="keep detections above this (SAM3 scores "
                    "are 0-1; 0.4-0.5 works well)")
    ap.add_argument("--imgsz", type=int, default=1024, help="inference size (higher helps the small ball)")
    ap.add_argument("--stride", type=int, default=5, help="write every Nth frame (SAM still tracks every frame)")
    ap.add_argument("--min-bytes", type=int, default=1_000_000)
    ap.add_argument("--shard", default="0/1", help="i/n: process only videos[i::n] — one shard per GPU")
    args = ap.parse_args()

    pairs = [p.split(":", 1) for p in args.prompts.split(",") if ":" in p]
    texts = [t.strip() for t, _ in pairs]
    class_names = [c.strip() for _, c in pairs]           # list index == YOLO class id
    i, n = (int(x) for x in args.shard.split("/"))

    root = Path(args.videos_dir)
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(class_names) + "\n")

    vids = sorted(p for p in root.rglob("*.mp4") if p.stat().st_size >= args.min_bytes)[i::n]
    print(f"shard {i}/{n}: {len(vids)} videos | prompts {texts} -> classes {class_names}", flush=True)

    for v in vids:
        vid = "_".join(v.relative_to(root).with_suffix("").parts)      # collision-safe id from the path
        predictor = SAM3VideoSemanticPredictor(overrides=dict(
            conf=args.conf, task="segment", mode="predict", imgsz=args.imgsz,
            model=args.model, verbose=False))
        kept = 0
        for idx, r in enumerate(predictor(source=str(v), text=texts, stream=True)):
            if idx % args.stride != 0 or r.boxes is None or len(r.boxes) == 0:
                continue
            H, W = r.orig_img.shape[:2]
            lines = []
            for b in r.boxes:
                cls = int(b.cls)                                        # index into `texts` -> YOLO id
                if cls >= len(class_names) or float(b.conf) < args.conf:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not lines:
                continue
            stem = f"{vid}_f{idx:06d}"
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), r.orig_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            kept += 1
        print(f"  {v.name} -> {kept} labeled frames", flush=True)


if __name__ == "__main__":
    main()
