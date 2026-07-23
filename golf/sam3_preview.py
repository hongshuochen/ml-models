#!/usr/bin/env python3
"""Quick SAM 3 / 3.1 PREVIEW on ONE clip — text-prompt an object, track it, draw the boxes to an mp4
so you can eyeball how well SAM does (built to sanity-check the small golf ball before trusting SAM
to auto-label). Unlike sam3_label_golf.py (which writes YOLO labels for training), this just renders.

Prereq: the SAM 3 checkpoint `sam3.pt` is GATED — request access + download from
https://huggingface.co/facebook/sam3 (the SAM 3.1 checkpoint), then pass --model /path/to/sam3.pt.
(if CLIP errors:  pip uninstall -y clip && pip install git+https://github.com/ultralytics/CLIP.git)

Run (on the 3090 box, in the repo venv):
    ~/ml-models/.venv/bin/python golf/sam3_preview.py \
        ~/2026/dataset/golf/Outdoor/Michael/<a-clip>.mp4 michael_ball_sam.mp4 \
        --prompt "golf ball" --model sam3.pt --imgsz 1024
"""
import argparse
from pathlib import Path

import cv2

try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError:
    raise SystemExit("need ultralytics with SAM3 (from ultralytics.models.sam import SAM3VideoSemanticPredictor)")

BGR = [(255, 255, 0), (11, 158, 245), (94, 197, 34), (200, 120, 255)]   # cyan, amber, green, violet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--prompt", default="golf ball",
                    help="comma list of prompts. SAM 3 wants a SHORT NOUN PHRASE (Meta's own examples: "
                         "'ear','handle','laptop') — NOT an article or a descriptive sentence. The FP lever "
                         "is --conf: 'golf ball' @0.25 over-fires on round objects, @0.5 it's clean.")
    ap.add_argument("--model", default="sam3.pt", help="gated SAM 3/3.1 checkpoint")
    ap.add_argument("--imgsz", type=int, default=1024, help="raise (e.g. 1280) to help the small ball")
    ap.add_argument("--conf", type=float, default=0.5, help="SAM3 scores ARE 0-1; 0.5+ recommended")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    texts = [t.strip() for t in args.prompt.split(",") if t.strip()]
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30.0
    print(f"SAM preview | {Path(args.video).name} | prompts={texts} | imgsz={args.imgsz} conf={args.conf}", flush=True)

    predictor = SAM3VideoSemanticPredictor(overrides=dict(
        conf=args.conf, task="segment", mode="predict", imgsz=args.imgsz,
        model=args.model, device=args.device, verbose=False))

    writer = None
    frames = hit = nboxes = 0
    conf_sum = 0.0
    for r in predictor(source=args.video, text=texts, stream=True):
        img = r.orig_img.copy()
        n = 0
        if r.boxes is not None:
            for b in r.boxes:
                cls = int(b.cls)
                c = float(b.conf)
                if c < args.conf:
                    continue
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                color = BGR[cls % len(BGR)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                label = f"{texts[cls] if cls < len(texts) else cls} {int(c * 100)}%"
                cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                n += 1
                conf_sum += c
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (15, 15, 15), -1)
        cv2.putText(img, f"SAM  frame {frames}  dets {n}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            H, W = img.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        writer.write(img)
        frames += 1
        hit += 1 if n else 0
        nboxes += n
        if frames % 50 == 0:
            print(f"  {frames} frames, {hit} with a det, {nboxes} boxes total", flush=True)

    if writer:
        writer.release()
    print(f"\n✅ wrote {args.out}  ({frames} frames)")
    print(f"   frames with >=1 box: {hit}/{frames} ({hit/max(frames,1):.0%}) | "
          f"{nboxes} boxes total, {nboxes/max(hit,1):.1f} boxes per detected frame"
          + (f" | avg box conf {conf_sum/max(nboxes,1):.2f}" if nboxes else ""))
    print("   >1 box/frame usually = over-firing on round objects; use a more specific --prompt + higher --conf")
    print("   watch the mp4: does the box sit on the ball, and does it hold through the swing/roll?")


if __name__ == "__main__":
    main()
