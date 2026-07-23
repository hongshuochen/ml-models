#!/usr/bin/env python3
"""SAM 2 preview on ONE clip -> annotated mp4. SAM 2 has NO text prompts (that's SAM 3), so we seed
it with our trained golf detector: the detector finds the ball box each frame, SAM 2 turns each box
into a precise segmentation mask, and we overlay mask + box -> mp4. Lets you see what SAM 2's masks
add over plain detector boxes (tight pixel outline of the ball), on real golf footage.

sam2 checkpoints auto-download (NOT gated): sam2_t.pt (tiny, fast) / sam2_b.pt / sam2_l.pt.

Run (3090 box, repo venv):
    ~/ml-models/.venv/bin/python golf/sam2_preview.py \
        <a-clip>.mp4 ball_sam2.mp4 \
        --detector runs/detect/golf_ego_v5_nomined/weights/best.pt \
        --model sam2_b.pt --classes ball
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM, YOLO

BGR = {"ball": (255, 255, 0), "club_head": (11, 158, 245), "hole": (94, 197, 34)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--detector", default="runs/detect/golf_ego_v5_nomined/weights/best.pt",
                    help="golf detector that seeds SAM with boxes")
    ap.add_argument("--model", default="sam2_b.pt", help="SAM2 checkpoint (auto-downloads; sam2_t/_b/_l)")
    ap.add_argument("--classes", default="ball", help="comma detector classes to segment (e.g. ball,hole)")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence floor")
    ap.add_argument("--imgsz", type=int, default=1280, help="detector input size")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    want = {c.strip() for c in args.classes.split(",") if c.strip()}
    det = YOLO(args.detector)
    sam = SAM(args.model)
    names = det.names
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30.0
    print(f"SAM2 preview | {Path(args.video).name} | seed={Path(args.detector).name} "
          f"classes={sorted(want)} | model={args.model}", flush=True)

    cap = cv2.VideoCapture(args.video)
    writer = None
    idx = frames = seg = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        # 1) detector -> boxes of the wanted classes
        r = det.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        boxes, labels = [], []
        for b in r.boxes:
            cls = names[int(b.cls)]
            if cls in want:
                boxes.append([float(v) for v in b.xyxy[0].tolist()])
                labels.append(cls)
        img = frame.copy()
        # 2) SAM2: each box -> mask
        if boxes:
            sr = sam.predict(frame, bboxes=boxes, device=args.device, verbose=False)[0]
            masks = sr.masks.data.cpu().numpy() if sr.masks is not None else np.zeros((0,) + frame.shape[:2])
            for i, (box, cls) in enumerate(zip(boxes, labels)):
                color = BGR.get(cls, (200, 200, 200))
                if i < len(masks):
                    m = masks[i].astype(bool)
                    overlay = img.copy(); overlay[m] = color
                    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, cls, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            seg += 1
        cv2.rectangle(img, (0, 0), (img.shape[1], 32), (15, 15, 15), -1)
        cv2.putText(img, f"SAM2  frame {frames}  boxes {len(boxes)}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            H, W = img.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (W, H))
        writer.write(img)
        frames += 1
        if frames % 50 == 0:
            print(f"  {frames} frames, {seg} segmented", flush=True)
        idx += 1

    cap.release()
    if writer:
        writer.release()
    print(f"\n✅ wrote {args.out}  ({frames} frames, {seg} had a segmented object)")
    print("   watch the mp4: does SAM's mask hug the ball tightly through the swing/roll?")


if __name__ == "__main__":
    main()
