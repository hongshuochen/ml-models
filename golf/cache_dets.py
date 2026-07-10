"""Cache per-frame YOLO golf detections (ball + club_head boxes) for a video to JSON.

Output: {"fps": float, "frames": [{"b": [[x1,y1,x2,y2],...], "c": [[x1,y1,x2,y2,conf],...]}, ...]}
(coords in full-res pixels — the same layout hit_detector.detect_hits consumes).

Run: uv run python golf/cache_dets.py <video> <out.json> [--imgsz 1280]
"""
import argparse
import json

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--model", default="runs/detect/golf_ego_v2_1280/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    m = YOLO(args.model)
    frames = []
    for r in m.predict(source=args.video, stream=True, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False):
        b = [box.xyxy[0].tolist() for box in r.boxes if int(box.cls) == 0]
        c = [box.xyxy[0].tolist() + [float(box.conf)] for box in r.boxes if int(box.cls) == 1]
        frames.append({"b": b, "c": c})
    with open(args.out, "w") as f:
        json.dump({"fps": fps, "frames": frames}, f)
    print(f"{args.video}: {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
