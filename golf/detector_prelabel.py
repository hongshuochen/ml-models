#!/usr/bin/env python3
"""Pre-label EVERY sampled frame of every video with the trained golf detector (fast, stable — no
SAM, no OOM), for full Label-Studio review. Unlike select_review_frames (active-learning subset) or
mine_golf_videos (keeps only confident frames), this labels the whole set at --fps so a human can
review all of it.

For each .mp4 under VIDEOS_DIR: down-sample to --fps, run the detector, write image + YOLO box labels
(same layout sam3_label_golf uses, so make_ls_projects.py splits it into train/val/test LS projects).

Run (on the 3090 box, repo venv):
    python golf/detector_prelabel.py ~/ml-models/data/golf out_prelabel \
        --model runs/detect/golf_ego_v5_nomined/weights/best.pt --fps 3 --imgsz 1280

Output:
    out_prelabel/images/<video-id>_f######.jpg
    out_prelabel/labels/<video-id>_f######.txt   (YOLO: cls cx cy w h)
    out_prelabel/classes.txt
"""
import argparse
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("pip/uv install ultralytics opencv-python (run golf/setup_offline_env.sh)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="runs/detect/golf_ego_v5_nomined/weights/best.pt",
                    help="golf detector best.pt (3-class ball/club_head/hole)")
    ap.add_argument("--fps", type=float, default=3.0, help="sample this many frames/sec from each video")
    ap.add_argument("--conf", type=float, default=0.35, help="keep detections above this")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--keep-empty", action="store_true",
                    help="also save frames with NO detection (lets reviewers add misses — needed for a "
                         "thorough val/test GT, but many more frames)")
    ap.add_argument("--min-bytes", type=int, default=500_000)
    ap.add_argument("--shard", default="0/1", help="i/n: process only videos[i::n]")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    root = Path(args.videos_dir).expanduser()
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in args.shard.split("/"))

    model = YOLO(args.model)
    names = model.names
    (out / "classes.txt").write_text("\n".join(names[k] for k in sorted(names)) + "\n")

    vids = sorted(p for p in root.rglob("*.mp4") if p.stat().st_size >= args.min_bytes)[i::n]
    print(f"shard {i}/{n}: {len(vids)} videos | detector {Path(args.model).name} classes {names} | "
          f"fps {args.fps} conf {args.conf} imgsz {args.imgsz} keep_empty {args.keep_empty}", flush=True)

    grand = 0
    for v in vids:
        vid = "_".join(v.relative_to(root).with_suffix("").parts)
        cap = cv2.VideoCapture(str(v))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / args.fps))
        idx = kept = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if idx % step == 0:
                r = model.predict(f, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                H, W = f.shape[:2]
                lines = []
                for b in r.boxes:
                    cls = int(b.cls)
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    bw, bh = (x2 - x1) / W, (y2 - y1) / H
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if lines or args.keep_empty:
                    stem = f"{vid}_f{idx:06d}"
                    cv2.imwrite(str(out / "images" / f"{stem}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                    kept += 1
            idx += 1
        cap.release()
        grand += kept
        print(f"  {v.name} -> {kept} frames", flush=True)
    print(f"\n✅ {grand} frames labeled -> {out}/  (feed to make_ls_projects.py for train/val/test LS projects)")


if __name__ == "__main__":
    main()
