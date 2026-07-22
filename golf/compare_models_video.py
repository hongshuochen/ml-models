#!/usr/bin/env python3
"""Run ONE video through TWO detectors side-by-side to eyeball the difference (built for the 'hole'
class, which the fixed val can't measure). Left panel = model A, right panel = model B; each frame
gets ball/club_head/hole boxes + a per-frame hole count, and the end prints how many frames each
model found a hole in.

Colors match the Android app: ball=cyan, club_head=amber, hole=green (thick).

Run (offline box):
    uv run python golf/compare_models_video.py CLIP.mp4 compare.mp4 \
        --a v3=~/golf_offline/best.pt \
        --b v5=~/golf_offline/runs/detect/golf_ego_v5_nomined/weights/best.pt \
        --imgsz 1280 --conf 0.25
Add --stride 2 to go ~2x faster (skips frames), --max-secs N to cut it short.
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

BGR = {"ball": (255, 255, 0), "club_head": (11, 158, 245), "hole": (94, 197, 34)}
FALLBACK = (220, 220, 220)


def load_spec(spec):
    """'name=path' -> (name, YOLO). Bare 'path' -> (stem, YOLO)."""
    name, _, path = spec.partition("=")
    if not path:
        path, name = name, Path(name).stem
    return name, YOLO(str(Path(path).expanduser()))


def draw(frame, res, title):
    """Draw one model's boxes on a copy of frame; return (panel, n_holes)."""
    img = frame.copy()
    names = res.names
    n_holes = 0
    for b in res.boxes:
        cls = names[int(b.cls)]
        if cls == "hole":
            n_holes += 1
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        color = BGR.get(cls, FALLBACK)
        thick = 5 if cls == "hole" else 3
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        tag = f"{cls} {int(float(b.conf) * 100)}%"
        cv2.putText(img, tag, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    # header bar: model name + hole count
    bar = f"{title}   holes: {n_holes}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (15, 15, 15), -1)
    hc = BGR["hole"] if n_holes else (200, 200, 200)
    cv2.putText(img, bar, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hc, 2, cv2.LINE_AA)
    return img, n_holes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--a", required=True, help="left model 'name=weights.pt'")
    ap.add_argument("--b", required=True, help="right model 'name=weights.pt'")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--panel-h", type=int, default=720, help="height each panel is resized to")
    ap.add_argument("--max-secs", type=float, default=0.0, help="stop after N seconds of source (0=all)")
    args = ap.parse_args()

    na, ma = load_spec(args.a)
    nb, mb = load_spec(args.b)
    print(f"A(left)={na}  B(right)={nb}  | imgsz={args.imgsz} conf={args.conf} stride={args.stride}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frame = int(args.max_secs * src_fps) if args.max_secs else 0

    writer = None
    idx = written = 0
    holes_a = holes_b = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frame and idx > max_frame):
            break
        if idx % args.stride != 0:
            idx += 1
            continue
        ra = ma.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        rb = mb.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        pa, ha = draw(frame, ra, na)
        pb, hb = draw(frame, rb, nb)
        holes_a += 1 if ha else 0
        holes_b += 1 if hb else 0

        # resize both panels to a common height, stack horizontally
        def fit(p):
            h = args.panel_h
            w = int(p.shape[1] * h / p.shape[0])
            return cv2.resize(p, (w, h))
        combo = cv2.hconcat([fit(pa), fit(pb)])
        if writer is None:
            H, W = combo.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     src_fps / args.stride, (W, H))
        writer.write(combo)
        written += 1
        if written % 50 == 0:
            print(f"  {written} frames  (holes so far: {na}={holes_a} {nb}={holes_b})", flush=True)
        idx += 1

    cap.release()
    if writer:
        writer.release()
    print(f"\n✅ wrote {args.out}  ({written} frames)")
    print(f"   frames-with-a-hole:  {na} = {holes_a}   |   {nb} = {holes_b}")
    print("   (higher = detects the cup in more frames; watch the video for false holes too)")


if __name__ == "__main__":
    main()
