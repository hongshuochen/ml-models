"""Annotate golf video(s) with the ball + club_head detector.

Input : a single video file OR a folder (all videos inside are processed).
Output: for each input, an annotated MP4 (`<name>_annotated.mp4`) with boxes drawn.

Works on any golf video; note the model is trained on THIRD-PERSON data, so a first-person
(AR-glasses) view will be weaker until we fine-tune on our own egocentric footage.

Examples:
    uv run python golf/annotate_video.py my_swing.mp4
    uv run python golf/annotate_video.py datasets/glasses_raw/ --out runs/golf_annotated
    uv run python golf/annotate_video.py clip.mov --model runs/detect/golf_detect_x_640/weights/best.pt --conf 0.3
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
DEFAULT_MODEL = "runs/detect/golf_detect_s_640/weights/best.pt"


def find_videos(inp: Path):
    if inp.is_dir():
        # skip our own outputs so re-running a folder doesn't re-annotate annotated clips
        return sorted(
            p for p in inp.iterdir()
            if p.suffix.lower() in VIDEO_EXTS and not p.stem.endswith("_annotated")
        )
    if inp.is_file() and inp.suffix.lower() in VIDEO_EXTS:
        return [inp]
    raise SystemExit(f"No video found at: {inp} (a file with a video extension, or a folder of them)")


def annotate_one(model, src: Path, out_dir: Path, conf, imgsz, device, line_width, max_frames):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src.stem}_annotated.mp4"
    writer = None
    n = 0
    counts = {}  # class name -> total instances drawn

    stream = model.predict(
        source=str(src), stream=True, imgsz=imgsz, conf=conf, device=device, verbose=False
    )
    for r in stream:
        frame = r.plot(line_width=line_width)  # BGR frame with boxes + labels
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        for c in r.boxes.cls.tolist():
            name = model.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        n += 1
        if total and n % 50 == 0:
            print(f"  {src.name}: {n}/{total} frames", end="\r")
        if max_frames and n >= max_frames:
            break

    if writer is not None:
        writer.release()
    tally = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no detections"
    print(f"  {src.name}: {n} frames @ {fps:.0f}fps -> {out_path}  ({tally})")
    return out_path, n, counts


def main():
    ap = argparse.ArgumentParser(description="Annotate golf video(s) with ball + club_head boxes.")
    ap.add_argument("input", type=Path, help="video file or folder of videos")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"detector weights (.pt/.tflite/.onnx) [default {DEFAULT_MODEL}]")
    ap.add_argument("--out", type=Path, default=None, help="output dir [default: <input>/annotated or the file's folder]")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold [0.25]")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size [640]")
    ap.add_argument("--device", default=None, help="cuda device e.g. 0, or cpu [auto]")
    ap.add_argument("--line-width", type=int, default=2, help="box line width [2]")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per video (0 = all; for quick tests)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}\nTrain it first (TRAINING.md §11) or pass --model.")
    videos = find_videos(args.input)
    # default output dir: <folder>/annotated for a folder, else next to the file
    out_dir = args.out or ((args.input / "annotated") if args.input.is_dir() else args.input.parent)

    print(f"Model: {args.model} | {len(videos)} video(s) | out -> {out_dir}")
    model = YOLO(args.model)
    done = 0
    for v in videos:
        annotate_one(model, v, out_dir, args.conf, args.imgsz, args.device, args.line_width, args.max_frames)
        done += 1
    print(f"Done: {done}/{len(videos)} video(s) -> {out_dir}")


if __name__ == "__main__":
    main()
