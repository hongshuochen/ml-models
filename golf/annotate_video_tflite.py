#!/usr/bin/env python3
"""Annotate a video with the RAW-HEAD golf .tflite EXACTLY as the phone does — true device parity.

The deployed golf_v*.tflite is a raw-head export (3 NHWC maps [1,G,G,4+nc], strides 8/16/32, reg_max=1,
no DFL/NMS on-graph). Ultralytics' YOLO()/annotate_video.py CANNOT decode it — the decode lives in
GolfDetector.kt. This script mirrors that Kotlin decode 1:1 (RGB/255 input, per-cell sigmoid+argmax,
box = (cell+0.5 ± ltrb)*stride/INPUT, per-class greedy NMS) so the boxes match what the phone draws.

Use this when you want the PHONE's behavior (640, f16, raw-head). For a quick approximation the v5 .pt
at --imgsz 640 is very close (parity-verified) — annotate_video.py does that.

    python annotate_video_tflite.py <video-or-folder> --model golf_v5.tflite --conf 0.5

Standalone (no other repo files needed) — shippable as just this script + the .tflite + RUN_VIDEO_GUIDE.md.
Needs a TFLite runtime: tries ai_edge_litert, tflite_runtime, then tensorflow.lite.
Install one: `pip install ai-edge-litert opencv-python-headless numpy`.
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
# BGR, matches the app overlay: ball cyan, club_head amber, hole green (#22c55e)
COLORS = [(255, 255, 0), (11, 158, 245), (94, 197, 34), (200, 120, 255), (0, 165, 255)]


def load_interpreter(model):
    for mod, cls in (("ai_edge_litert.interpreter", "Interpreter"),
                     ("tflite_runtime.interpreter", "Interpreter"),
                     ("tensorflow.lite", "Interpreter")):
        try:
            m = __import__(mod, fromlist=[cls])
            it = getattr(m, cls)(model_path=str(model))
            it.allocate_tensors()
            return it
        except ImportError:
            continue
    raise SystemExit("no TFLite runtime — `uv pip install --python ~/ml-models/.venv/bin/python ai-edge-litert`")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(outputs, inp, nc, score_thr):
    """outputs: list of [G,G,4+nc] arrays. -> list of (x1,y1,x2,y2,score,cls) normalized [0,1]."""
    dets = []
    for arr in outputs:
        G = arr.shape[0]
        stride = inp / G
        box = arr[:, :, :4]                       # l,t,r,b  (direct distances, grid units)
        logits = arr[:, :, 4:4 + nc]
        cls = logits.argmax(-1)
        best = np.take_along_axis(logits, cls[..., None], -1)[..., 0]
        score = sigmoid(best)
        ys, xs = np.where(score > score_thr)      # axis0=row(y), axis1=col(x)
        for y, x in zip(ys.tolist(), xs.tolist()):
            l, t, r, b = box[y, x]
            ax, ay = x + 0.5, y + 0.5
            x1 = min(max((ax - l) * stride / inp, 0.0), 1.0)
            y1 = min(max((ay - t) * stride / inp, 0.0), 1.0)
            x2 = min(max((ax + r) * stride / inp, 0.0), 1.0)
            y2 = min(max((ay + b) * stride / inp, 0.0), 1.0)
            dets.append((x1, y1, x2, y2, float(score[y, x]), int(cls[y, x])))
    return dets


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_per_class(dets, iou_thr):
    keep = []
    for c in {d[5] for d in dets}:
        cand = sorted([d for d in dets if d[5] == c], key=lambda d: -d[4])
        dead = [False] * len(cand)
        for i in range(len(cand)):
            if dead[i]:
                continue
            keep.append(cand[i])
            for j in range(i + 1, len(cand)):
                if not dead[j] and iou(cand[i], cand[j]) > iou_thr:
                    dead[j] = True
    return keep


def process(video, it, args, names, out_dir):
    inp = args.input_size
    in_idx = it.get_input_details()[0]["index"]
    out_details = sorted(it.get_output_details(), key=lambda d: -d["shape"][1])  # big grid first
    nc = int(out_details[0]["shape"][-1]) - 4

    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = out_dir / f"{video.stem}_tflite.mp4"
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tally, n = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # preprocess EXACTLY like the app: resize (squash) to INPUT, RGB, /255
        rgb = cv2.cvtColor(cv2.resize(frame, (inp, inp)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        it.set_tensor(in_idx, rgb[None])
        it.invoke()
        outs = [it.get_tensor(d["index"])[0] for d in out_details]
        dets = nms_per_class(decode(outs, inp, nc, args.conf), args.iou)
        for x1, y1, x2, y2, sc, c in dets:
            col = COLORS[c % len(COLORS)]
            p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
            cv2.rectangle(frame, p1, p2, col, args.line_width)
            nm = names[c] if c < len(names) else str(c)
            cv2.putText(frame, f"{nm} {sc:.2f}", (p1[0], max(14, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            tally[nm] = tally.get(nm, 0) + 1
        vw.write(frame)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release(); vw.release()
    print(f"  {video.name}: {n} frames -> {out_path.name}  | dets {tally}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="video file or folder")
    ap.add_argument("--model", required=True, help="raw-head golf_v*.tflite")
    ap.add_argument("--out", type=Path, default=None, help="output dir [default: alongside input]")
    ap.add_argument("--conf", type=float, default=0.5, help="score threshold (app SCORE_THRESHOLD=0.5)")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU (app NMS_IOU=0.5)")
    ap.add_argument("--input-size", type=int, default=640, help="model input (app INPUT=640)")
    ap.add_argument("--names", default="ball,club_head,hole")
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames/video (0=all)")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",")]
    it = load_interpreter(args.model)
    print(f"model {Path(args.model).name} | input "
          f"{it.get_input_details()[0]['shape'].tolist()} | classes {names}")

    if args.input.is_dir():
        vids = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    else:
        vids = [args.input]
    out_dir = args.out or (args.input if args.input.is_dir() else args.input.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(vids)} video(s) -> {out_dir}/")
    for v in vids:
        process(v, it, args, names, out_dir)


if __name__ == "__main__":
    main()
