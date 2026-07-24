#!/usr/bin/env python3
"""SAM 3 / 3.1 batch auto-labeler for golf videos — STANDALONE (run on the H100 box).

For each .mp4 under VIDEOS_DIR: down-sample to --fps, cut long clips into fresh-tracked chunks,
text-prompt SAM with the class concepts, track them, and export BOTH YOLO detection boxes AND
segmentation masks (the box detector only needs boxes, but SAM gives masks for free and re-running
is many hours — save both as cheap insurance / tighter boxes / future seg). Output feeds
build_and_train_golf.py as a --mined-style source AFTER a human spot-check in Label Studio.

Setup on the box (needs internet once):
    bash golf/setup_offline_env.sh          # uv env + torch + ultralytics + opencv
    # download the gated checkpoint (HF token):
    #   ~/ml-models/.venv/bin/hf download facebook/sam3.1 --local-dir ~/sam3_1
    # if CLIP errors:  uv pip install --python .../python "git+https://github.com/ultralytics/CLIP.git"

Run (pick a free GPU with nvidia-smi first):
    CUDA_VISIBLE_DEVICES=5 ~/ml-models/.venv/bin/python golf/sam3_label_golf.py \
        ~/ml-models/data/golf out_sam --model ~/sam3_1/sam3.1_multiplex.pt --imgsz 1024
    # multi-GPU: one process per free card with --shard i/n (e.g. --shard 0/3, 1/3, 2/3)

Output (feeds golf/build_and_train_golf.py after review):
    out_sam/images/<video-id>_f######.jpg
    out_sam/labels/<video-id>_f######.txt        (YOLO box:  cls cx cy w h, normalized)
    out_sam/labels_seg/<video-id>_f######.txt    (YOLO-seg:  cls x1 y1 x2 y2 ... normalized polygon)
    out_sam/classes.txt
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2

try:
    import torch
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError:
    sys.exit("need ultralytics with SAM3 + opencv (run golf/setup_offline_env.sh)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="sam3.1_multiplex.pt", help="SAM 3 / 3.1 checkpoint (.pt)")
    ap.add_argument("--prompts", default="golf ball:ball,golf club head:club_head,golf hole:hole",
                    help="comma 'short noun phrase:class_name'; ORDER = YOLO class id. Short nouns, no article.")
    ap.add_argument("--conf", type=float, default=0.5, help="keep detections above this (0.5+ = fewer FPs)")
    ap.add_argument("--imgsz", type=int, default=1024, help="1024 is the speed/quality sweet spot; higher = slower")
    ap.add_argument("--fps", type=float, default=5.0, help="down-sample each video to this fps before tracking")
    ap.add_argument("--max-clip-secs", type=float, default=20.0,
                    help="cut clips into fresh-tracked chunks of this many source-seconds. Tracking memory "
                         "GROWS within a chunk (many tracked objects), so long chunks OOM even on 80GB with "
                         "3 concepts @1024 — keep it short (10-20s). Lower on OOM.")
    ap.add_argument("--no-masks", action="store_true", help="save only boxes (skip the labels_seg/ polygons)")
    ap.add_argument("--min-bytes", type=int, default=500_000)
    ap.add_argument("--shard", default="0/1", help="i/n: process only videos[i::n] — one shard per GPU")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    pairs = [p.split(":", 1) for p in args.prompts.split(",") if ":" in p]
    texts = [t.strip() for t, _ in pairs]
    class_names = [c.strip() for _, c in pairs]
    i, n = (int(x) for x in args.shard.split("/"))

    root = Path(args.videos_dir).expanduser()
    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if not args.no_masks:
        (out / "labels_seg").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(class_names) + "\n")
    tmpdir = out / "_tmp"; tmpdir.mkdir(exist_ok=True)

    vids = sorted(p for p in root.rglob("*.mp4") if p.stat().st_size >= args.min_bytes)[i::n]
    print(f"shard {i}/{n}: {len(vids)} videos | prompts {texts} -> {class_names} | "
          f"fps {args.fps} imgsz {args.imgsz} conf {args.conf} masks {not args.no_masks}", flush=True)

    def predictor():
        return SAM3VideoSemanticPredictor(overrides=dict(
            conf=args.conf, task="segment", mode="predict", imgsz=args.imgsz,
            model=str(Path(args.model).expanduser()), device=args.device, verbose=False))

    for v in vids:
        vid = "_".join(v.relative_to(root).with_suffix("").parts)   # collision-safe id from the path
        # 1) down-sample -> temp chunk videos (each <= max-clip-secs), remembering each frame's source index
        cap = cv2.VideoCapture(str(v))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / args.fps))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_frames = int(args.max_clip_secs * args.fps) if args.max_clip_secs > 0 else 10 ** 9
        chunks, writer, idx = [], None, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if idx % step == 0:
                if writer is None or len(chunks[-1][1]) >= cap_frames:
                    if writer is not None:
                        writer.release()
                    cpath = str(tmpdir / f"{vid}_c{len(chunks):03d}.mp4")
                    writer = cv2.VideoWriter(cpath, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                    chunks.append([cpath, []])
                writer.write(f); chunks[-1][1].append(idx)
            idx += 1
        cap.release()
        if writer is not None:
            writer.release()

        # 2) track each chunk with a FRESH predictor; write image + box + seg per detected frame
        kept = 0
        for cpath, src_idxs in chunks:
            pred = predictor()
            for fi, r in enumerate(pred(source=cpath, text=texts, stream=True)):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                H, W = r.orig_img.shape[:2]
                box_lines, seg_lines = [], []
                polys = (r.masks.xy if (r.masks is not None and not args.no_masks) else None)
                for bi, b in enumerate(r.boxes):
                    cls = int(b.cls)
                    if cls >= len(class_names) or float(b.conf) < args.conf:
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    bw, bh = (x2 - x1) / W, (y2 - y1) / H
                    box_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    if polys is not None and bi < len(polys) and len(polys[bi]) >= 3:
                        pts = " ".join(f"{px / W:.6f} {py / H:.6f}" for px, py in polys[bi])
                        seg_lines.append(f"{cls} {pts}")
                if not box_lines:
                    continue
                src = src_idxs[fi] if fi < len(src_idxs) else fi
                stem = f"{vid}_f{src:06d}"
                cv2.imwrite(str(out / "images" / f"{stem}.jpg"), r.orig_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / "labels" / f"{stem}.txt").write_text("\n".join(box_lines) + "\n")
                if seg_lines:
                    (out / "labels_seg" / f"{stem}.txt").write_text("\n".join(seg_lines) + "\n")
                kept += 1
            del pred                                   # free the chunk's tracker state + weights
            torch.cuda.empty_cache()
            Path(cpath).unlink(missing_ok=True)
        print(f"  {v.name} -> {kept} labeled frames ({len(chunks)} chunks)", flush=True)

    try:
        tmpdir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
