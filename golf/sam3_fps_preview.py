#!/usr/bin/env python3
"""SAM 3 / 3.1 on ONE clip, DOWN-SAMPLED to N fps, optionally CHUNKED -> annotated JPGs + mp4.

Why downsample the INPUT: SAM's video tracker runs inference on EVERY frame it's given, so skipping
saves doesn't skip compute — we first write temp video(s) at --fps (e.g. 5) and SAM only tracks those.
Why chunk (--max-clip-secs): a long clip is slow (linear in frames) AND the concept tracker drifts
over minutes / scene changes, so we cut long inputs into fresh-tracked segments. Segments
ball/club_head/hole in one pass; saves each annotated frame as a JPG to flip through (+ preview.mp4).

Run (3090/H100, repo venv):
    python golf/sam3_fps_preview.py <clip>.mp4 out_sam_preview --fps 5 --max-clip-secs 60 \
        --model ~/sam3_1/sam3.1_multiplex.pt --imgsz 1024
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut OOM from fragmentation

import cv2
import torch
from ultralytics.models.sam import SAM3VideoSemanticPredictor

BGR = {"ball": (255, 255, 0), "club_head": (11, 158, 245), "hole": (94, 197, 34)}


def new_predictor(args):
    return SAM3VideoSemanticPredictor(overrides=dict(
        conf=args.conf, task="segment", mode="predict", imgsz=args.imgsz,
        model=str(Path(args.model).expanduser()), device=args.device, verbose=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=5.0, help="down-sample the input to this fps (the speed lever)")
    ap.add_argument("--max-clip-secs", type=float, default=60.0,
                    help="cut inputs longer than this (source seconds) into fresh-tracked chunks; 0 = no chunking")
    ap.add_argument("--model", default="sam3.1_multiplex.pt", help="SAM3/3.1 checkpoint")
    ap.add_argument("--prompts", default="golf ball:ball",
                    help="comma 'short noun phrase:class'; each extra concept multiplies tracking memory")
    ap.add_argument("--conf", type=float, default=0.5, help="FP lever (0.5+); low conf over-fires")
    ap.add_argument("--imgsz", type=int, default=1024, help="lower on OOM; higher helps the small ball but is slower")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    pairs = [p.split(":", 1) for p in args.prompts.split(",") if ":" in p]
    texts = [t.strip() for t, _ in pairs]
    names = [c.strip() for _, c in pairs]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) down-sample the input to temp chunk video(s) at --fps, each <= max-clip-secs of source
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / args.fps))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    chunk_cap = int(args.max_clip_secs * args.fps) if args.max_clip_secs > 0 else 10 ** 9
    chunks, writer, in_chunk, idx, kept = [], None, 0, 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if writer is None or in_chunk >= chunk_cap:
                if writer is not None:
                    writer.release()
                cpath = str(out / f"_chunk{len(chunks):03d}.mp4")
                writer = cv2.VideoWriter(cpath, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                chunks.append(cpath); in_chunk = 0
            writer.write(f); in_chunk += 1; kept += 1
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    print(f"input {idx} frames @{src_fps:.0f}fps -> {kept} frames @{args.fps:.0f}fps in {len(chunks)} chunk(s)", flush=True)

    # 2) SAM tracks each chunk with a FRESH tracker (no drift across chunks)
    print(f"running SAM ({Path(args.model).name}) prompts={texts} conf={args.conf} imgsz={args.imgsz}...", flush=True)
    t0 = time.time()
    out_writer = None
    gi = nboxes = 0
    for ci, chunk in enumerate(chunks):
        pred = new_predictor(args)
        for r in pred(source=chunk, text=texts, stream=True):
            img = r.orig_img.copy()
            n = 0
            for b in (r.boxes or []):
                cls = int(b.cls); nm = names[cls] if cls < len(names) else str(cls)
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                color = BGR.get(nm, (220, 220, 220))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, f"{nm} {float(b.conf):.2f}", (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                n += 1
            nboxes += n
            cv2.rectangle(img, (0, 0), (img.shape[1], 32), (15, 15, 15), -1)
            cv2.putText(img, f"SAM  chunk {ci}  frame {gi}  boxes {n}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / f"f{gi:04d}.jpg"), img)
            if out_writer is None:
                out_writer = cv2.VideoWriter(str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.fps, (img.shape[1], img.shape[0]))
            out_writer.write(img)
            gi += 1
        del pred
        torch.cuda.empty_cache()
        if gi % 50 < args.fps:  # occasional heartbeat
            print(f"  chunk {ci + 1}/{len(chunks)} done, {gi} frames, {nboxes} boxes", flush=True)
    if out_writer:
        out_writer.release()
    dt = time.time() - t0

    for c in chunks:
        Path(c).unlink(missing_ok=True)
    print(f"\n✅ {gi} frames in {dt:.1f}s ({dt / max(gi, 1):.2f}s/frame, {gi / max(dt, 1e-9):.1f} fps) "
          f"| {len(chunks)} chunks | {nboxes} boxes total")
    print(f"   JPGs + preview.mp4 in {out}/  — flip through the f####.jpg to eyeball labels")


if __name__ == "__main__":
    main()
