#!/usr/bin/env python3
"""SAM 3 / 3.1 on ONE clip, DOWN-SAMPLED to N fps first (the real speed-up) -> annotated JPGs + mp4.

Why downsample the INPUT (not just the output): SAM's video tracker runs inference on EVERY frame of
whatever source it's given — skipping saves doesn't skip compute. So we first write a temp video at
--fps (e.g. 5), then SAM only tracks those frames -> ~6x faster than 30 fps, tracking still coherent
at 0.2 s gaps. Segments ball/club_head/hole in one pass and saves each annotated frame as a JPG you
can flip through (+ a preview mp4). Prints wall-clock so you can see the speed.

Run (3090, repo venv):
    ~/ml-models/.venv/bin/python golf/sam3_fps_preview.py \
        <clip>.mp4 out_sam_preview --fps 5 --model ~/sam3_1/sam3.1_multiplex.pt --imgsz 1280
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # cut OOM from fragmentation

import cv2
from ultralytics.models.sam import SAM3VideoSemanticPredictor

BGR = {"ball": (255, 255, 0), "club_head": (11, 158, 245), "hole": (94, 197, 34)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=5.0, help="down-sample the input to this fps (the speed lever)")
    ap.add_argument("--model", default="sam3.1_multiplex.pt", help="SAM3/3.1 checkpoint")
    ap.add_argument("--prompts", default="golf ball:ball",
                    help="comma 'short noun phrase:class'; SAM wants short nouns, no article. Each extra "
                         "concept multiplies tracking memory — 3 concepts @1280 OOMs even on 24GB, so the "
                         "default is ONE (ball). Add more (…,golf club head:club_head,golf hole:hole) only "
                         "if it fits, or run one concept per pass.")
    ap.add_argument("--conf", type=float, default=0.5, help="FP lever (0.5+); low conf over-fires")
    ap.add_argument("--imgsz", type=int, default=1024, help="lower this on OOM (1280 needs lots of VRAM "
                    "with multi-concept tracking; 1024/960/768 are safer)")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    pairs = [p.split(":", 1) for p in args.prompts.split(",") if ":" in p]
    texts = [t.strip() for t, _ in pairs]
    names = [c.strip() for _, c in pairs]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) down-sample the input to a temp video at --fps
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / args.fps))
    tmp = str(out / "_downsampled.mp4")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    idx = kept = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx % step == 0:
            vw.write(f); kept += 1
        idx += 1
    cap.release(); vw.release()
    print(f"input {idx} frames @{src_fps:.0f}fps -> {kept} frames @{args.fps:.0f}fps (every {step}th)", flush=True)

    # 2) SAM tracks the down-sampled video
    p = SAM3VideoSemanticPredictor(overrides=dict(
        conf=args.conf, task="segment", mode="predict", imgsz=args.imgsz,
        model=str(Path(args.model).expanduser()), device=args.device, verbose=False))
    print(f"running SAM ({Path(args.model).name}) prompts={texts} conf={args.conf} imgsz={args.imgsz}...", flush=True)

    t0 = time.time()
    writer = None
    nboxes = 0
    for i, r in enumerate(p(source=tmp, text=texts, stream=True)):
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
        cv2.imwrite(str(out / f"f{i:04d}.jpg"), img)           # flip through these
        if writer is None:
            writer = cv2.VideoWriter(str(out / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer:
        writer.release()
    dt = time.time() - t0

    Path(tmp).unlink(missing_ok=True)
    print(f"\n✅ {kept} frames in {dt:.1f}s ({dt/max(kept,1):.2f}s/frame, {kept/max(dt,1e-9):.1f} fps) "
          f"| {nboxes} boxes total")
    print(f"   JPGs + preview.mp4 in {out}/  — flip through the f####.jpg to eyeball labels")


if __name__ == "__main__":
    main()
