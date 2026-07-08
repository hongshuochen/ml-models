"""Pre-label golf frames with GroundingDINO (open-vocab) + optional YOLO union.

Recovered plan (see golf/sample_sheets/groundingdino_test2.jpg):
  - GroundingDINO-base, SEPARATE prompts per class ("a golf ball." / "a golf club head."),
    threshold ~0.30. (The single combined prompt bleeds tokens -> junk; test.jpg was the bad one.)
  - Class is decided by WHICH prompt pass produced the box (not by parsing DINO's text labels).
  - Union with YOLO s_v3_1280 boxes (open-vocab DINO catches balls the student misses; the
    trained YOLO is tighter on club_head) -> class-wise NMS dedup -> maximize recall so the
    human only deletes, rarely adds.

Out: datasets/golf_prelabels/labels/<name>.txt  (YOLO xywhn; ball=0, club_head=1)
Run:
  # validate on a spread-out sample + render a QA sheet, write NOTHING:
  uv run python golf/prelabel_gdino.py --sample 24 --sheet golf/sample_sheets/gdino_prelabel.jpg --dry
  # full run (writes txt, background):
  uv run python golf/prelabel_gdino.py
"""
import argparse, glob, time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

BALL, HEAD = 0, 1
PROMPTS = {BALL: "a golf ball.", HEAD: "a golf club head."}
COLORS = {BALL: (0, 0, 255), HEAD: (0, 200, 0)}  # BGR: ball=red, head=green


def load_gdino(device):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(device).eval()
    return proc, model


def gdino_boxes(proc, model, pil, device, res, box_thr, text_thr):
    """Return list of (cls, x1,y1,x2,y2, score) from two separate-prompt passes."""
    proc.image_processor.size = {"shortest_edge": res, "longest_edge": max(2048, res + 512)}
    out = []
    W, H = pil.size
    for cls, prompt in PROMPTS.items():
        inp = proc(images=pil, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            res_out = model(**inp)
        post = proc.post_process_grounded_object_detection(
            res_out, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
            target_sizes=[(H, W)],
        )[0]
        for box, score in zip(post["boxes"].tolist(), post["scores"].tolist()):
            x1, y1, x2, y2 = box
            out.append((cls, x1, y1, x2, y2, float(score)))
    return out


def yolo_boxes(model, path, res):
    r = model.predict(path, imgsz=res, conf=0.20, device=0, verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append((int(b.cls), x1, y1, x2, y2, float(b.conf)))
    return out


def nms_per_class(dets, iou_thr):
    """dets: list of (cls,x1,y1,x2,y2,score). Class-wise NMS, keep highest score."""
    from torchvision.ops import nms
    kept = []
    for cls in (BALL, HEAD):
        cd = [d for d in dets if d[0] == cls]
        if not cd:
            continue
        boxes = torch.tensor([[d[1], d[2], d[3], d[4]] for d in cd], dtype=torch.float32)
        scores = torch.tensor([d[5] for d in cd], dtype=torch.float32)
        keep = nms(boxes, scores, iou_thr).tolist()
        kept += [cd[i] for i in keep]
    return kept


def to_yolo_txt(dets, W, H):
    lines = []
    for cls, x1, y1, x2, y2, _ in dets:
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def draw(img, dets, srcs):
    for (cls, x1, y1, x2, y2, sc), src in zip(dets, srcs):
        c = COLORS[cls]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        tag = ("ball" if cls == BALL else "head") + f":{src}{sc:.2f}"
        cv2.putText(img, tag, (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return img


def make_sheet(cells, path, cols=6, cell_h=360):
    rows_imgs = []
    row = []
    for im in cells:
        h, w = im.shape[:2]
        im = cv2.resize(im, (int(w * cell_h / h), cell_h))
        row.append(im)
        if len(row) == cols:
            rows_imgs.append(row); row = []
    if row:
        rows_imgs.append(row)
    maxw = max(sum(im.shape[1] for im in r) for r in rows_imgs)
    canvas = []
    for r in rows_imgs:
        strip = np.zeros((cell_h, maxw, 3), np.uint8)
        x = 0
        for im in r:
            strip[:, x:x + im.shape[1]] = im; x += im.shape[1]
        canvas.append(strip)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, np.vstack(canvas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/golf_frames")
    ap.add_argument("--outl", default="datasets/golf_prelabels/labels")
    ap.add_argument("--limit", type=int, default=0, help="first N frames (0=all)")
    ap.add_argument("--sample", type=int, default=0, help="evenly-spread N frames (for QA)")
    ap.add_argument("--sheet", default="", help="render a QA grid to this path")
    ap.add_argument("--dry", action="store_true", help="do not write txt (QA only)")
    ap.add_argument("--box-thr", type=float, default=0.30)
    ap.add_argument("--text-thr", type=float, default=0.25)
    ap.add_argument("--res", type=int, default=1280)
    ap.add_argument("--yolo", default="runs/detect/golf_detect_s_v3_1280/weights/best.pt")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--iou", type=float, default=0.6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = sorted(glob.glob(args.src + "/*.jpg"))
    if args.sample:
        step = max(1, len(frames) // args.sample)
        frames = frames[::step][:args.sample]
    elif args.limit:
        frames = frames[:args.limit]

    print(f"GDINO pre-label: {len(frames)} frames | res {args.res} | box_thr {args.box_thr} "
          f"| yolo_union={not args.no_yolo} | dry={args.dry}", flush=True)
    proc, gd = load_gdino(device)
    yolo = None
    if not args.no_yolo:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)

    outl = Path(args.outl)
    if not args.dry:
        outl.mkdir(parents=True, exist_ok=True)
    cells, stats = [], {"ball": 0, "head": 0, "empty": 0, "gd": 0, "yl": 0}
    t0 = time.time()
    for i, p in enumerate(frames):
        pil = Image.open(p).convert("RGB")
        W, H = pil.size
        gd_d = gdino_boxes(proc, gd, pil, device, args.res, args.box_thr, args.text_thr)
        yl_d = yolo_boxes(yolo, p, args.res) if yolo else []
        # tag source before union so the sheet can show provenance
        tagged = [(d, "D") for d in gd_d] + [(d, "Y") for d in yl_d]
        merged = nms_per_class([d for d, _ in tagged], args.iou)
        # recover source tag for each surviving det (match by identity)
        src_of = {id(d): s for d, s in tagged}
        srcs = [src_of[id(d)] for d in merged]
        stats["gd"] += len(gd_d); stats["yl"] += len(yl_d)
        nb = sum(d[0] == BALL for d in merged); nh = sum(d[0] == HEAD for d in merged)
        stats["ball"] += nb > 0; stats["head"] += nh > 0; stats["empty"] += (nb + nh) == 0
        if not args.dry:
            (outl / f"{Path(p).stem}.txt").write_text(to_yolo_txt(merged, W, H))
        if args.sheet:
            img = cv2.imread(p)
            cells.append(draw(img, merged, srcs))
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(frames)}  ({(i+1)/dt:.2f} img/s, eta {dt/(i+1)*(len(frames)-i-1)/60:.0f} min)", flush=True)

    n = len(frames)
    print("\n=== GDINO PRE-LABEL STATS ===")
    print(f"  frames            : {n}   ({time.time()-t0:.0f}s, {n/(time.time()-t0):.2f} img/s)")
    print(f"  with >=1 ball     : {stats['ball']} ({100*stats['ball']/n:.0f}%)")
    print(f"  with >=1 club_head: {stats['head']} ({100*stats['head']/n:.0f}%)")
    print(f"  empty (0 det)     : {stats['empty']} ({100*stats['empty']/n:.0f}%)")
    print(f"  raw boxes: gdino {stats['gd']}  yolo {stats['yl']}  (pre-NMS)")
    if args.sheet:
        make_sheet(cells, args.sheet)
        print(f"  QA sheet -> {args.sheet}")
    if not args.dry:
        print(f"  labels -> {outl}/")


if __name__ == "__main__":
    main()
