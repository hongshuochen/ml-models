#!/usr/bin/env python3
"""Mine golf training data from a folder of videos — STANDALONE (run it on any machine).

Recursively finds every .mp4 under VIDEOS_DIR, runs the golf detector on each, and mines
frames + YOLO labels that actually IMPROVE the model, instead of naively pseudo-labeling
everything:

  T2 "recovered"  — the high-value tier, aimed straight at the ball-recall weakness:
      detections are chained into per-class tracks; a short GAP inside a consistent track
      (ball seen before & after, missed in between) is a proven false negative. The missing
      box is interpolated, then ZOOM-VERIFIED: a crop around the expected spot is re-run
      through the model (small ball becomes big in the crop); only a confirmed detection is
      written, with the crop's tight box. Low-confidence detections (in a track) are promoted
      the same way. These are "the model missed it at full frame, but it is provably there".
  T1 "confident"  — ordinary pseudo-labels: frames with confident detections, time-spaced and
      capped per video (validated earlier: pseudo-label expansion raised precision).
  Frames with no detections are never written (auto-empties reinforce the model's misses).

Temporal consistency runs BOTH directions (both validated on real clips):
  * a GAP in a track (present before & after, missing between) -> a missed ball -> RECOVER (T2).
  * an ISOLATED detection (present in one frame, no same-class neighbour before OR after) -> a
    likely FALSE POSITIVE (sun glint, a smartwatch face, a one-frame motion-blur blob) -> DROPPED,
    never written. "Support" = a nearby same-class detection within +/-SUPPORT_WIN frames (presence,
    not track-linking), so a fast rolling ball whose tracker fails to link still counts as supported.

Output (copy this folder back to the training machine):
  OUT_DIR/images/<video-id>_f######.jpg     full-res frames
  OUT_DIR/labels/<video-id>_f######.txt     YOLO labels (class cx cy w h, normalized)
  OUT_DIR/manifest.csv                      per-frame provenance (video, tier, confs)
  OUT_DIR/stats.json                        per-video mining stats (also the resume record)

The <video-id> encodes the video's relative path, so a later dataset build can split by VIDEO
(never put mined frames in val — keep the fixed egocentric val set for comparable metrics).

Setup on the mining machine (GPU used automatically if present; CPU works, just slower):
    pip install ultralytics opencv-python
    # copy the model:  runs/detect/golf_ego_v3_hole/weights/best.pt   (3-class, incl hole)
    python mine_golf_videos.py /path/to/videos out_mined --model best.pt

Classes: CLASS-COUNT-AWARE — mines whatever the model's head has (2-class ball/club, OR the
3-class ball/club_head/hole model). Tracks, gap-recovery and low-conf promotion all run per
class, so on a putting frame the cup is recovered/promoted alongside the ball → the frame stays
FULLY labeled (no partial-label trap where a written frame silently teaches hole=background).
Interpolation is only trusted on SLOW track segments (a resting/rolling ball, a grounded club,
a static cup) — fast segments (a swinging club) are never interpolated.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("pip install ultralytics opencv-python")

SUPPORT_R = 6.0   # diameters: a neighbour-frame same-class det within this supports a detection
                  # (generous, so a fast rolling ball's next-frame detection still supports it)


def vid_id(root: Path, video: Path) -> str:
    """datasets/x/2025-07-02/golf_12.mp4 -> 2025-07-02_golf_12 (safe, collision-resistant)."""
    rel = video.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rel))


def load_done(out: Path) -> dict:
    p = out / "stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


class Track:
    """One object's detections over (processed) frames: [(k, box, conf)] with gaps."""
    __slots__ = ("cls", "obs")

    def __init__(self, cls):
        self.cls = cls
        self.obs = []   # (frame_idx, [x1,y1,x2,y2], conf)


def build_tracks(dets_per_frame, cls, gate_mult=4.0, max_skip=8):
    """Greedy nearest-neighbor tracks for one class. dets_per_frame[k] = [(box, conf, cls)].
    gate = gate_mult x box 'radius'; a track is extended across <= max_skip processed frames."""
    tracks, active = [], []
    for k, dets in enumerate(dets_per_frame):
        cands = [(b, c) for (b, c, cl) in dets if cl == cls]
        used = set()
        for tr in active:
            kk, bb, _ = tr.obs[-1]
            if k - kk > max_skip:
                continue
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            r = max((bb[2] - bb[0] + bb[3] - bb[1]) / 4, 4.0)
            best, bd = None, gate_mult * r * (k - kk)
            for i, (b, c) in enumerate(cands):
                if i in used:
                    continue
                d = math.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy)
                if d < bd:
                    best, bd = i, d
            if best is not None:
                used.add(best)
                tr.obs.append((k, cands[best][0], cands[best][1]))
        for i, (b, c) in enumerate(cands):
            if i not in used:
                tr = Track(cls)
                tr.obs.append((k, b, c))
                tracks.append(tr)
                active.append(tr)
        active = [t for t in active if k - t.obs[-1][0] <= max_skip]
    return tracks


def interp_box(b0, b1, a):
    return [b0[i] + (b1[i] - b0[i]) * a for i in range(4)]


def zoom_verify(model, frame, exp_box, cls, crop=512, conf=0.25, imgsz=640):
    """Re-detect in an upscaled crop around the expected box. Returns (box, conf) in FRAME
    coords if a same-class detection lands near the crop center, else None."""
    H, W = frame.shape[:2]
    cx = (exp_box[0] + exp_box[2]) / 2
    cy = (exp_box[1] + exp_box[3]) / 2
    x0 = int(min(max(cx - crop / 2, 0), max(W - crop, 0)))
    y0 = int(min(max(cy - crop / 2, 0), max(H - crop, 0)))
    patch = frame[y0:y0 + crop, x0:x0 + crop]
    if patch.shape[0] < 64 or patch.shape[1] < 64:
        return None
    r = model.predict(patch, imgsz=imgsz, conf=conf, verbose=False)[0]
    exp_d = max((exp_box[2] - exp_box[0] + exp_box[3] - exp_box[1]) / 2, 8.0)
    best = None
    for box in r.boxes:
        if int(box.cls) != cls:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bx, by = x0 + (x1 + x2) / 2, y0 + (y1 + y2) / 2
        if math.hypot(bx - cx, by - cy) > 2.0 * exp_d:
            continue    # a different object, not the one we expected
        c = float(box.conf)
        if best is None or c > best[1]:
            best = ([x0 + x1, y0 + y1, x0 + x2, y0 + y2], c)
    return best


def yolo_line(cls, box, W, H):
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(x1, W - 1)), max(0, min(x2, W - 1))
    y1, y2 = max(0, min(y1, H - 1)), max(0, min(y2, H - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
            f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")


def mine_video(model, video: Path, vid: str, out: Path, args):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    # ---- pass 1: detect on every STRIDE-th frame, keep frames in a bounded pixel cache ----
    dets, frames_px, frame_ids = [], {}, []
    for k, r in enumerate(model.predict(source=str(video), stream=True, imgsz=args.imgsz,
                                        conf=args.conf_low, vid_stride=args.stride,
                                        device=args.device or None, verbose=False)):
        dd = [(box.xyxy[0].tolist(), float(box.conf), int(box.cls)) for box in r.boxes]
        dets.append(dd)
        frame_ids.append(k * args.stride)
        if dd:
            frames_px[k] = r.orig_img.copy()
    n = len(dets)
    if n == 0:
        return {"frames": 0, "t1": 0, "t2": 0, "labels": 0}

    H, W = next(iter(frames_px.values())).shape[:2] if frames_px else (0, 0)
    sample_gap = max(1, int(round(args.sample_every * fps / args.stride)))

    # per-frame label pools
    labels = [dict() for _ in range(n)]     # key (cls, round(cx), round(cy)) -> (box, conf, tier)

    def add(k, cls, box, conf, tier):
        key = (cls, int((box[0] + box[2]) / 2 / 8), int((box[1] + box[3]) / 2 / 8))
        cur = labels[k].get(key)
        if cur is None or conf > cur[1]:
            labels[k][key] = (box, conf, tier)

    # ---- tracks (shared by T2 recovery and the temporal-support FP filter) ----
    # class-count-aware: mine whatever the model detects (2-class ball/club, or 3-class incl hole)
    tracks = {cls: build_tracks(dets, cls) for cls in range(len(model.names))}

    # temporal support: a detection is TRUSTED only if a same-class detection also appears in a
    # NEARBY position within +/- SUPPORT_WIN neighbour frames. A one-frame blip with no neighbour
    # is almost always a false positive (sun glint / white flower / blur) -> never written.
    # This is NEIGHBOUR PRESENCE, not track linking: a fast rolling ball whose tracker fails to
    # link across stride still has a nearby detection next frame, so it stays supported (not dropped).
    def is_supported(k, cls, b):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        D = max((b[2] - b[0] + b[3] - b[1]) / 2, 8.0)
        hits = 0
        for kk in range(max(0, k - args.support_win), min(n, k + args.support_win + 1)):
            if kk == k:
                continue
            for (bb, _c, cl2) in dets[kk]:
                if cl2 == cls and math.hypot((bb[0] + bb[2]) / 2 - cx, (bb[1] + bb[3]) / 2 - cy) < SUPPORT_R * D:
                    hits += 1
                    break
        return hits >= args.min_support

    # ---- T2: track gaps + low-conf promotions, zoom-verified ----
    # runs for every model class: for hole (3-class model) this recovers missed/weak cups on kept
    # putting frames, which is exactly what keeps those frames FULLY labeled (no partial-label trap).
    t2 = 0
    for cls in range(len(model.names)):
        for tr in tracks[cls]:
            if len(tr.obs) < 3:
                continue
            # (a) gap recovery
            for (k0, b0, c0), (k1, b1, c1) in zip(tr.obs, tr.obs[1:]):
                dk = k1 - k0
                if dk <= 1 or dk > args.max_gap:
                    continue
                d0 = (b0[2] - b0[0] + b0[3] - b0[1]) / 2
                move = math.hypot((b1[0] + b1[2]) / 2 - (b0[0] + b0[2]) / 2,
                                  (b1[1] + b1[3]) / 2 - (b0[1] + b0[3]) / 2)
                if move / dk > 1.2 * max(d0, 8):       # fast segment: interpolation not trusted
                    continue
                for kk in range(k0 + 1, k1):
                    if kk in frames_px:                 # only frames we still have pixels for
                        exp = interp_box(b0, b1, (kk - k0) / dk)
                        got = zoom_verify(model, frames_px[kk], exp, cls,
                                          crop=args.crop, conf=args.verify_conf)
                        if got:
                            add(kk, cls, got[0], got[1], "recovered")
                            t2 += 1
            # (b) low-conf promotion
            for (k, b, c) in tr.obs:
                if c < args.conf_keep and k in frames_px:
                    got = zoom_verify(model, frames_px[k], b, cls,
                                      crop=args.crop, conf=args.verify_conf)
                    if got:
                        add(k, cls, got[0], got[1], "promoted")
                        t2 += 1

    # frames worth keeping: any T2 label, or T1 sampling (confident AND temporally supported)
    keep = set(k for k in range(n) if any(v[2] != "confident" for v in labels[k].values()))
    last = -10 ** 9
    t1 = 0
    for k in range(n):
        if k in frames_px and k - last >= sample_gap and \
                any(c >= args.conf_keep and is_supported(k, cl, b) for (b, c, cl) in dets[k]):
            keep.add(k)
            last = k
            t1 += 1
    keep = set(list(sorted(keep))[: args.cap_per_video])

    # on kept frames, write every confident detection that has temporal support (a saved frame
    # must be FULLY labeled; isolated single-frame detections are dropped as likely FPs)
    fp_dropped = 0
    for k in keep:
        for (b, c, cl) in dets[k]:
            if c >= args.conf_keep:
                if is_supported(k, cl, b):
                    add(k, cl, b, c, "confident")
                else:
                    fp_dropped += 1        # confident but temporally isolated -> likely FP

    # ---- write ----
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rows, wrote = [], 0
    for k in sorted(keep):
        if k not in frames_px or not labels[k]:
            continue
        img = frames_px[k]
        H, W = img.shape[:2]
        lines = []
        tiers = []
        for (cls_key, _cx, _cy), (box, conf, tier) in labels[k].items():
            ln = yolo_line(cls_key, box, W, H)
            if ln:
                lines.append(ln)
                tiers.append(tier)
        if not lines:
            continue
        name = f"{vid}_f{frame_ids[k]:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        rows.append([name, vid, frame_ids[k], "+".join(sorted(set(tiers))), len(lines)])
        wrote += 1

    mf = out / "manifest.csv"
    new = not mf.exists()
    with mf.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["frame", "video", "src_frame", "tiers", "n_labels"])
        w.writerows(rows)
    return {"frames": wrote, "t1": t1, "t2": t2, "fp_dropped": fp_dropped,
            "labels": sum(len(labels[k]) for k in keep if k in frames_px)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--conf-keep", type=float, default=0.40, help="confident pseudo-label threshold")
    ap.add_argument("--conf-low", type=float, default=0.12, help="detection floor (low-conf mining)")
    ap.add_argument("--verify-conf", type=float, default=0.30, help="zoom-verify acceptance conf")
    ap.add_argument("--crop", type=int, default=512, help="zoom-verify crop size (px)")
    ap.add_argument("--max-gap", type=int, default=6, help="max track gap (processed frames) to recover")
    ap.add_argument("--min-support", type=int, default=1,
                    help="a written detection needs a same-class detection in >= this many neighbour "
                         "frames; isolated one-frame blips are dropped as likely FPs")
    ap.add_argument("--support-win", type=int, default=2,
                    help="neighbour frames each side to look for temporal support")
    ap.add_argument("--sample-every", type=float, default=1.5, help="s between T1 confident samples")
    ap.add_argument("--cap-per-video", type=int, default=80, help="max frames written per video")
    ap.add_argument("--min-bytes", type=int, default=1_000_000,
                    help="skip files smaller than this (fake/thumbnail mp4s)")
    ap.add_argument("--device", default="", help="'' auto, or e.g. 0 / cpu")
    args = ap.parse_args()

    root = Path(args.videos_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    videos = sorted(p for p in root.rglob("*.mp4") if p.stat().st_size >= args.min_bytes)
    if not videos:
        sys.exit(f"no .mp4 >= {args.min_bytes} bytes under {root}")
    stats = load_done(out)
    print(f"{len(videos)} videos under {root} ({sum(1 for v in videos if vid_id(root, v) in stats)} already mined)")

    for i, v in enumerate(videos):
        vid = vid_id(root, v)
        if vid in stats:
            continue
        try:
            s = mine_video(model, v, vid, out, args)
        except Exception as e:                     # keep the batch alive; record the failure
            s = {"error": str(e)}
        stats[vid] = s
        (out / "stats.json").write_text(json.dumps(stats, indent=1))
        print(f"[{i + 1}/{len(videos)}] {vid}: {s}")

    tot = {k: sum(s.get(k, 0) for s in stats.values() if isinstance(s, dict))
           for k in ("frames", "t1", "t2", "labels", "fp_dropped")}
    print(f"\nDONE: {tot['frames']} frames, {tot['labels']} labels "
          f"({tot['t2']} zoom-verified recoveries/promotions, {tot['t1']} confident samples, "
          f"{tot['fp_dropped']} isolated FPs dropped)")
    print(f"Copy '{out}' back to the training machine (images/ + labels/ + manifest.csv).")


if __name__ == "__main__":
    main()
