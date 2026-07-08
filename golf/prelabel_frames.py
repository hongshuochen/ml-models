"""Auto pre-label all extracted frames with golf_v3 (for human correction later).

Writes YOLO-format txt per frame (ball=0, club_head=1) so it can go into any tool.
No frame filtering (we keep every frame); empty predictions = empty .txt (a negative / to fix by hand).

In:  datasets/golf_frames/*.jpg
Out: datasets/golf_prelabels/labels/<name>.txt  (YOLO boxes)
Run: uv run python golf/prelabel_frames.py [conf]   (default conf=0.20)
"""
import glob, os, sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

SRC = "datasets/golf_frames"
OUTL = Path("datasets/golf_prelabels/labels")
# 1280 teacher (best v3 data, high-res -> catches the small egocentric ball). GDINO was
# tried as an open-vocab pre-labeler but is non-viable on this footage (0.30 silent / 0.15
# giant garbage boxes; clean only on public front-view frames). No sun/sky FP filter -- the
# human deletes sun->ball FPs in Label Studio (they're high-conf, conf threshold won't kill them).
MODEL = "runs/detect/golf_detect_s_v3_1280/weights/best.pt"
IMGSZ = 1280
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20


def main():
    OUTL.mkdir(parents=True, exist_ok=True)
    frames = sorted(glob.glob(SRC + "/*.jpg"))
    print(f"pre-labeling {len(frames)} frames with golf_v3_1280 @ imgsz {IMGSZ} conf {CONF} ...")
    m = YOLO(MODEL)
    st = Counter()
    n = len(frames)
    # IMPORTANT: pass the DIRECTORY STRING, never a Python list. predict(list) batches the WHOLE
    # list into one tensor (19,700 paths -> ~80GB CPU balloon; a 256-list -> a 12.5GB single-conv
    # alloc -> OOM on the 10GB GPU). predict(source=<dir>, stream=True) streams ONE image at a time
    # (bs=1): bounded RAM, GPU stays warm (~33 img/s @1280 -> ~10 min for 19,700).
    for i, r in enumerate(m.predict(source=SRC, imgsz=IMGSZ, conf=CONF, device=0, stream=True, verbose=False)):
        name = Path(r.path).stem
        lines, nb, nh = [], 0, 0
        for b in r.boxes:
            c = int(b.cls)
            x, y, w, h = b.xywhn[0].tolist()
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            nb += (c == 0); nh += (c == 1)
        (OUTL / f"{name}.txt").write_text("\n".join(lines))
        st["has_ball"] += nb > 0
        st["has_club"] += nh > 0
        st["has_any"] += (nb + nh) > 0
        st["empty"] += (nb + nh) == 0
        st["balls"] += nb; st["clubs"] += nh
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n}", flush=True)

    print("\n=== PRE-LABEL STATS ===")
    print(f"  frames            : {n}")
    print(f"  with >=1 ball     : {st['has_ball']} ({100*st['has_ball']/n:.0f}%)  | total ball boxes: {st['balls']}")
    print(f"  with >=1 club_head: {st['has_club']} ({100*st['has_club']/n:.0f}%)  | total club boxes: {st['clubs']}")
    print(f"  with any detection: {st['has_any']} ({100*st['has_any']/n:.0f}%)")
    print(f"  empty (0 det)     : {st['empty']} ({100*st['empty']/n:.0f}%)  <- negatives / to fix by hand")
    print(f"  labels -> {OUTL}/")


if __name__ == "__main__":
    main()
