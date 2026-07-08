"""Auto-label (pseudo-label) the un-hand-labeled first-person frames with s_v1, to expand the
training set for v2 (weak supervision). Hand-labeled v1 frames stay the clean anchor + val set.

In:  first-person frames (fpclass) NOT in v1_frames.txt
Out: datasets/golf_prelabels/autolabels/*.txt  (YOLO; ball=0, club_head=1)
Run: uv run python golf/autolabel_firstperson.py [conf]   (default 0.35)
"""
import glob, json, os, sys
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

FRAMES = Path("datasets/golf_frames")
FPCLASS = Path("datasets/golf_prelabels/all_frames.fpclass.json")
V1 = Path("datasets/golf_prelabels/v1_frames.txt")
OUTL = Path("datasets/golf_prelabels/autolabels")
MODEL = "runs/detect/runs/detect/golf_ego_v1_1280/weights/best.pt"  # s_v1
TMP = Path("/tmp/claude-1000/-home-max-2026-ml-models/ef3aba0d-9d1b-4678-8a09-02d72865ef6e/scratchpad/autolabel_frames")
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
IMGSZ = 1280


def main():
    cls = json.loads(FPCLASS.read_text())
    fp = {os.path.basename(p) for p, v in cls.items() if not (v["maxh"] > 0.15 and v["top"] < 0.60)}
    labeled = {os.path.basename(l) for l in V1.read_text().splitlines() if l.strip()}
    todo = sorted(FRAMES / f for f in (fp - labeled))
    print(f"auto-labeling {len(todo)} first-person frames with s_v1 @ imgsz {IMGSZ} conf {CONF}", flush=True)

    if TMP.exists():
        for f in TMP.glob("*"):
            f.unlink()
    TMP.mkdir(parents=True, exist_ok=True)
    for p in todo:
        (TMP / p.name).symlink_to(p.resolve())

    OUTL.mkdir(parents=True, exist_ok=True)
    m = YOLO(MODEL)
    st = Counter()
    for i, r in enumerate(m.predict(source=str(TMP), imgsz=IMGSZ, conf=CONF, device=0, stream=True, verbose=False)):
        name = Path(r.path).stem
        lines, nb, nh = [], 0, 0
        for b in r.boxes:
            c = int(b.cls); x, y, w, h = b.xywhn[0].tolist()
            lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            nb += c == 0; nh += c == 1
        (OUTL / f"{name}.txt").write_text("\n".join(lines))
        st["ball"] += nb > 0; st["club"] += nh > 0; st["empty"] += (nb + nh) == 0
        st["nb"] += nb; st["nh"] += nh
        if (i + 1) % 3000 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)

    n = len(todo)
    print(f"\n=== AUTO-LABEL (s_v1, conf {CONF}) ===")
    print(f"  frames {n} | with ball {st['ball']} ({100*st['ball']//n}%) | with club_head {st['club']} ({100*st['club']//n}%) | empty {st['empty']} ({100*st['empty']//n}%)")
    print(f"  boxes: ball {st['nb']}  club_head {st['nh']}")
    print(f"  -> {OUTL}/")


if __name__ == "__main__":
    main()
