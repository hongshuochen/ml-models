"""Build golf egocentric v2 dataset = v1 hand-labels + s_v1 pseudo-labels (weak supervision).

val  = EXACT v1 val (290 hand-labeled, held out by video) -> v1/v2 compared on the same ruler.
train = v1 train hand-labels (1,622, clean anchor)
      + auto-labels (s_v1) for frames whose VIDEO is not a val video AND that have >=1 detection
        (drop auto-empty frames: s_v1 misses ~23% of balls -> an empty pseudo-frame could reinforce
         a miss; we already have clean hand negatives, so only add confident positive pseudo-signal).

Out: datasets/golf_ego_v2/{images,labels}/{train,val} + golf_ego_v2.yaml
Run: uv run python golf/build_golf_v2_dataset.py
"""
import glob, os, re, shutil
from pathlib import Path

FRAMES = Path("datasets/golf_frames")
V1 = Path("datasets/golf_ego_v1")
AUTO = Path("datasets/golf_prelabels/autolabels")
OUT = Path("datasets/golf_ego_v2")


def vid_of(stem):
    return re.sub(r"_f\d+$", "", stem)


def main():
    val_vids = {vid_of(Path(p).stem) for p in glob.glob(str(V1 / "images/val/*.jpg"))}
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = OUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    cnt = {"val": 0, "train_hand": 0, "train_auto": 0}

    # val = v1 val unchanged
    for lp in glob.glob(str(V1 / "labels/val/*.txt")):
        stem = Path(lp).stem
        os.symlink((FRAMES / f"{stem}.jpg").resolve(), OUT / f"images/val/{stem}.jpg")
        shutil.copy(lp, OUT / f"labels/val/{stem}.txt")
        cnt["val"] += 1

    # train hand = v1 train unchanged
    for lp in glob.glob(str(V1 / "labels/train/*.txt")):
        stem = Path(lp).stem
        os.symlink((FRAMES / f"{stem}.jpg").resolve(), OUT / f"images/train/{stem}.jpg")
        shutil.copy(lp, OUT / f"labels/train/{stem}.txt")
        cnt["train_hand"] += 1

    # train auto = pseudo-labels from non-val videos, with >=1 box
    for lp in glob.glob(str(AUTO / "*.txt")):
        stem = Path(lp).stem
        if vid_of(stem) in val_vids:
            continue
        if os.path.getsize(lp) == 0:
            continue
        img = FRAMES / f"{stem}.jpg"
        link = OUT / f"images/train/{stem}.jpg"
        if not img.exists() or link.exists():
            continue
        os.symlink(img.resolve(), link)
        shutil.copy(lp, OUT / f"labels/train/{stem}.txt")
        cnt["train_auto"] += 1

    Path("golf_ego_v2.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n  1: club_head\n")
    print(f"golf_ego_v2: val {cnt['val']} (hand) | train {cnt['train_hand']+cnt['train_auto']} "
          f"= {cnt['train_hand']} hand + {cnt['train_auto']} auto(pseudo)")
    print("  yaml -> golf_ego_v2.yaml")


if __name__ == "__main__":
    main()
