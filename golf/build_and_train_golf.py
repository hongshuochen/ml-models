#!/usr/bin/env python3
"""Build golf_ego_v3 and fine-tune from golf_ego_v2 — STANDALONE (run on the video machine).

Merges data sources into one training set WITHOUT copying images (uses image-list .txt files, so
Ultralytics finds each label at the source's images/->labels/ sibling), then fine-tunes the
deployed detector on it. Anti-drift guardrails are baked in:

  * VAL IS FIXED and comes from HELD-OUT videos (carry the existing egocentric val over unchanged)
    -> metrics stay comparable to golf_ego_v2 and forgetting is detectable.
  * The OLD hand-labeled train set is mixed back in (prevents catastrophic forgetting of what v2
    already knows while it learns the new people/scenes).
  * AUTO-mined labels are lower trust -> capped to a fraction of the train set (human-reviewed and
    hand-labeled frames dominate).
  * Fine-tune from v2 weights with early stopping on the fixed val (never train past the val peak).

Trust order (highest first): --old (hand-labeled) = --reviewed (human-corrected) > --mined (auto).

Sources must each be a directory with images/ + labels/ siblings (YOLO txt: `cls cx cy w h`):
    --val       fixed egocentric val (REQUIRED, carry over unchanged; e.g. golf_ego_v1/val)
    --old       existing hand-labeled ego train (recommended: golf_ego_v2 train)
    --reviewed  human-corrected frames from Label Studio export (select_review_frames.py -> LS)
    --mined     auto-mined frames (mine_golf_videos.py output)

Example (on the video machine):
    python build_and_train_golf.py \
        --base-weights golf_ego_v2_best.pt \
        --val      carried/golf_ego_v1_val \
        --old      carried/golf_ego_v2_train \
        --reviewed out_review_corrected \
        --mined    out_mined \
        --name golf_ego_v3 --imgsz 1280 --epochs 40 --batch 6
"""
import argparse
import os
import random
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("pip install ultralytics opencv-python")

IMG_EXT = (".jpg", ".jpeg", ".png")


def images_in(d):
    """Image paths from a source dir — accepts either a flat dir of images (e.g. an existing
    dataset's images/train split) or a dir with an images/ subdir (miner / selector output).
    Labels resolve via Ultralytics' images/->labels/ path rule, so the paths must live under
    a '.../images/...' directory (both layouts do)."""
    # abspath (NOT resolve): the listed path must keep its own '.../images/...' so Ultralytics
    # can swap images/->labels/ — resolve() would follow symlinks (datasets often symlink frames)
    p = Path(d)
    def listing(dir_):
        return sorted(os.path.abspath(str(f)) for f in dir_.iterdir()
                      if f.is_file() and f.suffix.lower() in IMG_EXT)
    flat = listing(p) if p.is_dir() else []
    if flat:
        if "images" not in Path(flat[0]).parts:
            sys.exit(f"{d}: images must live under an 'images/' dir so labels resolve (found {flat[0]})")
        return flat
    if (p / "images").is_dir():
        return listing(p / "images")
    sys.exit(f"{d}: no image files and no images/ subdir")


def video_of(path):
    """Recover the <video-id> prefix from a mined/reviewed frame name (…/<vid>_f######.jpg)."""
    stem = Path(path).stem
    return stem.rsplit("_f", 1)[0] if "_f" in stem else stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-weights", required=True, help="golf_ego_v2 best.pt (fine-tune start)")
    ap.add_argument("--val", required=True, help="FIXED held-out val dir (images/ + labels/)")
    ap.add_argument("--old", help="existing hand-labeled ego train dir")
    ap.add_argument("--reviewed", help="human-corrected review dir")
    ap.add_argument("--mined", help="auto-mined dir")
    ap.add_argument("--out", default="datasets/golf_ego_v3")
    ap.add_argument("--name", default="golf_ego_v3")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--patience", type=int, default=12, help="early-stop patience on the fixed val")
    ap.add_argument("--auto-cap-frac", type=float, default=0.5,
                    help="max fraction of TRAIN that may be auto-mined labels")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="assemble + check the dataset, don't train")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    val_imgs = images_in(args.val)
    val_vids = {video_of(p) for p in val_imgs}

    trusted = []                                   # hand-labeled + human-reviewed (dominant)
    for src in (args.old, args.reviewed):
        if src:
            trusted += images_in(src)
    mined = images_in(args.mined) if args.mined else []

    # leakage guard: no train frame may come from a val video
    def drop_val(imgs, tag):
        keep = [p for p in imgs if video_of(p) not in val_vids]
        if len(keep) != len(imgs):
            print(f"  [guard] dropped {len(imgs) - len(keep)} {tag} frames from val videos")
        return keep
    trusted = drop_val(trusted, "trusted")
    mined = drop_val(mined, "mined")

    # cap auto-mined to a fraction of the final train set
    if mined and args.auto_cap_frac < 1.0:
        max_mined = int(args.auto_cap_frac / (1 - args.auto_cap_frac) * max(len(trusted), 1))
        if len(mined) > max_mined:
            rng.shuffle(mined)
            print(f"  [cap] auto-mined {len(mined)} -> {max_mined} ({args.auto_cap_frac:.0%} train cap)")
            mined = mined[:max_mined]

    train = trusted + mined
    if not train:
        sys.exit("no training images (need at least --old/--reviewed or --mined)")
    rng.shuffle(train)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "val.txt").write_text("\n".join(val_imgs) + "\n")
    yaml = out / f"{args.name}.yaml"
    yaml.write_text(
        f"# {args.name}: fine-tune from golf_ego_v2 with new people/scenes\n"
        f"train: {(out / 'train.txt').resolve()}\n"
        f"val: {(out / 'val.txt').resolve()}\n"
        f"names:\n  0: ball\n  1: club_head\n")

    print(f"\ntrain {len(train)} imgs ({len(trusted)} trusted + {len(mined)} auto-mined) | "
          f"val {len(val_imgs)} imgs (FIXED, {len(val_vids)} videos)")
    print(f"dataset yaml: {yaml}")
    if args.dry_run:
        print("dry-run: not training.")
        return

    # ---- baseline: the current deployed model on the fixed val (for a fair before/after) ----
    base = YOLO(args.base_weights)
    print("\n== baseline (golf_ego_v2) on the fixed val ==")
    b = base.val(data=str(yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, verbose=False)
    print(f"  mAP50={b.box.map50:.3f} mAP50-95={b.box.map:.3f} "
          f"ball R={b.box.r[0]:.3f} club R={b.box.r[1]:.3f}")

    # ---- fine-tune from v2 weights ----
    model = YOLO(args.base_weights)
    model.train(data=str(yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                patience=args.patience, device=args.device, name=args.name, seed=args.seed,
                mixup=0.1, copy_paste=0.1)

    print("\n== golf_ego_v3 on the fixed val ==")
    r = model.val(data=str(yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, verbose=False)
    print(f"  mAP50={r.box.map50:.3f} mAP50-95={r.box.map:.3f} "
          f"ball R={r.box.r[0]:.3f} club R={r.box.r[1]:.3f}")
    print(f"  Δ vs v2: mAP50 {r.box.map50 - b.box.map50:+.3f}, ball recall {r.box.r[0] - b.box.r[0]:+.3f}")
    print(f"\nweights: runs/detect/{args.name}/weights/best.pt")
    print("if ball recall went UP and mAP held/rose -> export & deploy (see GOLF_YOLO.md).")
    print("if it DROPPED -> forgetting/drift: raise --auto-cap-frac trust (fewer auto), add more "
          "reviewed frames, or lower --epochs.")


if __name__ == "__main__":
    main()
