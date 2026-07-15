# Offline golf pipeline (3-class: ball / club_head / hole)

Run-book for the **video machine** (no internet / no Claude). Label new footage and fine-tune the
3-class detector entirely offline. Everything here is self-contained: three standalone scripts +
`pip install ultralytics opencv-python` + a few carried-over files. GPU is used automatically if
present; CPU works, just slower.

## 0. What to copy onto the machine

Scripts (from `golf/`):
- `select_review_frames.py`  — pick the most useful frames to hand-label + write Label Studio tasks
- `mine_golf_videos.py`      — auto-mine high-confidence frames (incl hole) as extra train labels
- `build_and_train_golf.py`  — merge everything and fine-tune
- `annotate_status.py`       — (optional) sanity-check overlay video (boxes / MADE PUTT / ball speed)

Weights (git-ignored — copy the actual files):
- `runs/detect/golf_ego_v3_hole/weights/best.pt`  → the **3-class** model. Used for BOTH pre-labeling
  (`--model`) and as the fine-tune start (`--base-weights`). This one file is all you need.

Carry-over anchors (keep metrics comparable + prevent forgetting):
- `datasets/golf_ego_v1/{images,labels}/val`  → the **FIXED val** (`--val`, REQUIRED). It is ball+club
  only (no hole) — that's fine, see the note in step 3.
- `datasets/golf_ego_v2/{images,labels}/train` → hand-labeled ego train (`--old`, recommended).
  2-class ball/club; holes are legitimately absent there, so it is a correct negative for hole.

> ⚠️ Do **not** use `datasets/golf_hole` as `--old`: that set is the 1-class hole teacher where
> **class 0 = hole**, so its ids collide with ball. Only `golf_ego_v2` / `golf_ego_v1` are safe.

```
pip install ultralytics opencv-python
```

## 1. Pick frames to hand-label  →  correct in Label Studio

```
python select_review_frames.py /path/to/new_videos out_review \
    --model golf_ego_v3_hole_best.pt --total 500
```
Writes `out_review/images/*.jpg`, `out_review/labels/*.txt` (YOLO pre-labels), and
`out_review/ls_tasks.json` (Label Studio import with pre-annotations, **including hole**).

- Import `ls_tasks.json` into a **local** Label Studio project whose labeling config has three
  rectangle labels: `ball`, `club_head`, `hole` (order matters — it must map 0/1/2).
- Fix the boxes (the model is weakest on ball recall + the new hole class — the selector already
  favors those frames). **Label every visible hole** — a frame that shows a cup but leaves it
  unlabeled teaches "hole = background" (partial-label trap).
- Export **YOLO** format → a dir with `images/` + `labels/`. Call it `out_review_corrected`.

The 3-class model emits hole directly, so no separate `--hole-model` teacher is needed. (The
`--hole-model` flag is the legacy bootstrap for when only a 2-class model existed.)

## 2. Auto-mine extra labels (optional but recommended)

```
python mine_golf_videos.py /path/to/new_videos out_mined --model golf_ego_v3_hole_best.pt
```
Class-count-aware: mines ball, club **and** hole. It only keeps frames it can label *fully*
(confident + temporally supported detections; track-gap recovery + low-conf promotion are
zoom-verified), so mined putting frames get their cup recovered alongside the ball. Auto labels are
lower-trust than human ones — step 3 caps them to a fraction of the train set automatically.

## 3. Fine-tune the 3-class model

```
python build_and_train_golf.py \
    --base-weights golf_ego_v3_hole_best.pt \
    --val      golf_ego_v1_val \
    --old      golf_ego_v2_train \
    --reviewed out_review_corrected \
    --mined    out_mined \
    --names ball,club_head,hole \
    --name golf_ego_v4_hole --imgsz 1280 --epochs 40 --batch 6
```
Fixed val + old-set mix-in + capped auto labels + early stopping guard against drift. The report
prints per-class recall, e.g.:
```
  mAP50=0.83 mAP50-95=0.61  ball R=0.90  club_head R=0.88  hole R=n/a
  NOTE: the fixed val has NO 'hole' labels, so hole recall is not evaluated here...
```
`hole R=n/a` is expected — the carried val predates hole. To actually **track cup recall**, add a
few held-out putting clips (hand-labeled with holes) into the `--val` dir; then it prints a real
`hole R=…` and the note disappears. Otherwise you're flying blind on the new class.

Output weights: `runs/detect/golf_ego_v4_hole/weights/best.pt`.

## 4. (optional) Eyeball the result

```
python annotate_status.py /path/to/a_clip.mp4 out.mp4 --model runs/detect/golf_ego_v4_hole/weights/best.pt
```
Overlays ball/club/hole boxes, trail, ball speed, and a MADE-PUTT / HITS tally — a fast way to
confirm the new model behaves before copying `best.pt` back for export/deployment.

## Bring back to the training/deploy machine
Copy `runs/detect/golf_ego_v4_hole/weights/best.pt` back; export + deploy as before
(the raw-head TFLite → QNN NPU path in `android-golf`).
