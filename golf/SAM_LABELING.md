# SAM 3.1 auto-labeling (H100 server)

Auto-label golf clips with **SAM 3.1** (Meta's Segment Anything 3.1, 848M): text-prompt the class
concepts, let SAM detect + **track them across every frame**, export YOLO labels. Runs fully on the
H100 box (no Claude there). Script: `golf/sam3_label_golf.py` (standalone: ultralytics + opencv).

## When SAM helps (vs our golf detector)
- ✅ **hole / hard classes** — a concept prompt needs no training; great for the new `hole` class.
- ✅ **propagation** — label a clip end-to-end from one prompt (tracking), not frame-by-frame.
- ✅ **the small ball — but ONLY with a specific prompt.** Verified on `facebook/sam3` (RTX 3080/3090):
  bare `"golf ball"` OVER-FIRES on any round object (GPS-watch icons → 5-7 false boxes/frame, scores
  don't separate). **`"a small white golf ball on the grass"` + conf 0.4-0.5 → finds the real ball
  tightly, ZERO false positives, and tracks it (caught it 9 frames before the trained detector).** Use
  the specific phrasing (the script defaults are now specific). imgsz 1280 needs a 24GB card (OOMs at
  10GB → 1024); tracking is ~1.3 s/frame so sample frames / pick clips rather than labeling everything.

## GPU note
SAM 3.1 = **848M params (~1.7 GB FP16 weights)** — trivial on an H100 (80 GB). One H100 runs it with
room for long clips; use the **2nd GPU for throughput** (split the videos), NOT model-parallel.

## 1. Setup (once, needs internet)
```bash
pip install -U ultralytics opencv-python
# sam3.pt is GATED — request access + download from https://huggingface.co/facebook/sam3
#   (use the SAM 3.1 checkpoint), put it next to the script or pass --model /path/to/sam3.pt
# if CLIP errors:  pip uninstall -y clip && pip install git+https://github.com/ultralytics/CLIP.git
```

## 2. Run — one process per GPU (both H100s in parallel)
```bash
CUDA_VISIBLE_DEVICES=0 python golf/sam3_label_golf.py /path/to/videos out_sam --shard 0/2 &
CUDA_VISIBLE_DEVICES=1 python golf/sam3_label_golf.py /path/to/videos out_sam --shard 1/2 &
wait
```
`--shard i/n` processes `videos[i::n]`, so both write into the same `out_sam/`. Useful flags:
`--imgsz 1024` (raise for the small ball), `--stride 5` (write every 5th frame), `--conf 0.25`,
`--prompts "golf ball:ball,golf club head:club_head,golf hole:hole"` (order = YOLO class id).

## 3. ⚠️ VERIFY the class mapping on the FIRST clip
SAM returns a class index per box; the script maps it to the YOLO id by the **prompt order**. Sanity-
check one video's output before scaling: draw the labels back on with the existing tool —
```bash
uv run python golf/annotate_video.py <a-clip>.mp4 --model out_sam   # or eyeball a few labels/*.txt
```
Confirm ball/club_head/hole boxes land on the right objects. If a class is scrambled or a prompt
over/under-fires, adjust `--prompts` wording (e.g. "golf hole cup", "golf ball on grass") or `--conf`.

## 4. Feed into training
`out_sam/` is `images/` + `labels/` — hand it to the trainer as a source (spot-check / correct in
Label Studio first; SAM labels are auto, so treat them like `--mined`, not gold):
```bash
uv run python golf/build_and_train_golf.py --base-weights best.pt \
    --val data/golf_ego_v1/images/val --old data/golf_ego_v2/images/train \
    --mined out_sam --names ball,club_head,hole --name golf_ego_v5_sam
```

## Hybrid (recommended if SAM misses the small ball)
Best of both: **our golf detector for ball/club** (it's trained on them) + **SAM for hole /
propagation**. Run `mine_golf_videos.py` (detector) for ball+club and `sam3_label_golf.py
--prompts "golf hole:hole"` for hole, then merge the two `labels/` outputs per frame before training.
