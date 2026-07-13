# Golf — Improve the Model from an Offline Video Folder (Runbook)

You have many golf videos (more people & scenes) on a machine that can't run Claude Code. This is
the self-contained recipe to turn them into a better detector **on that machine**, then bring only
the trained weights back. It closes the domain gap the right way: auto-label what the model can
already almost see, put the scarce human effort only on the *new* diversity, and guard against
self-training drift.

Three standalone scripts (each: `pip install ultralytics opencv-python`, no repo needed):
`golf/mine_golf_videos.py` · `golf/select_review_frames.py` · `golf/build_and_train_golf.py`.

## Why this way (not naive pseudo-labeling)

- Self-training only helps if the labels beat the model's own guesses. The miner does this with
  **zoom-verify** (re-detect a small ball in an upscaled crop) — a stronger labeler than the 640
  student. It **cannot** invent a ball it misses entirely, though.
- **New people/scenes create blind spots the auto-labeler can't self-correct.** So a *small*,
  actively-selected human pass (a few hundred frames covering the new diversity + the model's
  uncertainty) is the highest-ROI effort — not thousands of random frames.
- **Anti-drift guardrails**: the val set stays FIXED and from held-out videos (metrics comparable,
  forgetting visible); the old hand-labeled data is mixed back in; auto-labels are capped; fine-tune
  with early stopping on that fixed val.

## Carry these to the offline machine (small — no videos moved)

| what | from | size |
|---|---|---|
| the 3 scripts | `golf/mine_golf_videos.py`, `golf/select_review_frames.py`, `golf/build_and_train_golf.py` | KB |
| base weights | `runs/detect/golf_ego_v2_1280/weights/best.pt` → e.g. `golf_ego_v2_best.pt` | 19 MB |
| **fixed val** (keep unchanged) | `datasets/golf_ego_v1/images/val` + `datasets/golf_ego_v1/labels/val` | ~290 imgs |
| old train (recommended, prevents forgetting) | `datasets/golf_ego_v1/images/train` + `.../labels/train` | few GB |

The videos stay put. Only the final `best.pt` comes back.

## Step 1 — Auto-mine hard examples

```bash
python golf/mine_golf_videos.py /path/to/videos out_mined --model golf_ego_v2_best.pt --imgsz 1280
```
Recursively mines every `.mp4` (subfolders included). Writes `out_mined/images` + `out_mined/labels`
(recovered/promoted balls the full-frame model missed, plus time-sampled confident frames) and a
`manifest.csv`. Resumable (re-run to continue; skips done videos via `stats.json`). GPU auto-used.

## Step 2 — Select a small review set + correct it

```bash
python golf/select_review_frames.py /path/to/videos out_review --model golf_ego_v2_best.pt --total 500
```
Picks ~500 frames that are simultaneously **uncertain** (club present but no ball → a missed ball;
green scene, no detection; mid-confidence) and **diverse** (spread across scenes/people). Writes
`out_review/images`, YOLO pre-labels, and `out_review/ls_tasks.json`.

Correct them (you've used Label Studio before — memory `golf-label-studio`):
1. New LS project, labeling = **Object Detection with 2 rectangle labels: `ball`, `club_head`**.
2. Point LS local storage at `out_review/images/` (or set `--ls-prefix` to match your LS
   `LOCAL_FILES_DOCUMENT_ROOT`), then **Import** `out_review/ls_tasks.json` — the detector's boxes
   come in as pre-annotations, so you only fix mistakes (add missed balls, delete false ones).
3. Fix, then **Export → YOLO** into `out_review_corrected/` (an `images/` + `labels/` pair).

Prefer no LS? Just hand-edit the YOLO `out_review/labels/*.txt` and use `out_review` as the
corrected set. Skipping this step entirely = the "pure auto" path (faster, weaker on new scenes).

## Step 3 — Build golf_ego_v3 and fine-tune

```bash
python golf/build_and_train_golf.py \
    --base-weights golf_ego_v2_best.pt \
    --val      carried/golf_ego_v1/images/val \
    --old      carried/golf_ego_v1/images/train \
    --reviewed out_review_corrected \
    --mined    out_mined \
    --name golf_ego_v3 --imgsz 1280 --epochs 40 --batch 6 --auto-cap-frac 0.5
```
It assembles the train set (no image copying — uses path lists), **drops any frame from a val
video** (leakage guard), **caps auto-mined labels** to ≤50 % of train so reviewed/hand labels
dominate, prints the **baseline v2 val**, fine-tunes from v2 weights with early stopping, then
prints the **v3 val and the Δ**. Add `--dry-run` first to sanity-check the counts.

Read the output:
- **ball recall ↑ and mAP50 held/↑** → success. Bring `runs/detect/golf_ego_v3/weights/best.pt`
  back; export & deploy per `GOLF_YOLO.md` (`yolo export … format=tflite half=True imgsz=640`).
- **ball recall ↓** → drift/forgetting. Trust humans more: raise `--auto-cap-frac` toward 0.3,
  add more reviewed frames, or lower `--epochs`. Re-run. Do at most 1–2 rounds.

## Step 4 — Bring back

Only `golf_ego_v3/weights/best.pt` returns to the main machine. Update `GOLF_YOLO.md` (re-run its
`yolo val`), export TFLite f16 @640, drop into `android/app/src/main/assets/golf.tflite`, rebuild.

## Knobs worth knowing

- `mine_golf_videos.py --cap-per-video 80 --max-gap 6 --conf-keep 0.40` — per-video frame cap, how
  long a track gap to recover, confident-sample threshold.
- `select_review_frames.py --total 500 --per-video-cap 40 --novelty 1.0` — review budget, anti one-
  video-dominates cap, diversity weight (raise to spread wider across scenes).
- `build_and_train_golf.py --auto-cap-frac 0.5 --patience 12` — auto-label trust cap, early-stop
  patience on the fixed val.

Related docs: `GOLF_YOLO.md` (detector card), `GOLF_APP.md` (hit counter), `GOLF_PLAN.md` (plan).
