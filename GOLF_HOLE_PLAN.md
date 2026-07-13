# Golf — Adding the `hole` Class (Plan)

Extend the detector from 2 classes `{ball, club_head}` to 3 `{ball, club_head, hole}` (the cup on
the green). **Why:** it lets the hit counter tell a *made putt / tap-in* (ball's track ends in the
hole) from a ball merely lost, and enables putting analytics (distance-to-hole). See GOLF_APP.md
for the counter, GOLF_YOLO.md for the detector.

## The two decisions that shape everything

1. **Public hole data is BOOTSTRAP-ONLY.** The public sets label `ball` + `hole` but have **no
   `club_head`** — dropping them raw into training would teach "club_head = background" and wreck
   our 0.90 club recall. So public data never enters the final training set. Instead it trains a
   throwaway **hole-only teacher**, which only ever pre-labels the `hole` class.
2. **Two-model pre-annotation.** On our egocentric footage, pre-labels come from *two* models —
   `golf_ego_v2` for ball+club_head (strong on our domain) and the hole-teacher for hole. Merged,
   they give 3-class pre-annotations the human corrects. The final 3-class model trains only on our
   own frames, where all three classes are labelled.

**The partial-label rule (applies to OUR data too):** any training/val frame that contains a
visible hole **must** get a hole box, or it teaches "hole = background". Holes only appear in
putting/approach frames, so this is a bounded sweep — but it **includes the existing labelled
putting frames already in train/val** (e.g. golf_010, ho_0703_011, ho_0704_005), which currently
have no hole label. The fixed val must get its holes labelled too, or hole recall is unmeasurable.

## Public datasets (vetted 2026-07-13, Phase 0 done)

| dataset | imgs | classes | hole boxes | license | notes |
|---|---|---|---|---|---|
| bosharluke/golf-ball-and-hole-detection (v6) | 415 | 0=ball, 1=hole | 239 | CC BY 4.0 | some low-angle/first-person putting views — decent |
| sai-gon-university/golf-ball-and-hole-detection-1k7 (v6) | 1171 | 0=ball, 1=hole | 637 | CC BY 4.0 | 2×2-tiled (a few empty junk tiles); good cup close-ups |
| golf-green-flag-stick/flag-stick | 164 | Ball, FlagStick | — | BY-NC-SA | flag ≠ hole; skip unless we want a flag class |

Neither has club_head. Viewpoints are mixed (some front-view, some usefully down-tilted) — fine for
a bootstrap teacher, **not** a substitute for egocentric hole labels.

## Pipeline

```
Phase 0  vet + download public sets                      [DONE]
Phase 1  hole-only teacher (public, ball dropped)         [DONE — training]
Phase 2  label hole in OUR putting footage (2-model pre-anno + Label Studio)   [offline machine]
Phase 3  build 3-class set (+ val hole-sweep) + fine-tune golf_ego_v2 -> v3    [offline machine]
Phase 4  deploy: GolfDetector 3-class, export TFLite, use hole in the counter
Phase 5  docs / memory
```

### Phase 1 — hole-only teacher (done here)
- `golf/prepare_hole_teacher.py` → `datasets/golf_hole/` : keeps only public `hole` boxes
  (rewritten to class 0), drops `ball`, keeps ball-only images as negatives. train 1969 / val 256.
- Train: fine-tune `golf_ego_v2` → 1-class `{hole}`:
  ```
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run yolo detect train \
    model=runs/detect/golf_ego_v2_1280/weights/best.pt \
    data=datasets/golf_hole/golf_hole_teacher.yaml imgsz=1280 epochs=40 batch=6 \
    patience=12 name=golf_hole_teacher device=0 mixup=0.1 copy_paste=0.1
  ```
  → `runs/detect/golf_hole_teacher/weights/best.pt` (carry-over artifact for Phase 2).

### Phase 2 — label hole in our putting footage (offline machine)
Only clips where a hole is visible (putting green / short approach — not tee/range).
```
python golf/select_review_frames.py /putting_videos out_hole \
    --model golf_ego_v2.pt --hole-model golf_hole_teacher.pt --total 500
```
The two-model pass merges ball/club (v2) + hole (teacher); the selector now also favours
`hole_seen` frames and `putt_no_hole` scenes (puttable green where the weak teacher missed the
hole). Import `out_hole/ls_tasks.json` into Label Studio with **3 labels: ball, club_head, hole**,
correct (add missed holes, fix ball/club), export YOLO → `out_hole_corrected/`.
Target ~300–600 egocentric hole instances.
**Also** hole-sweep the existing labelled putting frames in the current train/val (add hole boxes).

### Phase 3 — build 3-class dataset + fine-tune (offline machine)
```
python golf/build_and_train_golf.py \
    --base-weights golf_ego_v2.pt \
    --val   carried/golf_ego_v1/images/val   \   # fixed val, now WITH hole labels swept in
    --old   carried/golf_ego_v1/images/train \   # existing ego train, holes swept in
    --reviewed out_hole_corrected \
    --names ball,club_head,hole --name golf_ego_v3_hole --imgsz 1280 --epochs 40
```
Head re-inits for nc 2→3, backbone/neck transfer from v2. Prints per-class val incl. hole; confirm
ball/club recall holds on the same val frames.

### Phase 4 — deploy + use it
- `android-golf/.../GolfDetector.kt`: `LABELS = ["ball","club_head","hole"]`; OverlayView colour for hole.
- Export TFLite f16 @640 → `android-golf/app/src/main/assets/golf.tflite`.
- Counter: a putt whose ego-propagated track ends inside/at a `hole` box = **made putt** (distinct
  from vanish-elsewhere); tap-in becomes unambiguous.

## Code changes made for this (committed)
- `golf/prepare_hole_teacher.py` — build the hole-only teacher dataset.
- `golf/select_review_frames.py` — `--hole-model` (two-model pre-annotation) + hole-aware selection.
- `golf/build_and_train_golf.py` — `--names` (arbitrary class count; `ball,club_head,hole`).

## Risks
- **Viewpoint gap** on public hole data → bootstrap only (done).
- **Class imbalance** — hole only in putting frames; gather enough (aim 300–600).
- **Partial-label trap** — every hole-containing train/val frame must be hole-labelled (the val sweep
  is easy to forget → hole recall would read 0).
- **Metric comparability** — head changes with nc; compare ball/club recall on the *same* val frames.
