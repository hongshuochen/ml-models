# golf_ego_v6 — results log

v6 = fine-tune **v5** (`golf_ego_v5_nomined`) on the Label-Studio human-reviewed egocentric set, 3-class
{ball, club_head, hole}, imgsz 1280. `--old` = `golf_hole_reviewed` (842 frames, 434 hole — the v3-era
hand-labeled hole set; NOT v2, which has no hole). See [[golf-v6-dataset]] memory, `LABEL_STUDIO_REVIEW.md`,
`DATASET_SPLIT.md`.

## Frozen eval val
`golf_reviewed/val` — **3,391 images, 5,639 instances** (ball 3,166 / club_head 1,622 / **hole 851**).
By-person split (val people: Michael, Joy, Hiro, Ramu, Aryan). Same val across p50/p85 → directly comparable.

## Results on the frozen val (imgsz 1280)

| model | train frames (hole boxes) | mAP50 | mAP50-95 | ball R | club_head R | hole R |
|---|---|---|---|---|---|---|
| **v5** (baseline) | — | 0.880 | — | 0.894 | — | — |
| **v6_p50** (2026-07-29) — 50% train → `golf_v6.tflite` | 35,814 (9,761) | **0.940** | 0.844 | 0.909 | 0.862 | **0.868** |
| **v6_p85** (2026-07-30) — 85% train → `golf_v7.tflite` | 61,494 (15,323) | _TBD_ | | | | |
| **v8 / full** (2026-08-04) — 100% train → `golf_v8.tflite` | 69,024 (16,203) | _TBD_ | | | | |

Per-class @ v6_p50: ball P0.903 R0.909 mAP50 0.950 · club_head P0.887 R0.863 mAP50 0.923 · hole P0.960 R0.870 mAP50 0.948.
**Δ p50 vs v5:** mAP50 **+0.060**, ball recall **+0.015**. First real `hole` recall = **0.87** (was un-measurable before — no hole val).

## Caveats (absolute numbers may be optimistic; relative Δ is solid)
- **val is 1fps** — the fast-swing frames (hardest: impact, motion-blur) were subsampled out → metrics skew high vs the full distribution.
- **val is detector-prelabeled + human-reviewed** — a detector miss the reviewer didn't add isn't in GT → recall slightly overstated. (`--keep-empty` mitigates: reviewers saw empty frames too.)
- "best at epoch 1" on p50 = fine-tune from a strong v5 converges fast; normal, best.pt is good.

## Deploy
`golf/export_golf_rawhead_tflite.py --weights runs/detect/golf_ego_v6_p50-3/weights/best.pt --imgsz 640 --out golf_v6.tflite`
→ 19 MB raw-head f16 (3 maps `[1,G,G,7]`). App auto-loads highest `golf_v<N>.tflite`. Phone + tflite runner both **letterbox**.

## Version → tflite naming
p50 → `golf_v6.tflite`, p85 → `golf_v7.tflite`, full → `golf_v8.tflite`. The app auto-loads the highest
`golf_v<N>.tflite`. All are the same 3-class raw-head export at 640; only the training-data % differs.

## Data progression (train, on the same frozen val)
p50 35,814 (9,761 hole) → p85 61,494 (15,323) → **full 69,024 (16,203)**. val unchanged (3,391 / 851 hole) throughout.

## Next
- Fill p85 + v8 val rows once trained.
- test-set eval (`golf_reviewed/test`, people: Yujin, Alex, Kun, Madhu, AJ) → the honest held-out number.
- v8 = the full-data model; base is still v5 (full train ⊇ p85 ⊇ p50 → clean single-shot from v5, no chained drift).
