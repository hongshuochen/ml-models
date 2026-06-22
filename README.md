# ml-models — on-device face & hand detection + hand landmarks

Real-time, **on-device** face/hand ML: YOLO26 detectors plus a MobileNet 21-keypoint
hand-landmark regressor — trained with [uv](https://docs.astral.sh/uv/) + Ultralytics,
exported to **TFLite**, and runnable live in the browser
([`webcam-tflite/`](webcam-tflite/)).

The headline result: a **MediaPipe-style two-stage hand pipeline** (detect box → crop →
regress 21 keypoints) that fits in **~1.2 MB int8** and works on normal webcam framing —
after fixing a catastrophic webcam blind spot with the **HaGRID** dataset (see below).

Full numbers (P/R/mAP/params/FLOPs/size/latency across float32·float16·int8) live in
**[`MODELS_REPORT.md`](MODELS_REPORT.md)**; exact reproduce commands per model in
**[`TRAINING.md`](TRAINING.md)**.

---

## ⭐ Recommended on-device stack (compact two-stage, webcam-ready)

| Stage | Model | int8 size | Key metric (webcam domain) |
|-------|-------|----------:|----------------------------|
| 1. Detect | Face+Hand **Pico-P4P5 +HaGRID** | **0.807 MB** | mAP50 0.991 · hand AP **0.989** |
| 2. Landmark | **MNv3-small_025 +HaGRID** (21 kpts) | **0.413 MB** | PCK@0.1 **0.899** |
| | **total** | **~1.22 MB** | |

Both were retrained on HaGRID — the un-augmented baselines were nearly blind to
webcam-framed hands (detector hand AP **0.031**, landmark PCK **0.395**).

---

## Models

All YOLO models are **YOLO26**, input **640×640**. Metrics are mAP@50 unless noted.

### Detectors

| Model | Task | Params | mAP50 (per-class) | int8 size | Training data |
|-------|------|------:|-------------------|----------:|---------------|
| Face | detect (face) | 2.50 M | 0.682 | 2.83 MB | WIDER FACE |
| Hand pose | pose, 21 kpts | 3.12 M | 0.927 (pose) | — ¹ | hand-keypoints |
| Face+Hand (nano) | detect (face, hand) | 2.50 M | 0.831 (f 0.670 / h 0.992) | 2.84 MB | WIDER + hand-keypoints |
| Face+Hand **Pico** | detect | 0.68 M | 0.779 | **0.93 MB** | WIDER + hand-keypoints |
| Face+Hand **Pico-P4P5** | detect (drop-P3) | 0.64 M | 0.752 (h 0.989) | **0.808 MB** | WIDER + hand-keypoints |
| Face+Hand nano **+HaGRID** | detect | 2.50 M | orig 0.814 / **webcam 0.991** | 2.84 MB | + HaGRIDv2 |
| Face+Hand **Pico-P4P5 +HaGRID** ⭐ | detect | 0.64 M | orig 0.709 / **webcam 0.991** | **0.807 MB** | + HaGRIDv2 |

¹ int8 corrupts the NMS-free pose head — use float16 (6.86 MB) for the pose model.

### Hand-landmark regressors (stage 2 · 21 keypoints · 224² crop)

| Backbone | Params | PCK@0.1 (hand-keypoints) | PCK@0.1 (webcam) | f16 / int8 | Training data |
|----------|------:|-------------------------:|-----------------:|-----------:|---------------|
| MobileNetV3-small | 1.56 M | **0.971** | — | 3.16 MB | hand-keypoints |
| MobileNetV3-small_050 | 0.61 M | 0.959 | 0.460 | 1.27 MB | hand-keypoints |
| MobileNetV2_035 | 0.45 M | 0.951 | — | 0.92 MB | hand-keypoints |
| MobileNetV3-small_035 | 0.38 M | 0.942 | — | ~0.8 MB | hand-keypoints |
| MobileNetV3-small_025 | 0.29 M | 0.942 | 0.395 | 0.63 MB | hand-keypoints |
| MNv3-small_050 **+HaGRID** | 0.61 M | 0.945 | **0.927** | 1.27 MB | + HaGRIDv2 |
| MNv3-small_025 **+HaGRID** ⭐ | 0.29 M | 0.921 | **0.899** | 0.63 / **0.41 MB** | + HaGRIDv2 |

> **Two findings worth knowing:** (1) **pretraining isn't needed** here — from-scratch
> matched/beat ImageNet-pretrained, which freed us to use any width (incl. custom
> 0.35/0.25). (2) **FLOPs don't predict CPU latency** at this scale — MobileNetV2's plain
> depthwise convs beat MobileNetV3's SE + hard-swish on XNNPACK despite 2.5× the FLOPs.

---

## The HaGRID story — fixing the webcam-domain blind spot

The detectors learned hands from the **hand-keypoints** set (close-up, hand-centric
shots), so on normal webcam framing (person 0.5–4 m away) they **almost never detected
hands**. Measured on a held-out HaGRIDv2 val:

| | webcam hand AP (detector) | webcam landmark PCK |
|---|--:|--:|
| baseline (hand-keypoints only) | **0.012–0.031** 💀 | 0.40–0.46 |
| **+ HaGRIDv2** | **0.988–0.989** ✅ | 0.90–0.93 |

**Fix:** fine-tune on **HaGRIDv2** (official 512px, 34 gestures, ~1.08 M images,
subject-disjoint splits). Hand boxes come from HaGRID's own human annotations; **face
boxes are pseudo-labeled by InsightFace SCRFD-10G**, and the **21 landmarks are HaGRID's
bundled MediaPipe annotations**. Original-task accuracy drops only slightly (the easy
HaGRID faces pull off WIDER's tiny crowd faces). Scripts: `prepare_hagrid.py` (detector
data), `prepare_hagrid_landmark.py` (landmark data).

> ⚠️ **License / research-only:** the HaGRID-augmented models use **InsightFace**
> pretrained models, which are **non-commercial research only**, and **HaGRID** is
> CC-BY-SA-4.0. For a commercial product, swap the face-labeling teacher for **YuNet**
> (MIT) or **MediaPipe BlazeFace** (Apache-2.0). The non-HaGRID models above carry no
> such restriction.

---

## Datasets (auto/scripted, git-ignored)

| Dataset | Used for | Prep |
|---------|----------|------|
| WIDER FACE | face detection | `prepare_widerface.py` → `widerface.yaml` |
| Ultralytics hand-keypoints | hand pose + landmarks | auto; use `hand-keypoints-fixed.yaml` (corrected `flip_idx` — the stock one scrambles mirrored hands) |
| Face+Hand (WIDER + hand-keypoints) | 2-class detection | `build_face_hand.py` → `face-hand.yaml` |
| HaGRIDv2 (512px) | webcam-domain augmentation | `prepare_hagrid.py` / `prepare_hagrid_landmark.py` → `face-hand-hagrid-v2.yaml`, `hagrid-val.yaml` |

---

## Quickstart

```bash
uv sync                                   # Python 3.12 env (Ultralytics, torch, timm, ...)
# reproduce any model — see TRAINING.md for exact per-model commands, e.g.:
uv run yolo detect train model=yolo26n.pt data=face-hand.yaml epochs=100 imgsz=640
uv run python train_hand_landmark.py --backbone mobilenetv3_small_025 --no-pretrained
# export to TFLite (float32/float16/int8) — see MODELS_REPORT "Reproduce"
uv run yolo export model=<best.pt> format=tflite imgsz=640
```

### Web app — live in-browser inference
[`webcam-tflite/`](webcam-tflite/): Next.js + TF.js (WASM) running the TFLite models on
the webcam, fully on-device. `npm install && npm run dev`.

---

## Repo layout

```
*.yaml                     dataset / model-scale configs (one per model)
prepare_*.py build_*.py    dataset builders (widerface, face-hand, HaGRID detect/landmark)
train_hand_landmark.py     stage-2 landmark regressor trainer (multi-backbone, multi-data)
export_landmark.py         landmark torch → ONNX → TFLite
bench_latency.py           TFLite CPU latency benchmark
MODELS_REPORT.md           full metrics matrix (all models, all precisions)
TRAINING.md                exact reproduce commands per model
webcam-tflite/             in-browser demo app
```
Trained weights live under `runs/` and datasets under `datasets/` — both git-ignored
(regenerate via the commands in `TRAINING.md`).

## Environment
uv-managed Python 3.12 · Ultralytics 8.4 · torch 2.12 (cu130) · trained on an RTX 3080.
