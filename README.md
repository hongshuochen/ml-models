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

---

## Full benchmarks — every model × precision

Latency¹ = TFLite **CPU, 4 threads**, 640² input (AMD Ryzen 9 5950X, XNNPACK), mean of
40 runs. **int8 gives no x86/WASM CPU speedup** (often slightly slower) — its only benefit
on desktop/browser is file size; on ARM/mobile NPUs int8 does accelerate. All quantization
is post-training (PTQ): **float16** = half size, no accuracy loss; **int8 full-integer** =
smallest, needs calibration, can drop accuracy; **int8 dynamic-range** = int8 weights /
float activations, no calibration, ≈ float accuracy.

### Face (WIDER FACE) — 2.50 M params · 5.8 GFLOPs · 1 class
| Precision | Size | P | R | mAP@50 | mAP@50-95 | Latency¹ |
|-----------|-----:|---:|---:|-------:|----------:|---------:|
| float32 | 9.82 MB | 0.840 | 0.608 | **0.682** | 0.365 | 18.5 ms |
| float16 | 4.99 MB | 0.839 | 0.608 | 0.682 | 0.364 | 18.8 ms |
| int8 full-integer | 2.79 MB | 0.786 | 0.492 | 0.545 | 0.240 | 20.6 ms |
| int8 dynamic-range | 2.83 MB | 0.840 | 0.608 | 0.681 | 0.364 | 20.9 ms |

### Hand pose (21 kpts) — 3.12 M · 8.3 GFLOPs · Pose (OKS) metrics
| Precision | Size | P | R | mAP@50 | mAP@50-95 | Latency¹ |
|-----------|-----:|---:|---:|-------:|----------:|---------:|
| float32 | 13.2 MB | 0.928 | 0.919 | **0.925** | 0.789 | 29.6 ms |
| float16 | 6.86 MB | 0.928 | 0.919 | 0.924 | 0.790 | 29.3 ms |
| int8 full-integer | 3.90 MB | N/A² | N/A² | N/A² | N/A² | 27.9 ms |
| int8 dynamic-range | 4.03 MB | N/A² | N/A² | N/A² | N/A² | 28.7 ms |

² **int8 is not viable for the pose head** — both variants corrupt the NMS-free class
output (invalid class indices → `yolo val` crashes). Use float16 for pose.

### Face + Hand (nano) — 2.50 M · 5.8 GFLOPs · 2 classes
| Precision | Size | mAP@50 | mAP@50-95 | face AP | hand AP | Latency¹ |
|-----------|-----:|-------:|----------:|--------:|--------:|---------:|
| float32 | 9.83 MB | **0.831** | 0.605 | 0.670 | 0.992 | 17.1 ms |
| float16 | 5.01 MB | 0.831 | 0.605 | 0.670 | 0.992 | 17.1 ms |
| int8 full-integer | 2.80 MB | 0.721 | 0.527 | 0.454 | — | 19.1 ms |
| int8 dynamic-range | 2.84 MB | 0.830 | 0.605 | 0.669 | 0.992 | 17.7 ms |

### Face + Hand "Pico" — 0.68 M · 1.68 GFLOPs · 2 classes
| Precision | Size | mAP@50 | mAP@50-95 | face AP | hand AP | Latency¹ |
|-----------|-----:|-------:|----------:|--------:|--------:|---------:|
| float32 | 2.80 MB | 0.780 | 0.585 | ~0.57 | ~0.99 | 6.7 ms |
| float16 | 1.49 MB | 0.779 | 0.567 | 0.569 | 0.988 | 6.9 ms |
| int8 full-integer | **0.93 MB** | 0.654 | 0.476 | 0.383 | 0.926 | 11.6 ms |
| int8 dynamic-range | 1.04 MB | 0.779 | 0.567 | 0.569 | 0.988 | 8.0 ms |

### Face + Hand "Pico-P4P5" (drop-P3) — 0.64 M · 1.2 GFLOPs · 2 classes
| Precision | Size | mAP@50 | mAP@50-95 | face AP | hand AP | Latency¹ |
|-----------|-----:|-------:|----------:|--------:|--------:|---------:|
| float32 | 2.48 MB | 0.752 | 0.547 | 0.516 | 0.989 | 5.4 ms |
| float16 | 1.31 MB | 0.752 | 0.547 | 0.515 | 0.989 | 5.1 ms |
| int8 full-integer | 0.818 MB | 0.676 | 0.498 | 0.366 | 0.986 | 10.4 ms |
| int8 dynamic-range | **0.808 MB** | 0.752 | 0.546 | 0.516 | 0.989 | 5.8 ms |

### Face + Hand **+ HaGRID** (webcam-domain) — 2 classes
Same architectures/sizes as their base models above; only the training data changed.
Two scorecards: **orig** = WIDER+hand-keypoints val, **webcam** = held-out HaGRIDv2 val.

| Model | Precision | Size | orig mAP@50 (face/hand) | webcam mAP@50 (face/hand) |
|-------|-----------|-----:|------------------------:|--------------------------:|
| nano + HaGRID | float32 | 9.83 MB | 0.814 (0.638 / 0.989) | 0.991 (0.994 / 0.988) |
| nano + HaGRID | float16 | 5.01 MB | 0.814 | 0.991 |
| **Pico-P4P5 + HaGRID** | float32 | 2.48 MB | 0.709 (0.438 / 0.980) | 0.991 (0.994 / 0.989) |
| **Pico-P4P5 + HaGRID** | float16 | 1.31 MB | 0.709 | 0.991 |
| **Pico-P4P5 + HaGRID** | int8 dynamic-range | **0.807 MB** | 0.709 | 0.991 |

(int8/f16/f32 accuracy are ≈ equal here — dynamic-range int8 keeps float accuracy.)

### Hand-landmark regressors — 224² crop · 21 keypoints
PCK@0.1 on hand-keypoints val (true GT) and the held-out HaGRID val (vs MediaPipe labels).

| Backbone | Params | Data | hk PCK | webcam PCK | f16 / int8 | Latency¹ |
|----------|------:|------|-------:|-----------:|-----------:|---------:|
| MobileNetV3-small | 1.56 M | hk | **0.971** | — | 3.16 MB | 1.2 ms |
| MobileNetV3-small_050 | 0.61 M | hk | 0.959 | 0.460 | 1.27 MB | 1.1 ms |
| MobileNetV2_035 | 0.45 M | hk | 0.951 | — | 0.92 MB | 0.9 ms |
| MobileNetV3-small_035 | 0.38 M | hk | 0.942 | — | ~0.8 MB | — |
| MobileNetV3-small_025 | 0.29 M | hk | 0.942 | 0.395 | 0.63 MB | — |
| MNv3-small_050 + HaGRID | 0.61 M | hk+HaGRID | 0.945 | **0.927** | 1.27 MB | 1.1 ms |
| **MNv3-small_025 + HaGRID** | 0.29 M | hk+HaGRID | 0.921 | **0.899** | 0.63 / **0.41 MB** | ~0.9 ms |

(Landmark f32 ≈ 2× the f16 size. int8 landmark loads on native/ARM but not in the
browser's tfjs-tflite WASM — use f16 there.)

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

## Known limitations & roadmap
- **Landmark "OK" gesture tracking is weak.** The hand-landmark regressor is *distilled
  from MediaPipe pseudo-labels*, so it inherits MediaPipe's errors and is capped by its
  quality — the thumb+index pinch/occlusion of the OK gesture is where it degrades most.
  Next: benchmark our landmark vs **MediaPipe Hands** as a strong baseline (it's the
  teacher → the natural upper bound), then improve via more OK-heavy training data, a
  stronger teacher (e.g. HaMeR / WiLoR) for hard poses, or a higher-capacity backbone.
- **The Android app is detector-only** for now; the landmark stage ships once the above lands.
- **Browser int8:** tfjs-tflite's WASM runtime can't load dynamic-range int8 → the web app
  uses float16 (Android/native handle int8 fine).

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
