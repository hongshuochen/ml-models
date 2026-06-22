# Models Report — YOLO26-nano (face / hand)

On-device models trained locally and exported to TensorFlow Lite for the
`webcam-tflite` web app. All models: **YOLO26-nano**, input **640×640**, Ultralytics
8.4.70 (`torch 2.12.1`), TFLite via onnx2tf / TF 2.19.

**Last updated:** 2026-06-19 — complete: face-detect, hand-pose, face+hand, hand-landmark.

## Quantization types used (all PTQ, not QAT)

| Type | Weights | Activations | Calibration data | Notes |
|------|---------|-------------|------------------|-------|
| float32 | f32 | f32 | — | baseline |
| float16 | f16 | f16 | — | half size, no accuracy loss |
| **int8 (full-integer)** | int8 | int8 | **required** | smallest; accuracy can drop |
| **int8 (dynamic-range)** | int8 | f32 (quantized at runtime) | **none** | between f16 and full-int8 |

All of the above are **post-training quantization (PTQ)** — applied at export, no
retraining. **QAT (quantization-aware training)** is different: it inserts
fake-quant ops *during training* so the model learns to tolerate quantization, and
would require re-training each model. We are **not** using QAT here.

## 1. Face Detection — 1 class (face)
Weights `runs/detect/widerface_yolo26n/weights/best.pt` · **2.50 M** params · **5.8 GFLOPs** ·
WIDER val (3,226 imgs / 39,696 faces) · output `(1,300,6)` end-to-end (NMS-free)

| Precision | File | P | R | mAP@50 | mAP@50-95 | Latency¹ |
|-----------|-----:|---:|---:|-------:|----------:|---------:|
| float32 | 9.82 MB | 0.840 | 0.608 | **0.682** | 0.365 | 18.5 ms |
| float16 | 4.99 MB | 0.839 | 0.608 | 0.682 | 0.364 | 18.8 ms |
| int8 (full-int) | 2.79 MB | 0.786 | 0.492 | 0.545 | 0.240 | 20.6 ms |
| **int8 (dyn-range)** | 2.83 MB | 0.840 | 0.608 | **0.681** | 0.364 | 20.9 ms |

## 2. Hand Pose — 1 class (hand) + 21 keypoints
Weights `runs/pose/hand_pose_fixed/weights/best.pt` (corrected `flip_idx`) · **3.12 M** params ·
**8.3 GFLOPs** · hand val (7,847 imgs) · output `(1,300,69)`. Metrics below are **Pose** (OKS); Box mAP@50 ≈ 0.991 for f32/f16.

| Precision | File | Pose P | Pose R | Pose mAP@50 | Pose mAP@50-95 | Latency¹ |
|-----------|-----:|-------:|-------:|------------:|---------------:|---------:|
| float32 | 13.2 MB | 0.928 | 0.919 | **0.925** | 0.789 | 29.6 ms |
| float16 | 6.86 MB | 0.928 | 0.919 | 0.924 | 0.790 | 29.3 ms |
| int8 (full-int) | 3.90 MB | N/A² | N/A² | N/A² | N/A² | 27.9 ms |
| int8 (dyn-range) | 4.03 MB | N/A² | N/A² | N/A² | N/A² | 28.7 ms |

## 3. Face + Hand Detection — 2 classes (face, hand)
Weights `runs/detect/face_hand_yolo26n/weights/best.pt` · **2.50 M** params · **5.8 GFLOPs** ·
combined `datasets/face-hand` (31,656 train / 11,218 val) · output `(1,300,6)`. Overall (both classes):

| Precision | File | P | R | mAP@50 | mAP@50-95 | Latency¹ |
|-----------|-----:|---:|---:|-------:|----------:|---------:|
| float32 | 9.83 MB | 0.905 | 0.787 | **0.831** | 0.605 | 17.1 ms |
| float16 | 5.01 MB | 0.905 | 0.787 | 0.831 | 0.605 | 17.1 ms |
| int8 (full-int) | 2.80 MB | 0.804 | 0.717 | 0.721 | 0.527 | 19.1 ms |
| **int8 (dyn-range)** | 2.84 MB | 0.905 | 0.786 | **0.830** | 0.605 | 17.7 ms |

**Per-class AP@50** (float32): **face 0.670 · hand 0.992**. Hands are easy; WIDER crowd
faces are hard. Full-int8 hits faces hardest (face 0.454); dynamic-range int8 keeps it
(face 0.669, hand 0.992) — matching the float models, no calibration needed.

## 4. Hand Landmark Regressor — stage 2 · 21 keypoints
A small MobileNet on a **224×224** hand crop regresses 21 (x,y) keypoints. Trained on
hand-keypoints crops (18.8 k hands; box-jitter + flip + color aug). Backbones compared
(f16 TFLite; PCK/err³ from the torch model; latency = TFLite CPU, 4-thread):

| Backbone | Params | MFLOPs | Init | f16 size | PCK@0.1³ | mean err | f16 latency¹ |
|----------|------:|------:|------|---------:|--------:|---------:|-------------:|
| MobileNetV3-small (torchvision) | 1.56 M | 120 | ImageNet | 3.16 MB | **0.971** | 0.0221 | 1.2 ms |
| MobileNetV3-small_050 (timm) | 0.61 M | 46 | ImageNet (60 ep) | 1.27 MB | 0.949 | 0.0314 | 1.1 ms |
| MobileNetV3-small_050 (timm) | 0.61 M | 46 | scratch (100 ep) | 1.27 MB | 0.959 | 0.0281 | 1.1 ms |
| **MobileNetV2_035 (timm)** | **0.45 M** | 116 | scratch (100 ep) | 0.92 MB | 0.951 | 0.0298 | 0.9 ms |
| MobileNetV3-small_035 (custom) | 0.38 M | — | scratch (100 ep) | ~0.8 MB | 0.942 | 0.0329 | — |
| **MobileNetV3-small_025 (custom)** | **0.29 M** | — | scratch (100 ep) | **0.63 MB** | 0.942 | 0.0335 | — |

(f32 files ≈ 2× the f16 size. _035/_025 widths aren't registered in timm — built via
`_gen_mobilenet_v3`; only possible because we train from scratch.)

**Findings:**
- **Pretraining is not needed for this task.** Scratch MNv3-small_050 (PCK 0.959 @100 ep)
  matched/beat ImageNet-pretrained (0.949 @60 ep). Keypoint regression is far enough from
  ImageNet classification, and the data large enough, that pretrained features add no
  final-accuracy edge — which also frees us to use *any* width (incl. unregistered
  _035/_025), not just widths that ship pretrained checkpoints.
- **FLOPs don't predict latency at this scale.** MNv2_035 has 2.5× the FLOPs of
  MNv3-small_050 yet runs *faster* (0.9 vs 1.1 ms) — plain depthwise convs beat MNv3's
  SE + hard-swish on XNNPACK.
- **Width vs accuracy:** 0.61 M → 0.29 M costs only ~1.7 pts (0.959 → 0.942), and
  0.38 M ≈ 0.29 M (both 0.942) → **MNv3-small_025 (0.29 M) is the efficient floor.**

Weights: `runs/landmark/hand_landmark*/` (1.56 M / _mnv3s050* / _mnv2_035_scratch /
_mnv3s035_scratch / _mnv3s025_scratch).

### 4b. HaGRID landmark — fixing webcam-domain keypoints
The hand-keypoints-only regressors do poorly on webcam-framed hands — on a held-out
HaGRID landmark val: MNv3-small_050 PCK@0.1 **0.460**, _025 **0.395**. Retrain on
`hand-keypoints + HaGRIDv2` combined (HaGRID's MediaPipe 21-pt labels via
`prepare_hagrid_landmark.py`; 84.6 k train hands; 40 ep from scratch).

| Backbone (data) | Params | hand-keypoints PCK | **webcam PCK** | f16 / int8 |
|-----------------|------:|-------------------:|---------------:|-----------:|
| MNv3-small_050 (hk only) | 0.61 M | 0.959 | 0.460 | 1.27 MB |
| **MNv3-small_050 (+HaGRID)** | 0.61 M | 0.945 | **0.927** | 1.27 MB |
| MNv3-small_025 (hk only) | 0.29 M | 0.942 | 0.395 | 0.63 MB |
| **MNv3-small_025 (+HaGRID)** | 0.29 M | 0.921 | **0.899** | 0.63 / **0.41 MB** |

**Webcam landmark PCK 0.40–0.46 → ~0.90–0.93**, small cost to the original val. Caveat:
webcam PCK is vs MediaPipe pseudo-labels (agreement, not true GT); hand-keypoints val is
true GT. **Compact deploy:** MNv3-small_025 +HaGRID, **int8 0.41 MB**.

³ Accuracy from the torch model; the f16 TFLite is numerically near-identical.
**Stage-2 role:** YOLO (model 1 or 3) gives the hand box → crop → this regressor
predicts the 21 keypoints. Far lighter than the single-shot pose model (**0.12 vs
8.3 GFLOPs**, 1.2 vs 29 ms) and works on a tight crop for higher precision.
(TFLite export: torch → ONNX → onnxsim → onnx2tf; the simplify step is required.)

¹ Latency: TFLite CPU, single image, **4 threads** (matches the app's threaded
WASM), mean of 40 runs, AMD Ryzen 9 5950X, `tf.lite.Interpreter` (XNNPACK),
measured under concurrent GPU training (treat as upper bound; relative comparison
is consistent).

² **Both int8 variants are not viable for the Hand-Pose model.** Quantization
injects invalid class indices into padding detections, crashing `yolo val`'s
confusion matrix — full-integer (`index 75`) and dynamic-range (`index 124`)
alike, even with a conf filter. The model still yields correct top detections in
direct inference, but int8 is **not recommended** for the end-to-end pose head.
Use float16 for the pose model.

## 5. Face + Hand "pico" — sub-1 MB int8 detector (custom compound scale)
Stock **YOLO26 architecture**, custom scale **`[0.5, 0.125, 1024]`** (`yolo26n-pico.yaml`;
backbone/head byte-identical to official yolo26.yaml — only `nc`+`scales` changed) ·
2 classes · **0.68 M params · 1.68 GFLOPs** · `runs/detect/face_hand_pico/`. Built to
meet a <1 MB int8 requirement. (max_channels kept at 1024 so the C2PSA attention stays
valid; width 0.125 is the dominant size lever.)

| Precision | Size | mAP@50 | mAP@50-95 | face AP@50 | hand AP@50 | Latency |
|-----------|-----:|-------:|----------:|-----------:|-----------:|--------:|
| float32 | 2.80 MB | 0.780 | 0.585 | ~0.57 | ~0.99 | 6.7 ms |
| float16 | 1.49 MB | 0.779 | 0.567 | 0.569 | 0.988 | 6.9 ms |
| **int8 full-integer** | **0.93 MB** (930,219 B) | 0.654 | 0.476 | 0.383 | 0.926 | 11.6 ms |
| int8 dynamic-range | 1.04 MB (1,037,041 B) | 0.779 | 0.567 | 0.569 | 0.988 | 8.0 ms |

**Only full-integer int8 is under 1 MB decimal (0.93 MB)** — the headline deliverable —
but full-int8 hurts the hard face class (0.383); hands stay strong (0.926). Dynamic-range
int8 keeps **full float accuracy** (0.779 / face 0.569) and is faster, but is 1.04 MB
decimal (under 1 MiB only). vs the 2.5 M nano face+hand (mAP@50 0.831), the 0.68 M pico
loses ~5 pts at float — a good capacity/size trade.

## 6. Face + Hand "pico-P4P5" — drop-P3 (medium/large only)
`yolo26n-pico-p45.yaml`: pico with the **P3 (small-object) head removed** → 2 detection
scales, **strides [16, 32]** (min detectable ~16 px). **0.64 M params · 1.2 GFLOPs**
(28% fewer FLOPs than the 3-scale pico) · `runs/detect/face_hand_pico_p45/`. For use
cases that need medium/large objects and must keep large — small objects are dropped
by design.

| Precision | Size | mAP@50 | mAP@50-95 | face AP@50 | hand AP@50 | Latency |
|-----------|-----:|-------:|----------:|-----------:|-----------:|--------:|
| float32 | 2.48 MB | 0.752 | 0.547 | 0.516 | 0.989 | 5.4 ms |
| float16 | 1.31 MB | 0.752 | 0.547 | 0.515 | 0.989 | 5.1 ms |
| int8 full-integer | 0.818 MB | 0.676 | 0.498 | 0.366 | 0.986 | 10.4 ms |
| **int8 dynamic-range** | **0.808 MB** | **0.752** | 0.546 | 0.516 | **0.989** | 5.8 ms |

**Hands are unaffected by dropping P3** (AP@50 0.989). **int8 dynamic-range is the best
choice here**: it matches float32 accuracy (0.752 / hand 0.989) at **0.808 MB — strictly
< 1 MB (both decimal and MiB)** and 5.8 ms, beating full-integer int8 on size, speed, *and*
accuracy. Faces drop (most WIDER faces are tiny → undetectable; medium/large still caught
at 0.516). This is the smallest, fastest sub-1 MB face+hand detector with full-accuracy hands.

## 7. HaGRID augmentation — fixing the webcam-domain hand blind spot

**Problem found.** The detectors learned hands from the Ultralytics hand-keypoints set
(close-up, hand-centric shots), so on normal webcam framing (person 0.5–4 m away) they
**almost completely miss hands**. Measured on a held-out HaGRIDv2 val: baseline nano
hand AP@50 **0.012**, pico-P4P5 **0.031**. Faces transferred fine (~0.98); hands were a
near-total blind spot.

**Fix.** Fine-tune on **HaGRIDv2** (official 512px, 34 gestures, ~1.08 M imgs): HaGRID's
own human hand boxes (class 1) + face boxes pseudo-labeled by **InsightFace SCRFD-10G**
(`buffalo_l`, det_score ≥ 0.5, min-face-area filter to drop tiny photo/poster faces).
Subject-disjoint official splits → 67,464 train imgs added to the original 31,656;
**held-out webcam eval** = 10,100 imgs. Warm-start fine-tune (lr0 0.005, patience 12;
`prepare_hagrid.py` + `face-hand-hagrid-v2.yaml` + `hagrid-val.yaml`).

| Model (checkpoint) | original val — mAP50 / face / hand | webcam val — mAP50 / face / **hand** |
|--------------------|-----------------------------------:|-------------------------------------:|
| nano baseline | 0.831 / 0.670 / 0.992 | 0.498 / 0.984 / **0.012** |
| **nano + HaGRID** (`best.pt`, ep1) | 0.814 / 0.638 / 0.989 | 0.991 / 0.994 / **0.988** |
| pico-P4P5 baseline | 0.753 / 0.516 / 0.991 | 0.508 / 0.986 / **0.031** |
| **pico-P4P5 + HaGRID** (`last.pt`, ep13) | 0.709 / 0.438 / 0.980 | 0.991 / 0.994 / **0.989** |

**Result: webcam hand detection 0.01–0.03 → ~0.99** on both models, at a small cost to
the WIDER tiny-face task (HaGRID's large/close faces pull the model off WIDER's tiny
crowd faces). The bigger nano absorbs both (balanced already at ep1 → `best.pt`); the
smaller pico needs more adaptation (`last.pt`, ep13) and pays a bit more on the old task.

**Deploy (pico-P4P5 + HaGRID, `last.pt`):** TFLite float16 1.31 MB · **int8 dyn-range
0.807 MB** (same size as the pre-HaGRID pico, now with working webcam hands) · float32
2.48 MB. `runs/detect/face_hand_pico_p45_hagrid_ft/`.

**Caveat:** webcam-val *face* AP is vs InsightFace pseudo-labels (HaGRID has no face GT),
so it measures teacher-agreement; webcam *hand* AP is against HaGRID's real human boxes.

## Key findings

- **float16 = best deployment choice**: identical accuracy to float32 at **half the
  size**, same speed. Used in the app.
- **int8 gives no CPU speedup** on x86/AVX2 (equal or slightly slower); its only
  benefit here is file size. int8 speedups appear on ARM / mobile / edge NPUs.
- **For detection, dynamic-range int8 is the best int8 option**: Face mAP@50
  **0.681** (≈ float32's 0.682) at 2.83 MB, and it needs **no calibration**.
  Full-integer int8 loses far more (0.545) for the same size.
- **For the pose model, int8 is not viable** (both variants corrupt the class
  output → val crashes). Use float16 for pose.
- Net recommendation: **float16 everywhere** for accuracy parity; if you need the
  smallest detector, **dynamic-range int8** is a strong option.

## Reproduce

```bash
# float32 + float16
yolo export model=<best.pt> format=tflite imgsz=640 device=cpu
# int8 — separate dir (avoid clobbering f32/f16), limited calibration
yolo export model=<copy.pt> format=tflite int8=True data=<data.yaml> fraction=0.05 imgsz=640 device=cpu
#   produces *_integer_quant.tflite (full-int) and *_dynamic_range_quant.tflite
yolo val model=<model.tflite> data=<data.yaml> imgsz=640 device=cpu batch=1   # accuracy
python bench_latency.py <model.tflite> 4 40                                    # latency
```
