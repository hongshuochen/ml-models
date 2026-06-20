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
| **MobileNetV2_035 (timm)** | **0.45 M** | 116 | scratch (100 ep) | **0.92 MB** | 0.951 | 0.0298 | **0.9 ms** |

(f32 files ≈ 2× the f16 size.)

**Findings:**
- **Pretraining is not needed for this task.** At the *same* 60-epoch budget, scratch
  MNv3-small_050 (PCK 0.952) matched ImageNet-pretrained (0.949); given 100 epochs it
  reached 0.959. The hand-crop set is large enough, and keypoint regression is far
  enough from ImageNet classification, that pretrained features give no final-accuracy
  edge here — only slightly faster early convergence.
- **FLOPs don't predict latency at this scale.** MNv2_035 has 2.5× the FLOPs of
  MNv3-small_050 (116 vs 46) yet runs *faster* (0.9 vs 1.1 ms): its plain
  depthwise-separable convs are more XNNPACK-friendly than MNv3's SE + hard-swish.
- **Best compact pick: MNv2_035 from scratch** — smallest (0.45 M / 0.92 MB f16),
  fastest (0.9 ms), PCK 0.951 (≈ pretrained _050). For max accuracy at 0.61 M, use
  MNv3-small_050 from scratch (0.959). (MobileNetV4's smallest is *larger*, 2.55 M —
  wrong direction for shrinking.)

Weights: `runs/landmark/hand_landmark/` (1.56 M), `_mnv3s050/` (pretrained),
`_mnv3s050_scratch/`, `_mnv2_035_scratch/`.

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
