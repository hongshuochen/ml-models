# Golf YOLO Detector — Architecture, Benchmarks, Size & Latency

Model card for the golf object detector used by the egocentric (AR-glasses / head-cam) golf
pipeline: offline video annotation + hit counting (`golf/hit_detector.py`) and the Android live
counter (`android/` GolfActivity). All numbers below were **measured on 2026-07-10** on this
repo's hardware unless marked otherwise.

## 1. Task & deployment targets

- **Classes (2):** `ball` (0), `club_head` (1). Boxes only; no shaft class (dropped by design —
  head + ball are sufficient for impact/tempo/putt counting).
- **Input:** RGB, square letterboxed. **1280×1280** offline (small egocentric ball needs the
  resolution), **640×640** on device.
- **Consumers:** per-frame detections → `golf/hit_detector.py` v3 (ego-compensated hit counter,
  14/15 real hits, 0 false fires on 8 real clips) and `golf/annotate_status.py` (annotated video).

## 2. Architecture

| | |
|---|---|
| Family | **YOLO26** (Ultralytics), scale **s** |
| Layers / params | 260 layers, **9.95 M** params (fused for inference: 122 layers, 9.47 M) |
| Compute | **22.5 GFLOPs @640** (20.5 fused) · ~**90 GFLOPs @1280** (scales ×4 with area) |
| Head | anchor-free, **end-to-end NMS-free** — no NMS post-processing step at all |
| Output tensor | `(1, 300, 6)` = 300 detections × `[x1, y1, x2, y2, conf, cls]`, already final |
| Weights file | `runs/detect/golf_ego_v2_1280/weights/best.pt` (PyTorch), `android/app/src/main/assets/golf.tflite` (TFLite **float16**) |

Notes:
- NMS-free output makes the mobile runtime trivial (no NMS kernel / thresholds on device) and the
  output shape static — the Android `FaceHandDetector` consumes it directly.
- float16 is the deployed quantization. int8 was **not** used: on this repo's stack int8 gives no
  x86/WASM speedup and pose/NMS-free heads have int8 corruption issues (see `CLAUDE.md`); f16 is
  ≈lossless and 2× smaller than f32.
- A YOLO26**x** (112.8 MB) was trained once as a public-data teacher; it is **not** part of the
  current pipeline (fine-tuned s models beat it once the data improved — data > model size here).

## 3. Training data lineage & accuracy benchmarks

Two eval domains — do **not** compare numbers across them:
**public** (third-person Roboflow golf imagery, de-duplicated) and **egocentric** (our own
AR-glasses footage, 2048×1536 portrait fisheye; val split is video-disjoint, 290 images).

| Model (run dir) | Train data | Eval set | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|---|
| `golf_detect_x_640` (teacher, historical) | public v1 (leaky) | public, leakage-free test | 0.658 | — | — | — |
| `golf_detect_s_v3_640` | public v3 (12,155 unique scenes, cross-source dedup) | public, strict-dedup novel test | 0.898 (ball 0.918 / club 0.879) | — | — | ball R 0.855 |
| `golf_detect_s_v3_1280` | public v3 @1280 | public v3 val | 0.940 | 0.655 | 0.899 | 0.909 |
| `golf_ego_v1_1280` | + 1,622 hand-labeled egocentric frames | **egocentric val** | 0.87 (club 0.877) | 0.604 | 0.906 | ball R 0.772 |
| **`golf_ego_v2_1280` (DEPLOYED)** | + 4,735 pseudo-labeled egocentric frames (6,357 total ego) | **egocentric val** | **0.881** | **0.623** | **0.933** | **0.824** |

**Deployed model per-class (measured 2026-07-10, `yolo val` @1280 on the egocentric val):**

| Class | mAP50 | Precision | Recall |
|---|---|---|---|
| ball | 0.858 | 0.909 | 0.787 |
| club_head | 0.903 | 0.957 | 0.860 |
| **all** | **0.881** | **0.933** | **0.824** |

Key facts behind the lineage:
- The public-data model is **blind to egocentric club heads** (club mAP50 0.032 on our val) —
  the ego fine-tune's whole payoff is club_head 0.032 → 0.90.
- Pseudo-label expansion (self-labeling 9.8k unlabeled ego frames, keeping non-empty, non-val-video
  frames) **raised precision** across the board (ALL P 0.906 → 0.933) — it did not degrade it.
- Standing weakness: **ball recall 0.79** (small white ball, white-sky/fisheye edge cases). Known
  levers: more hand labels, SAHI tiling, higher res, temporal models (GOLF_PLAN §D).
- Roboflow-style random splits of video-sourced data leak (~48% near-dup test frames); every
  number above marked "dedup"/"video-disjoint" uses pHash dedup or video-level splits instead.

## 4. Model size

| Artifact | Format | Size |
|---|---|---|
| `golf_ego_v2_1280/weights/best.pt` (deployed, offline) | PyTorch f32 | **19.4 MB** |
| `android/.../golf.tflite` (deployed, Android) | TFLite **float16** | **18.9 MB** |
| any YOLO26s golf run | PyTorch f32 | 19.4–19.5 MB |
| `golf_detect_x_640` (historical teacher) | PyTorch f32 | 112.8 MB |

## 5. Latency (measured 2026-07-10)

| Runtime | Hardware | imgsz | Precision | Latency / frame | Throughput |
|---|---|---|---|---|---|
| PyTorch (`model.predict`, end-to-end, batch 1) | RTX 3080 (10 GB) | 1280 | f32 | **10.2 ms** | ~98 fps |
| PyTorch (same) | RTX 3080 | 640 | f32 | **5.7 ms** | ~175 fps |
| PyTorch `yolo val` inference-only, batch 4 | RTX 3080 | 1280 | f32 | 8.2 ms | ~122 fps |
| TFLite + XNNPACK | x86 CPU, **4 threads** | 640 | f16 | **58.1 ms** | ~17 fps |
| TFLite + XNNPACK | x86 CPU, 1 thread | 640 | f16 | 177.1 ms | ~5.6 fps |
| TFLite on phone (ARM / NNAPI / GPU delegate) | — | 640 | f16 | **not yet benchmarked** | x86 4-thread is the current proxy |

Method: `bench_latency.py` (8 warmup + 50 timed invokes, random input) for TFLite; 10 warmup +
60 timed `predict()` calls with CUDA sync for GPU. End-to-end predict includes pre/post-processing.

Offline pipeline context: at 1280 the detector sustains ~100 fps on the 3080, so annotating a
30 fps clip runs ~3× real time; the camera-motion affine estimation used by the hit detector
(`golf/cam_affine.py`, half-res LK optical flow) adds ~25–30 ms/frame on CPU and is the offline
bottleneck, not YOLO. The Android app uses CameraX `KEEP_ONLY_LATEST`, so detection latency
bounds the live loop rate; the on-device number is the next thing to measure.

## 6. Reproduce

- Exact training commands: `TRAINING.md` §11 (public v3) and the egocentric fine-tune section.
- Dataset builders: `golf/build_golf_v3.py` (public), `golf/build_golf_v1_dataset.py` (egocentric),
  `golf/autolabel_firstperson.py` (pseudo-labels).
- Export: `yolo export model=<best.pt> format=tflite half=True imgsz=640`.
- Latency: `uv run python bench_latency.py android/app/src/main/assets/golf.tflite 4 50`.
