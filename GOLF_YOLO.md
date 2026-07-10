# Golf YOLO Detector — Model Card

The object detector behind the egocentric (AR-glasses / head-cam) golf pipeline: per-frame
`ball` / `club_head` boxes, consumed by the hit counter (`golf/hit_detector.py`) and the Android
live counter. Numbers measured 2026-07-10.

## Architecture

| | |
|---|---|
| Model | **YOLO26s**, fine-tuned on our egocentric footage (`golf_ego_v2_1280`) |
| Classes | 2 — `ball` (0), `club_head` (1) |
| Parameters / compute | **9.95 M** params · 22.5 GFLOPs @640 (~90 GFLOPs @1280) |
| Head | anchor-free, **end-to-end NMS-free** — no NMS post-processing on device |
| Output | `(1, 300, 6)` = 300 detections × `[x1, y1, x2, y2, conf, cls]`, already final |
| Input | RGB square letterbox — **1280×1280** offline, **640×640** on device |
| Weights | `runs/detect/golf_ego_v2_1280/weights/best.pt` → exported TFLite **float16** (`android/app/src/main/assets/golf.tflite`) |

## Benchmark

Egocentric validation set: **290 images / 448 boxes**, real AR-glasses footage (2048×1536
portrait, fisheye), split **video-disjoint** from training (no frame leakage).
`yolo val` @1280:

| Class | mAP50 | Precision | Recall |
|---|---|---|---|
| ball | 0.858 | 0.909 | 0.787 |
| club_head | 0.903 | 0.957 | 0.860 |
| **all** | **0.881** | **0.933** | **0.824** |

Known weakness: ball recall (small white ball at distance); club_head is strong.

## Model size

| Format | Size |
|---|---|
| TFLite float16 (deployed on Android) | **18.9 MB** |
| PyTorch f32 (`best.pt`) | 19.4 MB |

## Latency

Closest-to-phone setting we can measure today — the deployed TFLite f16 model at the deployed
input size, CPU only:

| Runtime | Setting | Latency / frame | Throughput |
|---|---|---|---|
| TFLite + XNNPACK, CPU, 4 threads | 640×640, f16 | **58 ms** | **~17 fps** |

Method: `bench_latency.py` — 8 warmup + 50 timed invokes. On-device (ARM / GPU delegate) is not
yet benchmarked; this CPU number is the working proxy, and the app's CameraX `KEEP_ONLY_LATEST`
loop is bounded by it.

## Reproduce

- Training command: `TRAINING.md` (egocentric fine-tune section).
- Export: `yolo export model=runs/detect/golf_ego_v2_1280/weights/best.pt format=tflite half=True imgsz=640`
- Latency: `uv run python bench_latency.py android/app/src/main/assets/golf.tflite 4 50`
