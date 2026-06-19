# ml-models

On-device face/hand ML: train YOLO26 models with [uv](https://docs.astral.sh/uv/)
+ Ultralytics, export to TFLite, and run them live in the browser
(see [`webcam-tflite/`](webcam-tflite/)).

## Models

| Model | Task | Classes | Weights |
|-------|------|---------|---------|
| Face detection | detect | face | `runs/detect/widerface_yolo26n/` |
| Hand pose | pose (21 kpts) | hand | `runs/pose/hand_pose_fixed/` |
| Face + Hand detection | detect | face, hand | `runs/detect/face_hand_yolo26n/` |
| Hand landmark regressor | keypoint regression | — | _(in progress)_ |

Full benchmark (mAP / P / R / params / FLOPs / size / latency across
float32·float16·int8) is in **[`MODELS_REPORT.md`](MODELS_REPORT.md)**.

The end goal is a two-stage pipeline (MediaPipe-style): **YOLO detects face/hand
boxes → crop the hand → landmark regressor predicts the 21 keypoints**.

## Datasets (auto/scripted, git-ignored)

- **WIDER FACE** → `widerface.yaml` (face detection)
- **Hand keypoints** → `hand-keypoints-fixed.yaml` (corrected `flip_idx`; the
  stock Ultralytics one is broken and scrambles mirrored-hand keypoints)
- **Face + Hand** → `face-hand.yaml`, built by `build_face_hand.py` (WIDER faces
  class 0 + hand-keypoints boxes class 1, images symlinked)

Prep scripts: `prepare_widerface.py`, `build_face_hand.py`. Benchmark helper:
`bench_latency.py`.

## Setup

```bash
uv sync                       # .venv from pyproject.toml + uv.lock (Python 3.12)
```

## Train (examples)

```bash
# hand pose (corrected flip_idx)
uv run yolo pose train model=yolo26n-pose.pt data=hand-keypoints-fixed.yaml \
  epochs=100 imgsz=640 batch=32 device=0 name=hand_pose_fixed

# face + hand detection
uv run yolo detect train model=yolo26n.pt data=face-hand.yaml \
  epochs=100 imgsz=640 batch=16 device=0 name=face_hand_yolo26n
```

## Export + benchmark

```bash
uv run yolo export model=<best.pt> format=tflite imgsz=640 device=cpu          # f32 + f16
uv run yolo export model=<copy.pt> format=tflite int8=True data=<d.yaml> fraction=0.05 device=cpu
uv run yolo val model=<model.tflite> data=<d.yaml> imgsz=640 device=cpu batch=1 # accuracy
uv run python bench_latency.py <model.tflite> 4 40                              # latency
```

## What's tracked

Source, configs, lockfiles, the web app, and `MODELS_REPORT.md`. Git-ignored:
`datasets/`, `runs/`, `bench/`, `.venv/`, `*.pt`, `*.onnx`, `*.tflite`, `*.npy`,
`*.log` (all regenerable).

## Notes

- Hardware: NVIDIA RTX 3080 (10 GB). Node/git/uv installed under `~/.local`.
- torch installs CUDA wheels by default and works with the system driver.
