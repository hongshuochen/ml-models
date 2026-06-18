# ml-models

YOLO model training workspace (managed with [uv](https://docs.astral.sh/uv/)).

## First model: hand keypoints pose

Trains a YOLO26-nano pose model on the Ultralytics
[Hand Keypoints dataset](https://docs.ultralytics.com/datasets/pose/hand-keypoints/)
(21 keypoints, 1 `hand` class).

### Setup

```bash
uv sync          # create .venv from pyproject.toml + uv.lock (Python 3.12)
```

### Train

```bash
uv run yolo pose train \
  model=yolo26n-pose.pt \
  data=hand-keypoints.yaml \
  epochs=100 imgsz=640 batch=32 device=0 \
  name=hand_pose_yolo26n patience=30
```

The dataset (~369 MB) is auto-downloaded on first run to `datasets/`.
Results, plots, and checkpoints land in `runs/pose/<name>/`
(`weights/best.pt`, `weights/last.pt`).

### Notes

- Hardware: NVIDIA RTX 3080 (10 GB). `batch=32` uses ~6 GB at `imgsz=640`.
- torch installs CUDA 13 wheels by default and works with the system driver — no special index needed.
- ~2% of dataset labels have slightly out-of-bounds coordinates that Ultralytics
  automatically skips; the warnings during scanning are expected.

## What's tracked

Source + lockfiles only. `datasets/`, `runs/`, `.venv/`, `*.pt`, and `*.log`
are git-ignored (see `.gitignore`).
