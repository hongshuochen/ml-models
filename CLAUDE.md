# CLAUDE.md

Guidance for working in this repo with Claude Code. See **README.md** (overview),
**MODELS_REPORT.md** (full metrics), **TRAINING.md** (exact per-model commands).

## What this is
On-device face/hand ML: **YOLO26** detectors + a **MobileNet** 21-keypoint hand-landmark
regressor, trained with **uv + Ultralytics**, exported to **TFLite**, run live in the
browser (`webcam-tflite/`). Pipeline: detect hand box → crop → regress 21 keypoints.

## Environment & commands
- Python via **uv** (Python 3.12): `uv sync`, then `uv run yolo ...` / `uv run python ...`.
- GPU training on an RTX 3080. Web app: Node, `cd webcam-tflite && npm install && npm run dev`.
- Reproduce any model from **TRAINING.md** (one exact command per model).

## Repo map
- `*.yaml` — dataset/model-scale configs (one per model; `*-hagrid*` = HaGRID-augmented).
- `prepare_*.py`, `build_face_hand.py` — dataset builders (WIDER, face-hand, HaGRID detect/landmark).
- `train_hand_landmark.py` — landmark regressor trainer (multi-backbone, multi-data, `--eval-only`).
- `export_landmark.py` — landmark torch→ONNX→TFLite. `bench_latency.py` — TFLite CPU latency.
- `webcam-tflite/` — in-browser app. `runs/` (weights) + `datasets/` are **git-ignored**.

## Gotchas (hard-won — read before changing related code)
- **YOLO26 is NMS-free / end-to-end.** Detect output `(1,300,6)` = `[x1,y1,x2,y2,conf,cls]`; pose `(1,300,69)`.
- **hand-keypoints `flip_idx` is broken upstream** → always use `hand-keypoints-fixed.yaml` (identity flip_idx); the stock one scrambles mirrored-hand keypoints.
- **Web app must use float16/float32 — NOT int8.** tfjs-tflite's WASM runtime can't initialize
  dynamic-range int8 (hybrid) ops ("Can't initialize model"). int8 is fine for native/ARM only.
- **Spawn a fresh worker per model load** (`useInference.ts`). tfjs-tflite hangs when a 2nd model
  is loaded into an already-used worker → stuck "warming up". Don't reuse the worker across switches.
- **Landmark TFLite layout varies**: onnx2tf may emit NHWC `[1,224,224,3]` or NCHW `[1,3,224,224]`.
  The worker auto-detects from the input shape; don't hard-code a transpose.
- **Pose model int8 is not viable** (corrupts the NMS-free class output → `yolo val` crashes). Use f16.
- **int8 gives no x86/WASM CPU speedup** — its only benefit is file size.
- **Landmark backbones don't need ImageNet pretraining** (from-scratch matches). Custom MobileNetV3-small
  widths (`_035`, `_025`) aren't in timm — build via `timm.models.mobilenetv3._gen_mobilenet_v3`.
- **TFLite int8 export gotchas**: export int8 to a SEPARATE dir (it clobbers f32/f16), use a small
  `fraction` for calibration (full val → huge calib array / OOM/hang). Landmark export needs
  `onnxsim` before `onnx2tf` (else onnx2tf errors).

## Conventions
- Keep a config + an exact reproduce command for **every** model (configs committed; commands in TRAINING.md).
- New models → add to MODELS_REPORT.md (metrics) and README.md (catalog). Separate `runs/` dir per model (don't overwrite).
- Commit code/configs/docs; never commit `runs/`, `datasets/`, `*.tflite/*.pt/*.onnx`, or logs (all git-ignored).

## Agent workflow notes
- Kill background jobs by **numeric PID** (`ss -ltnp | grep :PORT`, then `kill <pid>`); a `pkill -f "<pattern>"`
  can match the agent's own shell when the command text contains the pattern.
- Long jobs: launch with the background runner so completion notifies; bare `&` detaches without notifying.
- GitHub auth here is via SSH (`git@github.com:hongshuochen/ml-models.git`); the push key lives in `~/.ssh/id_ed25519`.
