# Training recipes — reproduce every model

Each model below lists its **exact command**, config files, output dir, and result.
Scripts/configs are versioned; trained weights live under `runs/` (git-ignored —
regenerate with these commands). Ultralytics also writes the full hyperparameter
set to `runs/<task>/<name>/args.yaml` for each run.

Setup: `uv sync` (Python 3.12 env). GPU: RTX 3080. All YOLO models are YOLO26.

## Datasets (build first)
```bash
# WIDER FACE -> YOLO detection (face)
uv run python prepare_widerface.py            # -> datasets/widerface, widerface.yaml
# Hand keypoints: dataset auto-downloads on first pose train; use the CORRECTED yaml
#   hand-keypoints-fixed.yaml (identity flip_idx; stock Ultralytics flip_idx is broken)
# Face+Hand 2-class detection (face=0, hand=1; images symlinked)
uv run python build_face_hand.py              # -> datasets/face-hand, face-hand.yaml
```

## 1. Face detection — YOLO26n (WIDER FACE)
```bash
uv run yolo detect train model=yolo26n.pt data=widerface.yaml \
  epochs=100 imgsz=640 batch=16 device=0 name=widerface_yolo26n patience=30
```
→ `runs/detect/widerface_yolo26n/` · mAP@50 0.682

## 2. Hand pose — YOLO26n-pose, 21 kpts (corrected flip_idx)
```bash
uv run yolo pose train model=yolo26n-pose.pt data=hand-keypoints-fixed.yaml \
  epochs=100 imgsz=640 batch=32 device=0 name=hand_pose_fixed patience=30 fliplr=0.5
```
→ `runs/pose/hand_pose_fixed/` · pose mAP@50 0.927 (fixes left/mirrored-hand bug)

## 3. Face + Hand detection — YOLO26n, 2-class
```bash
uv run yolo detect train model=yolo26n.pt data=face-hand.yaml \
  epochs=100 imgsz=640 batch=16 device=0 name=face_hand_yolo26n patience=30
```
→ `runs/detect/face_hand_yolo26n/` · mAP@50 0.831 (face 0.670 / hand 0.992)

## 4. Face + Hand "pico" — sub-1MB int8 (custom scale [0.5,0.125,1024])
```bash
uv run yolo detect train model=yolo26n-pico.yaml data=face-hand.yaml \
  epochs=100 imgsz=640 batch=16 device=0 name=face_hand_pico patience=30
```
→ `runs/detect/face_hand_pico/` · 0.68M params · int8 0.93MB

## 5. Face + Hand "pico-P4P5" — drop-P3, medium/large only
```bash
uv run yolo detect train model=yolo26n-pico-p45.yaml data=face-hand.yaml \
  epochs=100 imgsz=640 batch=16 device=0 name=face_hand_pico_p45 patience=30
```
→ `runs/detect/face_hand_pico_p45/` · 0.64M params · dyn-range int8 0.808MB (hand AP 0.989)

## 6. Hand landmark regressor — stage 2 (one script, `--backbone`/`--out` per model)
```bash
# 6a. MobileNetV3-small (torchvision, ImageNet-pretrained)
uv run python train_hand_landmark.py --epochs 60 --batch 64 --device cuda --workers 8 \
  --out runs/landmark/hand_landmark
# 6b. MobileNetV3-small_050 (timm, ImageNet-pretrained) — ~0.6M, smaller
uv run python train_hand_landmark.py --backbone mobilenetv3_small_050 \
  --epochs 60 --batch 64 --device cuda --workers 8 --out runs/landmark/hand_landmark_mnv3s050
```
→ `runs/landmark/<out>/best.pt` · 6a: 1.56M, PCK@0.1 0.971 · 6b: 0.61M (see MODELS_REPORT)

## Export to TFLite (per model)
```bash
# YOLO: float32 + float16
uv run yolo export model=<best.pt> format=tflite imgsz=640 device=cpu
# YOLO int8 (separate dir + limited calibration to avoid a huge calib array / hang)
uv run yolo export model=<copy.pt> format=tflite int8=True data=<data.yaml> fraction=0.05 device=cpu
#   -> *_integer_quant.tflite (full-int8). For dynamic-range int8:
uv run python -c "import tensorflow as tf; c=tf.lite.TFLiteConverter.from_saved_model('<saved_model>'); \
  c.optimizations=[tf.lite.Optimize.DEFAULT]; open('dynrange.tflite','wb').write(c.convert())"
# Landmark: torch -> ONNX (dynamo=False) -> onnxsim -> onnx2tf  (see export_landmark.py)
uv run python export_landmark.py
```

## Benchmark
```bash
uv run yolo val model=<model.tflite> data=<data.yaml> imgsz=640 device=cpu batch=1   # mAP
uv run python bench_latency.py <model.tflite> 4 40                                    # latency (4-thread CPU)
```
