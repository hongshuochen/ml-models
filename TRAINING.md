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
# 6c. MobileNetV2_035 from scratch (no ImageNet weights) — smallest/fastest
uv run python train_hand_landmark.py --backbone mobilenetv2_035 --no-pretrained \
  --epochs 100 --batch 64 --device cuda --workers 8 --out runs/landmark/hand_landmark_mnv2_035_scratch
# 6d. MobileNetV3-small_050 from scratch (control: pretrain vs scratch) — best PCK at 0.61M
uv run python train_hand_landmark.py --backbone mobilenetv3_small_050 --no-pretrained \
  --epochs 100 --batch 64 --device cuda --workers 8 --out runs/landmark/hand_landmark_mnv3s050_scratch
```
→ `runs/landmark/<out>/best.pt` (PCK@0.1, see MODELS_REPORT §4): 6a 1.56M 0.971 ·
6b 0.61M 0.949 · 6c 0.45M 0.951 (scratch) · 6d 0.61M 0.959 (scratch, best compact).
**Pretraining is not needed here** — scratch matches/beats pretrained at this data size.

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

## 7. HaGRID augmentation — fix webcam-domain hands
Baselines miss webcam-framed hands (hand AP@50 ~0.01-0.03 on a HaGRID val) because they
learned hands from close-up hand-keypoints crops. Fine-tune on HaGRIDv2 to fix it.
```bash
# Data (official HaGRIDv2, sbercloud): annotations (719MB) + 512px images (119GB monolith)
#   datasets/hagrid_raw/annotations/{train,val,test}/<gesture>.json  (UUID-keyed hand bboxes)
#   datasets/hagrid_raw/HaGRIDv2_dataset_512/<gesture>/<uuid>.jpg     (34 gestures, ~1.08M imgs)
# Pseudo-label: HaGRID hand boxes (class 1) + InsightFace SCRFD-10g faces (class 0).
uv add insightface                                  # + onnxruntime (CPU ok)
ROOT=datasets/hagrid_raw/HaGRIDv2_dataset_512; ANN=datasets/hagrid_raw/annotations
uv run python prepare_hagrid.py --ann-dir $ANN/train --img-root $ROOT \
  --per-gesture-limit 2000 --shuffle --seed 0 --target-split train --out datasets/hagrid_det_v2
uv run python prepare_hagrid.py --ann-dir $ANN/val --img-root $ROOT \
  --per-gesture-limit 300  --shuffle --seed 0 --target-split val   --out datasets/hagrid_det_v2
# Fine-tune (warm start from baselines) on face-hand-hagrid-v2.yaml (val = original face-hand val)
uv run yolo detect train model=runs/detect/face_hand_yolo26n/weights/best.pt \
  data=face-hand-hagrid-v2.yaml epochs=40 imgsz=640 batch=16 device=0 lr0=0.005 patience=12 \
  name=face_hand_hagrid_v2_ft            # nano  -> keep best.pt (balanced; webcam fixed by ep1)
uv run yolo detect train model=runs/detect/face_hand_pico_p45/weights/best.pt \
  data=face-hand-hagrid-v2.yaml epochs=40 imgsz=640 batch=16 device=0 lr0=0.005 patience=12 \
  name=face_hand_pico_p45_hagrid_ft      # pico  -> keep last.pt (needs more epochs than best.pt=ep1)
# Eval BOTH scorecards (subject-disjoint webcam val is clean):
uv run yolo val model=<ckpt> data=face-hand.yaml imgsz=640 device=0   # original task
uv run yolo val model=<ckpt> data=hagrid-val.yaml imgsz=640 device=0  # webcam domain
```
→ webcam hand AP@50 0.01-0.03 → ~0.99 (see MODELS_REPORT §7). Deploy pico `last.pt`
int8 dyn-range 0.807 MB.

## 7.5. QR + barcode — one 4-class detector (face / hand / qr / barcode)
```bash
# Synthesize codes onto no-face COCO backgrounds (needs datasets/face_cls_cache.json from
# prepare_coco_face_cls.py) -> datasets/qrbar (15k train / 1.5k val, qr=2 barcode=3)
uv run python synth_qr_barcode.py --n-train 15000 --n-val 1500
# Warm-start from the 2-class pico (head auto re-inits for nc=4, backbone kept)
uv run yolo detect train model=runs/detect/face_hand_pico_p45_hagrid_ft/weights/pico_hagrid.pt \
  data=face-hand-qr-bar.yaml epochs=30 imgsz=640 batch=32 device=0 lr0=0.005 patience=15 \
  name=face_hand_qr_bar_pico             # -> runs/detect/face_hand_qr_bar_pico, keep best.pt=pico_qrbar.pt
uv run yolo detect val model=<best.pt> data=face-hand-qr-bar.yaml imgsz=640 device=0  # per-class AP
# Export (same as §5/§9): f32+f16 saved_model, then dyn-range int8 via TF Optimize.DEFAULT
uv run yolo export model=runs/detect/face_hand_qr_bar_pico/weights/pico_qrbar.pt format=tflite imgsz=640 device=cpu
```
→ face 0.451 / hand 0.989 / qr 0.984 / barcode 0.973 mAP@50 (MODELS_REPORT §7.5); no face/hand
regression. Deploy int8 dyn-range 0.78 MB.

## 8. Face landmark — 5 ArcFace points (for face alignment)
InsightFace gives the 5 points free per detected face; distill them into a tiny regressor.
```bash
# Build dataset: largest face per HaGRID image + 5 kps (subject-disjoint splits)
uv run python prepare_hagrid_face.py --device cpu --shuffle --seed 0   # -> datasets/hagrid-face
# Train (note the face flip-idx: swap L/R eye & mouth on hflip)
uv run python train_hand_landmark.py --backbone mobilenetv3_small_025 --no-pretrained \
  --num-kpts 5 --flip-idx 1,0,2,4,3 --data datasets/hagrid-face \
  --epochs 50 --batch 64 --device cuda --out runs/landmark/face_landmark_mnv3s025
# Export (5-pt)
uv run python export_landmark.py --backbone mobilenetv3_small_025 --num-kpts 5 \
  --ckpt runs/landmark/face_landmark_mnv3s025/best.pt --onnx runs/landmark/face_landmark_mnv3s025/face.onnx
```
→ PCK@0.1 0.998 vs InsightFace · 0.26M params · f16 0.56MB / int8 0.38MB.

## 9. Landmark head-256 trim (deployed) — smaller, no accuracy loss
MobileNetV3's head conv (144→1024) is overkill for keypoint regression; `--head-dim 256`
shrinks it with no accuracy cost (slightly better, see MODELS_REPORT §4c/§4d). The deployed
hand + face landmark `.tflite` use head 256.
```bash
# hand (21-pt): same recipe as §7 + --head-dim 256
uv run python train_hand_landmark.py --backbone mobilenetv3_small_025 --no-pretrained \
  --num-kpts 21 --head-dim 256 --data datasets/hand-keypoints datasets/hagrid-landmark \
  --epochs 40 --batch 64 --device cuda --out runs/landmark/hand_landmark_mnv3s025_h256
# face (5-pt): same recipe as §8 + --head-dim 256
uv run python train_hand_landmark.py --backbone mobilenetv3_small_025 --no-pretrained \
  --num-kpts 5 --flip-idx 1,0,2,4,3 --head-dim 256 --data datasets/hagrid-face \
  --epochs 50 --batch 64 --device cuda --out runs/landmark/face_landmark_mnv3s025_h256
# export each: export_landmark.py --head-dim 256 --num-kpts {21|5} --ckpt <best.pt> -> onnxsim -> onnx2tf
#   then dynamic-range int8 (Optimize.DEFAULT). Deployed: hand 0.25MB, face 0.24MB int8.
```
→ hand webcam PCK 0.899→0.905, int8 0.41→0.25MB · face PCK 0.998→0.999, int8 0.38→0.24MB.

## 10. Back-of-hand (egocentric) — add FreiHAND to the hand landmark
HaGRID is palm-toward-camera, so dorsal (back-of-hand) is OOD — bad for a head/glasses camera.
Add FreiHAND (real 21-joint labels incl. dorsal). See MODELS_REPORT §4c-bis.
```bash
# data: FreiHAND_pub_v2.zip (3.9GB) -> datasets/freihand_raw/ ; convert (project 3D->2D)
curl -L -o datasets/freihand_raw/FreiHAND_pub_v2.zip \
  https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
cd datasets/freihand_raw && unzip -q FreiHAND_pub_v2.zip && cd -
uv run python prepare_freihand.py --root datasets/freihand_raw --versions 2   # -> datasets/freihand (~63k)
# retrain hand landmark with FreiHAND mixed in (everything else = §9)
uv run python train_hand_landmark.py --backbone mobilenetv3_small_025 --no-pretrained \
  --num-kpts 21 --head-dim 256 --data datasets/hand-keypoints datasets/hagrid-landmark datasets/freihand \
  --epochs 40 --batch 64 --device cuda --out runs/landmark/hand_landmark_mnv3s025_h256_frei
# export (same as §9) -> int8 -> assets/hand_landmark.tflite
```
→ dorsal PCK **0.44 → 0.82**; palm unchanged (webcam 0.905→0.898, hand-keypoints 0.923→0.920).
The L-gesture MLP is not retrained (mirror-invariant features).
