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
- `synth_qr_barcode.py` — synth QR/barcode detection data (perspective/blur/small + adjacent pairs;
  `--real-patches` pastes real code crops). `prepare_real_codes.py` — convert real QR/barcode
  datasets → our 4-class YOLO layout (qr=2, barcode=3). Configs: `face-hand-qr-bar*.yaml`.
- `android/` — native face/hand app (now 4-class detector + model selector + ML Kit code decode).
  `android-caption/` — separate on-device captioning app (Florence-2-base-ft via ONNX Runtime).
- `webcam-tflite/` — in-browser app. `runs/` + `datasets/` + `models/` are **git-ignored**.

## Gotchas (hard-won — read before changing related code)
- **Synthetic-only code detectors collapse on REAL photos.** The synth-only 4-class
  face/hand/qr/barcode pico scored 0.95 on synthetic but real qr 0.17 / barcode 0.15. Always eval on
  a real held-out set; fixed with real data + a hardened synth (MODELS_REPORT §7.6). **2D codes
  (DataMatrix/Aztec/PDF417) are NOT trained** — only QR + 1D. Vertical barcodes are fine (real data
  is ~41% vertical). **Small/dense QR is missed at the deployed 640** when <~25–30% of frame (dense
  modules blur after downscale; barcodes survive) — detects fine close-up / at imgsz 1280.
- **Florence-2 ONNX caption (android-caption/): the KV-cache export is broken** — `decoder_with_past_model`
  has a static-16 `inputs_embeds`. Decode WITHOUT a cache (run `decoder_model` on the growing
  sequence; vision encoder runs once). Tokenizer is DIY (hardcoded prompt ids + byte-level decode
  from `vocab.json`) — DJL's native tokenizer has no Android ABIs. Spec verified in
  `android-caption/florence_onnx_demo.py`.
- **YOLO26 is NMS-free / end-to-end.** Detect output `(1,300,6)` = `[x1,y1,x2,y2,conf,cls]`; pose `(1,300,69)`.
- **The golf detector's phone build needs the RAW head, not e2e.** YOLO26's e2e head bakes
  TopK/GatherNd/INT64 that NO mobile delegate (GPU or NPU) runs → CPU fallback. Deploy path =
  `golf/export_golf_rawhead_tflite.py` (monkeypatches `Detect.forward`→raw one2one maps +
  `Attention.forward`→unrolled matmuls, then `onnx2tf enable_batchmatmul_unfold=False` — else C2PSA
  attention explodes into ~1600 `FULLY_CONNECTED` and the delegate rejects it). Emits 3 raw NHWC maps
  `[1,G,G,4+nc]` (per cell `[l,t,r,b, cls logits]`, reg_max=1, no DFL); Kotlin (`GolfDetector.kt`) does
  sigmoid+argmax+per-class NMS and reads `nc` from the shape (2- or 3-class generic — only `LABELS`
  changes). Parity-verified vs the e2e `.pt` on ball/club/hole. See [[android-golf-npu-deploy]].
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
- **Landmark head-256 beats head-1024.** MobileNetV3's head conv (→1024, sized for 1000-class ImageNet)
  dominates params at low width; `train_hand_landmark.py --head-dim 256` cuts the int8 model ~38% with
  no accuracy loss (slightly better — mild regularization). Deployed hand/face landmark use head 256.
- **Back-of-hand is OOD for HaGRID-trained landmarks** (HaGRID gestures are all palm-toward-camera) →
  dorsal PCK only ~0.44, bad for an egocentric/glasses camera (sees hand *backs*). Fix: mix in FreiHAND
  (`prepare_freihand.py`, real 21-joint labels incl. dorsal) → dorsal 0.82, palm unchanged. The
  L-gesture MLP needs no retrain (pairwise-distance features are mirror-invariant; palm-L ≈ dorsal-L).
- **Face landmarks need a flip_idx swap** on horizontal-flip aug (`--flip-idx 1,0,2,4,3`: swap
  L/R eye & mouth corners); hands stay identity. The same trainer does both via `--num-kpts`.
- **TFLite int8 export gotchas**: export int8 to a SEPARATE dir (it clobbers f32/f16), use a small
  `fraction` for calibration (full val → huge calib array / OOM/hang). Landmark export needs
  `onnxsim` before `onnx2tf` (else onnx2tf errors).
- **int8 dyn-range FC is op v12 → needs `tensorflow-lite` ≥ 2.17.0.** int8 dynamic-range emits a
  *hybrid* `FULLY_CONNECTED` at **op version 12**; the old `tensorflow-lite:2.16.1` only knew FC up
  to v11 → `Didn't find op for builtin code FULLY_CONNECTED version 12` at model load. The Android
  app pins **2.17.0** (the last `org.tensorflow:tensorflow-lite` on Maven — newer is the LiteRT
  rebrand `com.google.ai.edge.litert`), which loads v12, so the landmark + gesture models ship as
  int8 dyn-range (≈lossless: landmark ~0.5px, gesture 300/300 same decision). Conv-only models
  (the YOLO detector) have no FC and load on any runtime. float16 (FC v1) is the fallback only if
  you must stay on an older runtime.

## Conventions
- Keep a config + an exact reproduce command for **every** model (configs committed; commands in TRAINING.md).
- New models → add to MODELS_REPORT.md (metrics) and README.md (catalog). Separate `runs/` dir per model (don't overwrite).
- Commit code/configs/docs; never commit `runs/`, `datasets/`, `*.tflite/*.pt/*.onnx`, or logs (all git-ignored).

## Agent workflow notes
- Kill background jobs by **numeric PID** (`ss -ltnp | grep :PORT`, then `kill <pid>`); a `pkill -f "<pattern>"`
  can match the agent's own shell when the command text contains the pattern.
- Long jobs: launch with the background runner so completion notifies; bare `&` detaches without notifying.
- GitHub auth here is via SSH (`git@github.com:hongshuochen/ml-models.git`); the push key lives in `~/.ssh/id_ed25519`.
