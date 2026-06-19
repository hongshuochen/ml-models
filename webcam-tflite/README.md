# Webcam · YOLO TFLite

Real-time, **on-device** ML inference on your webcam feed using your own
**YOLO26** models exported to **TensorFlow Lite**. Built with Next.js (App
Router) + React + TypeScript. Every frame is processed locally in a Web Worker —
nothing ever leaves the browser.

## Bundled models

Both are YOLO26-nano models trained locally and exported to TFLite with NMS baked
in (`yolo export format=tflite nms=True imgsz=640`):

| Model | Type | Input | Output `[1,300,N]` | Overlay |
| --- | --- | --- | --- | --- |
| Face Detection (WIDER FACE) | detection | 640×640 f32 | N=6: `x1,y1,x2,y2,conf,cls` | bounding boxes |
| Hand Keypoints (21 pts) | pose | 640×640 f32 | N=69: box + conf/cls + 21×`(x,y,score)` | box + hand skeleton |

Coordinates are normalized to `[0,1]`; input is float32 RGB ÷ 255.

## Features

- 📷 Requests camera permission and shows the live video feed.
- 🧠 **Model selector** for the bundled `.tflite` models.
- ⚡ Inference on video frames with a **configurable interval** (0–1000 ms) and
  one-frame-in-flight backpressure.
- 🖼️ Boxes for detection; box + 21-point skeleton for hand pose; live latency/fps.
- 🧵 Inference runs in a **Web Worker** (off the main thread), with
  **multi-threaded WASM** when the page is cross-origin-isolated (see below).
- 🛡️ Graceful handling of unsupported browsers, denied/missing/disconnected
  camera, missing model files, and inference failures.

## Requirements

- **Node.js ≥ 18** (developed on Node 24).
- A modern browser with `getUserMedia`, Web Workers, and WebAssembly.
- A **secure context** for the camera: `http://localhost` (dev) or `https://`.

## Setup & run

```bash
npm install      # installs deps AND copies the TFLite runtime + models into /public
npm run dev      # http://localhost:3000
```

`npm install`'s post-install step (`scripts/setup-assets.mjs`):
1. copies the tfjs UMD bundles + TFLite WASM runtime from `node_modules` into
   `public/vendor/tflite/`, and
2. copies the exported `.tflite` models into `public/models/`.

The model files are already present in `public/models/`. To regenerate them from
the trained weights:

```bash
# from the training project root (one level up)
yolo export model=runs/detect/widerface_yolo26n/weights/best.pt format=tflite nms=True imgsz=640
yolo export model=runs/pose/hand_pose_yolo26n/weights/best.pt   format=tflite nms=True imgsz=640
npm run setup:assets   # copies the fresh exports into public/models
```

Production build:

```bash
npm run build && npm start    # http://localhost:3000
```

## Performance (multi-threading)

`next.config.mjs` sets `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. These make the page
**cross-origin-isolated**, which enables `SharedArrayBuffer` and lets the TFLite
WASM runtime run XNNPACK on multiple threads. Measured on a desktop (headless
Chrome), YOLO26n @ 640:

| | single-thread | threaded |
| --- | --- | --- |
| Face detect | ~283 ms | **~112 ms** |
| Hand pose | ~557 ms | **~131 ms** |

All app assets are same-origin, so `require-corp` is safe. If you add a
cross-origin resource later, it must send the appropriate CORP/CORS headers, or
switch COEP to `credentialless`.

## How it works

```
┌─ main thread (React) ────────────┐        ┌─ Web Worker (classic) ──────────┐
│ getUserMedia → <video>           │        │ importScripts(): tf-core,       │
│ rAF loop (throttled by interval) │  load  │   tf-backend-cpu, tf-tflite     │
│   draw frame → 640² canvas       │ ─────▶ │ tflite.loadTFLiteModel(         │
│   getImageData → RGBA buffer     │ infer  │   {numThreads})                 │
│   postMessage(buffer, transfer)  │ ─────▶ │ predict(float32 NHWC 0..1)      │
│ draw boxes/skeleton ◀────────────│ result │ decode [1,300,N] (NMS baked in) │
└──────────────────────────────────┘        └──────────────────────────────────┘
```

- **TensorFlow is never bundled** — the UMD bundles + WASM are copied to
  `public/vendor/` and loaded via `importScripts()`. `@tensorflow/*` are
  devDependencies used only at setup.
- Frames are resized on the main thread (cheap `drawImage`) and only a small
  RGBA buffer is *transferred* (zero-copy) to the worker.
- Results are tagged with a model id so a result from a previous selection is
  dropped after a model switch.

## Add your own model

1. Export a YOLO model: `yolo export model=your.pt format=tflite nms=True imgsz=640`
   (detection or pose).
2. Copy `best_float32.tflite` into `public/models/your_model.tflite`.
3. Add an entry to `src/lib/models.ts` with `type: 'detection' | 'pose'`,
   `inputWidth/Height: 640`, and `classNames`.

For an entirely different output format, add a decoder branch in
`public/workers/inference.worker.js`.

## Project structure

```
public/workers/inference.worker.js   # YOLO TFLite inference worker (the engine)
public/vendor/tflite/                # tfjs UMD + WASM runtime (generated at setup)
public/models/                       # bundled YOLO .tflite models
scripts/setup-assets.mjs             # copies runtime + models into /public
src/app/                             # Next.js page, layout, styles
src/components/                      # CameraStage, ModelSelector, Controls, ResultsPanel, ErrorBanner
src/hooks/useCamera.ts               # getUserMedia lifecycle + friendly errors
src/hooks/useInference.ts            # worker + capture loop + backpressure + stats
src/lib/models.ts                    # the bundled-model manifest
src/lib/types.ts                     # shared types + worker protocol
test/headless-infer.mjs              # optional end-to-end test (needs puppeteer-core)
```

## Optional: end-to-end test

Verifies the worker loads the runtime and decodes both models (feeds a face image
and a hand image, asserts faces + a 21-keypoint hand). Needs a local Chrome:

```bash
npm install --no-save puppeteer-core
npm run build && npm start &
cp <face>.jpg public/_test_face.jpg && cp <hand>.jpg public/_test_hand.jpg
ORIGIN=http://localhost:3000 CHROME_PATH=/path/to/chrome node test/headless-infer.mjs
rm public/_test_face.jpg public/_test_hand.jpg
```

## Privacy

All inference is on-device. The camera stream and every frame stay in your
browser; nothing is uploaded.
