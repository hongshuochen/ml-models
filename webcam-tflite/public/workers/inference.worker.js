/* eslint-disable no-undef */
/**
 * Classic Web Worker that runs Ultralytics YOLO26 TFLite models off the main
 * thread. Loads the tfjs UMD bundles + the TFLite WASM runtime from /vendor
 * (copied from node_modules by scripts/setup-assets.mjs) via importScripts().
 *
 * Models are exported with NMS baked in, so output is a fixed [1, 300, N] tensor
 * with normalized coordinates:
 *   detection (N=6):  [x1, y1, x2, y2, conf, cls]
 *   pose      (N=69): [x1, y1, x2, y2, conf, cls, 21 x (kx, ky, kconf)]
 * Input is float32 NHWC [1, H, W, 3] with values in [0, 1] (RGB / 255).
 *
 * Protocol (see src/lib/types.ts):
 *   main -> worker: { type:'load', model }
 *                   { type:'infer', requestId, modelId, width, height, buffer }  (RGBA, transferred)
 *   worker -> main: { type:'ready' | 'loadError' | 'result' | 'inferError', ... }
 */

const VENDOR = '/vendor/tflite';

let scriptsLoaded = false;
let model = null;
let landmarkModel = null; // two-stage: regressor run on each hand crop
let landmarkNCHW = true; // landmark input layout: channels-first [1,3,H,W] vs -last [1,H,W,3]
let modelDef = null;

function loadScriptsOnce() {
  if (scriptsLoaded) return;
  importScripts(
    `${VENDOR}/tf-core.min.js`,
    `${VENDOR}/tf-backend-cpu.min.js`,
    `${VENDOR}/tf-tflite.min.js`,
  );
  scriptsLoaded = true;
}

async function handleLoad(def) {
  loadScriptsOnce();
  await tf.setBackend('cpu');
  await tf.ready();
  tflite.setWasmPath(`${VENDOR}/wasm/`);
  // Use threaded WASM when the page is cross-origin-isolated (COOP/COEP set),
  // which makes these 640px YOLO models several times faster. Falls back to 1.
  const isolated = typeof self.crossOriginIsolated !== 'undefined' && self.crossOriginIsolated;
  const numThreads = isolated ? Math.min(self.navigator?.hardwareConcurrency || 4, 4) : 1;
  model = null;
  landmarkModel = null;
  modelDef = def;
  model = await tflite.loadTFLiteModel(def.modelUrl, { numThreads });
  if (def.landmarkUrl) {
    landmarkModel = await tflite.loadTFLiteModel(def.landmarkUrl, { numThreads });
    // Detect input layout from the model (fallback to config): onnx2tf may emit
    // channels-first [1,3,H,W] or channels-last [1,H,W,3] depending on the export.
    const lshape = landmarkModel.inputs?.[0]?.shape;
    landmarkNCHW =
      Array.isArray(lshape) && lshape.length === 4 ? lshape[1] === 3 : def.landmarkLayout !== 'nhwc';
  }
  postMessage({ type: 'ready', modelId: def.id });
}

/** RGBA Uint8 buffer -> float32 NHWC tensor with values 0..1 (YOLO input). */
function toInputTensor(rgba, width, height) {
  const rgb = new Float32Array(width * height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i] / 255;
    rgb[j + 1] = rgba[i + 1] / 255;
    rgb[j + 2] = rgba[i + 2] / 255;
  }
  return tf.tensor4d(rgb, [1, height, width, 3], 'float32');
}

/** result -> the single output tensor's flat data + [numDet, stride]. */
function outputData(result) {
  const t = typeof result.dataSync === 'function' ? result : Object.values(result)[0];
  const numDet = t.shape[1];
  const stride = t.shape[2];
  return { data: t.dataSync(), numDet, stride };
}

const labelFor = (cls) => modelDef.classNames[Math.round(cls)] || `class ${Math.round(cls)}`;

function clampBox(x1, y1, x2, y2) {
  const x = Math.max(0, Math.min(1, x1));
  const y = Math.max(0, Math.min(1, y1));
  const width = Math.max(0, Math.min(1, x2)) - x;
  const height = Math.max(0, Math.min(1, y2)) - y;
  return { x, y, width, height };
}

function decodeDetections(result) {
  const { data, numDet, stride } = outputData(result);
  const out = [];
  for (let i = 0; i < numDet; i++) {
    const b = i * stride;
    const score = data[b + 4];
    if (score < modelDef.scoreThreshold) continue; // NMS rows are sorted desc
    const box = clampBox(data[b], data[b + 1], data[b + 2], data[b + 3]);
    if (box.width <= 0 || box.height <= 0) continue;
    out.push({ label: labelFor(data[b + 5]), score, box });
  }
  return out;
}

function decodePoses(result) {
  const { data, numDet, stride } = outputData(result);
  const kThresh = modelDef.keypointThreshold ?? 0.3;
  const numKpts = (stride - 6) / 3; // 4 box + conf + cls, then x,y,score triples
  const out = [];
  for (let i = 0; i < numDet; i++) {
    const b = i * stride;
    const score = data[b + 4];
    if (score < modelDef.scoreThreshold) continue;
    const box = clampBox(data[b], data[b + 1], data[b + 2], data[b + 3]);
    const keypoints = [];
    for (let k = 0; k < numKpts; k++) {
      const o = b + 6 + k * 3;
      const ks = data[o + 2];
      keypoints.push({ x: data[o], y: data[o + 1], score: ks < kThresh ? 0 : ks });
    }
    out.push({ label: labelFor(data[b + 5]), score, box, keypoints });
  }
  return out;
}

// Two-stage: detect faces/hands, then regress 21 keypoints on each hand crop.
function runLandmark(input, box) {
  const size = modelDef.landmarkInput || 224;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const side = Math.max(box.width, box.height) * 1.3; // padded square crop (matches training)
  const x1 = cx - side / 2;
  const y1 = cy - side / 2;
  // crop from the detector's input tensor and resize (NHWC); transpose to NCHW
  // only if the landmark model expects channels-first.
  const crop = tf.tidy(() => {
    const resized = tf.image.cropAndResize(input, [[y1, x1, y1 + side, x1 + side]], [0], [size, size]);
    return landmarkNCHW ? tf.transpose(resized, [0, 3, 1, 2]) : resized;
  });
  const out = landmarkModel.predict(crop);
  const kp = out.dataSync(); // 42 values in [0,1] of the crop
  crop.dispose();
  tf.dispose(out);
  const pts = [];
  for (let k = 0; k < 21; k++) {
    pts.push({ x: x1 + kp[k * 2] * side, y: y1 + kp[k * 2 + 1] * side, score: 1 });
  }
  return pts;
}

function decodeTwoStage(result, input) {
  const { data, numDet, stride } = outputData(result);
  const thr = modelDef.scoreThreshold ?? 0.5;
  const faces = [];
  const handBoxes = [];
  for (let i = 0; i < numDet; i++) {
    const b = i * stride;
    const score = data[b + 4];
    if (score < thr) continue;
    const box = clampBox(data[b], data[b + 1], data[b + 2], data[b + 3]);
    if (box.width <= 0 || box.height <= 0) continue;
    if (Math.round(data[b + 5]) === 0) faces.push({ label: modelDef.classNames[0], score, box });
    else handBoxes.push({ score, box });
  }
  handBoxes.sort((a, b) => b.score - a.score);
  const poses = [];
  for (const h of handBoxes.slice(0, 4)) {
    if (!landmarkModel) break;
    poses.push({ label: modelDef.classNames[1], score: h.score, box: h.box, keypoints: runLandmark(input, h.box) });
  }
  return { detections: faces, poses };
}

function handleInfer(msg) {
  const { requestId, width, height } = msg;
  const modelId = modelDef ? modelDef.id : '';
  if (!model) {
    postMessage({ type: 'inferError', requestId, modelId, error: 'Model is not loaded yet.' });
    return;
  }
  const t0 = performance.now();
  let input = null;
  let result = null;
  try {
    input = toInputTensor(new Uint8ClampedArray(msg.buffer), width, height);
    result = model.predict(input);
    const payload = { type: 'result', requestId, modelId, modelType: modelDef.type, inferenceMs: 0 };
    if (modelDef.type === 'pose') {
      payload.poses = decodePoses(result);
    } else if (modelDef.type === 'twostage') {
      const ts = decodeTwoStage(result, input);
      payload.detections = ts.detections;
      payload.poses = ts.poses;
    } else {
      payload.detections = decodeDetections(result);
    }
    payload.inferenceMs = performance.now() - t0;
    postMessage(payload);
  } catch (err) {
    postMessage({ type: 'inferError', requestId, modelId, error: String(err && err.message ? err.message : err) });
  } finally {
    if (input) input.dispose();
    if (result) tf.dispose(result);
  }
}

self.addEventListener('message', (e) => {
  const msg = e.data;
  if (msg.type === 'load') {
    handleLoad(msg.model).catch((err) => {
      postMessage({
        type: 'loadError',
        modelId: msg.model.id,
        error: String(err && err.message ? err.message : err),
      });
    });
  } else if (msg.type === 'infer') {
    handleInfer(msg);
  }
});
