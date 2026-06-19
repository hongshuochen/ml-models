#!/usr/bin/env node
/**
 * Populates ./public with everything the app needs at runtime:
 *
 *   public/vendor/tflite/   - tfjs-core + cpu backend + tfjs-tflite UMD bundles
 *                             and the TFLite WASM runtime (copied from node_modules).
 *   public/models/          - the bundled YOLO26 .tflite models.
 *
 * The models are produced by exporting the locally-trained Ultralytics weights:
 *   yolo export model=runs/detect/widerface_yolo26n/weights/best.pt format=tflite nms=True imgsz=640
 *   yolo export model=runs/pose/hand_pose_yolo26n/weights/best.pt   format=tflite nms=True imgsz=640
 * This script copies those exported .tflite files into public/models if present.
 * If a model is already in public/models it is left as-is, so the app works even
 * without the training tree.
 *
 * Idempotent and dependency-free (Node core only).
 */
import { createRequire } from 'node:module';
import { mkdirSync, copyFileSync, existsSync, readdirSync } from 'node:fs';
import { dirname, join, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const VENDOR = join(ROOT, 'public', 'vendor', 'tflite');
const WASM_OUT = join(VENDOR, 'wasm');
const MODELS = join(ROOT, 'public', 'models');

// Training tree lives one level up (../runs) on the dev machine.
const RUNS = join(ROOT, '..', 'runs');

const log = (...a) => console.log('[setup]', ...a);

function pkgFile(spec) {
  const parts = spec.split('/');
  const pkg = parts.slice(0, 2).join('/');
  const sub = parts.slice(2).join('/');
  return join(dirname(require.resolve(`${pkg}/package.json`)), sub);
}

function copyVendor() {
  mkdirSync(WASM_OUT, { recursive: true });
  const umd = [
    '@tensorflow/tfjs-core/dist/tf-core.min.js',
    '@tensorflow/tfjs-backend-cpu/dist/tf-backend-cpu.min.js',
    '@tensorflow/tfjs-tflite/dist/tf-tflite.min.js',
  ];
  for (const spec of umd) {
    const src = pkgFile(spec);
    copyFileSync(src, join(VENDOR, basename(src)));
  }
  const wasmDir = dirname(pkgFile('@tensorflow/tfjs-tflite/wasm/tflite_web_api_client.js'));
  for (const f of readdirSync(wasmDir)) copyFileSync(join(wasmDir, f), join(WASM_OUT, f));
  log(`vendor + wasm copied -> ${VENDOR}`);
}

const REPO = join(ROOT, '..');
const MODEL_SOURCES = [
  {
    out: 'face_yolo26n.tflite',
    src: join(RUNS, 'detect/widerface_yolo26n/weights/best_saved_model/best_float32.tflite'),
  },
  {
    out: 'hand_pose_yolo26n.tflite', // corrected flip_idx model
    src: join(RUNS, 'pose/hand_pose_fixed/weights/best_saved_model/best_float32.tflite'),
  },
  {
    out: 'face_hand_yolo26n.tflite',
    src: join(REPO, 'bench/fh/fh_saved_model/fh_float16.tflite'),
  },
  {
    out: 'hand_landmark.tflite', // stage-2 landmark regressor
    src: join(RUNS, 'landmark/hand_landmark/saved_model/hand_landmark_sim_float16.tflite'),
  },
];

function ensureModels() {
  mkdirSync(MODELS, { recursive: true });
  for (const { out, src } of MODEL_SOURCES) {
    const dest = join(MODELS, out);
    if (existsSync(dest)) {
      log(`already present: models/${out}`);
      continue;
    }
    if (existsSync(src)) {
      copyFileSync(src, dest);
      log(`copied exported model -> models/${out}`);
    } else {
      log(`WARNING: models/${out} missing and no export found at ${src}. ` +
        `Export it with: yolo export model=<weights>.pt format=tflite nms=True imgsz=640`);
    }
  }
}

copyVendor();
ensureModels();
log('done.');
