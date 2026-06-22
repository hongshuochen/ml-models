// Headless load check: confirm a model def loads in the real tfjs-tflite WASM
// runtime (this is where "Can't initialize model" surfaces). Usage:
//   DET=/models/x.tflite LM=/models/y.tflite node test/load-check.mjs
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_PATH || '/opt/google/chrome/chrome';
const ORIGIN = process.env.ORIGIN || 'http://localhost:3100';
const MODEL = {
  id: 'load-check',
  type: 'twostage',
  modelUrl: process.env.DET || '/models/face_hand_pico_p45_hagrid_f16.tflite',
  landmarkUrl: process.env.LM || '/models/hand_landmark_mnv3s025_hagrid_f16.tflite',
  landmarkInput: 224,
  landmarkLayout: 'nhwc',
  inputWidth: 640,
  inputHeight: 640,
  classNames: ['face', 'hand'],
  scoreThreshold: 0.5,
};

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
await page.goto(ORIGIN, { waitUntil: 'domcontentloaded' });
const res = await page
  .evaluate(
    (model) =>
      new Promise((resolve, reject) => {
        const w = new Worker('/workers/inference.worker.js');
        const t = setTimeout(() => reject(new Error('timeout')), 40000);
        w.onerror = (e) => {
          clearTimeout(t);
          reject(new Error('worker: ' + e.message));
        };
        w.onmessage = (e) => {
          const m = e.data;
          if (m.type === 'loadError') {
            clearTimeout(t);
            reject(new Error('loadError: ' + m.error));
          }
          if (m.type === 'ready') {
            clearTimeout(t);
            resolve('ready: ' + m.modelId);
          }
        };
        w.postMessage({ type: 'load', model });
      }),
    MODEL,
  )
  .catch((e) => 'FAILED: ' + e.message);
console.log('DET =', MODEL.modelUrl);
console.log('LM  =', MODEL.landmarkUrl);
console.log('RESULT:', res);
await browser.close();
process.exit(String(res).startsWith('ready') ? 0 : 1);
