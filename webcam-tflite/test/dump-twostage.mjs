import puppeteer from 'puppeteer-core';
const CHROME = process.env.CHROME_PATH || '/opt/google/chrome/chrome';
const ORIGIN = process.env.ORIGIN || 'http://localhost:3100';
const IMG = process.env.IMG || '/_test_hand.jpg';
const MODEL = {
  id: 'face-hand-2stage', type: 'twostage',
  modelUrl: process.env.DET || '/models/face_hand_yolo26n.tflite',
  landmarkUrl: '/models/hand_landmark.tflite',
  landmarkInput: 224, inputWidth: 640, inputHeight: 640,
  classNames: ['face', 'hand'], scoreThreshold: 0.5,
};
const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('console', (m) => console.log('  [page]', m.text()));
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
await page.goto(ORIGIN, { waitUntil: 'domcontentloaded' });
const res = await page.evaluate((model, img) => new Promise((resolve, reject) => {
  const w = new Worker('/workers/inference.worker.js');
  const t = setTimeout(() => reject(new Error('timeout')), 30000);
  let loaded = false;
  w.onerror = (e) => { clearTimeout(t); reject(new Error('worker: ' + e.message)); };
  w.onmessage = async (e) => {
    const m = e.data;
    if (m.type === 'loadError' || m.type === 'inferError') { clearTimeout(t); reject(new Error(m.error)); }
    if (m.type === 'ready' && !loaded) {
      loaded = true;
      const bmp = await createImageBitmap(await (await fetch(img)).blob());
      const c = document.createElement('canvas'); c.width = 640; c.height = 640;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bmp, 0, 0, 640, 640);
      const d = ctx.getImageData(0, 0, 640, 640);
      w.postMessage({ type: 'infer', requestId: 1, modelId: model.id, width: 640, height: 640, buffer: d.data.buffer }, [d.data.buffer]);
    } else if (m.type === 'result') { clearTimeout(t); resolve(m); }
  };
  w.postMessage({ type: 'load', model });
}), MODEL, IMG);
const faces = res.detections?.length ?? 0;
const poses = res.poses ?? [];
console.log(`TWO-STAGE: ${faces} faces, ${poses.length} hands, latency ${res.inferenceMs.toFixed(0)}ms`);
if (poses[0]) {
  const k = poses[0].keypoints;
  console.log(`  hand#1 score=${poses[0].score.toFixed(2)} keypoints=${k.length} wrist=(${k[0].x.toFixed(2)},${k[0].y.toFixed(2)}) midtip=(${k[12].x.toFixed(2)},${k[12].y.toFixed(2)})`);
}
const ok = poses.length >= 1 && poses[0].keypoints.length === 21;
await browser.close();
console.log(ok ? '\nTWO-STAGE TEST PASSED' : '\nTWO-STAGE TEST FAILED');
process.exit(ok ? 0 : 1);
