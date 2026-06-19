// End-to-end headless test of the YOLO inference worker (no camera needed).
// Feeds a real image through each bundled model in a real browser and checks the
// worker loads the TFLite runtime, runs predict, and decodes outputs.
import puppeteer from 'puppeteer-core';

const ORIGIN = process.env.ORIGIN || 'http://localhost:3100';
const CHROME = process.env.CHROME_PATH || '/opt/google/chrome/chrome';

const MODELS = [
  {
    id: 'face-detect',
    type: 'detection',
    modelUrl: '/models/face_yolo26n.tflite',
    inputWidth: 640,
    inputHeight: 640,
    classNames: ['face'],
    scoreThreshold: 0.45,
    img: '/_test_face.jpg',
  },
  {
    id: 'hand-pose',
    type: 'pose',
    modelUrl: '/models/hand_pose_yolo26n.tflite',
    inputWidth: 640,
    inputHeight: 640,
    classNames: ['hand'],
    scoreThreshold: 0.35,
    keypointThreshold: 0.3,
    img: '/_test_hand.jpg',
  },
];

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});
const page = await browser.newPage();
page.on('console', (m) => console.log('  [page]', m.text()));
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
await page.goto(ORIGIN, { waitUntil: 'domcontentloaded' });

const results = await page.evaluate(async (models) => {
  const run = (model) =>
    new Promise((resolve, reject) => {
      const worker = new Worker('/workers/inference.worker.js');
      const fail = (m) => { worker.terminate(); reject(new Error(m)); };
      const timer = setTimeout(() => fail('timeout'), 30000);
      let loaded = false;
      worker.onerror = (e) => fail('worker error: ' + e.message);
      worker.onmessage = async (e) => {
        const msg = e.data;
        if (msg.type === 'loadError') return fail('loadError: ' + msg.error);
        if (msg.type === 'inferError') return fail('inferError: ' + msg.error);
        if (msg.type === 'ready' && !loaded) {
          loaded = true;
          const blob = await (await fetch(model.img)).blob();
          const bmp = await createImageBitmap(blob);
          const c = document.createElement('canvas');
          c.width = model.inputWidth;
          c.height = model.inputHeight;
          const ctx = c.getContext('2d', { willReadFrequently: true });
          ctx.drawImage(bmp, 0, 0, c.width, c.height);
          const img = ctx.getImageData(0, 0, c.width, c.height);
          worker.postMessage(
            { type: 'infer', requestId: 1, modelId: model.id, width: c.width, height: c.height, buffer: img.data.buffer },
            [img.data.buffer],
          );
        } else if (msg.type === 'result') {
          clearTimeout(timer);
          worker.terminate();
          resolve(msg);
        }
      };
      worker.postMessage({ type: 'load', model });
    });

  const out = [];
  for (const m of models) out.push({ id: m.id, result: await run(m) });
  return out;
}, MODELS);

let ok = true;
for (const { id, result } of results) {
  if (id === 'face-detect') {
    const dets = result.detections ?? [];
    console.log(`FACE-DETECT: ${dets.length} faces, top=${(dets[0]?.score * 100 || 0).toFixed(0)}%, latency ${result.inferenceMs.toFixed(0)}ms`);
    if (dets.length === 0 || dets[0].label !== 'face') { console.log('  !! expected face detections'); ok = false; }
    else console.log('  ✓ faces detected');
  } else if (id === 'hand-pose') {
    const poses = result.poses ?? [];
    const kpts = poses[0]?.keypoints.filter((k) => k.score > 0).length ?? 0;
    console.log(`HAND-POSE: ${poses.length} hands, ${kpts} visible kpts on hand#1, latency ${result.inferenceMs.toFixed(0)}ms`);
    if (poses.length === 0 || kpts < 5) { console.log('  !! expected a hand with keypoints'); ok = false; }
    else console.log('  ✓ hand + keypoints detected');
  }
}
await browser.close();
console.log(ok ? '\nE2E TEST PASSED' : '\nE2E TEST FAILED');
process.exit(ok ? 0 : 1);
