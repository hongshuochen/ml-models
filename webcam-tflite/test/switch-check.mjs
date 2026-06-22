// Reproduce the model-switch hang: load two models sequentially in ONE worker
// (what the app does when you switch). Reports which load reaches `ready`.
import puppeteer from 'puppeteer-core';
const CHROME = process.env.CHROME_PATH || '/opt/google/chrome/chrome';
const ORIGIN = process.env.ORIGIN || 'http://localhost:3000';

const COMPACT = {
  id: 'compact', type: 'twostage',
  modelUrl: '/models/face_hand_pico_p45_hagrid_f16.tflite',
  landmarkUrl: '/models/hand_landmark_mnv3s025_hagrid_f16.tflite',
  landmarkInput: 224, landmarkLayout: 'nhwc', inputWidth: 640, inputHeight: 640,
  classNames: ['face', 'hand'], scoreThreshold: 0.5,
};
const POSE = {
  id: 'hand-pose', type: 'pose', modelUrl: '/models/hand_pose_yolo26n.tflite',
  inputWidth: 640, inputHeight: 640, classNames: ['hand'], scoreThreshold: 0.35,
};

const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('  [pageerror]', e.message));
await page.goto(ORIGIN, { waitUntil: 'domcontentloaded' });
const out = await page.evaluate(
  (a, b) =>
    new Promise((resolve) => {
      const log = [];
      let w = null;
      let stage = 0;
      const onmsg = (e) => {
        const msg = e.data;
        if (msg.type === 'loadError') { log.push('loadError ' + msg.modelId + ': ' + msg.error); clearTimeout(t); resolve(log); }
        if (msg.type === 'ready') {
          log.push('ready ' + msg.modelId);
          if (stage === 0) { stage = 1; log.push('-- switch: fresh worker --'); spawn(b); }
          else { clearTimeout(t); resolve(log); }
        }
      };
      // Mirror the hook fix: terminate the old worker and load into a fresh one.
      const spawn = (m) => {
        if (w) w.terminate();
        w = new Worker('/workers/inference.worker.js');
        w.onmessage = onmsg;
        log.push('load ' + m.id);
        w.postMessage({ type: 'load', model: m });
      };
      const t = setTimeout(() => { log.push('TIMEOUT at stage ' + stage); resolve(log); }, 35000);
      spawn(a);
    }),
  COMPACT, POSE,
);
console.log(out.join('\n'));
await browser.close();
process.exit(out.some((l) => l.startsWith('ready hand-pose')) ? 0 : 1);
