import puppeteer from 'puppeteer-core';
const CHROME = process.env.CHROME_PATH || '/opt/google/chrome/chrome';
const ORIGIN = process.env.ORIGIN || 'http://localhost:3100';
const MODEL = {
  id: 'hand-pose', type: 'pose', modelUrl: '/models/hand_pose_yolo26n.tflite',
  inputWidth: 640, inputHeight: 640, classNames: ['hand'], scoreThreshold: 0.35, keypointThreshold: 0,
};
const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('console', (m) => console.log('  [page]', m.text()));
await page.goto(ORIGIN, { waitUntil: 'domcontentloaded' });
const res = await page.evaluate((model) => new Promise((resolve, reject) => {
  const w = new Worker('/workers/inference.worker.js');
  const t = setTimeout(() => reject(new Error('timeout')), 30000);
  let loaded = false;
  w.onmessage = async (e) => {
    const m = e.data;
    if (m.type === 'loadError' || m.type === 'inferError') { clearTimeout(t); reject(new Error(m.error)); }
    if (m.type === 'ready' && !loaded) {
      loaded = true;
      const bmp = await createImageBitmap(await (await fetch('/_test_hand.jpg')).blob());
      const c = document.createElement('canvas'); c.width = 640; c.height = 640;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(bmp, 0, 0, 640, 640);
      const img = ctx.getImageData(0, 0, 640, 640);
      w.postMessage({ type: 'infer', requestId: 1, modelId: model.id, width: 640, height: 640, buffer: img.data.buffer }, [img.data.buffer]);
    } else if (m.type === 'result') { clearTimeout(t); resolve(m.poses); }
  };
  w.postMessage({ type: 'load', model });
}), MODEL);
const names = ['wrist','thumb_cmc','thumb_mcp','thumb_ip','thumb_tip','index_mcp','index_pip','index_dip','index_tip','middle_mcp','middle_pip','middle_dip','middle_tip','ring_mcp','ring_pip','ring_dip','ring_tip','pinky_mcp','pinky_pip','pinky_dip','pinky_tip'];
const kp = res[0].keypoints;
console.log('BROWSER keypoints (top pose):');
kp.forEach((k, i) => console.log(`  ${i.toString().padStart(2)} ${names[i].padEnd(11)} x=${k.x.toFixed(3)} y=${k.y.toFixed(3)} c=${k.score.toFixed(2)}`));
await browser.close();
