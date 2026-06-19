import puppeteer from 'puppeteer-core';
import { resolve } from 'node:path';
const CHROME = process.env.CHROME_PATH || '/opt/google/chrome/chrome';
const ORIGIN = process.env.ORIGIN || 'http://localhost:3100';
const Y4M = resolve(process.env.Y4M || 'hand.y4m');

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: [
    '--no-sandbox',
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    `--use-file-for-fake-video-capture=${Y4M}`,
    '--window-size=900,820',
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 900, height: 820, deviceScaleFactor: 1 });
await page.goto(ORIGIN, { waitUntil: 'networkidle2' });
// switch to Hand Pose
await page.evaluate(() => {
  const btn = [...document.querySelectorAll('.seg')].find((b) => /pose/i.test(b.textContent));
  btn?.click();
});
await new Promise((r) => setTimeout(r, 5000));
await page.screenshot({ path: process.env.OUT || 'pose.png' });
await browser.close();
console.log('saved', process.env.OUT || 'pose.png');
