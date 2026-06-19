// Captures a screenshot of the running app in headless Chrome with a fake camera.
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_PATH || '/opt/google/chrome/chrome';
const ORIGIN = process.env.ORIGIN || 'http://localhost:3100';
const OUT = process.env.OUT || 'ui.png';

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: [
    '--no-sandbox',
    '--use-fake-device-for-media-stream',
    '--use-fake-ui-for-media-stream',
    '--window-size=1280,800',
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 800, deviceScaleFactor: 1 });
await page.goto(ORIGIN, { waitUntil: 'networkidle2' });
await new Promise((r) => setTimeout(r, 5000)); // camera + model load + a few inferences
await page.screenshot({ path: OUT });
await browser.close();
console.log('saved', OUT);
