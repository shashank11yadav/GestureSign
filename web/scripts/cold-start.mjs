// Cold start: how long a first-time visitor waits from opening the page to the
// first tracked landmarks.
//
// Serves the production build (dist/, so run `npm run build` first) with
// `vite preview`, and for each run opens it in a fresh browser context (an
// empty HTTP cache, like a first visit) with Chrome's fake camera playing the
// synthetic clip from fake-camera.mjs. It clicks "Start camera" a set time
// after navigation and reads the page's performance marks (milliseconds since
// navigation start):
//   gesturesign:model-ready      the sign model is loaded in its worker
//   gesturesign:camera-click     "Start camera" was clicked
//   gesturesign:first-landmarks  MediaPipe returned the first camera frame
// "Ready" is the later of model-ready and first-landmarks: from then on a sign
// can be recognised.
//
// Usage: npm run cold-start [-- --runs 3 --click-after 0 --network broadband|none]
//   --network broadband throttles every request to 50 Mbit/s down, 10 up,
//   20 ms latency (DevTools network emulation).

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';
import { preview } from 'vite';
import { writeY4m } from './fake-camera.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const Y4M = join(ROOT, 'bench', 'fake-camera.y4m');
const PORT = 5335;
const NETWORKS = {
  none: null,
  broadband: { latency: 20, downloadThroughput: 50e6 / 8, uploadThroughput: 10e6 / 8 },
};
const MARKS = ['model-ready', 'camera-click', 'first-landmarks'];

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
}

function launch() {
  return chromium.launch({
    headless: true,
    args: [
      '--use-fake-ui-for-media-stream',
      '--use-fake-device-for-media-stream',
      `--use-file-for-fake-video-capture=${Y4M}`,
      '--enable-gpu',
      '--ignore-gpu-blocklist',
      '--use-angle=metal',
    ],
  });
}

async function runOne(clickAfterMs, network) {
  // A new browser per run: no HTTP, code or GPU shader cache carries over.
  const browser = await launch();
  const context = await browser.newContext({ permissions: ['camera'] });
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  if (network) {
    // Applies to the page's dedicated workers too, which fetch the runtimes and models.
    await cdp.send('Network.enable');
    await cdp.send('Network.emulateNetworkConditions', { offline: false, ...network });
  }
  await page.goto(`http://localhost:${PORT}/GestureSign/`);
  const start = page.getByRole('button', { name: 'Start camera' });
  await start.waitFor();
  const now = await page.evaluate(() => performance.now());
  if (clickAfterMs > now) await page.waitForTimeout(clickAfterMs - now);
  await start.click();
  // Wait for both: a sign can be recognised only once the model is ready too.
  await page.waitForFunction(
    () =>
      performance.getEntriesByName('gesturesign:first-landmarks').length > 0 &&
      performance.getEntriesByName('gesturesign:model-ready').length > 0,
    null,
    { timeout: 120_000, polling: 50 },
  );
  const marks = await page.evaluate((names) => {
    const out = {};
    for (const name of names) {
      const entry = performance.getEntriesByName(`gesturesign:${name}`)[0];
      out[name] = entry ? Math.round(entry.startTime) : null;
    }
    return out;
  }, MARKS);
  await browser.close();
  return marks;
}

const ready = (r) => Math.max(r['model-ready'], r['first-landmarks']);

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
};

async function main() {
  const runs = Number(arg('runs', '3'));
  const clickAfter = Number(arg('click-after', '0'));
  const networkName = arg('network', 'none');
  if (!(networkName in NETWORKS)) throw new Error(`unknown --network ${networkName}`);
  await writeY4m(Y4M);
  const server = await preview({
    root: ROOT,
    configFile: join(ROOT, 'vite.config.ts'),
    preview: { port: PORT, strictPort: true },
    logLevel: 'error',
  });
  const results = [];
  try {
    for (let i = 0; i < runs; i++) {
      const r = await runOne(clickAfter, NETWORKS[networkName]);
      console.log(
        `run ${i + 1}: model ready ${r['model-ready']} ms, click ${r['camera-click']} ms, ` +
          `first landmarks ${r['first-landmarks']} ms, ready ${ready(r)} ms`,
      );
      results.push(r);
    }
  } finally {
    await new Promise((resolve) => {
      server.httpServer.close(resolve);
    });
  }
  const first = median(results.map((r) => r['first-landmarks']));
  const model = median(results.map((r) => r['model-ready']));
  console.log(
    `median over ${runs} runs (network ${networkName}, click at ${clickAfter} ms): ` +
      `model ready ${model} ms, first landmarks ${first} ms, ready ${median(results.map(ready))} ms`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
