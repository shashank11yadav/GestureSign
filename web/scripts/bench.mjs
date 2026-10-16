/* global window -- referenced inside page.evaluate callbacks, which run in the browser */
// A1 benchmark: decides the model's feature set (spec 4.1).
//
// Serves bench.html with Vite, opens it in Chrome with a synthetic fake camera
// (a synthetic y4m clip of a drawn figure) and DevTools CPU throttling, and
// measures landmark frames per second for:
//   - hands_pose_lips: the Holistic landmarker (hands, pose and face mesh)
//   - hands_pose:      the Hand and Pose landmarkers
// Rule: Holistic >= 15 fps at 4x throttling -> FEATURE_SET = "hands_pose_lips",
// otherwise "hands_pose". Writes pipeline/reports/bench.json and bench.md.
//
// Usage: npm run bench [-- --seconds 12 --channel chrome]

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { chromium } from '@playwright/test';
import { createServer } from 'vite';
import { writeY4m } from './fake-camera.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const REPORTS = join(ROOT, '..', 'pipeline', 'reports');
const Y4M = join(ROOT, 'bench', 'fake-camera.y4m');
const PORT = 5332;
const THRESHOLD_FPS = 15;
const REPEATS = 3;

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
}

async function runOne(browser, featureSet, throttle, seconds) {
  const context = await browser.newContext({ permissions: ['camera'] });
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: throttle });
  const url = `http://localhost:${PORT}/GestureSign/bench.html?featureSet=${featureSet}&seconds=${seconds}&warmup=4`;
  await page.goto(url);
  await page.waitForFunction(() => window.__bench?.done === true, null, {
    timeout: (seconds + 90) * 1000,
  });
  const bench = await page.evaluate(() => window.__bench);
  await context.close();
  if (bench.error) throw new Error(`${featureSet} @${throttle}x: ${bench.error}`);
  return { ...bench.result, throttle };
}

const pct = (x) => `${Math.round(x * 100)}%`;

function markdown(results, decision, holistic4x) {
  const rows = results
    .map(
      (r) =>
        `| ${r.featureSet === 'hands_pose_lips' ? 'Holistic' : 'Hand + Pose'} | ${r.throttle}x | ` +
        `${r.fps.toFixed(1)} | ${r.detectMsMean.toFixed(1)} | ${r.detectMsP95.toFixed(1)} | ` +
        `${r.inferMsMean.toFixed(1)} | ${r.videoFps.toFixed(1)} | ` +
        `${pct(r.detected.pose)} / ${pct(r.detected.hands)} / ${pct(r.detected.face)} | ${r.delegate} |`,
    )
    .join('\n');
  const first = results[0];
  return `# A1: landmark benchmark

Decides which landmarks the model uses (spec 4.1). Written by \`npm run bench\`
(web/scripts/bench.mjs); re-run it to reproduce.

**Rule:** Holistic at least ${THRESHOLD_FPS} fps with 4x CPU throttling -> \`hands_pose_lips\`,
otherwise \`hands_pose\` (Hand + Pose landmarkers, no lip features).

**Result:** Holistic ran at a median **${holistic4x.toFixed(1)} fps** over ${REPEATS} runs at 4x throttling, so
\`FEATURE_SET = "${decision}"\` (recorded in \`pipeline/configs/default.toml\`).

| Landmarks | CPU throttling | Landmark fps (median; runs) | Detect ms (mean) | Detect ms (p95) | Stub model ms | Camera fps | Detected pose / hands / face | MediaPipe delegate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
${rows}

## Setup

- Machine: Apple M4 (10-core CPU and GPU), ${first.userAgent.match(/Chrome\/[\d.]+/)?.[0] ?? 'Chrome'}, headless.
- WebGL renderer: ${first.gpu}.
- Input: Chrome's fake camera playing a synthetic 640x480, 30 fps y4m clip of a
  drawn, front-facing figure moving two open hands (web/scripts/fake-camera.mjs;
  no real person). The "detected" column says how often MediaPipe found a body,
  hands and a face in it.
- MediaPipe runs on the main thread for this measurement, because DevTools CPU
  throttling slows the page's main thread; the app itself runs it in a worker.
- The stub model (same I/O as the trained one) runs every 5th frame on a
  32-frame window, also on the main thread.
- Each 4x row is the median of ${REPEATS} runs (individual runs in brackets): run-to-run
  variation is a few fps, because other work on the machine competes for it.
- ${results[0].seconds} s measured after a 4 s warm-up, per run, after one discarded
  run per landmark set (WASM compilation and GPU shader setup). "Camera fps" is
  how many frames the fake camera delivered; landmark fps cannot exceed it.

## Caveats

- The figure is drawn, not filmed. Where the detected rates are below 100%,
  Holistic skipped its face and hand stages on those frames, so its fps is an
  upper bound for a real signer. Re-run with a person in front of a real
  webcam (\`npm run bench -- --real-camera\`) to confirm before relying on a
  close call.
- 4x throttling of an M4 stands in for a mid-range laptop; GPU speed is not
  throttled.
`;
}

async function main() {
  const seconds = Number(arg('seconds', '10'));
  const channel = arg('channel', 'chromium');
  execFileSync('node', [join(ROOT, 'scripts', 'fetch-assets.mjs'), '--bench'], {
    stdio: 'inherit',
  });
  const real = process.argv.includes('--real-camera');
  if (!real) await writeY4m(Y4M);
  const server = await createServer({
    root: ROOT,
    configFile: join(ROOT, 'vite.config.ts'),
    server: { port: PORT, strictPort: true },
    logLevel: 'error',
  });
  await server.listen();
  const browser = await chromium.launch({
    channel,
    headless: true,
    args: [
      '--use-fake-ui-for-media-stream',
      ...(real
        ? []
        : ['--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${Y4M}`]),
      '--enable-gpu',
      '--ignore-gpu-blocklist',
      '--use-angle=metal',
    ],
  });
  const results = [];
  try {
    for (const featureSet of ['hands_pose_lips', 'hands_pose']) {
      // Discarded run: the first page pays for WASM compilation and GPU shader setup.
      await runOne(browser, featureSet, 1, 3);
      for (const throttle of [1, 4]) {
        // Throttled runs decide the feature set, so they are repeated and the median is used.
        const runs = [];
        for (let i = 0; i < (throttle === 4 ? REPEATS : 1); i++) {
          runs.push(await runOne(browser, featureSet, throttle, seconds));
        }
        runs.sort((a, b) => a.fps - b.fps);
        const r = { ...runs[runs.length >> 1], runsFps: runs.map((x) => x.fps) };
        console.log(
          `${featureSet.padEnd(16)} ${throttle}x  ${r.fps.toFixed(1)} fps (runs ${r.runsFps.map((x) => x.toFixed(1)).join(', ')})  ` +
            `detect ${r.detectMsMean.toFixed(1)} ms  model ${r.inferMsMean.toFixed(1)} ms  ` +
            `camera ${r.videoFps.toFixed(1)} fps  ${r.delegate}  ` +
            `detected pose ${pct(r.detected.pose)} hands ${pct(r.detected.hands)} face ${pct(r.detected.face)}`,
        );
        results.push(r);
      }
    }
  } finally {
    await browser.close();
    await server.close();
  }
  const holistic4x = results.find(
    (r) => r.featureSet === 'hands_pose_lips' && r.throttle === 4,
  ).fps;
  const decision = holistic4x >= THRESHOLD_FPS ? 'hands_pose_lips' : 'hands_pose';
  await mkdir(REPORTS, { recursive: true });
  await writeFile(
    join(REPORTS, 'bench.json'),
    `${JSON.stringify({ thresholdFps: THRESHOLD_FPS, decision, results }, null, 2)}\n`,
  );
  await writeFile(join(REPORTS, 'bench.md'), markdown(results, decision, holistic4x));
  console.log(`Holistic at 4x: ${holistic4x.toFixed(1)} fps -> FEATURE_SET = "${decision}"`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
