// Puts every runtime file the page needs under public/vendor/, so the site
// serves them itself and the CSP can stay 'self' only:
//   - MediaPipe .task models, downloaded from Google's model bucket at pinned
//     versions and checked against pinned SHA-256 hashes (a mismatch fails the
//     build);
//   - the MediaPipe and onnxruntime WASM runtimes, copied from node_modules,
//     whose versions package-lock.json pins. Only onnxruntime's plain WASM
//     build ships (no WebGPU/JSEP builds): the model is small enough that the
//     CPU runs it in a few milliseconds.
//
// Which landmarker ships follows the model's feature set in
// public/models/model-meta.json (decided by the A1 benchmark): Holistic for
// "hands_pose_lips", the Hand and Pose landmarkers for "hands_pose".
//
// Usage: node scripts/fetch-assets.mjs [--bench]
//   --bench fetches every landmarker, for bench.html (dev only).

import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const VENDOR = join(ROOT, 'public', 'vendor');
const BUCKET = 'https://storage.googleapis.com/mediapipe-models';

export const MODELS = [
  {
    file: 'hand_landmarker.task',
    url: `${BUCKET}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
    sha256: 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    bytes: 7819105,
    featureSets: ['hands_pose'],
  },
  {
    file: 'pose_landmarker_lite.task',
    url: `${BUCKET}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
    sha256: '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a',
    bytes: 5777746,
    featureSets: ['hands_pose'],
  },
  {
    file: 'holistic_landmarker.task',
    url: `${BUCKET}/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task`,
    sha256: 'e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8',
    bytes: 13683609,
    featureSets: ['hands_pose_lips'],
  },
];

const COPIES = [
  ['@mediapipe/tasks-vision/wasm/vision_wasm_module_internal.js', 'mediapipe/wasm'],
  ['@mediapipe/tasks-vision/wasm/vision_wasm_module_internal.wasm', 'mediapipe/wasm'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.mjs', 'ort'],
  ['onnxruntime-web/dist/ort-wasm-simd-threaded.wasm', 'ort'],
];

// Classic (non-module) MediaPipe glue, only for the main-thread benchmark page.
const BENCH_COPIES = [
  ['@mediapipe/tasks-vision/wasm/vision_wasm_internal.js', 'mediapipe/wasm'],
  ['@mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm', 'mediapipe/wasm'],
];

async function sha256(path) {
  return createHash('sha256')
    .update(await readFile(path))
    .digest('hex');
}

async function exists(path) {
  try {
    await stat(path);
    return true;
  } catch {
    return false;
  }
}

async function fetchModel(model) {
  const dest = join(VENDOR, 'mediapipe', 'models', model.file);
  if ((await exists(dest)) && (await sha256(dest)) === model.sha256) return 'cached';
  await mkdir(dirname(dest), { recursive: true });
  const response = await fetch(model.url);
  if (!response.ok) throw new Error(`${model.url}: HTTP ${response.status}`);
  const body = Buffer.from(await response.arrayBuffer());
  const digest = createHash('sha256').update(body).digest('hex');
  if (digest !== model.sha256) {
    throw new Error(`${model.file}: SHA-256 ${digest} does not match the pinned ${model.sha256}`);
  }
  const tmp = `${dest}.part`;
  await writeFile(tmp, body);
  await rename(tmp, dest);
  return 'downloaded';
}

async function main() {
  const bench = process.argv.includes('--bench');
  const meta = JSON.parse(
    await readFile(join(ROOT, 'public', 'models', 'model-meta.json'), 'utf-8'),
  );
  for (const model of MODELS) {
    if (!bench && !model.featureSets.includes(meta.featureSet)) {
      await rm(join(VENDOR, 'mediapipe', 'models', model.file), { force: true });
      continue;
    }
    const how = await fetchModel(model);
    console.log(`${how.padEnd(10)} mediapipe/models/${model.file}`);
  }
  if (!bench) {
    for (const [source, folder] of BENCH_COPIES) {
      await rm(join(VENDOR, folder, source.split('/').pop()), { force: true });
    }
  }
  // Start the onnxruntime folder afresh, so runtimes that no longer ship do not linger.
  await rm(join(VENDOR, 'ort'), { recursive: true, force: true });
  for (const [source, folder] of bench ? [...COPIES, ...BENCH_COPIES] : COPIES) {
    const from = join(ROOT, 'node_modules', source);
    const to = join(VENDOR, folder, source.split('/').pop());
    await mkdir(dirname(to), { recursive: true });
    await rm(to, { force: true });
    await copyFile(from, to);
  }
  console.log(`copied     runtime files from node_modules`);
}

main().catch((error) => {
  console.error(`fetch-assets: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
