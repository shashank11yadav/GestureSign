// Puts every runtime file the page needs under public/vendor/, so the site
// serves them itself and the CSP can stay 'self' only:
//   - MediaPipe .task models, downloaded from Google's model bucket at pinned
//     versions and checked against a pinned size and SHA-256 before anything
//     is written to disk (a mismatch fails the build);
//   - the MediaPipe and onnxruntime WASM runtimes, copied from node_modules,
//     whose versions package-lock.json pins. Only onnxruntime's plain WASM
//     build ships (no WebGPU/JSEP builds): the model is small enough that the
//     CPU runs it in a few milliseconds.
//
// Which landmarker ships follows the model's feature set in
// public/models/model-meta.json (decided by the landmark benchmark): Holistic for
// "hands_pose_lips", the Hand and Pose landmarkers for "hands_pose".
//
// Usage: node scripts/fetch-assets.mjs [--bench]
//   --bench fetches every landmarker, for bench.html (dev only).

import { createHash, randomBytes } from 'node:crypto';
import { copyFile, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

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

/**
 * Downloads a pinned file into memory and returns it only if its size and
 * SHA-256 match the pins. Reading stops as soon as the body grows past the
 * pinned size, so a wrong or hostile response cannot fill memory.
 */
export async function downloadVerified(pinned, fetchImpl = fetch) {
  const response = await fetchImpl(pinned.url);
  if (!response.ok || !response.body) throw new Error(`${pinned.url}: HTTP ${response.status}`);
  const chunks = [];
  let size = 0;
  for await (const chunk of response.body) {
    size += chunk.byteLength;
    if (size > pinned.bytes) {
      await response.body.cancel().catch(() => undefined);
      throw new Error(`${pinned.file}: response is larger than the pinned ${pinned.bytes} bytes`);
    }
    chunks.push(chunk);
  }
  if (size !== pinned.bytes) {
    throw new Error(`${pinned.file}: got ${size} bytes, expected the pinned ${pinned.bytes}`);
  }
  const body = Buffer.concat(chunks);
  const digest = createHash('sha256').update(body).digest('hex');
  if (digest !== pinned.sha256) {
    throw new Error(`${pinned.file}: SHA-256 ${digest} does not match the pinned ${pinned.sha256}`);
  }
  return body;
}

/**
 * Writes through a new, uniquely named temporary file and renames it into
 * place, so `dest` only ever holds a complete, verified file.
 */
export async function writeAtomic(dest, body) {
  const tmp = `${dest}.${randomBytes(6).toString('hex')}.part`;
  try {
    await writeFile(tmp, body, { flag: 'wx' });
    await rename(tmp, dest);
  } catch (error) {
    await rm(tmp, { force: true });
    throw error;
  }
}

async function fetchModel(model) {
  const dest = join(VENDOR, 'mediapipe', 'models', model.file);
  if ((await exists(dest)) && (await sha256(dest)) === model.sha256) return 'cached';
  const body = await downloadVerified(model);
  await mkdir(dirname(dest), { recursive: true });
  await writeAtomic(dest, body);
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

// Run as a script; importing the module (from its tests) does nothing.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(`fetch-assets: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  });
}
