/**
 * A1 benchmark (dev only, served by `vite` from bench.html, never built).
 *
 * Runs a landmark pipeline on the main thread, where DevTools CPU throttling
 * applies, against the fake camera, and runs the stub model every 5th frame on
 * a 32-frame window. Reports landmark frames processed per second.
 *
 * Query: ?featureSet=hands_pose|hands_pose_lips&seconds=10&warmup=3&delegate=GPU|CPU
 */
import * as ort from 'onnxruntime-web/wasm';
import { modelInput, type FeatureSet } from '../ml/features';
import type { Frame } from '../ml/landmarks';
import { createTracker, type Delegate } from '../ml/tracker-core';

interface BenchResult {
  featureSet: FeatureSet;
  delegate: Delegate;
  fps: number;
  frames: number;
  seconds: number;
  detectMsMean: number;
  detectMsP95: number;
  inferMsMean: number;
  inferRuns: number;
  videoFps: number;
  /** Share of measured frames with a pose / at least one hand / a face detected. */
  detected: { pose: number; hands: number; face: number };
  gpu: string;
  userAgent: string;
}

declare global {
  interface Window {
    __bench?: { done: boolean; result?: BenchResult; error?: string };
  }
}

const params = new URLSearchParams(location.search);
const featureSet = (params.get('featureSet') ?? 'hands_pose') as FeatureSet;
const seconds = Number(params.get('seconds') ?? 10);
const warmup = Number(params.get('warmup') ?? 3);
const delegate = (params.get('delegate') ?? 'GPU') as Delegate;
const out = document.getElementById('out') as HTMLPreElement;
const video = document.getElementById('video') as HTMLVideoElement;
window.__bench = { done: false };

const site = (path: string) => new URL(`${import.meta.env.BASE_URL}${path}`, location.origin).href;

function gpuName(): string {
  const gl = document.createElement('canvas').getContext('webgl2');
  if (!gl) return 'no WebGL2';
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
}

function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))] ?? 0;
}

async function main(): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, frameRate: 30 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  const tracker = await createTracker({
    wasmBase: site('vendor/mediapipe/wasm'),
    modelBase: site('vendor/mediapipe/models/'),
    featureSet,
    delegate,
  });
  ort.env.wasm.wasmPaths = site('vendor/ort/');
  ort.env.wasm.numThreads = 1;
  const modelBytes = new Uint8Array(await (await fetch(site('models/model.onnx'))).arrayBuffer());
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] });

  const window32: Frame[] = [];
  const detectMs: number[] = [];
  const inferMs: number[] = [];
  let processed = 0;
  const presented = { first: -1, last: 0 };
  const seen = { pose: 0, hands: 0, face: 0 };
  let measuring = false;
  let inferBusy = false;
  const start = performance.now();

  await new Promise<void>((resolve) => {
    const onFrame = (_now: number, metadata: VideoFrameCallbackMetadata) => {
      const now = performance.now();
      const elapsed = (now - start) / 1000;
      if (!measuring && elapsed >= warmup) measuring = true;
      if (elapsed >= warmup + seconds) {
        resolve();
        return;
      }
      if (measuring) {
        if (presented.first < 0) presented.first = metadata.presentedFrames;
        presented.last = metadata.presentedFrames;
      }
      const t0 = performance.now();
      const { frame, poses, hands } = tracker.detect(video, now);
      const dt = performance.now() - t0;
      window32.push(frame);
      if (window32.length > 32) window32.shift();
      if (measuring) {
        processed++;
        detectMs.push(dt);
        if (poses > 0) seen.pose++;
        if (hands > 0) seen.hands++;
        if (Number.isFinite(frame[0])) seen.face++;
      }
      if (processed % 5 === 0 && window32.length >= 8 && !inferBusy) {
        inferBusy = true;
        const input = modelInput(window32, 30, {
          fps: 30,
          maxFrames: 64,
          featureSet: 'hands_pose',
        });
        const tensor = new ort.Tensor('float32', input.data, [1, input.frames, input.dim]);
        const i0 = performance.now();
        void session.run({ frames: tensor }).then(() => {
          if (measuring) inferMs.push(performance.now() - i0);
          inferBusy = false;
        });
      }
      out.textContent = `${featureSet} ${tracker.delegate}: ${processed} frames, ${elapsed.toFixed(1)} s`;
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  });

  const mean = (v: number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0);
  const result: BenchResult = {
    featureSet,
    delegate: tracker.delegate,
    fps: processed / seconds,
    frames: processed,
    seconds,
    detectMsMean: mean(detectMs),
    detectMsP95: percentile(detectMs, 95),
    inferMsMean: mean(inferMs),
    inferRuns: inferMs.length,
    videoFps: (presented.last - Math.max(presented.first, 0)) / seconds,
    detected: {
      pose: seen.pose / Math.max(1, processed),
      hands: seen.hands / Math.max(1, processed),
      face: seen.face / Math.max(1, processed),
    },
    gpu: gpuName(),
    userAgent: navigator.userAgent,
  };
  tracker.close();
  for (const track of stream.getTracks()) track.stop();
  out.textContent = JSON.stringify(result, null, 2);
  window.__bench = { done: true, result };
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  out.textContent = `error: ${message}`;
  window.__bench = { done: true, error: message };
});
