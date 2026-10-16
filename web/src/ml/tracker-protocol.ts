/** Messages between the page and tracker.worker.ts. */
import type { Delegate, TrackerOptions } from './tracker-core';

export type TrackerWorkerIn =
  { type: 'init'; options: TrackerOptions } | { type: 'frame'; bitmap: ImageBitmap; t: number };

export type TrackerWorkerOut =
  | { type: 'ready'; delegate: Delegate }
  | { type: 'init-error'; message: string }
  | { type: 'landmarks'; t: number; ms: number; frame: Float32Array; poses: number; hands: number }
  | { type: 'frame-error'; t: number; message: string };
