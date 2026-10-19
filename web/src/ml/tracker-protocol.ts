/** Messages between the page and tracker.worker.ts. */
import type { FeatureSet } from './features';
import type { Delegate } from './tracker-core';
import { isRecord, type Parsed } from './worker-guard';

/** The worker resolves its own file URLs (assets.ts), so an init request carries none. */
export type TrackerWorkerIn =
  | { type: 'init'; featureSet: FeatureSet; delegate: Delegate }
  | { type: 'frame'; bitmap: ImageBitmap; t: number };

export type TrackerWorkerOut =
  | { type: 'ready'; delegate: Delegate }
  | { type: 'init-error'; message: string }
  | { type: 'landmarks'; t: number; ms: number; frame: Float32Array; poses: number; hands: number }
  | { type: 'frame-error'; t: number; message: string };

const isFeatureSet = (v: unknown): v is FeatureSet => v === 'hands_pose' || v === 'hands_pose_lips';
const isDelegate = (v: unknown): v is Delegate => v === 'GPU' || v === 'CPU';

export function isImageBitmap(v: unknown): v is ImageBitmap {
  return typeof ImageBitmap !== 'undefined' && v instanceof ImageBitmap;
}

/** Checks a message from the page before the worker acts on it. */
export function parseTrackerRequest(data: unknown): Parsed<TrackerWorkerIn, TrackerWorkerOut> {
  if (!isRecord(data)) return { ok: false, reply: null };
  if (data['type'] === 'init') {
    const { featureSet, delegate } = data;
    if (!isFeatureSet(featureSet) || !isDelegate(delegate)) {
      return { ok: false, reply: { type: 'init-error', message: 'malformed init message' } };
    }
    return { ok: true, message: { type: 'init', featureSet, delegate } };
  }
  if (data['type'] !== 'frame') return { ok: false, reply: null };
  const { bitmap, t } = data;
  if (typeof t !== 'number' || !Number.isFinite(t)) return { ok: false, reply: null };
  if (!isImageBitmap(bitmap)) {
    return { ok: false, reply: { type: 'frame-error', t, message: 'malformed frame message' } };
  }
  return { ok: true, message: { type: 'frame', bitmap, t } };
}
