/** Messages between the page and model.worker.ts. */
import { isCount, isRecord, type Parsed } from './worker-guard';

/** The worker resolves its own file URLs (assets.ts), so a load request carries none. */
export type ModelWorkerIn =
  { type: 'load' } | { type: 'run'; id: number; data: Float32Array; frames: number; dim: number };

export type ModelWorkerOut =
  | { type: 'loaded' }
  | { type: 'load-error'; message: string; status: number | null }
  | { type: 'result'; id: number; logits: Float32Array; embedding: Float32Array; ms: number }
  | { type: 'run-error'; id: number; message: string };

/** Well above the model's window (model-meta.json, maxFrames) and feature size. */
const MAX_FRAMES = 1024;
const MAX_DIM = 4096;

/** Checks a message from the page before the worker acts on it. */
export function parseModelRequest(data: unknown): Parsed<ModelWorkerIn, ModelWorkerOut> {
  if (!isRecord(data)) return { ok: false, reply: null };
  if (data['type'] === 'load') return { ok: true, message: { type: 'load' } };
  if (data['type'] !== 'run') return { ok: false, reply: null };
  const { id, data: values, frames, dim } = data;
  if (typeof id !== 'number' || !Number.isSafeInteger(id) || id < 0) {
    return { ok: false, reply: null };
  }
  if (
    !(values instanceof Float32Array) ||
    !isCount(frames, MAX_FRAMES) ||
    !isCount(dim, MAX_DIM) ||
    values.length !== frames * dim
  ) {
    return { ok: false, reply: { type: 'run-error', id, message: 'malformed run message' } };
  }
  return { ok: true, message: { type: 'run', id, data: values, frames, dim } };
}
