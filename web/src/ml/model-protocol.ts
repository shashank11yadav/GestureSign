/** Messages between the page and model.worker.ts. */

export type ModelWorkerIn =
  | { type: 'load'; modelUrl: string; wasmPaths: string }
  | { type: 'run'; id: number; data: Float32Array; frames: number; dim: number };

export type ModelWorkerOut =
  | { type: 'loaded' }
  | { type: 'load-error'; message: string; status: number | null }
  | { type: 'result'; id: number; logits: Float32Array; embedding: Float32Array; ms: number }
  | { type: 'run-error'; id: number; message: string };
