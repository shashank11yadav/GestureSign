/** Page-side handle on the model worker, plus loading of model-meta.json and labels.json. */
import type { ModelInput } from './features';
import { parseLabels, parseMeta, type ModelMeta, type SignLabel } from './meta';
import type { ModelWorkerIn, ModelWorkerOut } from './model-protocol';
import type { Inference } from './recognizer';

export class ModelLoadError extends Error {
  override name = 'ModelLoadError';
  constructor(
    message: string,
    readonly status: number | null = null,
  ) {
    super(message);
  }
}

export interface LoadedModel {
  meta: ModelMeta;
  labels: SignLabel[];
  client: ModelClient;
}

/** Prefer WebGPU when the browser has it; the worker falls back to WASM on its own. */
export const PREFER_WEBGPU = true;

function siteUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, window.location.origin).href;
}

async function fetchJson(path: string): Promise<unknown> {
  const response = await fetch(siteUrl(path));
  if (!response.ok) {
    throw new ModelLoadError(
      `${path} could not be downloaded (HTTP ${response.status}).`,
      response.status,
    );
  }
  return response.json();
}

export class ModelClient {
  private nextId = 1;
  private readonly pending = new Map<
    number,
    { resolve: (r: Inference & { ms: number }) => void; reject: (e: Error) => void }
  >();

  private constructor(
    private readonly worker: Worker,
    readonly backend: 'webgpu' | 'wasm',
    readonly fallbackReason: string | null,
  ) {
    worker.onmessage = (event: MessageEvent<ModelWorkerOut>) => {
      const msg = event.data;
      if (msg.type !== 'result' && msg.type !== 'run-error') return;
      const waiter = this.pending.get(msg.id);
      if (!waiter) return;
      this.pending.delete(msg.id);
      if (msg.type === 'result') {
        waiter.resolve({ logits: msg.logits, embedding: msg.embedding, ms: msg.ms });
      } else {
        waiter.reject(new Error(msg.message));
      }
    };
  }

  static async load(): Promise<LoadedModel> {
    const [rawMeta, rawLabels] = await Promise.all([
      fetchJson('models/model-meta.json'),
      fetchJson('models/labels.json'),
    ]);
    const meta = parseMeta(rawMeta);
    const labels = parseLabels(rawLabels, meta.classes);
    const worker = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' });
    const loaded = await new Promise<ModelWorkerOut>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<ModelWorkerOut>) => {
        resolve(event.data);
      };
      worker.onerror = (event) => {
        reject(new ModelLoadError(event.message || 'the model worker failed to start'));
      };
      const msg: ModelWorkerIn = {
        type: 'load',
        modelUrl: siteUrl('models/model.onnx'),
        wasmPaths: siteUrl('vendor/ort/'),
        preferWebGpu: PREFER_WEBGPU,
      };
      worker.postMessage(msg);
    }).catch((error: unknown) => {
      worker.terminate();
      throw error;
    });
    if (loaded.type === 'load-error') {
      worker.terminate();
      const what =
        loaded.status !== null
          ? `The model file could not be downloaded (${loaded.message}).`
          : `The model could not be started: ${loaded.message}`;
      throw new ModelLoadError(what, loaded.status);
    }
    if (loaded.type !== 'loaded') {
      worker.terminate();
      throw new ModelLoadError('The model worker sent an unexpected reply.');
    }
    return { meta, labels, client: new ModelClient(worker, loaded.backend, loaded.fallbackReason) };
  }

  infer(input: ModelInput): Promise<Inference & { ms: number }> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      // Copy: the caller may still hold the features, and the buffer is transferred.
      const data = input.data.slice();
      const msg: ModelWorkerIn = { type: 'run', id, data, frames: input.frames, dim: input.dim };
      this.worker.postMessage(msg, [data.buffer]);
    });
  }

  dispose(): void {
    this.worker.terminate();
    for (const waiter of this.pending.values()) waiter.reject(new Error('model closed'));
    this.pending.clear();
  }
}
