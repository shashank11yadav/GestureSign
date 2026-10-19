/** Page-side handle on the model worker, plus loading of model-meta.json and labels.json. */
import { ASSETS, assetUrl } from './assets';
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

async function fetchJson(name: 'modelMeta' | 'labels'): Promise<unknown> {
  const response = await fetch(assetUrl(name));
  if (!response.ok) {
    throw new ModelLoadError(
      `${ASSETS[name]} could not be downloaded (HTTP ${response.status}).`,
      response.status,
    );
  }
  return response.json();
}

let metaRequest: Promise<ModelMeta> | null = null;

/**
 * model-meta.json, fetched once and shared: the landmark worker needs its
 * feature set while the model is still loading. A failed request is retried
 * on the next call.
 */
export function loadMeta(): Promise<ModelMeta> {
  if (!metaRequest) {
    const request = fetchJson('modelMeta').then(parseMeta);
    metaRequest = request;
    request.catch(() => {
      if (metaRequest === request) metaRequest = null;
    });
  }
  return metaRequest;
}

/** Starts the worker, which downloads the model and the WASM runtime in parallel. */
function startWorker(worker: Worker): Promise<void> {
  return new Promise<ModelWorkerOut>((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<ModelWorkerOut>) => {
      resolve(event.data);
    };
    worker.onerror = (event) => {
      reject(new ModelLoadError(event.message || 'the model worker failed to start'));
    };
    const msg: ModelWorkerIn = { type: 'load' };
    worker.postMessage(msg);
  }).then((loaded) => {
    if (loaded.type === 'load-error') {
      const what =
        loaded.status !== null
          ? `The model file could not be downloaded (${loaded.message}).`
          : `The model could not be started: ${loaded.message}`;
      throw new ModelLoadError(what, loaded.status);
    }
    if (loaded.type !== 'loaded') {
      throw new ModelLoadError('The model worker sent an unexpected reply.');
    }
  });
}

export class ModelClient {
  private nextId = 1;
  private readonly pending = new Map<
    number,
    { resolve: (r: Inference & { ms: number }) => void; reject: (e: Error) => void }
  >();

  private constructor(private readonly worker: Worker) {
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
    // The worker starts downloading straight away, alongside the two small JSON files.
    const worker = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' });
    const started = startWorker(worker);
    try {
      const [meta, rawLabels] = await Promise.all([loadMeta(), fetchJson('labels'), started]);
      const labels = parseLabels(rawLabels, meta.classes);
      return { meta, labels, client: new ModelClient(worker) };
    } catch (error) {
      worker.terminate();
      throw error;
    }
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
