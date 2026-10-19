/**
 * Runs the ONNX model off the main thread with ONNX Runtime's WebAssembly
 * backend. The model has about 1.4 M parameters and runs in a few
 * milliseconds on a CPU, so a GPU backend would not make it noticeably faster
 * and would add a much larger runtime download.
 *
 * The model file and the runtime's WASM binary are downloaded in parallel,
 * both from this site (public/models, public/vendor/ort), never from a CDN.
 */
import * as ort from 'onnxruntime-web/wasm';
import { assetUrl } from './assets';
import { parseModelRequest, type ModelWorkerOut } from './model-protocol';
import { isTrustedOrigin } from './worker-guard';

interface WorkerScope {
  location: { origin: string };
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage(message: ModelWorkerOut, transfer?: Transferable[]): void;
}
const scope = self as unknown as WorkerScope;

/** The SIMD build, run single-threaded: GitHub Pages cannot send the headers threads need. */
const WASM_FILE = 'ort-wasm-simd-threaded.wasm';

let session: ort.InferenceSession | null = null;

class HttpError extends Error {
  constructor(readonly status: number) {
    super(`HTTP ${status}`);
  }
}

async function download(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new HttpError(response.status);
  return new Uint8Array(await response.arrayBuffer());
}

async function load(): Promise<void> {
  const wasmPaths = assetUrl('ortRuntime');
  // Settle both downloads before reporting, so a failure in one is never left unhandled.
  const [model, wasm] = await Promise.allSettled([
    download(assetUrl('signModel')),
    download(`${wasmPaths}${WASM_FILE}`),
  ]);
  if (model.status === 'rejected') {
    const error: unknown = model.reason;
    scope.postMessage({
      type: 'load-error',
      message: error instanceof Error ? error.message : String(error),
      status: error instanceof HttpError ? error.status : null,
    });
    return;
  }
  if (wasm.status === 'rejected') {
    const error: unknown = wasm.reason;
    throw new Error(
      `the WebAssembly runtime could not be downloaded (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  // The glue script still comes from wasmPaths; the binary is the one downloaded above.
  ort.env.wasm.wasmPaths = wasmPaths;
  ort.env.wasm.wasmBinary = wasm.value;
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  session = await ort.InferenceSession.create(model.value, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  });
  scope.postMessage({ type: 'loaded' });
}

async function run(id: number, data: Float32Array, frames: number, dim: number): Promise<void> {
  if (!session) throw new Error('model not loaded');
  const started = performance.now();
  const input = new ort.Tensor('float32', data, [1, frames, dim]);
  const outputs = await session.run({ frames: input });
  const logits = outputs['logits']?.data;
  const embedding = outputs['embedding']?.data;
  if (!(logits instanceof Float32Array) || !(embedding instanceof Float32Array)) {
    throw new Error('model returned unexpected outputs');
  }
  const ms = performance.now() - started;
  const result = {
    type: 'result' as const,
    id,
    logits: logits.slice(),
    embedding: embedding.slice(),
    ms,
  };
  scope.postMessage(result, [result.logits.buffer, result.embedding.buffer]);
}

scope.onmessage = (event) => {
  if (!isTrustedOrigin(event.origin, scope.location.origin)) return;
  const parsed = parseModelRequest(event.data);
  if (!parsed.ok) {
    if (parsed.reply) scope.postMessage(parsed.reply);
    return;
  }
  const msg = parsed.message;
  if (msg.type === 'load') {
    load().catch((error: unknown) => {
      scope.postMessage({
        type: 'load-error',
        message: error instanceof Error ? error.message : String(error),
        status: null,
      });
    });
  } else {
    run(msg.id, msg.data, msg.frames, msg.dim).catch((error: unknown) => {
      scope.postMessage({
        type: 'run-error',
        id: msg.id,
        message: error instanceof Error ? error.message : String(error),
      });
    });
  }
};
