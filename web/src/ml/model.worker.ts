/**
 * Runs the ONNX model off the main thread. WebGPU is used when the browser
 * offers an adapter and the session builds; otherwise (or if WebGPU fails at
 * any point while loading) the plain WASM backend is used. The WASM runtime is
 * loaded from this site (public/vendor/ort), never from a CDN.
 */
import type * as Ort from 'onnxruntime-web';
import type { ModelWorkerIn, ModelWorkerOut } from './model-protocol';

interface WorkerScope {
  onmessage: ((event: MessageEvent<ModelWorkerIn>) => void) | null;
  postMessage(message: ModelWorkerOut, transfer?: Transferable[]): void;
}
const scope = self as unknown as WorkerScope;

type OrtModule = typeof Ort;
type Backend = 'webgpu' | 'wasm';

let session: Ort.InferenceSession | null = null;
let ort: OrtModule | null = null;

async function loadRuntime(backend: Backend, wasmPaths: string): Promise<OrtModule> {
  const mod: OrtModule =
    backend === 'webgpu'
      ? await import('onnxruntime-web/webgpu')
      : await import('onnxruntime-web/wasm');
  mod.env.wasm.wasmPaths = wasmPaths;
  // GitHub Pages cannot send the cross-origin isolation headers threads need.
  mod.env.wasm.numThreads = 1;
  mod.env.logLevel = 'error';
  return mod;
}

async function hasWebGpu(): Promise<boolean> {
  if (!('gpu' in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function createSession(backend: Backend, bytes: Uint8Array, wasmPaths: string) {
  const mod = await loadRuntime(backend, wasmPaths);
  const created = await mod.InferenceSession.create(bytes, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  });
  return { mod, created };
}

async function load(modelUrl: string, wasmPaths: string, preferWebGpu: boolean): Promise<void> {
  const response = await fetch(modelUrl);
  if (!response.ok) {
    scope.postMessage({
      type: 'load-error',
      message: `HTTP ${response.status}`,
      status: response.status,
    });
    return;
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  let backend: Backend = 'wasm';
  let fallbackReason: string | null = null;
  if (preferWebGpu && (await hasWebGpu())) {
    try {
      const { mod, created } = await createSession('webgpu', bytes, wasmPaths);
      ort = mod;
      session = created;
      backend = 'webgpu';
    } catch (error) {
      fallbackReason = error instanceof Error ? error.message : String(error);
    }
  }
  if (!session) {
    const { mod, created } = await createSession('wasm', bytes, wasmPaths);
    ort = mod;
    session = created;
  }
  scope.postMessage({ type: 'loaded', backend, fallbackReason });
}

async function run(id: number, data: Float32Array, frames: number, dim: number): Promise<void> {
  if (!session || !ort) throw new Error('model not loaded');
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
  const msg = event.data;
  if (msg.type === 'load') {
    load(msg.modelUrl, msg.wasmPaths, msg.preferWebGpu).catch((error: unknown) => {
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
