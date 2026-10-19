/**
 * The files the page and its workers download, as fixed paths under the
 * site's base URL. Workers build these URLs themselves rather than taking them
 * from messages, so no message can point a download somewhere else, and every
 * URL is checked to stay on this site's origin.
 */
export const ASSETS = {
  /** Folder with MediaPipe's WASM runtime. */
  mediapipeWasm: 'vendor/mediapipe/wasm',
  /** Folder with the MediaPipe .task models. */
  mediapipeModels: 'vendor/mediapipe/models/',
  /** Folder with onnxruntime's WASM runtime. */
  ortRuntime: 'vendor/ort/',
  signModel: 'models/model.onnx',
  modelMeta: 'models/model-meta.json',
  labels: 'models/labels.json',
} as const;

export type AssetName = keyof typeof ASSETS;

/**
 * Absolute URL of a site asset. `location` is the page's or the worker
 * script's URL (only its origin is used); `base` is Vite's base path.
 */
export function assetUrl(
  name: AssetName,
  location: string = globalThis.location.href,
  base: string = import.meta.env.BASE_URL,
): string {
  if (!Object.hasOwn(ASSETS, name)) throw new Error(`unknown asset ${name}`);
  const origin = new URL(location).origin;
  const url = new URL(`${base}${ASSETS[name]}`, origin);
  if (url.origin !== origin) throw new Error(`${name} must load from the same origin as the page`);
  return url.href;
}
