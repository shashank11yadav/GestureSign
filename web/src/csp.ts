/**
 * GitHub Pages cannot send response headers, so the policy ships as a <meta>
 * tag that Vite adds to builds. frame-ancestors, report-uri and sandbox are
 * ignored in a meta policy, which is why they are not here.
 *
 * Every script, style, worker, model and WASM file comes from this origin.
 * 'wasm-unsafe-eval' is the single relaxation: browsers require it to compile
 * WebAssembly (MediaPipe and onnxruntime). Plain 'unsafe-eval' is not allowed.
 * Camera frames arrive through a MediaStream, which needs no media-src entry.
 */
export const CSP_DIRECTIVES: Readonly<Record<string, readonly string[]>> = {
  'default-src': ["'self'"],
  'script-src': ["'self'", "'wasm-unsafe-eval'"],
  'style-src': ["'self'"],
  'img-src': ["'self'"],
  'font-src': ["'self'"],
  'connect-src': ["'self'"],
  'worker-src': ["'self'"],
  'media-src': ["'self'"],
  'manifest-src': ["'self'"],
  'frame-src': ["'none'"],
  'object-src': ["'none'"],
  'base-uri': ["'self'"],
  'form-action': ["'none'"],
};

export function buildCsp(directives = CSP_DIRECTIVES): string {
  return Object.entries(directives)
    .map(([name, values]) => `${name} ${values.join(' ')}`)
    .join('; ');
}
