import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defaultClientConditions, type Plugin } from 'vite';
import { defineConfig } from 'vitest/config';
import { buildCsp } from './src/csp.ts';

// GitHub Pages serves the site from /<repo>/.
const BASE = '/GestureSign/';

/**
 * Adds the CSP meta tag to builds only (production and the e2e test build).
 * The dev server injects inline scripts for hot reload, which the policy
 * would block.
 */
function contentSecurityPolicy(): Plugin {
  return {
    name: 'gesturesign:csp-meta',
    apply: 'build',
    transformIndexHtml: {
      order: 'post',
      handler: () => [
        {
          tag: 'meta',
          attrs: { 'http-equiv': 'Content-Security-Policy', content: buildCsp() },
          injectTo: 'head-prepend',
        },
      ],
    },
  };
}

export default defineConfig({
  base: BASE,
  plugins: [react(), tailwindcss(), contentSecurityPolicy()],
  resolve: {
    // onnxruntime-web's "extern wasm" builds load their WASM runtime from
    // ort.env.wasm.wasmPaths (public/vendor/ort) instead of bundling a copy.
    conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions],
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    sourcemap: false,
    // Keep every asset a file; inlined data: URIs would need a looser CSP.
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 1200,
  },
  server: { port: 5330, strictPort: true },
  preview: { port: 5331, strictPort: true },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
});
