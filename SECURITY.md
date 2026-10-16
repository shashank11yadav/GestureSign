# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://github.com/shashank11yadav/GestureSign/security/advisories/new)
rather than a public issue. I will acknowledge a report within a week.

## Scope

The deployed site is a static bundle (HTML, JS, CSS, WASM, model files) on
GitHub Pages. It has no server, no accounts, no cookies, no analytics and no
third-party scripts. Camera frames are processed in the page and never leave
the device, so the realistic risks are in the build and supply chain:

- Dependencies are pinned (`web/package-lock.json`, `pipeline/uv.lock`) and
  updated by Dependabot. CI runs `npm audit --omit=dev` and `pip-audit`.
- GitHub Actions are pinned to commit SHAs and run with `contents: read` unless
  a job needs more (Pages deploy, CodeQL upload).
- The MediaPipe `.task` models are downloaded at build time by
  `web/scripts/fetch-assets.mjs` and checked against pinned SHA-256 hashes; a
  mismatch fails the build. The MediaPipe and onnxruntime WASM files are copied
  from the locked npm packages, so the site serves every file itself.
- The page ships a Content-Security-Policy that allows only same-origin
  scripts, styles, workers and fetches. `'wasm-unsafe-eval'` is the one
  relaxation, and it is required to compile WebAssembly.
- Custom gestures taught in the demo are stored as embedding vectors in the
  browser's IndexedDB, never as video, and can be cleared from the page.
