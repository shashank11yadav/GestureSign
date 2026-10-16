# A1: landmark benchmark

Decides which landmarks the model uses (spec 4.1). Written by `npm run bench`
(web/scripts/bench.mjs); re-run it to reproduce.

**Rule:** Holistic at least 15 fps with 4x CPU throttling -> `hands_pose_lips`,
otherwise `hands_pose` (Hand + Pose landmarkers, no lip features).

**Result:** Holistic ran at a median **16.8 fps** over 3 runs at 4x throttling, so
`FEATURE_SET = "hands_pose_lips"` (recorded in `pipeline/configs/default.toml`).

| Landmarks | CPU throttling | Landmark fps (median; runs) | Detect ms (mean) | Detect ms (p95) | Stub model ms | Camera fps | Detected pose / hands / face | MediaPipe delegate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Holistic | 1x | 21.4 | 39.3 | 66.8 | 2.8 | 30.0 | 100% / 99% / 100% | GPU |
| Holistic | 4x | 16.8 | 54.1 | 67.9 | 10.7 | 29.7 | 100% / 99% / 100% | GPU |
| Hand + Pose | 1x | 23.0 | 40.8 | 55.3 | 3.0 | 29.9 | 99% / 100% / 0% | GPU |
| Hand + Pose | 4x | 18.1 | 48.2 | 63.0 | 11.2 | 29.7 | 100% / 100% / 0% | GPU |

## Setup

- Machine: Apple M4 (10-core CPU and GPU), Chrome/156.0.0.0, headless.
- WebGL renderer: ANGLE (Apple, ANGLE Metal Renderer: Apple M4, Unspecified Version).
- Input: Chrome's fake camera playing a synthetic 640x480, 30 fps y4m clip of a
  drawn, front-facing figure moving two open hands (web/scripts/fake-camera.mjs;
  no real person). The "detected" column says how often MediaPipe found a body,
  hands and a face in it.
- MediaPipe runs on the main thread for this measurement, because DevTools CPU
  throttling slows the page's main thread; the app itself runs it in a worker.
- The stub model (same I/O as the trained one) runs every 5th frame on a
  32-frame window, also on the main thread.
- Each 4x row is the median of 3 runs (individual runs in brackets): run-to-run
  variation is a few fps, because other work on the machine competes for it.
- 10 s measured after a 4 s warm-up, per run, after one discarded
  run per landmark set (WASM compilation and GPU shader setup). "Camera fps" is
  how many frames the fake camera delivered; landmark fps cannot exceed it.

## Caveats

- The figure is drawn, not filmed. Where the detected rates are below 100%,
  Holistic skipped its face and hand stages on those frames, so its fps is an
  upper bound for a real signer. Re-run with a person in front of a real
  webcam (`npm run bench -- --real-camera`) to confirm before relying on a
  close call.
- 4x throttling of an M4 stands in for a mid-range laptop; GPU speed is not
  throttled.
