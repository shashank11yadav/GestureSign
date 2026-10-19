/** Page-side handle on the MediaPipe worker: feeds camera frames, one in flight at a time. */
import type { FeatureSet } from './features';
import type { TrackedFrame } from './recognizer';
import type { Delegate } from './tracker-core';
import type { TrackerWorkerIn, TrackerWorkerOut } from './tracker-protocol';

export interface TrackerFrame extends TrackedFrame {
  ms: number;
}

export class TrackerClient {
  private running = false;
  private busy = false;
  private onFrame: ((frame: TrackerFrame) => void) | null = null;

  private constructor(
    private readonly worker: Worker,
    readonly delegate: Delegate,
  ) {
    worker.onmessage = (event: MessageEvent<TrackerWorkerOut>) => {
      const msg = event.data;
      if (msg.type === 'landmarks') {
        this.busy = false;
        if (this.running)
          this.onFrame?.({ frame: msg.frame, t: msg.t, poses: msg.poses, ms: msg.ms });
      } else if (msg.type === 'frame-error') {
        this.busy = false;
      }
    };
  }

  static async create(featureSet: FeatureSet): Promise<TrackerClient> {
    const worker = new Worker(new URL('./tracker.worker.ts', import.meta.url), { type: 'module' });
    const reply = await new Promise<TrackerWorkerOut>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<TrackerWorkerOut>) => {
        resolve(event.data);
      };
      worker.onerror = (event) => {
        reject(new Error(event.message || 'the landmark worker failed to start'));
      };
      const msg: TrackerWorkerIn = { type: 'init', featureSet, delegate: 'GPU' };
      worker.postMessage(msg);
    }).catch((error: unknown) => {
      worker.terminate();
      throw error;
    });
    if (reply.type !== 'ready') {
      worker.terminate();
      throw new Error(reply.type === 'init-error' ? reply.message : 'unexpected reply');
    }
    return new TrackerClient(worker, reply.delegate);
  }

  /** Track `video` until stop(); frames that arrive while one is being processed are skipped. */
  start(video: HTMLVideoElement, onFrame: (frame: TrackerFrame) => void): void {
    this.onFrame = onFrame;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      if (!this.busy && video.readyState >= 2 && video.videoWidth > 0) {
        this.busy = true;
        const t = performance.now();
        createImageBitmap(video)
          .then((bitmap) => {
            const msg: TrackerWorkerIn = { type: 'frame', bitmap, t };
            this.worker.postMessage(msg, [bitmap]);
          })
          .catch(() => {
            this.busy = false;
          });
      }
      if ('requestVideoFrameCallback' in video) video.requestVideoFrameCallback(tick);
      else requestAnimationFrame(tick);
    };
    tick();
  }

  stop(): void {
    this.running = false;
    this.onFrame = null;
  }

  dispose(): void {
    this.stop();
    this.worker.terminate();
  }
}
