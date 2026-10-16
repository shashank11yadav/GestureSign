/**
 * The live session: one landmark source at a time (camera, a replay clip, or
 * the e2e injection hook) feeding one Recognizer. Framework-free; React reads
 * it through useSyncExternalStore and draws frames through onFrame().
 */
import {
  CameraError,
  classifyCameraError,
  openCamera,
  stopStream,
  type CameraProblem,
} from '../lib/camera';
import { emptyFrame, type Frame } from '../ml/landmarks';
import { ModelClient, type LoadedModel } from '../ml/model-client';
import { Recognizer, type RecognitionResult, type RecognizerEvent } from '../ml/recognizer';
import type { FrameStatus } from '../ml/segmenter';
import { TrackerClient } from '../ml/tracker-client';
import type { ReplayClip } from '../replay/clips';

export type Source = 'off' | 'starting' | 'camera' | 'replay' | 'injected';

export type ModelState =
  | { status: 'loading' }
  | { status: 'ready'; model: LoadedModel }
  | { status: 'error'; message: string };

export interface SessionSnapshot {
  model: ModelState;
  source: Source;
  cameraProblem: CameraProblem | null;
  /** The camera was stopped because the tab was hidden. */
  paused: boolean;
  trackerError: string | null;
  status: FrameStatus | null;
  signing: boolean;
  replay: { clip: ReplayClip; done: boolean } | null;
  result: { id: number; result: RecognitionResult } | null;
  inferenceError: string | null;
}

export type FrameListener = (frame: Frame, source: Source) => void;
export type ResultListener = (result: RecognitionResult, id: number) => void;

/** Empty frames appended after a clip so the segmenter sees the hands leave. */
const TAIL_MS = 450;

export class Session {
  private snapshot: SessionSnapshot = {
    model: { status: 'loading' },
    source: 'off',
    cameraProblem: null,
    paused: false,
    trackerError: null,
    status: null,
    signing: false,
    replay: null,
    result: null,
    inferenceError: null,
  };
  private readonly listeners = new Set<() => void>();
  private readonly frameListeners = new Set<FrameListener>();
  private readonly resultListeners = new Set<ResultListener>();
  private recognizer: Recognizer | null = null;
  private tracker: Promise<TrackerClient> | null = null;
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private replayTimer: number | null = null;
  private resultId = 0;
  private clock = 0;
  private startToken = 0;

  constructor() {
    document.addEventListener('visibilitychange', () => {
      this.onVisibility();
    });
    window.addEventListener('pagehide', () => {
      this.stopCamera();
    });
  }

  // --- store plumbing -------------------------------------------------------
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): SessionSnapshot => this.snapshot;

  private set(patch: Partial<SessionSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const listener of this.listeners) listener();
  }

  onFrame(listener: FrameListener): () => void {
    this.frameListeners.add(listener);
    return () => this.frameListeners.delete(listener);
  }

  /** Called once per classified sign, whatever the mode. */
  onResult(listener: ResultListener): () => void {
    this.resultListeners.add(listener);
    return () => this.resultListeners.delete(listener);
  }

  // --- model ----------------------------------------------------------------
  async loadModel(): Promise<void> {
    if (this.snapshot.model.status === 'ready') return;
    this.set({ model: { status: 'loading' } });
    try {
      const model = await ModelClient.load();
      this.recognizer = new Recognizer(
        model.meta,
        model.labels,
        (input) => model.client.infer(input),
        (event) => {
          this.onRecognizer(event);
        },
      );
      this.set({ model: { status: 'ready', model } });
    } catch (error) {
      this.set({
        model: { status: 'error', message: error instanceof Error ? error.message : String(error) },
      });
    }
  }

  private onRecognizer(event: RecognizerEvent): void {
    switch (event.type) {
      case 'status':
        this.set({ status: event.status });
        break;
      case 'signing':
        this.set({ signing: true });
        break;
      case 'discarded':
        this.set({ signing: false });
        break;
      case 'result': {
        const id = ++this.resultId;
        this.set({ signing: false, inferenceError: null, result: { id, result: event.result } });
        for (const listener of this.resultListeners) listener(event.result, id);
        break;
      }
      case 'error':
        this.set({ signing: false, inferenceError: event.message });
        break;
    }
  }

  /** Forget any half-seen sign, e.g. when switching modes. */
  resetRecognition(): void {
    this.recognizer?.reset();
    this.set({ signing: false });
  }

  private push(frame: Frame, t: number, poses: number, source: Source): void {
    this.clock = Math.max(this.clock + 0.001, t);
    this.recognizer?.push({ frame, t: this.clock, poses });
    for (const listener of this.frameListeners) listener(frame, source);
  }

  // --- camera ---------------------------------------------------------------
  attachVideo(video: HTMLVideoElement | null): void {
    this.video = video;
  }

  async startCamera(): Promise<void> {
    const token = ++this.startToken;
    this.stopReplay();
    this.set({
      source: 'starting',
      cameraProblem: null,
      paused: false,
      trackerError: null,
      status: null,
    });
    const featureSet =
      this.snapshot.model.status === 'ready'
        ? this.snapshot.model.model.meta.featureSet
        : 'hands_pose_lips';
    // Start MediaPipe while the permission prompt is open; it is reused afterwards.
    const tracker = (this.tracker ??= TrackerClient.create(featureSet));
    tracker.catch(() => undefined);
    let stream: MediaStream;
    try {
      stream = await openCamera();
    } catch (error) {
      if (token !== this.startToken) return;
      const problem = error instanceof CameraError ? error.problem : classifyCameraError(error);
      this.set({ source: 'off', cameraProblem: problem });
      return;
    }
    if (token !== this.startToken) {
      stopStream(stream);
      return;
    }
    this.stream = stream;
    const video = this.video;
    if (!video) {
      this.stopCamera();
      return;
    }
    video.srcObject = stream;
    await video.play().catch(() => undefined);
    this.recognizer?.reset();
    this.set({ source: 'camera' });
    try {
      const client = await tracker;
      if (token !== this.startToken || this.snapshot.source !== 'camera') return;
      client.start(video, (f) => {
        this.push(f.frame, f.t, f.poses, 'camera');
      });
    } catch (error) {
      this.tracker = null;
      this.set({
        trackerError: `Hand tracking could not start: ${error instanceof Error ? error.message : String(error)}`,
      });
    }
  }

  stopCamera(): void {
    this.startToken++;
    void this.tracker
      ?.then((t) => {
        t.stop();
      })
      .catch(() => undefined);
    stopStream(this.stream);
    this.stream = null;
    if (this.video) this.video.srcObject = null;
    if (this.snapshot.source === 'camera' || this.snapshot.source === 'starting') {
      this.set({ source: 'off', status: null, signing: false });
    }
  }

  private onVisibility(): void {
    if (document.visibilityState === 'hidden' && this.snapshot.source === 'camera') {
      this.stopCamera();
      this.set({ paused: true });
    }
  }

  dismissCameraProblem(): void {
    this.set({ cameraProblem: null });
  }

  // --- replay and injection -------------------------------------------------
  playReplay(clip: ReplayClip): void {
    this.stopCamera();
    this.stopReplay();
    this.recognizer?.reset();
    this.set({
      source: 'replay',
      replay: { clip, done: false },
      cameraProblem: null,
      status: null,
    });
    const step = 1000 / clip.fps;
    const total = clip.frames.length + Math.ceil(TAIL_MS / step);
    let i = 0;
    const start = performance.now();
    const tick = () => {
      // Catch up on frames due by now, so slow timers do not slow the clip down.
      const due = Math.min(total, Math.floor((performance.now() - start) / step) + 1);
      while (i < due) {
        const frame = clip.frames[i] ?? emptyFrame();
        this.push(frame, start + i * step, i < clip.frames.length ? 1 : 0, 'replay');
        i++;
      }
      if (i >= total) {
        this.recognizer?.flush();
        this.replayTimer = null;
        this.set({ replay: { clip, done: true } });
        return;
      }
      this.replayTimer = window.setTimeout(tick, step / 2);
    };
    tick();
  }

  stopReplay(): void {
    if (this.replayTimer !== null) window.clearTimeout(this.replayTimer);
    this.replayTimer = null;
    if (this.snapshot.source === 'replay' || this.snapshot.source === 'injected') {
      this.set({ source: 'off', replay: null, status: null, signing: false });
    }
  }

  /** Test hook: push frames immediately with synthetic timestamps, then a quiet tail. */
  inject(frames: readonly Frame[], fps: number, poses = 1): void {
    if (this.snapshot.source === 'camera') this.stopCamera();
    if (this.snapshot.source !== 'injected') this.set({ source: 'injected', replay: null });
    const step = 1000 / fps;
    const t0 = this.clock + 1000;
    frames.forEach((frame, i) => {
      this.push(frame, t0 + i * step, poses, 'injected');
    });
    const tail = Math.ceil(TAIL_MS / step);
    for (let i = 0; i < tail; i++) {
      this.push(emptyFrame(), t0 + (frames.length + i) * step, 0, 'injected');
    }
  }
}
