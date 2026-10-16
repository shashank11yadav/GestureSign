/**
 * The live pipeline, independent of where frames come from (camera, Replay or
 * the e2e injection hook): frame status -> segmenter -> features -> model ->
 * decision. Inference is injected so the logic is unit-testable without
 * workers or WASM.
 */
import { decide, type Decision } from './decision';
import { modelInput, type ModelInput } from './features';
import type { Frame } from './landmarks';
import type { ModelMeta, SignLabel } from './meta';
import { Segmenter, frameStatus, type FrameStatus, type Segment } from './segmenter';

export interface Inference {
  logits: Float32Array;
  embedding: Float32Array;
}

export type Infer = (input: ModelInput) => Promise<Inference>;

export interface RecognitionResult {
  decision: Decision;
  inference: Inference;
  segment: Segment;
  /** True when the signer's shoulders were out of view for most of the sign. */
  partial: boolean;
}

export type RecognizerEvent =
  | { type: 'status'; status: FrameStatus }
  | { type: 'signing' }
  | { type: 'discarded' }
  | { type: 'result'; result: RecognitionResult }
  | { type: 'error'; message: string };

export interface TrackedFrame {
  frame: Frame;
  /** Milliseconds, monotonic. */
  t: number;
  /** People the pose landmarker found (0, 1 or more). */
  poses: number;
}

export class Recognizer {
  private readonly segmenter = new Segmenter();
  private statuses: FrameStatus[] = [];
  private lastStatus: FrameStatus | null = null;
  private generation = 0;

  constructor(
    private readonly meta: ModelMeta,
    private readonly labels: readonly SignLabel[],
    private readonly infer: Infer,
    private readonly emit: (event: RecognizerEvent) => void,
  ) {}

  get shipped(): boolean[] {
    return this.labels.map((l) => l.shipped);
  }

  reset(): void {
    this.generation++;
    this.segmenter.reset();
    this.statuses = [];
    this.lastStatus = null;
  }

  push({ frame, t, poses }: TrackedFrame): void {
    const status = frameStatus(frame, poses);
    if (status !== this.lastStatus) {
      this.lastStatus = status;
      this.emit({ type: 'status', status });
    }
    const event = this.segmenter.push(frame, t, { crowded: status === 'crowded' });
    if (this.segmenter.state === 'signing') this.statuses.push(status);
    this.handle(event);
  }

  /** End an open sign now, e.g. when a replay clip finishes. */
  flush(): void {
    this.handle(this.segmenter.flush());
  }

  private handle(event: ReturnType<Segmenter['push']>): void {
    if (!event) return;
    if (event.type === 'start') {
      this.statuses = [];
      this.emit({ type: 'signing' });
      return;
    }
    const statuses = this.statuses;
    this.statuses = [];
    if (event.type === 'discard') {
      this.emit({ type: 'discarded' });
      return;
    }
    const partialFrames = statuses.filter((s) => s === 'partial').length;
    const partial = partialFrames > statuses.length / 2;
    void this.classify(event.segment, partial, this.generation);
  }

  private async classify(segment: Segment, partial: boolean, generation: number): Promise<void> {
    const input = modelInput(segment.frames, segment.fps, this.meta);
    if (input.frames < this.meta.minFrames) {
      this.emit({ type: 'discarded' });
      return;
    }
    try {
      const inference = await this.infer(input);
      if (generation !== this.generation) return; // the mode or source changed meanwhile
      const decision = decide(inference.logits, this.shipped, this.meta.calibration, { partial });
      this.emit({ type: 'result', result: { decision, inference, segment, partial } });
    } catch (error) {
      this.emit({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    }
  }
}
