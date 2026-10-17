/**
 * Cuts a continuous landmark stream into single-sign segments.
 *
 * A segment starts when a hand is present and moving, and ends when the hands
 * are gone, or held still, for `gapMs` (300 ms). Short dropouts inside a sign
 * are bridged. After a "still" ending the segmenter waits for new motion, so
 * resting a hand in view does not produce a stream of empty guesses, and a
 * quick movement out of a still hold that ends with the hands leaving the
 * frame is taken as the hands dropping after the sign, not as a new sign.
 * Motion is measured in shoulder widths per second, so it does not depend on
 * how far the signer sits from the camera or on the frame rate.
 */
import { computeAnchor } from './features';
import {
  LEFT_HAND_START,
  N_HAND,
  POSE_LEFT_SHOULDER,
  POSE_RIGHT_SHOULDER,
  POSE_START,
  RIGHT_HAND_START,
  handsMissing,
  type Frame,
} from './landmarks';

export interface SegmenterOptions {
  /** Hands absent or still for this long ends a sign. */
  gapMs: number;
  /** Hand speed (shoulder widths per second) above which the hand counts as moving. */
  moveSpeed: number;
  /** Segments with fewer frames are discarded as noise. */
  minFrames: number;
  /** A segment is closed after this long even if signing continues. */
  maxMs: number;
  /** Frames with hands just before motion starts are kept (hands entering). */
  prerollMs: number;
  /**
   * A segment that starts from a still hold and ends with the hands leaving
   * in less than this is the hands dropping after a held sign; it is discarded.
   */
  leaveMs: number;
}

export const DEFAULT_SEGMENTER: SegmenterOptions = {
  gapMs: 300,
  moveSpeed: 0.6,
  minFrames: 8,
  maxMs: 4000,
  prerollMs: 200,
  leaveMs: 500,
};

export interface Segment {
  frames: Frame[];
  timestamps: number[];
  /** Frame rate estimated from the timestamps. */
  fps: number;
  durationMs: number;
}

export type EndReason = 'absent' | 'still' | 'max' | 'flush';

export type SegmentEvent =
  | { type: 'start'; t: number }
  | { type: 'end'; reason: EndReason; segment: Segment }
  | { type: 'discard'; reason: 'too-short' | 'hands-leaving' };

export type SegmenterState = 'idle' | 'signing' | 'resting';

interface Sample {
  frame: Frame;
  t: number;
}

const HAND_STARTS = [LEFT_HAND_START, RIGHT_HAND_START];

/** Mean displacement of hand points seen in both frames, in shoulder widths per second. */
export function handSpeed(prev: Sample, next: Sample): number {
  const dt = (next.t - prev.t) / 1000;
  if (!(dt > 0)) return 0;
  const { scale } = computeAnchor([next.frame]);
  let total = 0;
  let count = 0;
  for (const start of HAND_STARTS) {
    for (let i = 0; i < N_HAND; i++) {
      const k = (start + i) * 3;
      const dx = (next.frame[k] ?? Number.NaN) - (prev.frame[k] ?? Number.NaN);
      const dy = (next.frame[k + 1] ?? Number.NaN) - (prev.frame[k + 1] ?? Number.NaN);
      const d = Math.hypot(dx, dy);
      if (Number.isFinite(d)) {
        total += d;
        count++;
      }
    }
  }
  return count ? total / count / scale / dt : 0;
}

export class Segmenter {
  private readonly opts: SegmenterOptions;
  private samples: Sample[] = [];
  private recent: Sample[] = [];
  private last: Sample | null = null;
  private speed = 0;
  private quietSince: number | null = null;
  private quietReason: 'absent' | 'still' = 'absent';
  /** The open segment started from a still hold (state 'resting'). */
  private fromRest = false;
  state: SegmenterState = 'idle';

  constructor(options: Partial<SegmenterOptions> = {}) {
    this.opts = { ...DEFAULT_SEGMENTER, ...options };
  }

  reset(): void {
    this.samples = [];
    this.recent = [];
    this.last = null;
    this.speed = 0;
    this.quietSince = null;
    this.fromRest = false;
    this.state = 'idle';
  }

  /** Feed one frame; returns an event when a segment starts, ends or is discarded. */
  push(frame: Frame, t: number, flags: { crowded?: boolean } = {}): SegmentEvent | null {
    const sample = { frame, t };
    const present = !flags.crowded && !handsMissing(frame);
    if (present && this.last && !handsMissing(this.last.frame)) {
      // Light smoothing so one jittery frame neither starts nor ends a sign.
      this.speed = 0.5 * this.speed + 0.5 * handSpeed(this.last, sample);
    } else if (!present) {
      this.speed = 0;
    }
    this.last = present ? sample : { frame, t };
    const moving = present && this.speed >= this.opts.moveSpeed;

    if (this.state !== 'signing') {
      if (!present) this.state = 'idle';
      this.recent = present
        ? [...this.recent, sample].filter((s) => t - s.t <= this.opts.prerollMs)
        : [];
      if (!moving) return null;
      this.fromRest = this.state === 'resting';
      this.state = 'signing';
      this.samples = [...this.recent];
      this.quietSince = null;
      return { type: 'start', t };
    }

    this.samples.push(sample);
    const first = this.samples[0];
    if (first && t - first.t >= this.opts.maxMs) {
      const event = this.finish('max', this.samples.length);
      this.recent = [];
      return event;
    }
    if (moving) {
      this.quietSince = null;
      return null;
    }
    if (this.quietSince === null) {
      this.quietSince = t;
      this.quietReason = present ? 'still' : 'absent';
    }
    if (t - this.quietSince < this.opts.gapMs) return null;
    // Keep frames up to the start of the quiet gap; a still hold keeps its first frame.
    const quietStart = this.quietSince;
    const keep = this.samples.filter((s) => s.t < quietStart).length;
    const reason = this.quietReason;
    if (reason === 'absent' && this.fromRest) {
      const kept = this.samples.slice(0, keep);
      const span = (kept[kept.length - 1]?.t ?? 0) - (kept[0]?.t ?? 0);
      if (span < this.opts.leaveMs) {
        this.finish(reason, 0);
        return { type: 'discard', reason: 'hands-leaving' };
      }
    }
    const event = this.finish(reason, reason === 'still' ? keep + 1 : keep);
    this.state = present ? 'resting' : 'idle';
    return event;
  }

  /** End any open segment now (e.g. when a replay clip finishes). */
  flush(): SegmentEvent | null {
    if (this.state !== 'signing') return null;
    const keep =
      this.quietSince === null
        ? this.samples.length
        : this.samples.filter((s) => s.t < (this.quietSince ?? 0)).length;
    return this.finish('flush', keep);
  }

  private finish(reason: EndReason, keep: number): SegmentEvent {
    const kept = this.samples.slice(0, Math.max(0, keep));
    this.samples = [];
    this.quietSince = null;
    this.state = 'idle';
    if (kept.length < this.opts.minFrames) return { type: 'discard', reason: 'too-short' };
    const first = kept[0]?.t ?? 0;
    const lastT = kept[kept.length - 1]?.t ?? first;
    const durationMs = lastT - first;
    const fps = durationMs > 0 ? ((kept.length - 1) * 1000) / durationMs : 30;
    return {
      type: 'end',
      reason,
      segment: {
        frames: kept.map((s) => s.frame),
        timestamps: kept.map((s) => s.t),
        fps,
        durationMs,
      },
    };
  }
}

export type FrameStatus = 'nothing' | 'no-hands' | 'crowded' | 'partial' | 'ok';

function shoulderInView(frame: Frame, index: number): boolean {
  const k = (POSE_START + index) * 3;
  const x = frame[k] ?? Number.NaN;
  const y = frame[k + 1] ?? Number.NaN;
  return x >= -0.02 && x <= 1.02 && y >= -0.02 && y <= 1.02;
}

/**
 * What the camera sees, for the status line and for refusing to guess.
 * `poses` is how many people the pose landmarker found.
 */
export function frameStatus(frame: Frame, poses: number): FrameStatus {
  if (poses > 1) return 'crowded';
  const noHands = handsMissing(frame);
  if (noHands && poses === 0) return 'nothing';
  if (noHands) return 'no-hands';
  if (!shoulderInView(frame, POSE_LEFT_SHOULDER) || !shoulderInView(frame, POSE_RIGHT_SHOULDER)) {
    return 'partial';
  }
  return 'ok';
}
