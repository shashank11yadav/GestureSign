/**
 * Landmark features, a line-for-line port of pipeline/src/gesturesign/features.py
 * and resample.py. shared/fixtures/features.json is checked by both test
 * suites, so any drift between the two fails CI.
 *
 * Arithmetic is done in float64 (JS numbers) in the same order as NumPy and
 * stored as float32, like the Python side.
 */
import {
  FRAME_LENGTH,
  HAND_MIDDLE_MCP,
  HAND_WRIST,
  LEFT_HAND_START,
  LIPS,
  LIPS_MIRROR,
  N_HAND,
  POSE_LEFT_SHOULDER,
  POSE_RIGHT_SHOULDER,
  POSE_START,
  POSE_UPPER,
  POSE_UPPER_MIRROR,
  RIGHT_HAND_START,
  type Frame,
} from './landmarks';

export type FeatureSet = 'hands_pose' | 'hands_pose_lips';
export const FEATURE_VERSION = 2;

export interface Anchor {
  cx: number;
  cy: number;
  scale: number;
  /** Divisor for y: the shoulder width times the face's vertical proportion (see features.py). */
  yScale: number;
}

export const DEFAULT_ANCHOR: Readonly<Anchor> = { cx: 0.5, cy: 0.5, scale: 0.25, yScale: 0.25 };
const MIN_SCALE = 1e-3;
/** Median eye-to-mouth / ear-to-ear distance in asl-signs (normalised coordinates). */
export const FACE_RATIO_REF = 0.43;
const FACE_RATIO_MIN = 0.2;
const FACE_RATIO_MAX = 1.2;

export interface FeatureLayout {
  frameDim: number;
  sequenceDim: number;
  positions: number;
}

export function featureLayout(featureSet: FeatureSet): FeatureLayout {
  const hand = N_HAND * 2;
  const pose = POSE_UPPER.length * 2;
  const lips = featureSet === 'hands_pose_lips' ? LIPS.length * 2 : 0;
  const positions = 2 * hand + pose + lips;
  const parts = lips ? 4 : 3;
  const frameDim = positions + 2 * hand + parts;
  return { frameDim, sequenceDim: frameDim + positions, positions };
}

const x = (f: Frame, point: number) => f[point * 3] ?? Number.NaN;
const y = (f: Frame, point: number) => f[point * 3 + 1] ?? Number.NaN;

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  if (sorted.length % 2) return sorted[mid] ?? Number.NaN;
  return ((sorted[mid - 1] ?? Number.NaN) + (sorted[mid] ?? Number.NaN)) / 2;
}

/**
 * Median eye-to-mouth over ear-to-ear distance from pose points, or null if never seen.
 * MediaPipe normalises x by image width and y by height, so this ratio measures
 * how the camera's aspect ratio stretches vertical distances.
 */
export function faceRatio(frames: readonly Frame[]): number | null {
  const ratios: number[] = [];
  const p = (i: number) => POSE_START + i;
  for (const f of frames) {
    const eyeX = (x(f, p(2)) + x(f, p(5))) / 2;
    const eyeY = (y(f, p(2)) + y(f, p(5))) / 2;
    const mouthX = (x(f, p(9)) + x(f, p(10))) / 2;
    const mouthY = (y(f, p(9)) + y(f, p(10))) / 2;
    const vertical = Math.hypot(eyeX - mouthX, eyeY - mouthY);
    const horizontal = Math.hypot(x(f, p(7)) - x(f, p(8)), y(f, p(7)) - y(f, p(8)));
    if (Number.isFinite(vertical) && Number.isFinite(horizontal) && horizontal > 1e-4) {
      ratios.push(vertical / horizontal);
    }
  }
  if (ratios.length === 0) return null;
  return Math.min(Math.max(median(ratios), FACE_RATIO_MIN), FACE_RATIO_MAX);
}

/** Shoulder midpoint and width (medians over frames showing both shoulders) and y scale. */
export function computeAnchor(frames: readonly Frame[]): Anchor {
  const cxs: number[] = [];
  const cys: number[] = [];
  const widths: number[] = [];
  const l = POSE_START + POSE_LEFT_SHOULDER;
  const r = POSE_START + POSE_RIGHT_SHOULDER;
  for (const f of frames) {
    const lx = x(f, l);
    const ly = y(f, l);
    const rx = x(f, r);
    const ry = y(f, r);
    if (![lx, ly, rx, ry].every(Number.isFinite)) continue;
    cxs.push((lx + rx) / 2);
    cys.push((ly + ry) / 2);
    widths.push(Math.hypot(lx - rx, ly - ry));
  }
  let cx = DEFAULT_ANCHOR.cx;
  let cy = DEFAULT_ANCHOR.cy;
  let scale = DEFAULT_ANCHOR.scale;
  if (cxs.length > 0) {
    scale = median(widths);
    if (!(scale > MIN_SCALE)) scale = DEFAULT_ANCHOR.scale;
    cx = median(cxs);
    cy = median(cys);
  }
  const ratio = faceRatio(frames);
  return { cx, cy, scale, yScale: ratio === null ? scale : (scale * ratio) / FACE_RATIO_REF };
}

function handSeen(f: Frame, start: number): boolean {
  for (let i = 0; i < N_HAND; i++) if (Number.isFinite(x(f, start + i))) return true;
  return false;
}

function wristTravel(frames: readonly Frame[], start: number): number {
  let total = 0;
  for (let t = 1; t < frames.length; t++) {
    const a = frames[t - 1];
    const b = frames[t];
    if (!a || !b) continue;
    const step =
      Math.abs(x(b, start + HAND_WRIST) - x(a, start + HAND_WRIST)) +
      Math.abs(y(b, start + HAND_WRIST) - y(a, start + HAND_WRIST));
    if (Number.isFinite(step)) total += step;
  }
  return total;
}

/**
 * The signing hand is the one seen in more frames; on a tie, the one whose
 * wrist travels further. Both rules swap under a horizontal flip, so a
 * mirrored webcam image canonicalises to the same features.
 */
export function leftIsDominant(frames: readonly Frame[]): boolean {
  let left = 0;
  let right = 0;
  for (const f of frames) {
    if (handSeen(f, LEFT_HAND_START)) left++;
    if (handSeen(f, RIGHT_HAND_START)) right++;
  }
  if (left !== right) return left > right;
  return wristTravel(frames, LEFT_HAND_START) > wristTravel(frames, RIGHT_HAND_START);
}

function range(start: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => start + i);
}

interface Indices {
  dominant: number[];
  other: number[];
  body: number[];
  nPose: number;
}

function indices(featureSet: FeatureSet, mirror: boolean): Indices {
  const left = range(LEFT_HAND_START, N_HAND);
  const right = range(RIGHT_HAND_START, N_HAND);
  const pose = (mirror ? POSE_UPPER_MIRROR : POSE_UPPER).map((i) => POSE_START + i);
  const lips = featureSet === 'hands_pose_lips' ? [...(mirror ? LIPS_MIRROR : LIPS)] : [];
  return {
    dominant: mirror ? left : right,
    other: mirror ? right : left,
    body: [...pose, ...lips],
    nPose: POSE_UPPER.length,
  };
}

/** Per-frame features (NaN zeroed) plus the raw normalised positions (NaN kept). */
function frameBlock(
  f: Frame,
  featureSet: FeatureSet,
  anchor: Anchor,
  mirror: boolean,
): { features: Float64Array; positions: Float64Array } {
  const idx = indices(featureSet, mirror);
  const layout = featureLayout(featureSet);
  const sign = mirror ? -1 : 1;
  const positions = new Float64Array(layout.positions);
  let p = 0;
  for (const point of [...idx.dominant, ...idx.other, ...idx.body]) {
    positions[p++] = ((x(f, point) - anchor.cx) / anchor.scale) * sign;
    positions[p++] = (y(f, point) - anchor.cy) / anchor.yScale;
  }
  const features = new Float64Array(layout.frameDim);
  features.set(positions, 0);
  let o = layout.positions;
  for (const handOffset of [0, N_HAND * 2]) {
    const wx = positions[handOffset + HAND_WRIST * 2] ?? Number.NaN;
    const wy = positions[handOffset + HAND_WRIST * 2 + 1] ?? Number.NaN;
    const refX = (positions[handOffset + HAND_MIDDLE_MCP * 2] ?? Number.NaN) - wx;
    const refY = (positions[handOffset + HAND_MIDDLE_MCP * 2 + 1] ?? Number.NaN) - wy;
    let size = Math.hypot(refX, refY);
    if (!(Number.isFinite(size) && size > 1e-6)) size = 1;
    for (let i = 0; i < N_HAND; i++) {
      features[o++] = ((positions[handOffset + i * 2] ?? Number.NaN) - wx) / size;
      features[o++] = ((positions[handOffset + i * 2 + 1] ?? Number.NaN) - wy) / size;
    }
  }
  const anyX = (from: number, count: number) => {
    for (let i = 0; i < count; i++) {
      if (Number.isFinite(positions[(from + i) * 2] ?? Number.NaN)) return 1;
    }
    return 0;
  };
  features[o++] = anyX(0, N_HAND);
  features[o++] = anyX(N_HAND, N_HAND);
  features[o++] = anyX(2 * N_HAND, idx.nPose);
  if (featureSet === 'hands_pose_lips') features[o] = anyX(2 * N_HAND + idx.nPose, LIPS.length);
  for (let i = 0; i < features.length; i++) {
    if (!Number.isFinite(features[i] ?? Number.NaN)) features[i] = 0;
  }
  return { features, positions };
}

/**
 * Features of one 543 x 3 frame. Without an anchor and mirror flag it is
 * treated as a sequence of one: its own shoulders anchor it, and it is
 * mirrored when only the left hand shows.
 */
export function frameFeatures(
  points: Frame,
  featureSet: FeatureSet,
  anchor?: Anchor,
  mirror?: boolean,
): Float32Array {
  const a = anchor ?? computeAnchor([points]);
  const m = mirror ?? leftIsDominant([points]);
  return Float32Array.from(frameBlock(points, featureSet, a, m).features);
}

/** Per-frame features plus velocity, flattened row-major as [T, sequenceDim]. */
export function sequenceFeatures(frames: readonly Frame[], featureSet: FeatureSet): Float32Array {
  const layout = featureLayout(featureSet);
  const out = new Float32Array(frames.length * layout.sequenceDim);
  if (frames.length === 0) return out;
  const anchor = computeAnchor(frames);
  const mirror = leftIsDominant(frames);
  let previous: Float64Array | null = null;
  frames.forEach((f, t) => {
    const { features, positions } = frameBlock(f, featureSet, anchor, mirror);
    const row = t * layout.sequenceDim;
    out.set(features, row);
    if (previous) {
      for (let i = 0; i < layout.positions; i++) {
        const now = positions[i] ?? Number.NaN;
        const before = previous[i] ?? Number.NaN;
        out[row + layout.frameDim + i] =
          Number.isFinite(now) && Number.isFinite(before) ? now - before : 0;
      }
    }
    previous = positions;
  });
  return out;
}

export function resampledLength(T: number, srcFps: number, dstFps: number): number {
  if (T <= 1) return T;
  return Math.floor(((T - 1) * dstFps) / srcFps + 1e-9) + 1;
}

function interpolate(seq: readonly Float32Array[], positions: number[]): Float32Array[] {
  const T = seq.length;
  return positions.map((u) => {
    const i0 = Math.min(Math.floor(u), T - 1);
    const i1 = Math.min(i0 + 1, T - 1);
    const w = u - i0;
    const a = seq[i0];
    const b = seq[i1];
    if (!a || !b) throw new Error('interpolate: index out of range');
    const out = new Float32Array(a.length);
    for (let k = 0; k < a.length; k++) {
      const va = a[k] ?? Number.NaN;
      const vb = b[k] ?? Number.NaN;
      out[k] = Number.isFinite(va) && Number.isFinite(vb) ? va + (vb - va) * w : w < 0.5 ? va : vb;
    }
    return out;
  });
}

/** Resample frames recorded at srcFps to dstFps (first frame kept, never past the last). */
export function resample(
  seq: readonly Float32Array[],
  srcFps: number,
  dstFps: number,
): Float32Array[] {
  const T = seq.length;
  const n = resampledLength(T, srcFps, dstFps);
  if ((n === T && srcFps === dstFps) || T <= 1) return [...seq];
  return interpolate(
    seq,
    Array.from({ length: n }, (_, k) => (k * srcFps) / dstFps),
  );
}

/** Squeeze sequences longer than maxLen onto exactly maxLen frames. */
export function fitLength(seq: readonly Float32Array[], maxLen: number): Float32Array[] {
  const T = seq.length;
  if (maxLen >= T) return [...seq];
  return interpolate(
    seq,
    Array.from({ length: maxLen }, (_, k) => (k * (T - 1)) / (maxLen - 1)),
  );
}

export interface InputConfig {
  fps: number;
  maxFrames: number;
  featureSet: FeatureSet;
}

export interface ModelInput {
  data: Float32Array;
  frames: number;
  dim: number;
}

/** A captured segment -> the model's input: resample, squeeze, features. */
export function modelInput(frames: readonly Frame[], srcFps: number, cfg: InputConfig): ModelInput {
  const fitted = fitLength(resample(frames, srcFps, cfg.fps), cfg.maxFrames);
  const dim = featureLayout(cfg.featureSet).sequenceDim;
  return { data: sequenceFeatures(fitted, cfg.featureSet), frames: fitted.length, dim };
}

export { FRAME_LENGTH };
