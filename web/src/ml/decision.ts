/**
 * Turning model logits into what the page says. The rule is to never show a
 * confident wrong label: below the confidence threshold calibrated on
 * held-out signers (model-meta.json), the answer is "Not sure".
 */

export interface Calibration {
  temperature: number;
  notSureBelow: number;
}

export interface Guess {
  index: number;
  probability: number;
}

export type DecisionKind = 'sign' | 'not-sure' | 'nothing';

export interface Decision {
  kind: DecisionKind;
  /** Up to three guesses among shipped signs, best first (shown even when not sure). */
  top: Guess[];
  reason?: 'low-confidence' | 'partial';
}

/** Softmax over the shipped signs only; unshipped signs get probability 0. */
export function softmaxOver(
  logits: ArrayLike<number>,
  shipped: readonly boolean[],
  temperature: number,
): Float64Array {
  const out = new Float64Array(logits.length);
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; i++) {
    if (shipped[i]) max = Math.max(max, (logits[i] ?? Number.NaN) / temperature);
  }
  if (!Number.isFinite(max)) return out.fill(Number.NaN);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    if (!shipped[i]) continue;
    const e = Math.exp((logits[i] ?? Number.NaN) / temperature - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) / sum;
  return out;
}

export function decide(
  logits: ArrayLike<number>,
  shipped: readonly boolean[],
  calibration: Calibration,
  quality: { partial?: boolean } = {},
  k = 3,
): Decision {
  if (!shipped.some(Boolean)) return { kind: 'nothing', top: [] };
  for (let i = 0; i < logits.length; i++) {
    if (!Number.isFinite(logits[i] ?? Number.NaN)) return { kind: 'nothing', top: [] };
  }
  const probs = softmaxOver(logits, shipped, calibration.temperature);
  const top = Array.from(probs, (probability, index) => ({ index, probability }))
    .filter((g) => shipped[g.index])
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k);
  if (quality.partial) return { kind: 'not-sure', top, reason: 'partial' };
  const best = top[0]?.probability ?? 0;
  if (best < calibration.notSureBelow) return { kind: 'not-sure', top, reason: 'low-confidence' };
  return { kind: 'sign', top };
}

/** Practice mode: how strongly the model saw the target sign (0..1). */
export function practiceScore(
  logits: ArrayLike<number>,
  shipped: readonly boolean[],
  target: number,
  calibration: Calibration,
): number {
  if (!shipped[target]) return 0;
  const p = softmaxOver(logits, shipped, calibration.temperature)[target] ?? 0;
  return Number.isFinite(p) ? p : 0;
}
