/**
 * "Teach your own": nearest-prototype matching on the model's embeddings.
 *
 * Each gesture's prototype is the normalised mean of its takes' embeddings. A
 * new attempt matches the prototype with the highest cosine similarity, but
 * only if that similarity clears `minSimilarity` and beats the runner-up by
 * `margin`; otherwise the answer is "Not sure". Both thresholds come from
 * few-shot episodes on held-out signers (model-meta.json, `teach`).
 */

export const MIN_TAKES = 3;
export const MAX_TAKES = 5;
export const MAX_GESTURES = 8;
export const MAX_NAME_LENGTH = 40;

export interface Gesture {
  id: string;
  name: string;
  takes: Float32Array[];
}

export interface TeachThresholds {
  minSimilarity: number;
  margin: number;
}

export type GestureMatch =
  | { kind: 'match'; gesture: Gesture; similarity: number; runnerUp: number }
  | { kind: 'not-sure'; gesture: Gesture; similarity: number; runnerUp: number }
  | { kind: 'empty' };

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

export function prototypeOf(takes: readonly Float32Array[]): Float32Array {
  const dim = takes[0]?.length ?? 0;
  const mean = new Float64Array(dim);
  for (const take of takes) for (let i = 0; i < dim; i++) mean[i] = (mean[i] ?? 0) + (take[i] ?? 0);
  const norm = Math.hypot(...mean) || 1;
  return Float32Array.from(mean, (v) => v / norm);
}

export function isReady(g: Gesture): boolean {
  return g.takes.length >= MIN_TAKES;
}

export function addTake(g: Gesture, embedding: Float32Array): Gesture {
  if (g.takes.length >= MAX_TAKES) return g;
  return { ...g, takes: [...g.takes, embedding] };
}

export function matchGesture(
  embedding: ArrayLike<number>,
  gestures: readonly Gesture[],
  thresholds: TeachThresholds,
): GestureMatch {
  const ready = gestures.filter(isReady);
  if (ready.length === 0) return { kind: 'empty' };
  const scored = ready
    .map((gesture) => ({ gesture, similarity: cosine(embedding, prototypeOf(gesture.takes)) }))
    .sort((a, b) => b.similarity - a.similarity);
  const [best, second] = scored;
  if (!best) return { kind: 'empty' };
  const runnerUp = second?.similarity ?? -1;
  const confident =
    best.similarity >= thresholds.minSimilarity &&
    (second === undefined || best.similarity - runnerUp >= thresholds.margin);
  return { kind: confident ? 'match' : 'not-sure', ...best, runnerUp };
}
