import { describe, expect, it } from 'vitest';
import {
  MAX_GESTURES,
  MAX_TAKES,
  MIN_TAKES,
  addTake,
  cosine,
  matchGesture,
  prototypeOf,
  type Gesture,
} from './prototypes';

function unit(...values: number[]): Float32Array {
  const n = Math.hypot(...values);
  return Float32Array.from(values, (v) => v / n);
}

function gesture(name: string, takes: Float32Array[]): Gesture {
  return { id: name, name, takes };
}

const thresholds = { minSimilarity: 0.8, margin: 0.05 };

describe('prototypes', () => {
  it('is the normalised mean of the takes', () => {
    const p = prototypeOf([unit(1, 0), unit(0, 1)]);
    expect(Array.from(p)).toEqual([expect.closeTo(Math.SQRT1_2), expect.closeTo(Math.SQRT1_2)]);
  });

  it('matches the nearest prototype by cosine', () => {
    const wave = gesture('wave', [unit(1, 0, 0), unit(0.95, 0.1, 0), unit(1, 0, 0.05)]);
    const fist = gesture('fist', [unit(0, 1, 0), unit(0.05, 1, 0), unit(0, 0.97, 0.1)]);
    const m = matchGesture(unit(0.98, 0.05, 0), [wave, fist], thresholds);
    expect(m.kind).toBe('match');
    if (m.kind !== 'match') return;
    expect(m.gesture.name).toBe('wave');
    expect(m.similarity).toBeGreaterThan(0.9);
  });

  it('is not sure when nothing is similar enough', () => {
    const wave = gesture('wave', [unit(1, 0, 0), unit(1, 0, 0), unit(1, 0, 0)]);
    const m = matchGesture(unit(0, 0, 1), [wave], thresholds);
    expect(m.kind).toBe('not-sure');
  });

  it('is not sure when two gestures are nearly tied', () => {
    const a = gesture('a', [unit(1, 0.1), unit(1, 0.1), unit(1, 0.1)]);
    const b = gesture('b', [unit(1, -0.1), unit(1, -0.1), unit(1, -0.1)]);
    const m = matchGesture(unit(1, 0), [a, b], thresholds);
    expect(m.kind).toBe('not-sure');
  });

  it('ignores gestures with too few takes', () => {
    const half = gesture('half', [unit(1, 0)]);
    expect(matchGesture(unit(1, 0), [half], thresholds).kind).toBe('empty');
    expect(MIN_TAKES).toBe(3);
  });

  it('caps takes per gesture', () => {
    let g = gesture('g', []);
    for (let i = 0; i < MAX_TAKES + 2; i++) g = addTake(g, unit(1, i));
    expect(g.takes).toHaveLength(MAX_TAKES);
    expect(MAX_GESTURES).toBe(8);
  });

  it('computes cosine for unit and non-unit vectors', () => {
    expect(cosine(Float32Array.of(2, 0), Float32Array.of(5, 0))).toBeCloseTo(1);
    expect(cosine(Float32Array.of(1, 0), Float32Array.of(0, 3))).toBeCloseTo(0);
  });
});
