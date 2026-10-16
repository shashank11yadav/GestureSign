import { describe, expect, it } from 'vitest';
import { decide, practiceScore, softmaxOver } from './decision';

const calibration = { temperature: 1, notSureBelow: 0.5 };
const allShipped = (n: number) => Array.from({ length: n }, () => true);

describe('decide', () => {
  it('names the top sign when it is confident', () => {
    const d = decide(Float32Array.of(0, 6, 1, 0), allShipped(4), calibration);
    expect(d.kind).toBe('sign');
    expect(d.top.map((g) => g.index)).toEqual([1, 2, 0]);
    expect(d.top[0]?.probability).toBeGreaterThan(0.9);
  });

  it('says "not sure" below the calibrated threshold but still lists the guesses', () => {
    const d = decide(Float32Array.of(1, 1.2, 1.1, 0.9), allShipped(4), calibration);
    expect(d.kind).toBe('not-sure');
    expect(d.top).toHaveLength(3);
  });

  it('only ranks shipped signs', () => {
    const shipped = [true, false, true, true];
    const d = decide(Float32Array.of(0, 9, 3, 0), shipped, calibration);
    expect(d.top.map((g) => g.index)).toEqual([2, 0, 3]);
    expect(d.top.reduce((s, g) => s + g.probability, 0)).toBeCloseTo(1);
  });

  it('applies the temperature before the threshold', () => {
    const logits = Float32Array.of(2, 0, 0, 0);
    expect(decide(logits, allShipped(4), calibration).kind).toBe('sign');
    expect(decide(logits, allShipped(4), { temperature: 4, notSureBelow: 0.5 }).kind).toBe(
      'not-sure',
    );
  });

  it('refuses to answer for a partly visible signer', () => {
    const d = decide(Float32Array.of(0, 9, 0, 0), allShipped(4), calibration, { partial: true });
    expect(d.kind).toBe('not-sure');
    expect(d.reason).toBe('partial');
  });

  it('treats broken outputs and an empty sign list as nothing to say', () => {
    expect(decide(Float32Array.of(Number.NaN, 1), allShipped(2), calibration).kind).toBe('nothing');
    expect(decide(Float32Array.of(1, 2), [false, false], calibration).kind).toBe('nothing');
  });

  it('a near-uniform untrained model is never confident', () => {
    const logits = Float32Array.from({ length: 250 }, (_, i) => Math.sin(i) * 0.5);
    expect(decide(logits, allShipped(250), calibration).kind).toBe('not-sure');
  });
});

describe('practiceScore', () => {
  it('is the calibrated probability of the target sign among shipped signs', () => {
    const logits = Float32Array.of(0, 2, 0);
    const p = softmaxOver(logits, [true, true, true], 1);
    expect(practiceScore(logits, [true, true, true], 1, calibration)).toBeCloseTo(p[1] ?? 0);
    expect(practiceScore(logits, [true, true, true], 0, calibration)).toBeLessThan(0.2);
  });

  it('is zero for a target that is not shipped', () => {
    expect(practiceScore(Float32Array.of(0, 2), [true, false], 1, calibration)).toBe(0);
  });
});
