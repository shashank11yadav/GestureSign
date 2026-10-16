import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  FEATURE_VERSION,
  featureLayout,
  fitLength,
  frameFeatures,
  modelInput,
  resample,
  resampledLength,
  type FeatureSet,
} from './features';
import { LEFT_HAND_START, N_HAND, N_POINTS, RIGHT_HAND_START, emptyFrame } from './landmarks';

interface FixtureCase {
  name: string;
  kind: 'sequence' | 'frame';
  featureSet: FeatureSet;
  srcFps?: number;
  fps?: number;
  maxFrames?: number;
  frames: (number | null)[][];
  expected: { length?: number; features: number[] | number[][] };
}

interface Fixture {
  featureVersion: number;
  tolerance: number;
  points: number[];
  cases: FixtureCase[];
}

const fixture = JSON.parse(
  readFileSync(new URL('../../../shared/fixtures/features.json', import.meta.url), 'utf-8'),
) as Fixture;

function decode(frames: (number | null)[][]): Float32Array[] {
  return frames.map((flat) => {
    const out = emptyFrame();
    fixture.points.forEach((point, i) => {
      const x = flat[2 * i];
      const y = flat[2 * i + 1];
      out[point * 3] = x ?? Number.NaN;
      out[point * 3 + 1] = y ?? Number.NaN;
    });
    return out;
  });
}

function maxAbsDiff(got: ArrayLike<number>, expected: number[]): number {
  expect(got.length).toBe(expected.length);
  let worst = 0;
  for (let i = 0; i < expected.length; i++) {
    worst = Math.max(worst, Math.abs((got[i] ?? Number.NaN) - (expected[i] ?? Number.NaN)));
  }
  return worst;
}

describe('feature parity with the Python pipeline', () => {
  it('implements the fixture feature version', () => {
    expect(fixture.featureVersion).toBe(FEATURE_VERSION);
  });

  for (const testCase of fixture.cases) {
    it(`matches ${testCase.name}`, () => {
      const frames = decode(testCase.frames);
      if (testCase.kind === 'frame') {
        const [frame] = frames;
        if (!frame) throw new Error('empty frame case');
        const got = frameFeatures(frame, testCase.featureSet);
        expect(maxAbsDiff(got, testCase.expected.features as number[])).toBeLessThanOrEqual(
          fixture.tolerance,
        );
        return;
      }
      const input = modelInput(frames, testCase.srcFps ?? 30, {
        fps: testCase.fps ?? 30,
        maxFrames: testCase.maxFrames ?? 64,
        featureSet: testCase.featureSet,
      });
      expect(input.frames).toBe(testCase.expected.length);
      const expected = (testCase.expected.features as number[][]).flat();
      expect(maxAbsDiff(input.data, expected)).toBeLessThanOrEqual(fixture.tolerance);
    });
  }

  it('gives a mirrored clip the same features as the original', () => {
    const byName = new Map(fixture.cases.map((c) => [c.name, c]));
    const a = byName.get('two-hands');
    const b = byName.get('two-hands-mirrored');
    if (!a || !b) throw new Error('fixture is missing the mirrored pair');
    const cfg = { fps: 30, maxFrames: 64, featureSet: 'hands_pose' as const };
    const fa = modelInput(decode(a.frames), 30, cfg).data;
    const fb = modelInput(decode(b.frames), 30, cfg).data;
    expect(maxAbsDiff(fa, Array.from(fb))).toBeLessThanOrEqual(1e-4);
  });
});

describe('frameFeatures', () => {
  it('turns an all-NaN frame into zeros with no presence', () => {
    for (const set of ['hands_pose', 'hands_pose_lips'] as const) {
      const out = frameFeatures(emptyFrame(), set);
      const layout = featureLayout(set);
      expect(out.length).toBe(layout.frameDim);
      expect(out.every((v) => v === 0)).toBe(true);
    }
  });

  it('has the documented sizes', () => {
    expect(featureLayout('hands_pose').frameDim).toBe(193);
    expect(featureLayout('hands_pose').sequenceDim).toBe(299);
    expect(featureLayout('hands_pose_lips').frameDim).toBe(274);
  });

  it('mirrors a frame with only the left hand into the dominant slot', () => {
    const right = emptyFrame();
    const left = emptyFrame();
    for (let i = 0; i < N_HAND; i++) {
      const x = 0.3 + i * 0.004;
      const y = 0.4 - i * 0.003;
      right[(RIGHT_HAND_START + i) * 3] = x;
      right[(RIGHT_HAND_START + i) * 3 + 1] = y;
      left[(LEFT_HAND_START + i) * 3] = 1 - x;
      left[(LEFT_HAND_START + i) * 3 + 1] = y;
    }
    // No shoulders: both use the default anchor at x = 0.5, so the mirror is exact.
    const a = frameFeatures(right, 'hands_pose');
    const b = frameFeatures(left, 'hands_pose');
    expect(maxAbsDiff(a, Array.from(b))).toBeLessThan(1e-5);
    expect(a.length).toBe(193);
    expect(N_POINTS).toBe(543);
  });
});

describe('resample', () => {
  const ramp = (n: number) => Array.from({ length: n }, (_, i) => Float32Array.of(i, i));

  it('keeps the same rate unchanged', () => {
    const seq = ramp(5);
    expect(resample(seq, 30, 30)).toEqual(seq);
  });

  it('halves the frame count from 60 to 30 fps', () => {
    expect(resample(ramp(9), 60, 30).map((f) => f[0])).toEqual([0, 2, 4, 6, 8]);
  });

  it('interpolates when upsampling', () => {
    expect(resample(ramp(3), 10, 20).map((f) => f[0])).toEqual([0, 0.5, 1, 1.5, 2]);
  });

  it('matches the Python output length', () => {
    expect(resampledLength(26, 25, 30)).toBe(31);
    expect(resampledLength(1, 25, 30)).toBe(1);
    expect(resampledLength(0, 25, 30)).toBe(0);
  });

  it('takes the nearest frame next to a missing one', () => {
    const seq = ramp(3);
    seq[1] = Float32Array.of(Number.NaN, Number.NaN);
    const out = resample(seq, 10, 40).map((f) => f[0]);
    expect(out[1]).toBe(0);
    expect(out[2]).toBeNaN();
    expect(out[7]).toBe(2);
  });

  it('squeezes long sequences onto maxFrames, keeping both ends', () => {
    const out = fitLength(ramp(127), 64);
    expect(out).toHaveLength(64);
    expect(out[0]?.[0]).toBe(0);
    expect(out[63]?.[0]).toBeCloseTo(126);
  });
});
