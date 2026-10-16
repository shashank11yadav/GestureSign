import { describe, expect, it } from 'vitest';
import { assembleFrame, type Point } from './assemble';
import { LEFT_HAND_START, POSE_START, RIGHT_HAND_START } from './landmarks';

function hand(wristX: number, wristY: number): Point[] {
  return Array.from({ length: 21 }, (_, i) => ({
    x: wristX + i * 0.002,
    y: wristY - i * 0.003,
    z: 0,
  }));
}

function pose(rightWrist: [number, number], leftWrist: [number, number]): Point[] {
  const points = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  points[11] = { x: 0.62, y: 0.6, z: 0 };
  points[12] = { x: 0.38, y: 0.6, z: 0 };
  points[15] = { x: leftWrist[0], y: leftWrist[1], z: 0 };
  points[16] = { x: rightWrist[0], y: rightWrist[1], z: 0 };
  return points;
}

const at = (frame: Float32Array, point: number) => [frame[point * 3], frame[point * 3 + 1]];

describe('assembleFrame', () => {
  it('puts each hand on the side of the nearest pose wrist, ignoring handedness labels', () => {
    const frame = assembleFrame({
      hands: [
        { points: hand(0.3, 0.4), label: 'Right' },
        { points: hand(0.7, 0.45), label: 'Right' },
      ],
      pose: pose([0.31, 0.41], [0.69, 0.44]),
    });
    expect(at(frame, RIGHT_HAND_START)[0]).toBeCloseTo(0.3);
    expect(at(frame, LEFT_HAND_START)[0]).toBeCloseTo(0.7);
    expect(at(frame, POSE_START + 11)).toEqual([expect.closeTo(0.62), expect.closeTo(0.6)]);
  });

  it('resolves two hands near the same wrist by the cheaper overall pairing', () => {
    const frame = assembleFrame({
      hands: [
        { points: hand(0.35, 0.4), label: 'Left' },
        { points: hand(0.45, 0.4), label: 'Left' },
      ],
      pose: pose([0.3, 0.4], [0.6, 0.4]),
    });
    expect(at(frame, RIGHT_HAND_START)[0]).toBeCloseTo(0.35);
    expect(at(frame, LEFT_HAND_START)[0]).toBeCloseTo(0.45);
  });

  it('without a pose, flips MediaPipe handedness because the camera image is not mirrored', () => {
    const frame = assembleFrame({ hands: [{ points: hand(0.3, 0.4), label: 'Left' }], pose: null });
    expect(at(frame, RIGHT_HAND_START)[0]).toBeCloseTo(0.3);
    expect(Number.isNaN(at(frame, LEFT_HAND_START)[0])).toBe(true);
    expect(Number.isNaN(at(frame, POSE_START)[0])).toBe(true);
  });

  it('leaves everything NaN when nothing is detected', () => {
    const frame = assembleFrame({ hands: [], pose: null });
    expect(frame.every(Number.isNaN)).toBe(true);
  });

  it('copies face points when a face is given (Holistic), dropping iris points', () => {
    const face = Array.from({ length: 478 }, (_, i) => ({ x: i / 1000, y: 0.2, z: 0 }));
    const frame = assembleFrame({ hands: [], pose: null, face });
    expect(at(frame, 467)[0]).toBeCloseTo(0.467);
    expect(Number.isNaN(at(frame, LEFT_HAND_START)[0])).toBe(true);
  });

  it('can take hands already assigned to sides (Holistic)', () => {
    const frame = assembleFrame({
      hands: [{ points: hand(0.2, 0.3), side: 'left' }],
      pose: pose([0.2, 0.3], [0.8, 0.3]),
    });
    expect(at(frame, LEFT_HAND_START)[0]).toBeCloseTo(0.2);
  });
});
