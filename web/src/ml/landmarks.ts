/**
 * The 543-point frame layout of Kaggle asl-signs (MediaPipe Holistic), which
 * the pipeline trains on: 468 face points, 21 left-hand, 33 pose, 21
 * right-hand, each with x, y, z. A frame is a Float32Array of 543 * 3 values
 * with NaN for anything not detected. Mirrors pipeline/src/gesturesign/landmarks.py.
 */

export const N_FACE = 468;
export const N_HAND = 21;
export const N_POSE = 33;
export const N_POINTS = N_FACE + N_HAND + N_POSE + N_HAND;
export const FRAME_LENGTH = N_POINTS * 3;

export const LEFT_HAND_START = N_FACE;
export const POSE_START = LEFT_HAND_START + N_HAND;
export const RIGHT_HAND_START = POSE_START + N_POSE;

export const POSE_UPPER = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16] as const;
export const POSE_UPPER_MIRROR = [0, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15] as const;
export const POSE_LEFT_SHOULDER = 11;
export const POSE_RIGHT_SHOULDER = 12;
export const POSE_LEFT_WRIST = 15;
export const POSE_RIGHT_WRIST = 16;

const LIP_RUNS = [
  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
  [185, 40, 39, 37, 0, 267, 269, 270, 409],
  [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
  [191, 80, 81, 82, 13, 312, 311, 310, 415],
] as const;
export const LIPS: readonly number[] = LIP_RUNS.flat();
export const LIPS_MIRROR: readonly number[] = LIP_RUNS.flatMap((run) => [...run].reverse());

export const HAND_WRIST = 0;
export const HAND_MIDDLE_MCP = 9;

export const HAND_CONNECTIONS: readonly (readonly [number, number])[] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]; // prettier-ignore

/** Upper-body pose connections drawn in the overlay (pose landmark indices). */
export const POSE_CONNECTIONS: readonly (readonly [number, number])[] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [7, 2], [2, 0], [0, 5], [5, 8],
]; // prettier-ignore

/** Shoulder-to-hip lines, drawn fainter (hips are often out of a webcam's view). */
export const TORSO_CONNECTIONS: readonly (readonly [number, number])[] = [
  [11, 23],
  [12, 24],
];

export type Frame = Float32Array;

export function emptyFrame(): Frame {
  return new Float32Array(FRAME_LENGTH).fill(Number.NaN);
}

/** True when no hand point is present (the "nothing to read" case). */
export function handsMissing(frame: Frame): boolean {
  return (
    Number.isNaN(frame[LEFT_HAND_START * 3] ?? Number.NaN) &&
    Number.isNaN(frame[RIGHT_HAND_START * 3] ?? Number.NaN)
  );
}
