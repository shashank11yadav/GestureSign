/**
 * Turns MediaPipe results into the 543 x 3 frame the model was trained on.
 *
 * The training data comes from MediaPipe Holistic, which assigns each hand to
 * the pose wrist it belongs to. The browser's HandLandmarker labels hands by
 * appearance instead, and assumes a mirrored (selfie) image, which the raw
 * camera frame is not. To match training, hands are assigned to the nearest
 * pose wrist; only without a pose do we fall back to the flipped label.
 * Feature canonicalisation (features.ts) then makes left- and right-handed
 * signing, and mirrored or unmirrored cameras, look the same to the model.
 */
import {
  FRAME_LENGTH,
  LEFT_HAND_START,
  N_FACE,
  N_HAND,
  N_POSE,
  POSE_LEFT_WRIST,
  POSE_RIGHT_WRIST,
  POSE_START,
  RIGHT_HAND_START,
  type Frame,
} from './landmarks';

export interface Point {
  x: number;
  y: number;
  z: number;
}

export type Side = 'left' | 'right';

export interface DetectedHand {
  points: readonly Point[];
  /** MediaPipe handedness ('Left' | 'Right'), which assumes a mirrored image. */
  label?: string;
  /** Already known side of the body (Holistic). */
  side?: Side;
}

export interface Detection {
  hands: readonly DetectedHand[];
  pose: readonly Point[] | null;
  face?: readonly Point[] | null;
}

function write(frame: Frame, start: number, points: readonly Point[], count: number): void {
  for (let i = 0; i < count; i++) {
    const p = points[i];
    if (!p) continue;
    frame[(start + i) * 3] = p.x;
    frame[(start + i) * 3 + 1] = p.y;
    frame[(start + i) * 3 + 2] = p.z;
  }
}

function dist(a: Point | undefined, b: Point | undefined): number {
  if (!a || !b) return Number.POSITIVE_INFINITY;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function assignSides(hands: readonly DetectedHand[], pose: readonly Point[] | null): Side[] {
  if (hands.every((h) => h.side)) return hands.map((h) => h.side ?? 'right');
  if (pose && pose.length >= N_POSE) {
    const left = pose[POSE_LEFT_WRIST];
    const right = pose[POSE_RIGHT_WRIST];
    const toLeft = hands.map((h) => dist(h.points[0], left));
    const toRight = hands.map((h) => dist(h.points[0], right));
    if (hands.length === 1) return [(toLeft[0] ?? 0) < (toRight[0] ?? 0) ? 'left' : 'right'];
    // Two hands: pick the pairing with the smaller total distance.
    const straight = (toRight[0] ?? 0) + (toLeft[1] ?? 0);
    const swapped = (toLeft[0] ?? 0) + (toRight[1] ?? 0);
    return straight <= swapped ? ['right', 'left'] : ['left', 'right'];
  }
  // No pose: the camera frame is not mirrored, so MediaPipe's label is the opposite side.
  return hands.map((h) => h.side ?? (h.label === 'Left' ? 'right' : 'left'));
}

export function assembleFrame(detection: Detection): Frame {
  const frame = new Float32Array(FRAME_LENGTH).fill(Number.NaN);
  if (detection.face && detection.face.length >= N_FACE) write(frame, 0, detection.face, N_FACE);
  if (detection.pose && detection.pose.length >= N_POSE) {
    write(frame, POSE_START, detection.pose, N_POSE);
  }
  const hands = detection.hands.slice(0, 2);
  const sides = assignSides(hands, detection.pose);
  const used = new Set<Side>();
  hands.forEach((hand, i) => {
    const side = sides[i] ?? 'right';
    if (used.has(side)) return;
    used.add(side);
    write(frame, side === 'left' ? LEFT_HAND_START : RIGHT_HAND_START, hand.points, N_HAND);
  });
  return frame;
}
