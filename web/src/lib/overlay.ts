/** Draws the tracked landmarks: hands, upper body and (if present) lips. */
import {
  HAND_CONNECTIONS,
  LEFT_HAND_START,
  LIPS,
  N_HAND,
  POSE_CONNECTIONS,
  POSE_START,
  RIGHT_HAND_START,
  TORSO_CONNECTIONS,
  type Frame,
} from '../ml/landmarks';

export interface OverlayColors {
  rightHand: string;
  leftHand: string;
  body: string;
  torso: string;
  joint: string;
  lips: string;
}

export const OVERLAY_COLORS: OverlayColors = {
  rightHand: '#38bdf8',
  leftHand: '#f472b6',
  body: 'rgba(255, 255, 255, 0.72)',
  torso: 'rgba(255, 255, 255, 0.28)',
  joint: '#ffffff',
  lips: '#fbbf24',
};

/** Image-normalised point -> canvas pixels, or null if missing. */
function at(frame: Frame, point: number, w: number, h: number): [number, number] | null {
  const x = frame[point * 3] ?? Number.NaN;
  const y = frame[point * 3 + 1] ?? Number.NaN;
  return Number.isFinite(x) && Number.isFinite(y) ? [x * w, y * h] : null;
}

function line(
  ctx: CanvasRenderingContext2D,
  a: [number, number] | null,
  b: [number, number] | null,
) {
  if (!a || !b) return;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

/**
 * Draw `frame` scaled to the canvas. `fit` maps the image's normalised
 * coordinates into a box of the given aspect ratio (the camera's), centred.
 */
export function drawLandmarks(
  ctx: CanvasRenderingContext2D,
  frame: Frame,
  colors: OverlayColors = OVERLAY_COLORS,
  options: { clear?: boolean; aspect?: number } = {},
): void {
  const { width, height } = ctx.canvas;
  if (options.clear !== false) ctx.clearRect(0, 0, width, height);
  let w = width;
  let h = height;
  let ox = 0;
  let oy = 0;
  if (options.aspect) {
    // Letterbox: keep the source aspect ratio inside the canvas.
    if (width / height > options.aspect) {
      w = height * options.aspect;
      ox = (width - w) / 2;
    } else {
      h = width / options.aspect;
      oy = (height - h) / 2;
    }
  }
  ctx.save();
  ctx.translate(ox, oy);
  const scale = Math.max(1, Math.min(w, h) / 360);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  ctx.lineWidth = 3 * scale;
  ctx.strokeStyle = colors.torso;
  for (const [a, b] of TORSO_CONNECTIONS) {
    line(ctx, at(frame, POSE_START + a, w, h), at(frame, POSE_START + b, w, h));
  }
  ctx.strokeStyle = colors.body;
  for (const [a, b] of POSE_CONNECTIONS) {
    line(ctx, at(frame, POSE_START + a, w, h), at(frame, POSE_START + b, w, h));
  }
  const nose = at(frame, POSE_START, w, h);
  const leftShoulder = at(frame, POSE_START + 11, w, h);
  const rightShoulder = at(frame, POSE_START + 12, w, h);
  if (nose && leftShoulder && rightShoulder) {
    const r =
      Math.hypot(leftShoulder[0] - rightShoulder[0], leftShoulder[1] - rightShoulder[1]) * 0.32;
    ctx.beginPath();
    ctx.arc(nose[0], nose[1], r, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.fillStyle = colors.lips;
  for (const point of LIPS) {
    const p = at(frame, point, w, h);
    if (!p) continue;
    ctx.beginPath();
    ctx.arc(p[0], p[1], 1.4 * scale, 0, Math.PI * 2);
    ctx.fill();
  }

  for (const [start, color] of [
    [RIGHT_HAND_START, colors.rightHand],
    [LEFT_HAND_START, colors.leftHand],
  ] as const) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * scale;
    for (const [a, b] of HAND_CONNECTIONS) {
      line(ctx, at(frame, start + a, w, h), at(frame, start + b, w, h));
    }
    ctx.fillStyle = colors.joint;
    for (let i = 0; i < N_HAND; i++) {
      const p = at(frame, start + i, w, h);
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 2.4 * scale, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}
