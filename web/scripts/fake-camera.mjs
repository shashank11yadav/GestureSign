// Synthetic fake-camera clip for the A1 benchmark: a drawn, front-facing
// figure (face with eyes, brows, nose and mouth, hair, neck, shirt) moving
// two open hands in front of the chest. No real person is involved. The
// drawing is crude, so MediaPipe may or may not detect it; bench.mjs reports
// the detection rates next to the fps.
//
// Usage: node scripts/fake-camera.mjs out.y4m [--preview out.ppm]

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const W = 640;
const H = 480;

function canvas() {
  const rgb = new Float32Array(W * H * 3);
  const set = (x, y, c) => {
    const i = (y * W + x) * 3;
    rgb[i] = c[0];
    rgb[i + 1] = c[1];
    rgb[i + 2] = c[2];
  };
  const fill = (inside, color, box) => {
    const [x0, y0, x1, y1] = box ?? [0, 0, W, H];
    for (let y = Math.max(0, y0 | 0); y < Math.min(H, y1); y++) {
      for (let x = Math.max(0, x0 | 0); x < Math.min(W, x1); x++) {
        const c = inside(x + 0.5, y + 0.5);
        if (c) set(x, y, typeof color === 'function' ? color(x, y) : color);
      }
    }
  };
  const ellipse = (cx, cy, rx, ry, color, angle = 0) => {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const r = Math.max(rx, ry);
    fill(
      (x, y) => {
        const dx = x - cx;
        const dy = y - cy;
        const u = dx * c + dy * s;
        const v = -dx * s + dy * c;
        return (u / rx) ** 2 + (v / ry) ** 2 <= 1;
      },
      color,
      [cx - r, cy - r, cx + r + 1, cy + r + 1],
    );
  };
  // A capsule from (ax, ay) to (bx, by) with radius r (fingers, neck, arms).
  const capsule = (ax, ay, bx, by, r, color) => {
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy || 1;
    fill(
      (x, y) => {
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2));
        return (x - ax - t * dx) ** 2 + (y - ay - t * dy) ** 2 <= r * r;
      },
      color,
      [
        Math.min(ax, bx) - r,
        Math.min(ay, by) - r,
        Math.max(ax, bx) + r + 1,
        Math.max(ay, by) + r + 1,
      ],
    );
  };
  return { rgb, fill, ellipse, capsule };
}

const SKIN = [224, 172, 140];
const SKIN_SHADE = [196, 140, 110];
const HAIR = [52, 36, 28];
const SHIRT = [48, 84, 150];
const LIPS = [168, 76, 78];

function hand(g, wx, wy, angle, flip) {
  // Palm and five fingers, fingers pointing up (angle 0), thumb to one side.
  const rot = (dx, dy) => [
    wx + dx * Math.cos(angle) - dy * Math.sin(angle),
    wy + dx * Math.sin(angle) + dy * Math.cos(angle),
  ];
  const [pcx, pcy] = rot(0, -32);
  g.ellipse(pcx, pcy, 30, 34, SKIN_SHADE, angle);
  g.ellipse(pcx, pcy, 27, 31, SKIN, angle);
  const fingers = [
    [-20, -58, -26, -100],
    [-7, -62, -8, -112],
    [7, -62, 9, -108],
    [20, -56, 25, -94],
  ];
  for (const [bx, by, tx, ty] of fingers) {
    const [ax, ay] = rot(bx, by);
    const [cx, cy] = rot(tx, ty);
    g.capsule(ax, ay, cx, cy, 8, SKIN_SHADE);
    g.capsule(ax, ay, cx, cy, 6.5, SKIN);
  }
  const [t0x, t0y] = rot(flip * 24, -20);
  const [t1x, t1y] = rot(flip * 52, -46);
  g.capsule(t0x, t0y, t1x, t1y, 8.5, SKIN_SHADE);
  g.capsule(t0x, t0y, t1x, t1y, 7, SKIN);
}

export function renderFrame(f, frames) {
  const g = canvas();
  const a = (2 * Math.PI * f) / frames;
  g.fill(
    () => true,
    (_x, y) => [196 - y * 0.08, 192 - y * 0.08, 184 - y * 0.08],
  );
  const sway = 6 * Math.sin(a);
  const cx = 320 + sway;
  // Shoulders and shirt
  g.ellipse(cx, 470, 210, 130, SHIRT);
  g.capsule(cx - 150, 380, cx - 175, 480, 40, SHIRT);
  g.capsule(cx + 150, 380, cx + 175, 480, 40, SHIRT);
  g.capsule(cx, 250, cx, 330, 32, SKIN_SHADE);
  // Head
  g.ellipse(cx, 190, 82, 104, HAIR);
  g.ellipse(cx - 78, 200, 14, 24, SKIN_SHADE);
  g.ellipse(cx + 78, 200, 14, 24, SKIN_SHADE);
  g.ellipse(cx, 205, 74, 94, SKIN);
  g.ellipse(cx, 112, 80, 34, HAIR);
  for (const side of [-1, 1]) {
    g.capsule(cx + side * 18, 166, cx + side * 46, 160, 4, HAIR);
    g.ellipse(cx + side * 30, 186, 15, 8, [245, 245, 240]);
    g.ellipse(cx + side * 30, 186, 7, 7, [60, 40, 30]);
    g.ellipse(cx + side * 30, 186, 3, 3, [10, 10, 10]);
  }
  g.capsule(cx, 192, cx - 3, 228, 7, SKIN_SHADE);
  g.ellipse(cx, 236, 12, 5, SKIN_SHADE);
  g.ellipse(cx, 260, 26, 9, LIPS);
  g.capsule(cx - 22, 260, cx + 22, 260, 1.5, [110, 40, 45]);
  // Hands: the right hand (image left) traces a loop, the left hand bobs.
  hand(g, 230 + 70 * Math.cos(a), 420 + 40 * Math.sin(2 * a), -0.25 + 0.2 * Math.sin(a), 1);
  hand(g, 420 + 20 * Math.sin(a), 440 + 15 * Math.cos(a), 0.2, -1);
  return g.rgb;
}

function toYuv420(rgb) {
  const y = Buffer.alloc(W * H);
  const u = Buffer.alloc((W / 2) * (H / 2));
  const v = Buffer.alloc((W / 2) * (H / 2));
  const clamp = (n) => Math.max(0, Math.min(255, Math.round(n)));
  for (let i = 0; i < W * H; i++) {
    const r = rgb[i * 3];
    const gg = rgb[i * 3 + 1];
    const b = rgb[i * 3 + 2];
    y[i] = clamp(0.299 * r + 0.587 * gg + 0.114 * b);
  }
  for (let yy = 0; yy < H / 2; yy++) {
    for (let xx = 0; xx < W / 2; xx++) {
      const i = (2 * yy * W + 2 * xx) * 3;
      const r = rgb[i];
      const gg = rgb[i + 1];
      const b = rgb[i + 2];
      u[yy * (W / 2) + xx] = clamp(128 - 0.168736 * r - 0.331264 * gg + 0.5 * b);
      v[yy * (W / 2) + xx] = clamp(128 + 0.5 * r - 0.418688 * gg - 0.081312 * b);
    }
  }
  return [y, u, v];
}

export async function writeY4m(path, frames = 60) {
  const chunks = [Buffer.from(`YUV4MPEG2 W${W} H${H} F30:1 Ip A1:1 C420jpeg\n`)];
  for (let f = 0; f < frames; f++)
    chunks.push(Buffer.from('FRAME\n'), ...toYuv420(renderFrame(f, frames)));
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, Buffer.concat(chunks));
}

export async function writePreview(path) {
  const rgb = renderFrame(7, 60);
  const body = Buffer.alloc(W * H * 3);
  for (let i = 0; i < body.length; i++) body[i] = Math.max(0, Math.min(255, Math.round(rgb[i])));
  await writeFile(path, Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), body]));
}

if (process.argv[1]?.endsWith('fake-camera.mjs')) {
  const out = process.argv[2];
  const i = process.argv.indexOf('--preview');
  if (i > 0) await writePreview(process.argv[i + 1]);
  if (out && !out.startsWith('--')) await writeY4m(out);
}
