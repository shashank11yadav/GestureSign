import { describe, expect, it } from 'vitest';
import { Segmenter, frameStatus, type SegmentEvent } from './segmenter';
import { emptyFrame, LEFT_HAND_START, POSE_START, RIGHT_HAND_START, type Frame } from './landmarks';

const DT = 1000 / 30;

function body(): Frame {
  const f = emptyFrame();
  const set = (p: number, x: number, y: number) => {
    f[p * 3] = x;
    f[p * 3 + 1] = y;
    f[p * 3 + 2] = 0;
  };
  set(POSE_START + 11, 0.62, 0.6);
  set(POSE_START + 12, 0.38, 0.6);
  return f;
}

function withHand(x: number, y: number, start = RIGHT_HAND_START): Frame {
  const f = body();
  for (let i = 0; i < 21; i++) {
    f[(start + i) * 3] = x + i * 0.003;
    f[(start + i) * 3 + 1] = y - i * 0.004;
    f[(start + i) * 3 + 2] = 0;
  }
  return f;
}

/** Feed frames at 30 fps and collect events. */
function run(seg: Segmenter, frames: Frame[], t0 = 0): SegmentEvent[] {
  const events: SegmentEvent[] = [];
  frames.forEach((f, i) => {
    const e = seg.push(f, t0 + i * DT);
    if (e) events.push(e);
  });
  return events;
}

const moving = (n: number, from = 0) =>
  Array.from({ length: n }, (_, i) => withHand(0.3 + 0.01 * (from + i), 0.4));
const still = (n: number, x = 0.5) => Array.from({ length: n }, () => withHand(x, 0.4));
const absent = (n: number) => Array.from({ length: n }, () => body());

describe('Segmenter', () => {
  it('stays idle with no hands', () => {
    const seg = new Segmenter();
    expect(run(seg, absent(40))).toEqual([]);
    expect(seg.state).toBe('idle');
  });

  it('cuts a sign when the hands leave for 300 ms', () => {
    const seg = new Segmenter();
    const events = run(seg, [...absent(3), ...moving(20), ...absent(12)]);
    expect(events.map((e) => e.type)).toEqual(['start', 'end']);
    const end = events[1];
    if (end?.type !== 'end') throw new Error('expected an end event');
    expect(end.reason).toBe('absent');
    expect(end.segment.frames.length).toBeGreaterThanOrEqual(20);
    expect(end.segment.frames.length).toBeLessThanOrEqual(22);
    expect(end.segment.fps).toBeCloseTo(30, 0);
  });

  it('bridges a short dropout inside a sign', () => {
    const seg = new Segmenter();
    const events = run(seg, [...moving(10), ...absent(5), ...moving(10, 15), ...absent(12)]);
    expect(events.map((e) => e.type)).toEqual(['start', 'end']);
    const end = events[1];
    if (end?.type !== 'end') throw new Error('expected an end event');
    expect(end.segment.frames.length).toBeGreaterThanOrEqual(25);
  });

  it('does not start on a hand held still', () => {
    const seg = new Segmenter();
    expect(run(seg, still(40))).toEqual([]);
  });

  it('cuts a sign when the hand holds still for 300 ms, then waits for new motion', () => {
    const seg = new Segmenter();
    const events = run(seg, [...moving(20), ...still(30, 0.5)]);
    expect(events.map((e) => e.type)).toEqual(['start', 'end']);
    const end = events[1];
    if (end?.type !== 'end') throw new Error('expected an end event');
    expect(end.reason).toBe('still');
    const more = run(seg, still(20, 0.5), 50 * DT);
    expect(more).toEqual([]);
  });

  it('discards movements too short to be a sign', () => {
    const seg = new Segmenter();
    const events = run(seg, [...moving(4), ...absent(12)]);
    expect(events.map((e) => e.type)).toEqual(['start', 'discard']);
  });

  it('closes very long segments', () => {
    const seg = new Segmenter({ maxMs: 2000 });
    const events = run(seg, moving(90));
    expect(events.map((e) => e.type)).toEqual(['start', 'end', 'start']);
    const end = events[1];
    if (end?.type !== 'end') throw new Error('expected an end event');
    expect(end.reason).toBe('max');
  });

  it('estimates the frame rate from timestamps', () => {
    const seg = new Segmenter();
    const frames = [...moving(20), ...absent(12)];
    const events: SegmentEvent[] = [];
    frames.forEach((f, i) => {
      const e = seg.push(f, i * 50); // 20 fps webcam
      if (e) events.push(e);
    });
    const end = events.find((e) => e.type === 'end');
    if (end?.type !== 'end') throw new Error('expected an end event');
    expect(end.segment.fps).toBeCloseTo(20, 0);
  });

  it('ignores frames with more than one person', () => {
    const seg = new Segmenter();
    const events: SegmentEvent[] = [];
    moving(20).forEach((f, i) => {
      const e = seg.push(f, i * DT, { crowded: true });
      if (e) events.push(e);
    });
    expect(events).toEqual([]);
  });

  it('flush ends an open segment', () => {
    const seg = new Segmenter();
    run(seg, moving(15));
    const event = seg.flush();
    expect(event?.type).toBe('end');
  });
});

describe('frameStatus', () => {
  it('reports nothing, no hands, crowded, partial and ok', () => {
    expect(frameStatus(emptyFrame(), 0)).toBe('nothing');
    expect(frameStatus(body(), 1)).toBe('no-hands');
    expect(frameStatus(withHand(0.3, 0.4), 2)).toBe('crowded');
    const noShoulders = withHand(0.3, 0.4);
    noShoulders.fill(Number.NaN, (POSE_START + 11) * 3, (POSE_START + 13) * 3);
    expect(frameStatus(noShoulders, 1)).toBe('partial');
    const offscreen = withHand(0.3, 0.4);
    offscreen[(POSE_START + 11) * 3 + 1] = 1.4;
    expect(frameStatus(offscreen, 1)).toBe('partial');
    expect(frameStatus(withHand(0.3, 0.4, LEFT_HAND_START), 1)).toBe('ok');
  });
});
