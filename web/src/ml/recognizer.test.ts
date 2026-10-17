/**
 * Review focus 2: with nothing (or too much, or only part of a person) in
 * frame, or a low-confidence model, the page must say so instead of naming a sign.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it, vi } from 'vitest';
import { emptyFrame, POSE_START, RIGHT_HAND_START, type Frame } from './landmarks';
import { parseLabels, parseMeta, type ModelMeta } from './meta';
import { Recognizer, type Infer, type RecognizerEvent } from './recognizer';

const read = (name: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../public/models/${name}`, import.meta.url), 'utf-8'));
const meta: ModelMeta = parseMeta(read('model-meta.json'));
const labels = parseLabels(read('labels.json'), meta.classes);

function frameAt(handX: number, shoulders = true): Frame {
  const f = emptyFrame();
  const set = (p: number, x: number, y: number) => {
    f[p * 3] = x;
    f[p * 3 + 1] = y;
    f[p * 3 + 2] = 0;
  };
  if (shoulders) {
    set(POSE_START + 11, 0.62, 0.6);
    set(POSE_START + 12, 0.38, 0.6);
  } else {
    set(POSE_START + 0, 0.5, 0.2);
  }
  for (let i = 0; i < 21; i++) set(RIGHT_HAND_START + i, handX + i * 0.003, 0.4 - i * 0.004);
  return f;
}

function confidentAt(index: number): Infer {
  return () => {
    const logits = new Float32Array(meta.classes);
    logits[index] = 30;
    return Promise.resolve({ logits, embedding: new Float32Array(meta.embeddingDim) });
  };
}

async function drive(frames: { frame: Frame; poses: number }[], infer: Infer) {
  const events: RecognizerEvent[] = [];
  const rec = new Recognizer(meta, labels, infer, (e) => events.push(e));
  frames.forEach(({ frame, poses }, i) => {
    rec.push({ frame, t: i * (1000 / 30), poses });
  });
  await vi
    .waitFor(() => {
      if (!events.some((e) => e.type === 'result' || e.type === 'discarded')) {
        throw new Error('still waiting');
      }
    })
    .catch(() => undefined);
  return events;
}

const sign = (shoulders = true) =>
  Array.from({ length: 20 }, (_, i) => ({ frame: frameAt(0.3 + 0.01 * i, shoulders), poses: 1 }));
const gone = (n: number) => Array.from({ length: n }, () => ({ frame: emptyFrame(), poses: 0 }));

describe('Recognizer', () => {
  it('reports "nothing" and never runs the model on empty frames', async () => {
    const infer = vi.fn(confidentAt(1));
    const events = await drive(gone(30), infer);
    expect(events).toEqual([{ type: 'status', status: 'nothing' }]);
    expect(infer).not.toHaveBeenCalled();
  });

  it('names a sign when the model is confident', async () => {
    const events = await drive([...sign(), ...gone(12)], confidentAt(1));
    const result = events.find((e) => e.type === 'result');
    if (result?.type !== 'result') throw new Error('no result');
    expect(result.result.decision.kind).toBe('sign');
    expect(result.result.decision.top[0]?.index).toBe(1);
  });

  it('says "not sure" when the signer is only partly in view, however confident the model', async () => {
    const events = await drive([...sign(false), ...gone(12)], confidentAt(1));
    expect(events).toContainEqual({ type: 'status', status: 'partial' });
    const result = events.find((e) => e.type === 'result');
    if (result?.type !== 'result') throw new Error('no result');
    expect(result.result.decision.kind).toBe('not-sure');
    expect(result.result.decision.reason).toBe('partial');
  });

  it('says "partial" for a short sign too, ignoring the empty frames that end it', async () => {
    // 9 framed frames, then the 300 ms of empty frames that end the sign: the
    // empty frames must not outvote the partly framed ones.
    const events = await drive([...sign(false).slice(0, 9), ...gone(12)], confidentAt(1));
    const result = events.find((e) => e.type === 'result');
    if (result?.type !== 'result') throw new Error('no result');
    expect(result.result.decision.reason).toBe('partial');
  });

  it('says "not sure" below the calibrated threshold', async () => {
    const flat: Infer = () =>
      Promise.resolve({
        logits: new Float32Array(meta.classes).fill(0.1),
        embedding: new Float32Array(meta.embeddingDim),
      });
    const events = await drive([...sign(), ...gone(12)], flat);
    const result = events.find((e) => e.type === 'result');
    if (result?.type !== 'result') throw new Error('no result');
    expect(result.result.decision.kind).toBe('not-sure');
    expect(result.result.decision.top).toHaveLength(3);
  });

  it('does not segment while more than one person is in view', async () => {
    const infer = vi.fn(confidentAt(1));
    const crowded = sign().map((f) => ({ ...f, poses: 2 }));
    const events = await drive([...crowded, ...gone(12)], infer);
    expect(events[0]).toEqual({ type: 'status', status: 'crowded' });
    expect(infer).not.toHaveBeenCalled();
  });

  it('reports model errors instead of a result', async () => {
    const events = await drive([...sign(), ...gone(12)], () => Promise.reject(new Error('boom')));
    await vi.waitFor(() => {
      expect(events).toContainEqual({ type: 'error', message: 'boom' });
    });
  });
});
