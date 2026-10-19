import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { parseModelRequest } from './model-protocol';
import { parseTrackerRequest } from './tracker-protocol';
import { isTrustedOrigin } from './worker-guard';

const OWN = 'https://shashank11yadav.github.io';

describe('isTrustedOrigin', () => {
  it('accepts the creating page (empty origin) and this origin only', () => {
    expect(isTrustedOrigin('', OWN)).toBe(true);
    expect(isTrustedOrigin(OWN, OWN)).toBe(true);
    expect(isTrustedOrigin('https://evil.example', OWN)).toBe(false);
    expect(isTrustedOrigin(`${OWN}.evil.example`, OWN)).toBe(false);
    expect(isTrustedOrigin('null', OWN)).toBe(false);
  });
});

describe('parseModelRequest', () => {
  const run = (patch: Record<string, unknown> = {}) => ({
    type: 'run',
    id: 3,
    data: new Float32Array(2 * 4),
    frames: 2,
    dim: 4,
    ...patch,
  });

  it('accepts a load request and a well-formed run', () => {
    expect(parseModelRequest({ type: 'load' })).toEqual({ ok: true, message: { type: 'load' } });
    const parsed = parseModelRequest(run());
    expect(parsed.ok && parsed.message.type === 'run' && parsed.message.id).toBe(3);
  });

  it('ignores URLs a load request tries to carry', () => {
    const parsed = parseModelRequest({ type: 'load', modelUrl: 'https://evil.example/m.onnx' });
    expect(parsed).toEqual({ ok: true, message: { type: 'load' } });
  });

  it.each([
    ['a non-object', null],
    ['an unknown type', { type: 'eval', code: '1' }],
    ['an array', [1, 2]],
  ])('rejects %s without a reply', (_, data) => {
    expect(parseModelRequest(data)).toEqual({ ok: false, reply: null });
  });

  it.each([
    ['data that is not a Float32Array', { data: [0, 0, 0, 0, 0, 0, 0, 0] }],
    ['a length that does not match frames x dim', { data: new Float32Array(7) }],
    ['a fractional frame count', { frames: 1.5 }],
    ['zero frames', { frames: 0, data: new Float32Array(0) }],
    ['too many frames', { frames: 100_000, data: new Float32Array(4) }],
    ['a negative dim', { dim: -4 }],
  ])('rejects a run with %s and reports it against its id', (_, patch) => {
    const parsed = parseModelRequest(run(patch));
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.reply).toMatchObject({ type: 'run-error', id: 3 });
  });

  it('cannot reply to a run without a usable id', () => {
    expect(parseModelRequest(run({ id: 'x' }))).toEqual({ ok: false, reply: null });
    expect(parseModelRequest(run({ id: -1 }))).toEqual({ ok: false, reply: null });
    expect(parseModelRequest(run({ id: 2 ** 60 }))).toEqual({ ok: false, reply: null });
  });
});

describe('parseTrackerRequest', () => {
  class FakeBitmap {
    close() {
      return undefined;
    }
  }
  beforeEach(() => {
    vi.stubGlobal('ImageBitmap', FakeBitmap);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('accepts an init for a known feature set and delegate', () => {
    const msg = { type: 'init', featureSet: 'hands_pose_lips', delegate: 'GPU' };
    expect(parseTrackerRequest(msg)).toEqual({ ok: true, message: msg });
  });

  it('drops anything else an init carries, such as asset URLs', () => {
    const parsed = parseTrackerRequest({
      type: 'init',
      featureSet: 'hands_pose',
      delegate: 'CPU',
      wasmBase: 'https://evil.example/',
    });
    expect(parsed).toEqual({
      ok: true,
      message: { type: 'init', featureSet: 'hands_pose', delegate: 'CPU' },
    });
  });

  it.each([
    ['an unknown feature set', { featureSet: 'everything', delegate: 'GPU' }],
    ['an unknown delegate', { featureSet: 'hands_pose', delegate: 'TPU' }],
  ])('answers an init with %s with init-error', (_, patch) => {
    const parsed = parseTrackerRequest({ type: 'init', ...patch });
    expect(parsed.ok).toBe(false);
    if (!parsed.ok) expect(parsed.reply?.type).toBe('init-error');
  });

  it('accepts a frame with an ImageBitmap and a finite time', () => {
    const bitmap = new FakeBitmap();
    const parsed = parseTrackerRequest({ type: 'frame', bitmap, t: 12.5 });
    expect(parsed).toEqual({ ok: true, message: { type: 'frame', bitmap, t: 12.5 } });
  });

  it('answers a frame without a bitmap with frame-error, so the page can send the next one', () => {
    const parsed = parseTrackerRequest({ type: 'frame', bitmap: {}, t: 7 });
    expect(parsed).toEqual({
      ok: false,
      reply: { type: 'frame-error', t: 7, message: 'malformed frame message' },
    });
  });

  it('cannot reply to a frame without a finite time', () => {
    const bitmap = new FakeBitmap();
    expect(parseTrackerRequest({ type: 'frame', bitmap, t: Number.NaN })).toEqual({
      ok: false,
      reply: null,
    });
    expect(parseTrackerRequest({ type: 'frame', bitmap })).toEqual({ ok: false, reply: null });
  });

  it('ignores unknown messages', () => {
    expect(parseTrackerRequest('init')).toEqual({ ok: false, reply: null });
    expect(parseTrackerRequest({ type: 'shutdown' })).toEqual({ ok: false, reply: null });
  });
});
