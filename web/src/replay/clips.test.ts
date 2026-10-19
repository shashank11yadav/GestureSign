import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { RIGHT_HAND_START, LEFT_HAND_START, N_POINTS } from '../ml/landmarks';
import { decodeReplay, loadReplayClips } from './clips';

const raw: unknown = JSON.parse(
  readFileSync(new URL('../../public/replay/clips.json', import.meta.url), 'utf-8'),
);

describe('replay clips', () => {
  it('decode into full 543-point frames', () => {
    const { clips, source } = decodeReplay(raw);
    expect(clips.length).toBeGreaterThanOrEqual(4);
    for (const clip of clips) {
      expect(clip.frames.length).toBeGreaterThanOrEqual(8);
      expect(clip.frames[0]?.length).toBe(N_POINTS * 3);
      // Synthetic placeholders have no sign; real held-out clips have one and a credit.
      if (source === 'synthetic') expect(clip.sign).toBeNull();
      else expect(clip.sign && clip.credit).toBeTruthy();
    }
  });

  it('require the dataset credit on real clips', () => {
    const real = { ...(raw as object), source: 'asl-signs', credit: null };
    expect(() => decodeReplay(real)).toThrow(/credit/);
    const credited = decodeReplay({ ...(raw as object), source: 'asl-signs', credit: 'CC BY 4.0' });
    expect(credited.clips[0]?.credit).toBe('CC BY 4.0');
    const synthetic = { ...(raw as object), source: 'synthetic', credit: null };
    expect(decodeReplay(synthetic).clips[0]?.credit).toBeNull();
  });

  it('include a left-handed signer (synthetic placeholders)', () => {
    const { clips, source } = decodeReplay(raw);
    if (source !== 'synthetic') return;
    const handed = clips.map((clip) => {
      const left = clip.frames.filter((f) => Number.isFinite(f[LEFT_HAND_START * 3])).length;
      const right = clip.frames.filter((f) => Number.isFinite(f[RIGHT_HAND_START * 3])).length;
      return left > right ? 'left' : 'right';
    });
    expect(handed).toContain('left');
    expect(handed).toContain('right');
  });

  it('load through one function, relative to the site base', async () => {
    const urls: string[] = [];
    const fake = ((url: string) => {
      urls.push(url);
      return Promise.resolve(new Response(JSON.stringify(raw)));
    }) as typeof fetch;
    const clips = await loadReplayClips(fake, '/GestureSign/');
    expect(urls).toEqual(['/GestureSign/replay/clips.json']);
    expect(clips.length).toBeGreaterThan(0);
  });

  it('report a failed download', async () => {
    const fake = (() => Promise.resolve(new Response('', { status: 404 }))) as typeof fetch;
    await expect(loadReplayClips(fake, '/')).rejects.toThrow('404');
  });
});

describe('a damaged replay file', () => {
  const clip = (patch: Record<string, unknown> = {}) => ({
    id: 'c1',
    title: 'hello',
    sign: 'hello',
    note: '',
    fps: 30,
    frames: [
      [0.5, 0.5, null, null],
      [0.4, 0.6, 0.1, 0.2],
    ],
    ...patch,
  });
  const file = (patch: Record<string, unknown> = {}, clipPatch: Record<string, unknown> = {}) => ({
    source: 'synthetic',
    points: [0, 468],
    clips: [clip(clipPatch)],
    ...patch,
  });

  it('decodes a small valid file', () => {
    const { clips } = decodeReplay(file());
    expect(clips[0]?.frames[1]?.[468 * 3 + 1]).toBeCloseTo(0.2);
    expect(Number.isNaN(clips[0]?.frames[0]?.[468 * 3] ?? 0)).toBe(true);
  });

  it.each([
    ['a point outside the 543-point frame', { points: [0, 543] }],
    ['a negative point', { points: [-1, 0] }],
    ['a fractional point', { points: [0.5, 1] }],
    ['a point that is not a number', { points: ['0', 1] }],
    ['no clips array', { clips: 'none' }],
    ['too many clips', { clips: Array.from({ length: 101 }, () => clip()) }],
  ])('is refused when it has %s', (_, patch) => {
    expect(() => decodeReplay(file(patch))).toThrow(/replay clips/);
  });

  it.each([
    ['a frame rate of zero', { fps: 0 }],
    ['a negative frame rate', { fps: -30 }],
    ['an absurd frame rate', { fps: 1e9 }],
    ['no frames', { frames: [] }],
    ['a frame that is not an array', { frames: [null] }],
    ['a frame of the wrong length', { frames: [[0.5, 0.5]] }],
    ['a coordinate that is a string', { frames: [['0.5', 0.5, null, null]] }],
    ['an infinite coordinate', { frames: [[Number.POSITIVE_INFINITY, 0.5, null, null]] }],
    ['too many frames', { frames: Array.from({ length: 5000 }, () => [0, 0, 0, 0]) }],
    ['a missing id', { id: undefined }],
    ['a title that is an object', { title: { toString: 'x' } }],
  ])('is refused when a clip has %s', (_, patch) => {
    expect(() => decodeReplay(file({}, patch))).toThrow(/replay clips/);
  });
});
