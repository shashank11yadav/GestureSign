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
    expect(source).toBe('synthetic');
    expect(clips.length).toBeGreaterThanOrEqual(4);
    for (const clip of clips) {
      expect(clip.frames.length).toBeGreaterThan(20);
      expect(clip.frames[0]?.length).toBe(N_POINTS * 3);
      expect(clip.sign).toBeNull();
    }
  });

  it('require the dataset credit on real clips', () => {
    const real = { ...(raw as object), source: 'asl-signs', credit: null };
    expect(() => decodeReplay(real)).toThrow(/credit/);
    const credited = decodeReplay({ ...(raw as object), source: 'asl-signs', credit: 'CC BY 4.0' });
    expect(credited.clips[0]?.credit).toBe('CC BY 4.0');
    expect(decodeReplay(raw).clips[0]?.credit).toBeNull();
  });

  it('include a left-handed signer', () => {
    const { clips } = decodeReplay(raw);
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
