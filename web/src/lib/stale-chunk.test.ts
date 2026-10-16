import { describe, expect, it, vi } from 'vitest';
import { reloadOnStaleChunk } from './stale-chunk';

function fakeWindow(storage: Map<string, string> | null) {
  const target = new EventTarget();
  const reload = vi.fn();
  const win = {
    addEventListener: target.addEventListener.bind(target),
    location: { reload },
    get sessionStorage() {
      if (!storage) throw new Error('blocked');
      return {
        getItem: (k: string) => storage.get(k) ?? null,
        setItem: (k: string, v: string) => storage.set(k, v),
      };
    },
  } as unknown as Window;
  const fire = () => {
    const event = new Event('vite:preloadError', { cancelable: true });
    target.dispatchEvent(event);
    return event;
  };
  return { win, reload, fire };
}

describe('reloadOnStaleChunk', () => {
  it('reloads once when a chunk from an older build is missing', () => {
    const { win, reload, fire } = fakeWindow(new Map());
    reloadOnStaleChunk(win);
    expect(fire().defaultPrevented).toBe(true);
    expect(reload).toHaveBeenCalledTimes(1);
    fire();
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('does nothing without session storage, to avoid a reload loop', () => {
    const { win, reload, fire } = fakeWindow(null);
    reloadOnStaleChunk(win);
    fire();
    expect(reload).not.toHaveBeenCalled();
  });
});
