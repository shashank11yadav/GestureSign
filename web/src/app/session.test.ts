/**
 * The camera must not keep running in a hidden tab, including when the tab
 * is hidden while the camera is still starting.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const camera = vi.hoisted(() => ({
  open: null as null | ((stream: MediaStream) => void),
}));

vi.mock('../lib/camera', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../lib/camera')>()),
  openCamera: () =>
    new Promise<MediaStream>((resolve) => {
      camera.open = resolve;
    }),
}));
vi.mock('../ml/model-client', () => ({
  loadMeta: () => new Promise(() => undefined),
  ModelClient: { load: () => new Promise(() => undefined) },
}));
vi.mock('../ml/tracker-client', () => ({
  TrackerClient: { create: () => new Promise(() => undefined) },
}));

const { Session } = await import('./session');

class FakeDocument extends EventTarget {
  visibilityState: 'visible' | 'hidden' = 'visible';
  hide() {
    this.visibilityState = 'hidden';
    this.dispatchEvent(new Event('visibilitychange'));
  }
}

function fakeStream() {
  const track = { stop: vi.fn() };
  return { stream: { getTracks: () => [track] } as unknown as MediaStream, track };
}

function fakeVideo(play: () => Promise<void>) {
  return { srcObject: null as MediaStream | null, play } as unknown as HTMLVideoElement;
}

let doc: FakeDocument;

beforeEach(() => {
  doc = new FakeDocument();
  vi.stubGlobal('document', doc);
  vi.stubGlobal('window', new EventTarget());
  vi.stubGlobal('performance', { getEntriesByName: () => [], mark: () => undefined });
});

afterEach(() => {
  vi.unstubAllGlobals();
  camera.open = null;
});

describe('Session camera', () => {
  it('stops the camera when the tab is hidden', async () => {
    const session = new Session();
    session.attachVideo(fakeVideo(() => Promise.resolve()));
    // Never settles here: the mocked tracker never starts.
    void session.startCamera();
    const { stream, track } = fakeStream();
    camera.open?.(stream);
    await vi.waitFor(() => {
      expect(session.getSnapshot().source).toBe('camera');
    });
    doc.hide();
    expect(track.stop).toHaveBeenCalled();
    expect(session.getSnapshot()).toMatchObject({ source: 'off', paused: true });
  });

  it('does not start the camera if the tab is hidden while it is starting', async () => {
    const session = new Session();
    const play: { resolve: (() => void) | null } = { resolve: null };
    session.attachVideo(
      fakeVideo(
        () =>
          new Promise<void>((resolve) => {
            play.resolve = resolve;
          }),
      ),
    );
    const starting = session.startCamera();
    const { stream, track } = fakeStream();
    camera.open?.(stream);
    // The stream is open and the video is starting to play when the tab is hidden.
    await vi.waitFor(() => {
      expect(play.resolve).not.toBeNull();
    });
    doc.hide();
    play.resolve?.();
    await starting;
    expect(track.stop).toHaveBeenCalled();
    expect(session.getSnapshot()).toMatchObject({ source: 'off', paused: true });
  });

  it('stops the camera when the page is left', async () => {
    const session = new Session();
    session.attachVideo(fakeVideo(() => Promise.resolve()));
    void session.startCamera();
    const { stream, track } = fakeStream();
    camera.open?.(stream);
    await vi.waitFor(() => {
      expect(session.getSnapshot().source).toBe('camera');
    });
    (globalThis.window as unknown as EventTarget).dispatchEvent(new Event('pagehide'));
    expect(track.stop).toHaveBeenCalled();
  });
});
