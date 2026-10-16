/**
 * e2e-only: lets Playwright drive the live pipeline with landmark fixtures
 * instead of a camera. Imported only when import.meta.env.MODE === 'test'
 * (`npm run build:test`), so production builds do not contain it.
 */
import type { Session } from './app/session';
import { POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER, POSE_START, emptyFrame } from './ml/landmarks';
import { loadReplayClips } from './replay/clips';

export interface TestHook {
  modelReady(): boolean;
  /** Push replay clip `index` through the pipeline at once, followed by a quiet tail. */
  injectClip(index: number): Promise<void>;
  /** Push `count` frames in which nothing is detected. */
  injectEmpty(count: number): void;
  /** Push replay clip `index` with only the shoulders kept out of view (a partial signer). */
  injectPartial(index: number): Promise<void>;
}

declare global {
  interface Window {
    __gesturesignTest?: TestHook;
  }
}

export function installTestHook(session: Session): void {
  const clips = loadReplayClips();
  window.__gesturesignTest = {
    modelReady: () => session.getSnapshot().model.status === 'ready',
    async injectClip(index) {
      const clip = (await clips)[index];
      if (!clip) throw new Error(`no replay clip ${index}`);
      session.inject(clip.frames, clip.fps);
    },
    injectEmpty(count) {
      session.inject(
        Array.from({ length: count }, () => emptyFrame()),
        30,
        0,
      );
    },
    async injectPartial(index) {
      const clip = (await clips)[index];
      if (!clip) throw new Error(`no replay clip ${index}`);
      const frames = clip.frames.map((f) => {
        const copy = f.slice();
        // Shoulders below the bottom edge of the image.
        copy[(POSE_START + POSE_LEFT_SHOULDER) * 3 + 1] = 1.3;
        copy[(POSE_START + POSE_RIGHT_SHOULDER) * 3 + 1] = 1.3;
        return copy;
      });
      session.inject(frames, clip.fps);
    },
  };
  document.documentElement.dataset['testHook'] = 'ready';
}
