import { describe, expect, it } from 'vitest';
import { ASSETS, assetUrl, type AssetName } from './assets';

const ORIGIN = 'https://shashank11yadav.github.io';

describe('assetUrl', () => {
  it('resolves each fixed asset under the site base on this origin', () => {
    expect(assetUrl('signModel', ORIGIN, '/GestureSign/')).toBe(
      `${ORIGIN}/GestureSign/models/model.onnx`,
    );
    expect(assetUrl('mediapipeModels', ORIGIN, '/GestureSign/')).toBe(
      `${ORIGIN}/GestureSign/vendor/mediapipe/models/`,
    );
    for (const name of Object.keys(ASSETS) as AssetName[]) {
      expect(new URL(assetUrl(name, ORIGIN, '/')).origin).toBe(ORIGIN);
    }
  });

  it('works from a worker, whose location is the worker script', () => {
    expect(assetUrl('ortRuntime', `${ORIGIN}/GestureSign/assets/model.worker-abc.js`, '/x/')).toBe(
      `${ORIGIN}/x/vendor/ort/`,
    );
  });

  it('refuses a name that is not a known asset', () => {
    expect(() => assetUrl('toString' as AssetName, ORIGIN, '/')).toThrow(/unknown asset/);
    expect(() => assetUrl('../secret' as AssetName, ORIGIN, '/')).toThrow(/unknown asset/);
  });

  it('refuses a base that would point at another origin', () => {
    expect(() => assetUrl('signModel', ORIGIN, '//evil.example/')).toThrow(/same origin/);
    expect(() => assetUrl('signModel', ORIGIN, 'https://evil.example/')).toThrow(/same origin/);
  });
});
