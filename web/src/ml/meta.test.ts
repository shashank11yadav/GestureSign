import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MetaError, dictionaryUrl, parseLabels, parseMeta } from './meta';

const read = (name: string): unknown =>
  JSON.parse(readFileSync(new URL(`../../public/models/${name}`, import.meta.url), 'utf-8'));

describe('model files', () => {
  it('accepts the committed model-meta.json and labels.json', () => {
    const meta = parseMeta(read('model-meta.json'));
    const labels = parseLabels(read('labels.json'), meta.classes);
    expect(labels).toHaveLength(meta.classes);
    expect(meta.calibration.notSureBelow).toBeGreaterThan(0);
  });

  it('rejects a model built for another feature version', () => {
    const meta = { ...(read('model-meta.json') as object), featureVersion: 99 };
    expect(() => parseMeta(meta)).toThrow(MetaError);
  });

  it('rejects a feature size that does not match the feature set', () => {
    const meta = { ...(read('model-meta.json') as object), featureDim: 10 };
    expect(() => parseMeta(meta)).toThrow(/featureDim/);
  });

  it('rejects labels that do not match the class count', () => {
    expect(() => parseLabels(read('labels.json'), 3)).toThrow(/count/);
  });

  it('links signs to a dictionary search without fetching anything', () => {
    const url = dictionaryUrl({
      index: 0,
      sign: 'thankyou',
      label: 'thank you',
      lookup: 'thank you',
      shipped: true,
      heldOutTop1: null,
    });
    expect(url).toBe('https://www.signingsavvy.com/search/thank%20you');
  });
});
