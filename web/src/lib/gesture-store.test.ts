import { IDBFactory } from 'fake-indexeddb';
import { describe, expect, it } from 'vitest';
import { GestureStore } from './gesture-store';

const take = (...v: number[]) => Float32Array.from(v);

describe('GestureStore', () => {
  it('saves and loads gestures for the current model', async () => {
    const store = new GestureStore(new IDBFactory());
    await store.save({ id: 'a', name: 'wave', takes: [take(1, 0), take(0.5, 0.5)] }, 'v1');
    const { gestures, discarded } = await store.load('v1');
    expect(discarded).toBe(0);
    expect(gestures).toHaveLength(1);
    expect(gestures[0]?.name).toBe('wave');
    expect(Array.from(gestures[0]?.takes[1] ?? [])).toEqual([0.5, 0.5]);
  });

  it('drops gestures saved under another model version', async () => {
    const factory = new IDBFactory();
    const store = new GestureStore(factory);
    await store.save({ id: 'a', name: 'wave', takes: [take(1)] }, 'stub');
    expect(await store.load('trained')).toEqual({ gestures: [], discarded: 1 });
    expect((await new GestureStore(factory).load('stub')).gestures).toEqual([]);
  });

  it('removes one gesture or clears them all', async () => {
    const store = new GestureStore(new IDBFactory());
    await store.save({ id: 'a', name: 'a', takes: [] }, 'v');
    await store.save({ id: 'b', name: 'b', takes: [] }, 'v');
    await store.remove('a');
    expect((await store.load('v')).gestures.map((g) => g.id)).toEqual(['b']);
    await store.clear();
    expect((await store.load('v')).gestures).toEqual([]);
  });
});
