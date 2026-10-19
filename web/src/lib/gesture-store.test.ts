import { IDBFactory } from 'fake-indexeddb';
import { describe, expect, it } from 'vitest';
import { MAX_GESTURES } from '../ml/prototypes';
import { GestureStore } from './gesture-store';

const take = (...v: number[]) => Float32Array.from(v);

/** Writes raw records straight into the store, as a damaged or tampered database would hold. */
async function plant(factory: IDBFactory, records: unknown[]): Promise<void> {
  await new GestureStore(factory).clear();
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const req = factory.open('gesturesign', 1);
    req.onsuccess = () => {
      resolve(req.result);
    };
    req.onerror = () => {
      reject(new Error('open failed'));
    };
  });
  const tx = db.transaction('gestures', 'readwrite');
  for (const record of records) tx.objectStore('gestures').put(record);
  await new Promise<void>((resolve) => {
    tx.oncomplete = () => {
      resolve();
    };
  });
  db.close();
}

const valid = (id: string, patch: Record<string, unknown> = {}) => ({
  id,
  name: id,
  takes: [[1, 0]],
  modelVersion: 'v',
  updatedAt: 1,
  ...patch,
});

describe('GestureStore', () => {
  it('saves and loads gestures for the current model', async () => {
    const store = new GestureStore(new IDBFactory());
    await store.save({ id: 'a', name: 'wave', takes: [take(1, 0), take(0.5, 0.5)] }, 'v1');
    const { gestures, discarded, unreadable } = await store.load('v1', 2);
    expect(discarded).toBe(0);
    expect(unreadable).toBe(0);
    expect(gestures).toHaveLength(1);
    expect(gestures[0]?.name).toBe('wave');
    expect(Array.from(gestures[0]?.takes[1] ?? [])).toEqual([0.5, 0.5]);
  });

  it('drops gestures saved under another model version', async () => {
    const factory = new IDBFactory();
    const store = new GestureStore(factory);
    await store.save({ id: 'a', name: 'wave', takes: [take(1)] }, 'stub');
    expect(await store.load('trained', 1)).toEqual({ gestures: [], discarded: 1, unreadable: 0 });
    expect((await new GestureStore(factory).load('stub', 1)).gestures).toEqual([]);
  });

  it('removes one gesture or clears them all', async () => {
    const store = new GestureStore(new IDBFactory());
    await store.save({ id: 'a', name: 'a', takes: [] }, 'v');
    await store.save({ id: 'b', name: 'b', takes: [] }, 'v');
    await store.remove('a');
    expect((await store.load('v', 2)).gestures.map((g) => g.id)).toEqual(['b']);
    await store.clear();
    expect((await store.load('v', 2)).gestures).toEqual([]);
  });

  it('removes records it cannot read instead of failing', async () => {
    const factory = new IDBFactory();
    await plant(factory, [
      valid('ok'),
      valid('no-name', { name: undefined }),
      valid('long-name', { name: 'x'.repeat(10_000) }),
      valid('takes-not-array', { takes: 'oops' }),
      valid('take-wrong-length', { takes: [[1, 0, 0]] }),
      valid('take-huge', { takes: [new Array<number>(1_000_000).fill(0)] }),
      valid('take-nan', { takes: [[Number.NaN, 0]] }),
      valid('take-string', { takes: [['1', 0]] }),
      valid('too-many-takes', { takes: Array.from({ length: 50 }, () => [1, 0]) }),
      valid('bad-time', { updatedAt: 'yesterday' }),
      { id: 7, name: 'numeric id', takes: [], modelVersion: 'v', updatedAt: 1 },
    ]);
    const store = new GestureStore(factory);
    const { gestures, discarded, unreadable } = await store.load('v', 2);
    expect(gestures.map((g) => g.id)).toEqual(['ok']);
    expect(discarded).toBe(0);
    expect(unreadable).toBe(10);
    // They are gone, so the next load is clean.
    expect(await new GestureStore(factory).load('v', 2)).toMatchObject({ unreadable: 0 });
  });

  it('loads no more gestures than the page allows, oldest first', async () => {
    const factory = new IDBFactory();
    const many = Array.from({ length: MAX_GESTURES + 4 }, (_, i) =>
      valid(`g${String(i).padStart(2, '0')}`, { updatedAt: 100 - i }),
    );
    await plant(factory, many);
    const { gestures } = await new GestureStore(factory).load('v', 2);
    expect(gestures).toHaveLength(MAX_GESTURES);
    expect(gestures[0]?.id).toBe(`g${MAX_GESTURES + 3}`);
  });
});
