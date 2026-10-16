/**
 * Taught gestures live in this browser's IndexedDB only, as embedding
 * vectors (never video). Embeddings only make sense for the model that
 * produced them, so gestures saved under another model version are dropped
 * on load and the page says so.
 */
import type { Gesture } from '../ml/prototypes';

const DB_NAME = 'gesturesign';
const STORE = 'gestures';

interface StoredGesture {
  id: string;
  name: string;
  takes: number[][];
  modelVersion: string;
  updatedAt: number;
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => {
      resolve(req.result);
    };
    req.onerror = () => {
      reject(req.error ?? new Error('IndexedDB request failed'));
    };
  });
}

export class GestureStore {
  private db: Promise<IDBDatabase> | null = null;

  constructor(private readonly factory: IDBFactory = globalThis.indexedDB) {}

  private open(): Promise<IDBDatabase> {
    this.db ??= new Promise((resolve, reject) => {
      const req = this.factory.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore(STORE, { keyPath: 'id' });
      };
      req.onsuccess = () => {
        resolve(req.result);
      };
      req.onerror = () => {
        reject(req.error ?? new Error('IndexedDB is unavailable'));
      };
    });
    return this.db;
  }

  private async store(mode: IDBTransactionMode): Promise<IDBObjectStore> {
    return (await this.open()).transaction(STORE, mode).objectStore(STORE);
  }

  /** Gestures saved for this model version; others are deleted and counted. */
  async load(modelVersion: string): Promise<{ gestures: Gesture[]; discarded: number }> {
    const all = await request(
      (await this.store('readonly')).getAll() as IDBRequest<StoredGesture[]>,
    );
    const keep = all.filter((g) => g.modelVersion === modelVersion);
    const stale = all.filter((g) => g.modelVersion !== modelVersion);
    if (stale.length) {
      const store = await this.store('readwrite');
      await Promise.all(stale.map((g) => request(store.delete(g.id))));
    }
    keep.sort((a, b) => a.updatedAt - b.updatedAt);
    return {
      gestures: keep.map((g) => ({
        id: g.id,
        name: g.name,
        takes: g.takes.map((t) => Float32Array.from(t)),
      })),
      discarded: stale.length,
    };
  }

  async save(gesture: Gesture, modelVersion: string): Promise<void> {
    const stored: StoredGesture = {
      id: gesture.id,
      name: gesture.name,
      takes: gesture.takes.map((t) => Array.from(t)),
      modelVersion,
      updatedAt: Date.now(),
    };
    await request((await this.store('readwrite')).put(stored));
  }

  async remove(id: string): Promise<void> {
    await request((await this.store('readwrite')).delete(id));
  }

  async clear(): Promise<void> {
    await request((await this.store('readwrite')).clear());
  }
}
