/**
 * Taught gestures live in this browser's IndexedDB only, as embedding
 * vectors (never video). Embeddings only make sense for the model that
 * produced them, so gestures saved under another model version are dropped
 * on load and the page says so.
 *
 * Stored records are checked on load like any other input: one that is
 * damaged (or edited by hand in DevTools) is removed rather than allowed to
 * break the page, and nothing larger than the page itself would save is read.
 */
import { MAX_GESTURES, MAX_NAME_LENGTH, MAX_TAKES, type Gesture } from '../ml/prototypes';

const DB_NAME = 'gesturesign';
const STORE = 'gestures';

interface StoredGesture {
  id: string;
  name: string;
  takes: number[][];
  modelVersion: string;
  updatedAt: number;
}

/** IDs are crypto.randomUUID(); this is generous. */
const MAX_ID_LENGTH = 100;

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isTake(v: unknown, dim: number): v is number[] {
  return (
    Array.isArray(v) &&
    v.length === dim &&
    v.every((x) => typeof x === 'number' && Number.isFinite(x))
  );
}

/** The record if it has the shape save() writes, else null. */
export function readStoredGesture(raw: unknown, embeddingDim: number): StoredGesture | null {
  if (!isRecord(raw)) return null;
  const { id, name, takes, modelVersion, updatedAt } = raw;
  if (typeof id !== 'string' || id.length === 0 || id.length > MAX_ID_LENGTH) return null;
  if (typeof name !== 'string' || name.length === 0 || name.length > MAX_NAME_LENGTH) return null;
  if (typeof modelVersion !== 'string') return null;
  if (typeof updatedAt !== 'number' || !Number.isFinite(updatedAt)) return null;
  if (!Array.isArray(takes) || takes.length > MAX_TAKES) return null;
  if (!takes.every((t) => isTake(t, embeddingDim))) return null;
  return { id, name, takes, modelVersion, updatedAt };
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

  /** Run writes in one transaction and resolve only once it has committed. */
  private async write(run: (store: IDBObjectStore) => void): Promise<void> {
    const tx = (await this.open()).transaction(STORE, 'readwrite');
    const done = new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => {
        resolve();
      };
      tx.onerror = () => {
        reject(tx.error ?? new Error('IndexedDB write failed'));
      };
      tx.onabort = () => {
        reject(tx.error ?? new Error('IndexedDB write aborted'));
      };
    });
    run(tx.objectStore(STORE));
    return done;
  }

  /**
   * Gestures saved for this model version, oldest first and at most
   * MAX_GESTURES. Records for another model version (`discarded`) and records
   * that cannot be read (`unreadable`) are deleted and counted.
   */
  async load(
    modelVersion: string,
    embeddingDim: number,
  ): Promise<{ gestures: Gesture[]; discarded: number; unreadable: number }> {
    const store = await this.store('readonly');
    const [keys, values] = await Promise.all([
      request(store.getAllKeys()),
      request(store.getAll() as IDBRequest<unknown[]>),
    ]);
    const keep: StoredGesture[] = [];
    const stale: IDBValidKey[] = [];
    const broken: IDBValidKey[] = [];
    keys.forEach((key, i) => {
      const g = readStoredGesture(values[i], embeddingDim);
      if (!g || g.id !== key) broken.push(key);
      else if (g.modelVersion === modelVersion) keep.push(g);
      else stale.push(key);
    });
    if (stale.length || broken.length) {
      await this.write((s) => {
        for (const key of [...stale, ...broken]) s.delete(key);
      });
    }
    keep.sort((a, b) => a.updatedAt - b.updatedAt);
    return {
      gestures: keep.slice(0, MAX_GESTURES).map((g) => ({
        id: g.id,
        name: g.name,
        takes: g.takes.map((t) => Float32Array.from(t)),
      })),
      discarded: stale.length,
      unreadable: broken.length,
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
    await this.write((store) => store.put(stored));
  }

  async remove(id: string): Promise<void> {
    await this.write((store) => store.delete(id));
  }

  async clear(): Promise<void> {
    await this.write((store) => store.clear());
  }
}
