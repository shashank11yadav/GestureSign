/**
 * Runs MediaPipe off the main thread. The page sends one ImageBitmap at a
 * time and drops camera frames while one is in flight, so a slow device
 * processes fewer frames instead of falling behind.
 */
import { createTracker, type Tracker } from './tracker-core';
import { isImageBitmap, parseTrackerRequest, type TrackerWorkerOut } from './tracker-protocol';
import { isRecord, isTrustedOrigin } from './worker-guard';

interface WorkerScope {
  location: { origin: string };
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage(message: TrackerWorkerOut, transfer?: Transferable[]): void;
}
const scope = self as unknown as WorkerScope;

let tracker: Tracker | null = null;

scope.onmessage = (event) => {
  if (!isTrustedOrigin(event.origin, scope.location.origin)) return;
  const parsed = parseTrackerRequest(event.data);
  if (!parsed.ok) {
    // A transferred bitmap belongs to this worker now, so free it even when the message is bad.
    const bitmap = isRecord(event.data) ? event.data['bitmap'] : undefined;
    if (isImageBitmap(bitmap)) bitmap.close();
    if (parsed.reply) scope.postMessage(parsed.reply);
    return;
  }
  const msg = parsed.message;
  if (msg.type === 'init') {
    createTracker({ featureSet: msg.featureSet, delegate: msg.delegate })
      .then((created) => {
        tracker = created;
        scope.postMessage({ type: 'ready', delegate: created.delegate });
      })
      .catch((error: unknown) => {
        scope.postMessage({
          type: 'init-error',
          message: error instanceof Error ? error.message : String(error),
        });
      });
    return;
  }
  const { bitmap, t } = msg;
  try {
    if (!tracker) throw new Error('tracker not ready');
    const started = performance.now();
    const detected = tracker.detect(bitmap, t);
    const ms = performance.now() - started;
    scope.postMessage({ type: 'landmarks', t, ms, ...detected }, [detected.frame.buffer]);
  } catch (error) {
    scope.postMessage({
      type: 'frame-error',
      t,
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    bitmap.close();
  }
};
