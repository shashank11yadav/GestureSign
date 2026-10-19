/**
 * MediaPipe landmarkers -> 543-point frames. Shared by the tracker worker and
 * the dev-only benchmark page.
 *
 * Which landmarkers run depends on the model's feature set (decided by the
 * landmark benchmark, pipeline/reports/bench.md): `hands_pose` runs the Hand
 * and Pose landmarkers; `hands_pose_lips` runs the Holistic landmarker, which
 * also gives the face mesh the lip features need. Every file comes from this
 * site (assets.ts).
 */
import {
  FilesetResolver,
  HandLandmarker,
  HolisticLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { assembleFrame, type DetectedHand } from './assemble';
import { assetUrl } from './assets';
import type { FeatureSet } from './features';
import type { Frame } from './landmarks';

export type Delegate = 'GPU' | 'CPU';

export interface TrackerOptions {
  featureSet: FeatureSet;
  delegate: Delegate;
}

export interface Detected {
  frame: Frame;
  poses: number;
  hands: number;
}

export interface Tracker {
  delegate: Delegate;
  detect(image: ImageBitmap | HTMLVideoElement, timestampMs: number): Detected;
  close(): void;
}

type Image = ImageBitmap | HTMLVideoElement;

/**
 * In a module worker MediaPipe loads its WASM glue with import(), which needs
 * the ES-module build; on the main thread it injects a classic <script>,
 * which needs the classic build (only the dev benchmark does that).
 */
const useModule = typeof document === 'undefined';

interface Prefetched {
  stream(): ReadableStreamDefaultReader<Uint8Array>;
}

/**
 * Starts downloading a .task model now and hands MediaPipe a stream that
 * yields it once complete. MediaPipe loads its WASM runtime before it reads
 * the model, so the two downloads run in parallel instead of one after the
 * other. Call stream() again for a retry (e.g. on the CPU delegate).
 */
function prefetch(url: string): Prefetched {
  const bytes = fetch(url).then(async (response) => {
    if (!response.ok) throw new Error(`${url.split('/').pop() ?? url}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
  });
  // Handled where the stream is read; this only keeps an early failure from being "unhandled".
  bytes.catch(() => undefined);
  return {
    stream: () =>
      new ReadableStream<Uint8Array>({
        async start(controller) {
          try {
            controller.enqueue(await bytes);
            controller.close();
          } catch (error) {
            controller.error(error);
          }
        },
      }).getReader(),
  };
}

const points = (list: NormalizedLandmark[] | undefined) =>
  (list ?? []).map((p) => ({ x: p.x, y: p.y, z: p.z }));

async function handsAndPose(
  delegate: Delegate,
  models: { hand: Prefetched; pose: Prefetched },
): Promise<Tracker> {
  const fileset = await FilesetResolver.forVisionTasks(assetUrl('mediapipeWasm'), useModule);
  const hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: models.hand.stream(), delegate },
    runningMode: 'VIDEO',
    numHands: 2,
  });
  const pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: models.pose.stream(), delegate },
    runningMode: 'VIDEO',
    // Two, so a second person in view is noticed (and recognition pauses).
    numPoses: 2,
  });
  return {
    delegate,
    detect(image: Image, t: number): Detected {
      const h = hand.detectForVideo(image, t);
      const p = pose.detectForVideo(image, t);
      const hands: DetectedHand[] = h.landmarks.map((lm, i) => ({
        points: points(lm),
        label: h.handedness[i]?.[0]?.categoryName ?? '',
      }));
      const firstPose = p.landmarks[0];
      return {
        frame: assembleFrame({ hands, pose: firstPose ? points(firstPose) : null }),
        poses: p.landmarks.length,
        hands: hands.length,
      };
    },
    close() {
      hand.close();
      pose.close();
    },
  };
}

async function holistic(delegate: Delegate, task: Prefetched): Promise<Tracker> {
  const fileset = await FilesetResolver.forVisionTasks(assetUrl('mediapipeWasm'), useModule);
  const model = await HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: task.stream(), delegate },
    runningMode: 'VIDEO',
  });
  return {
    delegate,
    detect(image: Image, t: number): Detected {
      const r = model.detectForVideo(image, t);
      const hands: DetectedHand[] = [];
      const left = r.leftHandLandmarks[0];
      const right = r.rightHandLandmarks[0];
      if (left) hands.push({ points: points(left), side: 'left' });
      if (right) hands.push({ points: points(right), side: 'right' });
      const pose = r.poseLandmarks[0];
      const face = r.faceLandmarks[0];
      return {
        frame: assembleFrame({
          hands,
          pose: pose ? points(pose) : null,
          face: face ? points(face) : null,
        }),
        poses: r.poseLandmarks.length,
        hands: hands.length,
      };
    },
    close() {
      model.close();
    },
  };
}

/** Builds the landmarkers, falling back from the GPU to the CPU delegate. */
export async function createTracker(opts: TrackerOptions): Promise<Tracker> {
  const models = assetUrl('mediapipeModels');
  let build: (delegate: Delegate) => Promise<Tracker>;
  if (opts.featureSet === 'hands_pose_lips') {
    const model = prefetch(`${models}holistic_landmarker.task`);
    build = (delegate) => holistic(delegate, model);
  } else {
    const tasks = {
      hand: prefetch(`${models}hand_landmarker.task`),
      pose: prefetch(`${models}pose_landmarker_lite.task`),
    };
    build = (delegate) => handsAndPose(delegate, tasks);
  }
  if (opts.delegate === 'GPU') {
    try {
      return await build('GPU');
    } catch {
      // No WebGL2 (e.g. in a worker without OffscreenCanvas support): use the CPU.
    }
  }
  return build('CPU');
}
