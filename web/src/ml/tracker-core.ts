/**
 * MediaPipe landmarkers -> 543-point frames. Shared by the tracker worker and
 * the dev-only benchmark page.
 *
 * Which landmarkers run depends on the model's feature set (decided by the A1
 * benchmark, pipeline/reports/bench.md): `hands_pose` runs the Hand and Pose
 * landmarkers; `hands_pose_lips` runs the Holistic landmarker, which also
 * gives the face mesh the lip features need.
 */
import {
  FilesetResolver,
  HandLandmarker,
  HolisticLandmarker,
  PoseLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import { assembleFrame, type DetectedHand } from './assemble';
import type { FeatureSet } from './features';
import type { Frame } from './landmarks';

export type Delegate = 'GPU' | 'CPU';

export interface TrackerOptions {
  /** Absolute URL of the folder with the MediaPipe WASM files. */
  wasmBase: string;
  /** Absolute URL of the folder with the .task model files. */
  modelBase: string;
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

const points = (list: NormalizedLandmark[] | undefined) =>
  (list ?? []).map((p) => ({ x: p.x, y: p.y, z: p.z }));

async function handsAndPose(opts: TrackerOptions, delegate: Delegate): Promise<Tracker> {
  const fileset = await FilesetResolver.forVisionTasks(opts.wasmBase, useModule);
  const hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${opts.modelBase}hand_landmarker.task`, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
  });
  const pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${opts.modelBase}pose_landmarker_lite.task`, delegate },
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

async function holistic(opts: TrackerOptions, delegate: Delegate): Promise<Tracker> {
  const fileset = await FilesetResolver.forVisionTasks(opts.wasmBase, useModule);
  const model = await HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: `${opts.modelBase}holistic_landmarker.task`, delegate },
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
  const build = opts.featureSet === 'hands_pose_lips' ? holistic : handsAndPose;
  if (opts.delegate === 'GPU') {
    try {
      return await build(opts, 'GPU');
    } catch {
      // No WebGL2 (e.g. in a worker without OffscreenCanvas support): use the CPU.
    }
  }
  return build(opts, 'CPU');
}
