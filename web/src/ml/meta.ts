/**
 * model-meta.json and labels.json, written by `gesturesign export`. They are
 * validated before use: a model built for a different feature version or input
 * size would silently produce nonsense, so the page refuses it instead.
 */
import { FEATURE_VERSION, featureLayout, type FeatureSet } from './features';
import type { Calibration } from './decision';
import type { TeachThresholds } from './prototypes';

export interface ModelMetrics {
  top1: number;
  top5: number;
  testClips: number;
  shippedSigns: number;
  shippedTop1: number;
  shippedTop3: number;
  teachAccuracy: number;
}

export interface ModelMeta {
  version: string;
  featureVersion: number;
  featureSet: FeatureSet;
  fps: number;
  minFrames: number;
  maxFrames: number;
  featureDim: number;
  classes: number;
  embeddingDim: number;
  trainedOn: unknown;
  metrics: ModelMetrics | null;
  calibration: Calibration;
  teach: TeachThresholds;
}

export interface SignLabel {
  index: number;
  sign: string;
  label: string;
  lookup: string;
  shipped: boolean;
  heldOutTop1: number | null;
}

export class MetaError extends Error {
  override name = 'MetaError';
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function num(o: Record<string, unknown>, key: string): number {
  const v = o[key];
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new MetaError(`model-meta: bad ${key}`);
  return v;
}

export function parseMeta(raw: unknown): ModelMeta {
  if (!isRecord(raw)) throw new MetaError('model-meta: not an object');
  const featureSet = raw['featureSet'];
  if (featureSet !== 'hands_pose' && featureSet !== 'hands_pose_lips') {
    throw new MetaError('model-meta: unknown featureSet');
  }
  const featureVersion = num(raw, 'featureVersion');
  if (featureVersion !== FEATURE_VERSION) {
    throw new MetaError(
      `model-meta: feature version ${featureVersion}, but this page computes version ${FEATURE_VERSION}`,
    );
  }
  const featureDim = num(raw, 'featureDim');
  if (featureDim !== featureLayout(featureSet).sequenceDim) {
    throw new MetaError('model-meta: featureDim does not match the feature set');
  }
  const cal = raw['calibration'];
  const teach = raw['teach'];
  if (!isRecord(cal) || !isRecord(teach)) throw new MetaError('model-meta: missing thresholds');
  const metrics = raw['metrics'];
  if (metrics !== null && !isRecord(metrics)) throw new MetaError('model-meta: bad metrics');
  const version = raw['version'];
  if (typeof version !== 'string') throw new MetaError('model-meta: bad version');
  return {
    version,
    featureVersion,
    featureSet,
    fps: num(raw, 'fps'),
    minFrames: num(raw, 'minFrames'),
    maxFrames: num(raw, 'maxFrames'),
    featureDim,
    classes: num(raw, 'classes'),
    embeddingDim: num(raw, 'embeddingDim'),
    trainedOn: raw['trainedOn'] ?? null,
    metrics: metrics === null ? null : (metrics as unknown as ModelMetrics),
    calibration: { temperature: num(cal, 'temperature'), notSureBelow: num(cal, 'notSureBelow') },
    teach: { minSimilarity: num(teach, 'minSimilarity'), margin: num(teach, 'margin') },
  };
}

export function parseLabels(raw: unknown, classes: number): SignLabel[] {
  if (!isRecord(raw) || !Array.isArray(raw['signs'])) throw new MetaError('labels: no signs');
  const signs = raw['signs'].map((s: unknown, i): SignLabel => {
    if (!isRecord(s) || s['index'] !== i || typeof s['sign'] !== 'string') {
      throw new MetaError(`labels: bad entry ${i}`);
    }
    const top1 = s['heldOutTop1'];
    return {
      index: i,
      sign: s['sign'],
      label: typeof s['label'] === 'string' ? s['label'] : s['sign'],
      lookup: typeof s['lookup'] === 'string' ? s['lookup'] : s['sign'],
      shipped: s['shipped'] === true,
      heldOutTop1: typeof top1 === 'number' ? top1 : null,
    };
  });
  if (signs.length !== classes) throw new MetaError('labels: count does not match the model');
  return signs;
}

/** A sign's entry in an external ASL dictionary (a plain link; nothing is fetched). */
export function dictionaryUrl(label: SignLabel): string {
  return `https://www.signingsavvy.com/search/${encodeURIComponent(label.lookup.toLowerCase())}`;
}
