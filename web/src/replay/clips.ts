/**
 * Replay mode's clips: landmark sequences played through the same pipeline as
 * the camera, for visitors without a camera (or who would rather not use one).
 *
 * Everything goes through loadReplayClips(). Today it reads
 * public/replay/clips.json, written by the pipeline: synthetic motion
 * patterns from `gesturesign export-replay` until a model is trained, then
 * real held-out-signer clips of shipped signs from `gesturesign replay-clips`
 * (asl-signs is CC BY 4.0, so each real clip carries the dataset credit).
 * Swapping the source means changing only the pipeline command or this function.
 */
import { emptyFrame, N_POINTS, type Frame } from '../ml/landmarks';

export interface ReplayClip {
  id: string;
  title: string;
  /** The dataset sign, or null for a synthetic motion pattern. */
  sign: string | null;
  note: string;
  /** Attribution to show with the clip (real dataset clips only). */
  credit: string | null;
  fps: number;
  frames: Frame[];
}

export interface ReplayFile {
  source: 'synthetic' | 'asl-signs';
  clips: ReplayClip[];
}

/**
 * Limits well above what the pipeline writes (64 frames at 30 fps, six
 * clips), so a damaged or hostile file is refused instead of stalling the
 * page. In particular a frame rate of zero would make playback spin forever.
 */
const MAX_CLIPS = 100;
const MAX_FRAMES = 1000;
const MIN_FPS = 1;
const MAX_FPS = 240;

class ReplayFormatError extends Error {
  constructor(what: string) {
    super(`replay clips: ${what}`);
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isPointList(v: unknown): v is number[] {
  return (
    Array.isArray(v) &&
    v.length <= N_POINTS &&
    v.every((p) => typeof p === 'number' && Number.isInteger(p) && p >= 0 && p < N_POINTS)
  );
}

function decodeFrame(flat: unknown, points: readonly number[]): Frame {
  if (!Array.isArray(flat) || flat.length !== 2 * points.length) {
    throw new ReplayFormatError('bad frame');
  }
  const frame = emptyFrame();
  points.forEach((point, i) => {
    const x: unknown = flat[2 * i];
    const y: unknown = flat[2 * i + 1];
    if (x === null || y === null) return;
    if (typeof x !== 'number' || typeof y !== 'number' || !Number.isFinite(x + y)) {
      throw new ReplayFormatError('bad coordinate');
    }
    frame[point * 3] = x;
    frame[point * 3 + 1] = y;
    frame[point * 3 + 2] = 0;
  });
  return frame;
}

function decodeClip(c: unknown, points: readonly number[], credit: string | null): ReplayClip {
  if (!isRecord(c) || typeof c['id'] !== 'string' || typeof c['title'] !== 'string') {
    throw new ReplayFormatError('bad clip');
  }
  const { fps, frames } = c;
  if (typeof fps !== 'number' || !(fps >= MIN_FPS && fps <= MAX_FPS)) {
    throw new ReplayFormatError('bad frame rate');
  }
  if (!Array.isArray(frames) || frames.length === 0 || frames.length > MAX_FRAMES) {
    throw new ReplayFormatError('bad frame count');
  }
  return {
    id: c['id'],
    title: c['title'],
    sign: typeof c['sign'] === 'string' ? c['sign'] : null,
    note: typeof c['note'] === 'string' ? c['note'] : '',
    credit,
    fps,
    frames: frames.map((flat) => decodeFrame(flat, points)),
  };
}

export function decodeReplay(raw: unknown): ReplayFile {
  if (!isRecord(raw) || !isPointList(raw['points']) || !Array.isArray(raw['clips'])) {
    throw new ReplayFormatError('unexpected format');
  }
  const points = raw['points'];
  const source = raw['source'] === 'asl-signs' ? 'asl-signs' : 'synthetic';
  const credit = typeof raw['credit'] === 'string' ? raw['credit'] : null;
  if (source === 'asl-signs' && !credit) throw new ReplayFormatError('dataset clips need a credit');
  if (raw['clips'].length > MAX_CLIPS) throw new ReplayFormatError('too many clips');
  const clips = raw['clips'].map((c: unknown) => decodeClip(c, points, credit));
  return { source, clips };
}

export async function loadReplayClips(
  fetchImpl: typeof fetch = fetch,
  base: string = import.meta.env.BASE_URL,
): Promise<ReplayClip[]> {
  const response = await fetchImpl(`${base}replay/clips.json`);
  if (!response.ok) throw new Error(`replay clips: HTTP ${response.status}`);
  return decodeReplay(await response.json()).clips;
}
