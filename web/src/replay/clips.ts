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
import { emptyFrame, type Frame } from '../ml/landmarks';

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

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function decodeReplay(raw: unknown): ReplayFile {
  if (!isRecord(raw) || !Array.isArray(raw['points']) || !Array.isArray(raw['clips'])) {
    throw new Error('replay clips: unexpected format');
  }
  const points = raw['points'] as number[];
  const source = raw['source'] === 'asl-signs' ? 'asl-signs' : 'synthetic';
  const credit = typeof raw['credit'] === 'string' ? raw['credit'] : null;
  if (source === 'asl-signs' && !credit)
    throw new Error('replay clips: dataset clips need a credit');
  const clips = raw['clips'].map((c: unknown): ReplayClip => {
    if (!isRecord(c) || !Array.isArray(c['frames']) || typeof c['fps'] !== 'number') {
      throw new Error('replay clips: bad clip');
    }
    const frames = (c['frames'] as (number | null)[][]).map((flat) => {
      const frame = emptyFrame();
      points.forEach((point, i) => {
        const x = flat[2 * i];
        const y = flat[2 * i + 1];
        if (x == null || y == null) return;
        frame[point * 3] = x;
        frame[point * 3 + 1] = y;
        frame[point * 3 + 2] = 0;
      });
      return frame;
    });
    return {
      id: String(c['id']),
      title: String(c['title']),
      sign: typeof c['sign'] === 'string' ? c['sign'] : null,
      note: typeof c['note'] === 'string' ? c['note'] : '',
      credit,
      fps: c['fps'],
      frames,
    };
  });
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
