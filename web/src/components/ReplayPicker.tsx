import { Play, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { loadReplayClips, type ReplayClip } from '../replay/clips';
import { Button, Card } from './ui';

type Load = { status: 'loading' } | { status: 'ready'; clips: ReplayClip[] } | { status: 'error' };

/** No-camera path: recorded landmark clips played through the same recogniser. */
export function ReplayPicker({
  active,
  onPlay,
  onClose,
}: {
  active: string | null;
  onPlay: (clip: ReplayClip) => void;
  onClose: () => void;
}) {
  const [load, setLoad] = useState<Load>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    loadReplayClips()
      .then((clips) => {
        if (!cancelled) setLoad({ status: 'ready', clips });
      })
      .catch(() => {
        if (!cancelled) setLoad({ status: 'error' });
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const credit =
    load.status === 'ready' ? (load.clips.find((c) => c.credit)?.credit ?? null) : null;
  const synthetic = load.status === 'ready' && load.clips.every((c) => c.sign === null);

  return (
    <Card className="mt-3 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="font-semibold">Replay a clip</h3>
          <p className="mt-0.5 text-sm text-ink-muted">
            {synthetic
              ? 'Landmark animations made by the training pipeline. They show the recogniser working without a camera; they are motion patterns, not real signs.'
              : 'Signs recorded by people the model never trained on, played through the same recogniser.'}
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose} aria-label="Close the clip list">
          <X aria-hidden className="size-4" />
        </Button>
      </div>
      {load.status === 'loading' ? (
        <p role="status" className="mt-3 text-sm text-ink-muted">
          Loading clips…
        </p>
      ) : load.status === 'error' ? (
        <p role="alert" className="mt-3 text-sm text-danger">
          The clips could not be loaded.{' '}
          <button
            type="button"
            className="underline underline-offset-2"
            onClick={() => {
              setLoad({ status: 'loading' });
              setAttempt((n) => n + 1);
            }}
          >
            Retry
          </button>
        </p>
      ) : (
        <ul className="mt-3 grid gap-2 sm:grid-cols-2" aria-label="Replay clips">
          {load.clips.map((clip) => (
            <li key={clip.id}>
              <button
                type="button"
                onClick={() => {
                  onPlay(clip);
                }}
                aria-pressed={active === clip.id}
                className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2.5 text-left text-sm transition-colors ${
                  active === clip.id
                    ? 'border-accent bg-accent-soft'
                    : 'border-line bg-paper hover:border-accent/60'
                }`}
              >
                <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-accent text-accent-ink">
                  <Play aria-hidden className="size-4" />
                </span>
                <span className="min-w-0">
                  <span className="block truncate font-medium">{clip.title}</span>
                  <span className="block truncate text-xs text-ink-muted">
                    {clip.sign ? `Sign: ${clip.title}` : `${clip.frames.length} frames · synthetic`}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {credit ? <p className="mt-3 text-xs text-ink-muted">Clips: {credit}</p> : null}
    </Card>
  );
}
