import { HelpCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { Session, SessionSnapshot } from '../app/session';
import type { Decision } from '../ml/decision';
import type { LoadedModel } from '../ml/model-client';
import { GuessList } from './GuessList';
import { Percent } from './ui';

interface HistoryEntry {
  id: number;
  decision: Decision;
}

const HISTORY = 8;

function notSureText(decision: Decision): string {
  if (decision.reason === 'partial')
    return 'Move back so your shoulders are in view, then sign again.';
  return 'None of the shipped signs fit well enough to name one. Try again, a little slower.';
}

export function RecognisePanel({
  session,
  snap,
  model,
}: {
  session: Session;
  snap: SessionSnapshot;
  model: LoadedModel;
}) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  useEffect(
    () =>
      session.onResult((result, id) => {
        setHistory((h) => [{ id, decision: result.decision }, ...h].slice(0, HISTORY));
      }),
    [session],
  );

  const latest = history[0];
  const labels = model.labels;
  const shipped = labels.filter((l) => l.shipped).length;
  const decision = latest?.decision;
  const top = decision?.top[0];

  return (
    <div className="space-y-5">
      <p className="text-sm text-ink-muted">
        Sign one of the {shipped} supported ASL signs. Lower your hands or pause for a moment to
        finish a sign.
      </p>

      <div
        className="rounded-xl border border-line bg-paper p-4"
        aria-live="polite"
        data-testid="recognise-result"
      >
        {snap.signing ? (
          <p className="mb-2 flex items-center gap-2 text-sm font-medium text-accent">
            <span aria-hidden className="pulse-dot size-2 rounded-full bg-accent" />
            Reading your sign…
          </p>
        ) : null}
        {!decision && snap.signing ? null : !decision ? (
          <p className="text-sm text-ink-muted">Your result appears here after each sign.</p>
        ) : decision.kind === 'sign' && top ? (
          <div>
            <p className="text-xs font-medium tracking-wide text-ink-muted uppercase">You signed</p>
            <p className="mt-0.5 text-3xl font-semibold tracking-tight">
              {labels[top.index]?.label}
            </p>
            <p className="mt-1 text-sm text-ink-muted">
              <Percent value={top.probability} /> confident
            </p>
          </div>
        ) : decision.kind === 'not-sure' ? (
          <div>
            <p className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
              <HelpCircle aria-hidden className="size-6 text-warn" />
              Not sure
            </p>
            <p className="mt-1 text-sm text-ink-muted">{notSureText(decision)}</p>
          </div>
        ) : (
          <p className="text-sm text-ink-muted">Nothing to read in that clip.</p>
        )}
      </div>

      {decision && decision.top.length > 0 ? (
        <div>
          <h3 className="mb-2 text-sm font-semibold">Top guesses</h3>
          <GuessList guesses={decision.top} labels={labels} muted={decision.kind !== 'sign'} />
        </div>
      ) : null}

      <div>
        <h3 className="mb-2 text-sm font-semibold">Recent signs</h3>
        {history.length === 0 ? (
          <p className="text-sm text-ink-muted">None yet.</p>
        ) : (
          <ol
            className="flex gap-2 overflow-x-auto pb-1"
            aria-label="Recent signs"
            data-testid="history"
          >
            {history.map((entry) => {
              const best = entry.decision.top[0];
              const named = entry.decision.kind === 'sign' && best;
              return (
                <li
                  key={entry.id}
                  className={`shrink-0 rounded-full border px-3 py-1 text-sm ${
                    named
                      ? 'border-accent/40 bg-accent-soft'
                      : 'border-line bg-surface-muted text-ink-muted'
                  }`}
                >
                  {named ? labels[best.index]?.label : 'Not sure'}
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </div>
  );
}
