import { ArrowRight, ExternalLink, Shuffle } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import type { Session } from '../app/session';
import { practiceScore } from '../ml/decision';
import { dictionaryUrl } from '../ml/meta';
import type { LoadedModel } from '../ml/model-client';
import { Button, Percent } from './ui';

interface Attempt {
  id: number;
  score: number;
  sawIndex: number | null;
  partial: boolean;
}

function pick(indices: number[], not: number | null): number {
  const pool = indices.length > 1 ? indices.filter((i) => i !== not) : indices;
  return pool[Math.floor(Math.random() * pool.length)] ?? 0;
}

export function PracticePanel({ session, model }: { session: Session; model: LoadedModel }) {
  const { labels, meta } = model;
  const shipped = useMemo(() => labels.filter((l) => l.shipped).map((l) => l.index), [labels]);
  const [target, setTarget] = useState(() => pick(shipped, null));
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const [tries, setTries] = useState(0);

  useEffect(
    () =>
      session.onResult((result, id) => {
        const score = practiceScore(
          result.inference.logits,
          labels.map((l) => l.shipped),
          target,
          meta.calibration,
        );
        const best = result.decision.top[0];
        setAttempt({
          id,
          score,
          sawIndex: result.decision.kind === 'sign' && best ? best.index : null,
          partial: result.partial,
        });
        setTries((n) => n + 1);
      }),
    [session, labels, meta.calibration, target],
  );

  const sign = labels[target];
  if (!sign) return null;
  const passed =
    attempt !== null && !attempt.partial && attempt.score >= meta.calibration.notSureBelow;

  const next = () => {
    setTarget((t) => pick(shipped, t));
    setAttempt(null);
    setTries(0);
    session.resetRecognition();
  };

  return (
    <div className="space-y-5">
      <p className="text-sm text-ink-muted">
        Practise a sign. Your attempt is scored by how strongly the model saw this sign, the way the
        dataset was meant to help hearing parents learn ASL.
      </p>
      <div className="rounded-xl border border-line bg-paper p-4">
        <p className="text-xs font-medium tracking-wide text-ink-muted uppercase">Sign this</p>
        <p className="mt-0.5 text-3xl font-semibold tracking-tight" data-testid="practice-target">
          {sign.label}
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <a
            href={dictionaryUrl(sign)}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-surface px-3 text-sm font-medium hover:bg-surface-muted"
          >
            How to sign it
            <ExternalLink aria-hidden className="size-3.5" />
            <span className="sr-only">(opens an ASL dictionary in a new tab)</span>
          </a>
          <Button size="sm" variant="ghost" onClick={next}>
            <Shuffle aria-hidden className="size-4" />
            Another sign
          </Button>
        </div>
      </div>

      <div aria-live="polite" data-testid="practice-score">
        {attempt === null ? (
          <p className="text-sm text-ink-muted">
            Sign it to the camera, or replay a clip, to get a score.
          </p>
        ) : (
          <div
            className={`rounded-xl border p-4 ${passed ? 'border-success/40 bg-success-soft' : 'border-line bg-surface-muted'}`}
          >
            <div className="flex items-baseline justify-between gap-3">
              <p className="font-semibold">
                {attempt.partial
                  ? 'Move back so your shoulders are in view'
                  : passed
                    ? `That looked like “${sign.label}”`
                    : 'Not quite yet'}
              </p>
              <p className="text-2xl font-semibold">
                <Percent value={attempt.score} />
              </p>
            </div>
            <div aria-hidden className="mt-2 h-2 overflow-hidden rounded-full bg-surface">
              <div
                className={`h-full rounded-full ${passed ? 'bg-success' : 'bg-accent'}`}
                style={{ width: `${Math.max(2, Math.round(attempt.score * 100))}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-ink-muted">
              {attempt.sawIndex !== null && attempt.sawIndex !== target
                ? `It looked more like “${labels[attempt.sawIndex]?.label ?? '?'}”. `
                : ''}
              Attempt {tries}.
            </p>
            {passed ? (
              <Button className="mt-3" size="sm" variant="primary" onClick={next}>
                Next sign
                <ArrowRight aria-hidden className="size-4" />
              </Button>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
