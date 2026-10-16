import type { Guess } from '../ml/decision';
import type { SignLabel } from '../ml/meta';
import { Percent } from './ui';

/** Top guesses with confidence bars. Muted when the overall answer is "Not sure". */
export function GuessList({
  guesses,
  labels,
  muted,
}: {
  guesses: readonly Guess[];
  labels: readonly SignLabel[];
  muted: boolean;
}) {
  return (
    <ol className="space-y-2" aria-label="Top guesses">
      {guesses.map((g, i) => {
        const label = labels[g.index]?.label ?? `#${g.index}`;
        return (
          <li key={g.index} className="text-sm">
            <div className="flex items-baseline justify-between gap-3">
              <span className={`truncate ${i === 0 && !muted ? 'font-semibold' : ''}`}>
                <span className="mr-2 text-ink-muted tabular-nums">{i + 1}</span>
                {label}
              </span>
              <span className="text-ink-muted">
                <Percent value={g.probability} />
              </span>
            </div>
            <div aria-hidden className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-muted">
              <div
                className={`h-full rounded-full ${muted ? 'bg-ink-muted/40' : i === 0 ? 'bg-accent' : 'bg-accent/45'}`}
                style={{ width: `${Math.max(2, Math.round(g.probability * 100))}%` }}
              />
            </div>
          </li>
        );
      })}
    </ol>
  );
}
