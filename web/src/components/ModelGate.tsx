import { AlertTriangle, RotateCcw } from 'lucide-react';
import type { ReactNode } from 'react';
import type { LoadedModel } from '../ml/model-client';
import type { ModelState } from '../app/session';
import { Button } from './ui';

/** Renders children once the model is ready; otherwise a loading state or an error with Retry. */
export function ModelGate({
  state,
  onRetry,
  children,
}: {
  state: ModelState;
  onRetry: () => void;
  children: (model: LoadedModel) => ReactNode;
}) {
  if (state.status === 'ready') return children(state.model);
  if (state.status === 'loading') {
    return (
      <div role="status" className="space-y-3 py-2" aria-live="polite">
        <p className="text-sm text-ink-muted">Loading the sign model…</p>
        <div className="h-3 w-2/3 animate-pulse rounded bg-surface-muted" />
        <div className="h-3 w-1/2 animate-pulse rounded bg-surface-muted" />
        <div className="h-3 w-3/5 animate-pulse rounded bg-surface-muted" />
      </div>
    );
  }
  return (
    <div role="alert" className="rounded-xl border border-danger/30 bg-danger-soft p-4 text-sm">
      <p className="flex items-center gap-2 font-semibold text-danger">
        <AlertTriangle aria-hidden className="size-4" />
        The sign model did not load
      </p>
      <p className="mt-1 text-ink">{state.message}</p>
      <p className="mt-1 text-ink-muted">Check your connection, then try again.</p>
      <Button className="mt-3" variant="secondary" size="sm" onClick={onRetry}>
        <RotateCcw aria-hidden className="size-4" />
        Retry
      </Button>
    </div>
  );
}
