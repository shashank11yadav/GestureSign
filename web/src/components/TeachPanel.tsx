import { CircleDot, HelpCircle, Plus, Sparkles, Trash2 } from 'lucide-react';
import { useEffect, useId, useRef, useState, type SyntheticEvent } from 'react';
import type { Session } from '../app/session';
import { GestureStore } from '../lib/gesture-store';
import type { LoadedModel } from '../ml/model-client';
import {
  MAX_GESTURES,
  MAX_TAKES,
  MIN_TAKES,
  addTake,
  isReady,
  matchGesture,
  type Gesture,
  type GestureMatch,
} from '../ml/prototypes';
import { Button, Percent } from './ui';

const store = new GestureStore();

function TakeDots({ count }: { count: number }) {
  return (
    <span className="flex gap-1" aria-hidden>
      {Array.from({ length: MAX_TAKES }, (_, i) => (
        <span
          key={i}
          className={`size-2 rounded-full ${
            i < count ? 'bg-accent' : i < MIN_TAKES ? 'bg-ink-muted/35' : 'bg-ink-muted/15'
          }`}
        />
      ))}
    </span>
  );
}

export function TeachPanel({ session, model }: { session: Session; model: LoadedModel }) {
  const version = model.meta.version;
  const [gestures, setGestures] = useState<Gesture[]>([]);
  const [armed, setArmed] = useState<string | null>(null);
  const [match, setMatch] = useState<GestureMatch | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [formError, setFormError] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const inputId = useId();

  useEffect(() => {
    let cancelled = false;
    store
      .load(version)
      .then(({ gestures: saved, discarded }) => {
        if (cancelled) return;
        setGestures(saved);
        if (discarded) {
          setNotice(
            `${discarded} saved gesture${discarded === 1 ? ' was' : 's were'} taught with a different model version and had to be removed.`,
          );
        }
      })
      .catch(() => {
        if (!cancelled)
          setNotice('This browser is blocking storage, so gestures last until you leave the page.');
      });
    return () => {
      cancelled = true;
    };
  }, [version]);

  // Results arrive after renders commit, so the ref always holds the current list.
  const gesturesRef = useRef(gestures);
  useEffect(() => {
    gesturesRef.current = gestures;
  }, [gestures]);

  useEffect(
    () =>
      session.onResult((result) => {
        const embedding = result.inference.embedding;
        const list = gesturesRef.current;
        if (armed) {
          const gesture = list.find((g) => g.id === armed);
          if (!gesture) return;
          const updated = addTake(gesture, embedding);
          const next = list.map((g) => (g.id === armed ? updated : g));
          gesturesRef.current = next;
          setGestures(next);
          void store.save(updated, version).catch(() => undefined);
          const reachedMinimum =
            gesture.takes.length < MIN_TAKES && updated.takes.length >= MIN_TAKES;
          if (reachedMinimum || updated.takes.length >= MAX_TAKES) setArmed(null);
          return;
        }
        setMatch(matchGesture(embedding, list, model.meta.teach));
      }),
    [session, armed, version, model.meta.teach],
  );

  const ready = gestures.filter(isReady);
  const armedGesture = gestures.find((g) => g.id === armed);

  const validate = (trimmed: string): string | null => {
    if (!trimmed) return 'Give the gesture a name.';
    if (trimmed.length > 24) return 'Keep the name under 25 characters.';
    if (gestures.some((g) => g.name.toLowerCase() === trimmed.toLowerCase())) {
      return 'You already have a gesture with that name.';
    }
    if (gestures.length >= MAX_GESTURES) return `Up to ${MAX_GESTURES} gestures.`;
    return null;
  };

  const add = (event: SyntheticEvent) => {
    event.preventDefault();
    const trimmed = name.trim();
    const problem = validate(trimmed);
    setFormError(problem);
    if (problem) return;
    const gesture: Gesture = { id: crypto.randomUUID(), name: trimmed, takes: [] };
    setGestures((list) => [...list, gesture]);
    void store.save(gesture, version).catch(() => undefined);
    setName('');
    setArmed(gesture.id);
    setMatch(null);
    session.resetRecognition();
  };

  const storageFailed = () => {
    setNotice('This browser refused to update its storage; the change lasts until you leave.');
  };

  const remove = (id: string) => {
    if (armed === id) setArmed(null);
    setMatch(null);
    // Update the list once the deletion has committed, so leaving right away is safe.
    void store
      .remove(id)
      .catch(storageFailed)
      .finally(() => {
        setGestures((list) => list.filter((g) => g.id !== id));
      });
  };

  const clearAll = () => {
    setArmed(null);
    setMatch(null);
    setConfirmClear(false);
    void store
      .clear()
      .catch(storageFailed)
      .finally(() => {
        setGestures([]);
      });
  };

  return (
    <div className="space-y-5">
      <p className="text-sm text-ink-muted">
        Teach any gesture with {MIN_TAKES}–{MAX_TAKES} takes. It is matched by the model’s
        embedding, so it works for anyone, whether or not they sign ASL. Saved on this device only.
      </p>

      <form onSubmit={add} className="space-y-1.5" noValidate>
        <label htmlFor={inputId} className="block text-sm font-medium">
          New gesture
        </label>
        <div className="flex gap-2">
          <input
            id={inputId}
            value={name}
            onChange={(e) => {
              setName(e.target.value);
            }}
            maxLength={40}
            placeholder="e.g. wave, thumbs up"
            aria-invalid={formError ? true : undefined}
            aria-describedby={formError ? `${inputId}-error` : undefined}
            className="h-10 min-w-0 flex-1 rounded-lg border border-line bg-paper px-3 text-sm placeholder:text-ink-muted/70"
            disabled={gestures.length >= MAX_GESTURES}
          />
          <Button type="submit" variant="primary" disabled={gestures.length >= MAX_GESTURES}>
            <Plus aria-hidden className="size-4" />
            Add<span className="sr-only"> gesture</span>
          </Button>
        </div>
        {formError ? (
          <p id={`${inputId}-error`} className="text-sm text-danger">
            {formError}
          </p>
        ) : null}
      </form>

      <div aria-live="polite" data-testid="teach-status">
        {armedGesture ? (
          <p className="flex items-center gap-2 rounded-xl border border-accent/40 bg-accent-soft px-4 py-3 text-sm font-medium">
            <CircleDot aria-hidden className="pulse-dot size-4 text-accent" />
            Recording “{armedGesture.name}”: take{' '}
            {Math.min(armedGesture.takes.length + 1, MAX_TAKES)} of{' '}
            {Math.max(MIN_TAKES, armedGesture.takes.length + 1)}. Make the gesture, then lower your
            hands.
          </p>
        ) : ready.length === 0 ? (
          <p className="text-sm text-ink-muted">
            Add a gesture and record {MIN_TAKES} takes, then try it out.
          </p>
        ) : match === null || match.kind === 'empty' ? (
          <p className="text-sm text-ink-muted">Make one of your gestures to see it recognised.</p>
        ) : match.kind === 'match' ? (
          <div className="rounded-xl border border-success/40 bg-success-soft px-4 py-3">
            <p className="text-xs font-medium tracking-wide text-ink-muted uppercase">Recognised</p>
            <p className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
              <Sparkles aria-hidden className="size-5 text-success" />
              {match.gesture.name}
            </p>
            <p className="text-sm text-ink-muted">
              Similarity <Percent value={match.similarity} />
            </p>
          </div>
        ) : (
          <div className="rounded-xl border border-line bg-surface-muted px-4 py-3">
            <p className="flex items-center gap-2 text-xl font-semibold">
              <HelpCircle aria-hidden className="size-5 text-warn" />
              Not sure
            </p>
            <p className="text-sm text-ink-muted">
              Closest was “{match.gesture.name}”, but not by enough to say so.
            </p>
          </div>
        )}
      </div>

      {gestures.length > 0 ? (
        <ul
          className="divide-y divide-line rounded-xl border border-line"
          aria-label="Your gestures"
        >
          {gestures.map((g) => (
            <li key={g.id} className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2.5">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{g.name}</p>
                <div className="mt-1 flex items-center gap-2 text-xs text-ink-muted">
                  <TakeDots count={g.takes.length} />
                  <span>
                    {g.takes.length}/{MAX_TAKES} takes{isReady(g) ? '' : ` · needs ${MIN_TAKES}`}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-1">
                {g.takes.length < MAX_TAKES && armed !== g.id ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setArmed(g.id);
                      setMatch(null);
                      session.resetRecognition();
                    }}
                  >
                    <CircleDot aria-hidden className="size-4" />
                    {isReady(g) ? 'Add take' : 'Record'}
                    <span className="sr-only"> for {g.name}</span>
                  </Button>
                ) : null}
                {armed === g.id ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setArmed(null);
                    }}
                  >
                    Stop<span className="sr-only"> recording {g.name}</span>
                  </Button>
                ) : null}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    remove(g.id);
                  }}
                  aria-label={`Delete ${g.name}`}
                >
                  <Trash2 aria-hidden className="size-4" />
                </Button>
              </div>
            </li>
          ))}
        </ul>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-ink-muted">
        <span>
          {gestures.length}/{MAX_GESTURES} gestures · stored in this browser as numbers, never video
        </span>
        {gestures.length > 0 ? (
          confirmClear ? (
            <span className="flex gap-1">
              <Button size="sm" variant="danger" onClick={clearAll}>
                Delete all
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setConfirmClear(false);
                }}
              >
                Cancel
              </Button>
            </span>
          ) : (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setConfirmClear(true);
              }}
            >
              <Trash2 aria-hidden className="size-4" />
              Clear all
            </Button>
          )
        ) : null}
      </div>
      {notice ? <p className="text-sm text-warn">{notice}</p> : null}
    </div>
  );
}
