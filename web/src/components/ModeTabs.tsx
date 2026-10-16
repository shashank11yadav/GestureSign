import { useRef, type KeyboardEvent } from 'react';

export type Mode = 'recognise' | 'practice' | 'teach';

const MODES: readonly { id: Mode; label: string; short: string }[] = [
  { id: 'recognise', label: 'Recognise', short: 'Recognise' },
  { id: 'practice', label: 'Practice', short: 'Practice' },
  { id: 'teach', label: 'Teach your own', short: 'Teach' },
];

/** WAI-ARIA tabs: arrow keys move between tabs, Home/End jump to the ends. */
export function ModeTabs({ mode, onChange }: { mode: Mode; onChange: (mode: Mode) => void }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (event: KeyboardEvent, index: number) => {
    const last = MODES.length - 1;
    const target =
      event.key === 'ArrowRight'
        ? index === last
          ? 0
          : index + 1
        : event.key === 'ArrowLeft'
          ? index === 0
            ? last
            : index - 1
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? last
              : null;
    if (target === null) return;
    event.preventDefault();
    const next = MODES[target];
    if (!next) return;
    onChange(next.id);
    refs.current[target]?.focus();
  };
  return (
    <div
      role="tablist"
      aria-label="Mode"
      className="grid grid-cols-3 gap-1 rounded-xl bg-surface-muted p-1"
    >
      {MODES.map((m, i) => {
        const selected = m.id === mode;
        return (
          <button
            key={m.id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={`tab-${m.id}`}
            aria-label={m.label}
            aria-selected={selected}
            aria-controls={`panel-${m.id}`}
            tabIndex={selected ? 0 : -1}
            onClick={() => {
              onChange(m.id);
            }}
            onKeyDown={(e) => {
              onKey(e, i);
            }}
            className={`h-9 cursor-pointer truncate rounded-lg px-2 text-sm font-medium transition-colors ${
              selected ? 'bg-surface text-ink shadow-sm' : 'text-ink-muted hover:text-ink'
            }`}
          >
            <span aria-hidden className="sm:hidden">
              {m.short}
            </span>
            <span aria-hidden className="hidden sm:inline">
              {m.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
