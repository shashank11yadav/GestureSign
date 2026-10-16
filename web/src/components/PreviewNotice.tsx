import { FlaskConical } from 'lucide-react';

/** Shown while model-meta.json has `metrics: null`, i.e. the stub model is deployed. */
export function PreviewNotice() {
  return (
    <aside
      aria-label="About this model"
      className="border-b border-line bg-warn-soft text-sm"
      data-testid="untrained-notice"
    >
      <div className="mx-auto flex max-w-6xl items-start gap-2.5 px-4 py-2.5 leading-relaxed">
        <FlaskConical aria-hidden className="mt-0.5 size-4 shrink-0 text-warn" />
        <p>
          <strong className="font-semibold">Untrained preview model.</strong> Everything runs end to
          end, but the sign guesses are random until the trained model ships, so expect “Not sure”.
          Teach your own works already.
        </p>
      </div>
    </aside>
  );
}
