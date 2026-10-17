import { Boxes, BrainCircuit, Gauge, ScanLine, ShieldCheck } from 'lucide-react';
import type { ModelState } from '../app/session';
import { DATASET_CREDIT, DATASET_URL, REPO_URL } from '../links';
import { Card } from './ui';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function HowItWorks({ model }: { model: ModelState }) {
  const meta = model.status === 'ready' ? model.model.meta : null;
  const lips = meta?.featureSet !== 'hands_pose';
  const steps = [
    {
      icon: ScanLine,
      title: 'Landmarks',
      body: lips
        ? 'MediaPipe Holistic finds 543 body, face and hand points in each camera frame, in a Web Worker.'
        : 'MediaPipe’s hand and pose landmarkers find the hands and upper body in each frame, in a Web Worker.',
    },
    {
      icon: Boxes,
      title: 'Features',
      body: `Hands, shoulders and arms${lips ? ', and 40 lip points' : ''} are centred on the shoulders, scaled by their width and mirrored so the signing hand always looks like a right hand. Frames are resampled to ${meta?.fps ?? 30} fps.`,
    },
    {
      icon: BrainCircuit,
      title: 'Model',
      body: `A 1.4 M-parameter convolution + transformer reads up to ${meta?.maxFrames ?? 64} frames. It runs with ONNX Runtime in WebAssembly, in a second worker.`,
    },
    {
      icon: Gauge,
      title: 'Honest answers',
      body: 'Below a confidence threshold calibrated on people the model never saw, the page says “Not sure” instead of guessing.',
    },
  ];

  return (
    <section id="how-it-works" aria-labelledby="how-heading" className="mt-16 scroll-mt-6">
      <h2 id="how-heading" className="text-2xl font-semibold tracking-tight">
        How it works
      </h2>
      <p className="mt-2 max-w-2xl text-ink-muted">
        Signs are read from body landmarks rather than pixels, so lighting, background and camera
        quality matter far less, and the model is small enough to run on your device.
      </p>
      <ol className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {steps.map((step, i) => (
          <li key={step.title}>
            <Card className="h-full p-5">
              <div className="flex items-center gap-3">
                <span className="grid size-9 place-items-center rounded-lg bg-accent-soft text-accent">
                  <step.icon aria-hidden className="size-5" />
                </span>
                <span className="text-xs font-medium text-ink-muted">Step {i + 1}</span>
              </div>
              <h3 className="mt-3 font-semibold">{step.title}</h3>
              <p className="mt-1 text-sm leading-relaxed text-ink-muted">{step.body}</p>
            </Card>
          </li>
        ))}
      </ol>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card className="p-5">
          <h3 className="font-semibold">Held-out accuracy</h3>
          {meta?.metrics ? (
            <p className="mt-1 text-sm text-ink-muted">
              On {meta.metrics.testClips.toLocaleString('en')} clips from signers the model never
              saw in training.
            </p>
          ) : null}
          {meta?.metrics ? (
            <dl className="mt-3 grid grid-cols-2 gap-3 text-sm">
              {[
                ['Top-1, all signs', pct(meta.metrics.top1)],
                ['Top-5, all signs', pct(meta.metrics.top5)],
                // When every sign ships, a shipped-set top-1 would repeat the first tile;
                // show the top-3 instead, which matches the three guesses on this page.
                meta.metrics.shippedSigns === meta.classes
                  ? ['Top-3, all signs', pct(meta.metrics.shippedTop3)]
                  : [
                      `Top-1, ${meta.metrics.shippedSigns} shipped signs`,
                      pct(meta.metrics.shippedTop1),
                    ],
                ['Teach your own (8-way, 3 takes)', pct(meta.metrics.teachAccuracy)],
              ].map(([k, v]) => (
                <div key={k} className="rounded-lg bg-surface-muted p-3">
                  <dt className="text-ink-muted">{k}</dt>
                  <dd className="mt-0.5 text-xl font-semibold tabular-nums">{v}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              Not trained yet. Numbers appear here once the model is trained and evaluated on
              signers it never saw; every figure comes from the repository’s evaluation script.
            </p>
          )}
          <p className="mt-3 text-xs text-ink-muted">
            Training data:{' '}
            <a href={DATASET_URL} className="underline underline-offset-2">
              {DATASET_CREDIT}
            </a>
            .
          </p>
        </Card>
        <Card className="p-5">
          <h3 className="flex items-center gap-2 font-semibold">
            <ShieldCheck aria-hidden className="size-5 text-success" />
            Private by design
          </h3>
          <ul className="mt-2 space-y-1.5 text-sm leading-relaxed text-ink-muted">
            <li>The camera starts only when you click, and stops when you leave the tab.</li>
            <li>Video never leaves the page. There is no server, account or analytics.</li>
            <li>
              Taught gestures are stored as numbers in this browser only, and the Clear button
              removes them.
            </li>
            <li>
              All code, models and runtimes are served from this site.{' '}
              <a href={REPO_URL} className="underline underline-offset-2">
                Source on GitHub
              </a>
              .
            </li>
          </ul>
        </Card>
      </div>
    </section>
  );
}
