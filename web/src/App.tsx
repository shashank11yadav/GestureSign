import { Cpu, Lock, Sparkles } from 'lucide-react';
import { useState, useSyncExternalStore } from 'react';
import type { Session } from './app/session';
import { HowItWorks } from './components/HowItWorks';
import { ModeTabs, type Mode } from './components/ModeTabs';
import { ModelGate } from './components/ModelGate';
import { PracticePanel } from './components/PracticePanel';
import { PreviewNotice } from './components/PreviewNotice';
import { RecognisePanel } from './components/RecognisePanel';
import { SiteHeader } from './components/SiteHeader';
import { Stage } from './components/Stage';
import { TeachPanel } from './components/TeachPanel';
import { Card } from './components/ui';
import { AUTHOR_URL, DATASET_CREDIT, REPO_URL } from './links';

export function App({ session }: { session: Session }) {
  const snap = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const [mode, setMode] = useState<Mode>('recognise');
  const model = snap.model;
  const untrained = model.status === 'ready' && model.model.meta.metrics === null;
  const shipped =
    model.status === 'ready' ? model.model.labels.filter((l) => l.shipped).length : null;

  const changeMode = (next: Mode) => {
    setMode(next);
    session.resetRecognition();
  };

  return (
    <>
      <a href="#demo" className="skip-link">
        Skip to the demo
      </a>
      <SiteHeader />
      {untrained ? <PreviewNotice /> : null}
      <main id="main" className="mx-auto max-w-6xl px-4 pt-8 pb-16 sm:pt-12">
        <div className="max-w-3xl">
          <h1 className="text-3xl font-semibold tracking-tight text-balance sm:text-5xl">
            Sign to your camera. See ASL recognised as you go.
          </h1>
          <p className="mt-3 max-w-2xl text-base leading-relaxed text-ink-muted sm:text-lg">
            Live recognition of {shipped ?? 'hundreds of'} isolated American Sign Language signs, a
            practice mode, and gestures you teach yourself. It all runs in this tab.
          </p>
          <ul className="mt-4 flex flex-wrap gap-2 text-sm" aria-label="Highlights">
            {[
              { icon: Lock, text: 'Video stays on your device' },
              { icon: Cpu, text: 'Runs in your browser' },
              { icon: Sparkles, text: 'No install, no account' },
            ].map(({ icon: Icon, text }) => (
              <li
                key={text}
                className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1"
              >
                <Icon aria-hidden className="size-3.5 text-accent" />
                {text}
              </li>
            ))}
          </ul>
        </div>

        <div
          id="demo"
          className="mt-8 grid scroll-mt-4 gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]"
        >
          <Stage session={session} snap={snap} />
          <Card className="min-w-0 self-start p-4 sm:p-5">
            <ModeTabs mode={mode} onChange={changeMode} />
            <div
              role="tabpanel"
              id={`panel-${mode}`}
              aria-labelledby={`tab-${mode}`}
              tabIndex={0}
              className="mt-5 focus-visible:outline-offset-4"
            >
              <ModelGate
                state={model}
                onRetry={() => {
                  void session.loadModel();
                }}
              >
                {(loaded) =>
                  mode === 'recognise' ? (
                    <RecognisePanel session={session} snap={snap} model={loaded} />
                  ) : mode === 'practice' ? (
                    <PracticePanel session={session} model={loaded} />
                  ) : (
                    <TeachPanel session={session} model={loaded} />
                  )
                }
              </ModelGate>
              {snap.inferenceError ? (
                <p role="alert" className="mt-4 text-sm text-danger">
                  The model failed on that sign: {snap.inferenceError}
                </p>
              ) : null}
            </div>
          </Card>
        </div>

        <HowItWorks model={model} />
      </main>
      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-6 text-sm text-ink-muted sm:flex-row sm:justify-between">
          <p>
            GestureSign by{' '}
            <a href={AUTHOR_URL} className="underline underline-offset-2">
              Shashank Yadav
            </a>{' '}
            ·{' '}
            <a href={REPO_URL} className="underline underline-offset-2">
              source
            </a>{' '}
            · MIT licence
          </p>
          <p>Data: {DATASET_CREDIT}</p>
        </div>
      </footer>
    </>
  );
}
