import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { Session } from './app/session';
import { reloadOnStaleChunk } from './lib/stale-chunk';
import './styles.css';

reloadOnStaleChunk();

const session = new Session();
void session.loadModel();

// The e2e injection hook exists only in `vite build --mode test`; in production
// builds this branch is constant-false and the module is never bundled.
if (import.meta.env.MODE === 'test') {
  void import('./test-hook').then((hook) => {
    hook.installTestHook(session);
  });
}

const root = document.getElementById('root');
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App session={session} />
    </StrictMode>,
  );
}
