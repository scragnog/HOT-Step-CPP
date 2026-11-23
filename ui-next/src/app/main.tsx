// The page's entry: the concept's stylesheets, translation, the engine (the simulated one until the real adapter is
// built; docs/09, "The service boundary"), and in the test build the compatibility layer the concept's two tools drive
// the page through (testing/compat.ts, never in the normal builds).

import { createRoot } from 'react-dom/client';
import './styles/index.ts';
import './i18n/index.ts';
import './load.ts';
import { App } from './App.tsx';
import { mockEngine } from './services/mockEngine.ts';

const engine = mockEngine();

async function boot() {
  if (import.meta.env.MODE === 'test') await import('./testing/compat.ts');
  createRoot(document.getElementById('app')!).render(<App engine={engine} />);
}
void boot();
