// The page's entry: the server boundary's implementations (the simulated server, services/mock/index.ts, until package
// 10 registers the real adapters in its place: this one import), the concept's stylesheets, translation, every feature
// and part (load.ts), and in the test build the compatibility layer the concept's two tools drive the page through
// (testing/compat.ts, never in the normal builds). docs/09, "The service boundary".

import './services/mock/index.ts';
import { createRoot } from 'react-dom/client';
import './styles/index.ts';
import './i18n/index.ts';
import './load.ts';
import { App } from './App.tsx';

async function boot() {
  if (import.meta.env.MODE === 'test') await import('./testing/compat.ts');
  createRoot(document.getElementById('app')!).render(<App />);
}
void boot();
