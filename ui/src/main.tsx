import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { ToastContainer } from 'react-toastify'
import './index.css'
import './i18n'
import { resolveShell } from './app/shell'
import { AuthProvider } from './context/AuthContext.tsx'
import { startHeapWatch } from './utils/heapWatch'
import { ErrorBoundary } from './components/shared/ErrorBoundary'
import { queryClient } from './queries/client'

// Sampling the heap while we track down the idle out-of-memory crash. Costs one
// timer every 30 seconds and prints one line.
startHeapWatch()

// Only the chosen shell is downloaded (app/shell.ts). Both default-export a root.
// eslint-disable-next-line react-refresh/only-export-components -- entry file, nothing hot-reloads it
const Shell = lazy(() => resolveShell() === 'studio' ? import('./shells/studio/StudioApp') : import('./App'))

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      {/* One stable client (queries/client.ts), outside auth and every view. */}
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          {/* Global error toasts from the axios interceptor (axiosClient.ts). */}
          <ToastContainer />
          <Suspense fallback={null}>
            <Shell />
          </Suspense>
        </AuthProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>,
)

// Installable PWA: app-shell service worker, production builds only.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(err => console.warn('[sw] register failed', err));
  });
}
