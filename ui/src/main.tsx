import { StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { ToastContainer } from 'react-toastify'
import './index.css'
import './i18n'
import App from './App.tsx'
import { AuthProvider } from './context/AuthContext.tsx'
import { startHeapWatch } from './utils/audio/heapWatch'
import { ErrorBoundary } from './components/shared/feedback/ErrorBoundary'
import { queryClient } from './queries/client'

// Sampling the heap while we track down the idle out-of-memory crash. Costs one
// timer every 30 seconds and prints one line.
startHeapWatch()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      {/* One stable client (queries/client.ts), outside auth and every view. */}
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          {/* Global error toasts from the axios interceptor (axiosClient.ts).
              Every toast is dismissible and stays while hovered, including the
              long-running ones that never auto-close. */}
          {/* Above every modal and panel: an error raised inside one has to be
              readable while it is open. Still below ParamLabel's card. */}
          <ToastContainer closeOnClick pauseOnHover closeButton style={{ zIndex: 2147483000 }} />
          <Suspense fallback={null}>
            <App />
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
