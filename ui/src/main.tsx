// First import: puts zod in jitless mode before any contract builds a schema (the CSP refuses eval).
import '../../server/src/contracts/zodBrowser'
import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { ToastContainer } from 'react-toastify'
import './index.css'
import './i18n'
import { CLASSIC, resolveShell } from './app/shell'
import { loadShell } from './app/shellRegistry'
import { ShellHost } from './app/ShellHost'
import { AuthProvider } from './context/AuthContext.tsx'
import { startHeapWatch } from './utils/audio/heapWatch'
import { ErrorBoundary } from './components/shared/feedback/ErrorBoundary'
import { queryClient } from './queries/client'
import { dropForeignSkin } from './themes/registry'

// Sampling the heap while we track down the idle out-of-memory crash. Costs one
// timer every 30 seconds and prints one line.
startHeapWatch()

// theme-boot.js set data-skin from hs-skin; keep it only if the skin targets this shell.
const shellId = resolveShell()
dropForeignSkin(shellId === CLASSIC ? 'classic' : 'studio') // every other shell wears kit (studio) themes

// Only the chosen shell is downloaded (app/shell.ts). Classic is App.tsx and runs its own runtime;
// every other shell is registered in app/shellRegistry.ts and runs inside ShellHost.
// eslint-disable-next-line react-refresh/only-export-components -- entry file, nothing hot-reloads it
const Shell = lazy(() => shellId === CLASSIC ? import('./App') : loadShell(shellId))

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
            {shellId === CLASSIC ? <Shell /> : <ShellHost shellId={shellId}><Shell /></ShellHost>}
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
