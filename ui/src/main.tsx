import { StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './i18n'
import App from './App.tsx'
import { AuthProvider } from './context/AuthContext.tsx'
import { startHeapWatch } from './utils/heapWatch'

// Sampling the heap while we track down the idle out-of-memory crash. Costs one
// timer every 30 seconds and prints one line.
startHeapWatch()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AuthProvider>
      <Suspense fallback={null}>
        <App />
      </Suspense>
    </AuthProvider>
  </StrictMode>,
)
