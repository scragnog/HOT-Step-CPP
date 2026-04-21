import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { AuthProvider } from './context/AuthContext.tsx'
import { LanguageProvider } from './context/LanguageContext.tsx'
import { CreateProvider } from './context/CreateContext'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AuthProvider>
      <LanguageProvider>
        <CreateProvider>
          <App />
        </CreateProvider>
      </LanguageProvider>
    </AuthProvider>
  </StrictMode>,
)
