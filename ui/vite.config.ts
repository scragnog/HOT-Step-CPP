import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      // Proxy all /api, /audio and /references requests to the Node.js server
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
      '/audio': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
      '/references': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
})
