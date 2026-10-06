import { defineConfig, type ProxyOptions } from 'vite'
import react from '@vitejs/plugin-react'

// When the Node server dies mid-response (in-app restart, tsx watch reload),
// http-proxy leaves the browser's connection open: pipe() only ends the client
// response on a clean upstream 'end'. An EventSource (the terminal log stream,
// which is also the presence beacon) then sits on a dead stream forever
// instead of reconnecting, and a media request keeps its socket. Chrome allows
// six per host across all tabs, so a few of those stall every request.
const closeWithUpstream: ProxyOptions['configure'] = (proxy) => {
  proxy.on('proxyRes', (proxyRes, _req, res) => {
    proxyRes.on('close', () => { if (!res.writableEnded) res.destroy() })
  })
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      // Proxy all /api, /audio and /references requests to the Node.js server
      '/api': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
        configure: closeWithUpstream,
      },
      '/audio': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
        configure: closeWithUpstream,
      },
      '/references': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
        configure: closeWithUpstream,
      },
    },
  },
})
