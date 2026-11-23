import { defineConfig, type ProxyOptions, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { CSP_HEADER, contentSecurityPolicy } from '../server/src/contracts/csp.ts'

// Metadata-only releases also install a new worker and retire old assets.
function appShellWorker(): Plugin {
  let publicDir = ''
  return {
    name: 'hotstep-app-shell', apply: 'build', enforce: 'post',
    configResolved(config) { publicDir = config.publicDir },
    generateBundle: { order: 'post', handler(_options, bundle) {
      const fixed = ['index.html', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png', 'apple-touch-icon.png', 'logo.webp', 'theme-boot.js']
      const files = [...Object.keys(bundle), ...fixed.filter(f => !bundle[f])].sort()
      const template = readFileSync(resolve(publicDir, 'sw.js'), 'utf8')
      const hash = createHash('sha256').update(template)
      for (const file of files) {
        const entry = bundle[file]
        hash.update(file).update(entry ? (entry.type === 'chunk' ? entry.code : entry.source) : readFileSync(resolve(publicDir, file)))
      }
      this.emitFile({
        type: 'asset', fileName: 'sw.js',
        source: template.replace('__HOTSTEP_BUILD__', hash.digest('hex').slice(0, 20))
          .replace('/* __HOTSTEP_ASSETS__ */ []', JSON.stringify(files.map(f => `/${f}`))),
      })
    } },
  }
}

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
  plugins: [react(), appShellWorker()],
  // Production policy for `vite preview` of a build, so the service worker can be checked under it.
  preview: { headers: { [CSP_HEADER]: contentSecurityPolicy() } },
  server: {
    port: 3000,
    host: '0.0.0.0',
    // Dev variant of the policy Node sends with the production build.
    headers: { [CSP_HEADER]: contentSecurityPolicy(true) },
    proxy: {
      // Proxy all /api, /audio and /references requests to the Node.js server
      '/api': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
        // X-Forwarded-For lets the worker token gate tell a LAN browser from a local one.
        xfwd: true,
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
