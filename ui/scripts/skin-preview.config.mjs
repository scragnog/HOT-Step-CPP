import { defineConfig } from 'vite';
import baseConfig from '../vite.config.ts';
import { CORE_GENERATION_PARAM_DESCRIPTORS } from '../../server/src/contracts/generationControlDescriptors.ts';

const responses = new Map([
  ['/api/auth/auto', { user: { id: 'skin-preview', username: 'Preview' }, token: 'skin-preview' }],
  ['/api/health', { status: 'ok', version: 'Preview', engine: { ready: true } }],
  ['/api/backends', {
    activeId: 'ace',
    backends: [{ id: 'ace', displayName: 'ACE-Step (preview)', resourcePool: 'local', active: true }],
  }],
  ['/api/capabilities', {
    backend: 'ace', up: true,
    core: {
      duration: { min: 5, max: 240 }, batch: { min: 1, max: 4 },
      bpm: true, keyscale: true, negativePrompt: true,
    },
    features: { models: true },
    extensions: [],
  }],
  ['/api/backends/models', { backend: 'ace', buckets: {}, defaults: {}, adapters: [] }],
  ['/api/models', { models: { dit: [], lm: [], vae: [], understand: [] }, adapters: [], lmAdapters: [], config: {}, defaults: {} }],
  ['/api/model-manager/registry', { packs: [], files: [], modelsDir: '', variant: 'preview', cudaMajor: 0 }],
  ['/api/generation-controls/descriptors', CORE_GENERATION_PARAM_DESCRIPTORS],
  ['/api/logs/vram', { used_mb: 2048, free_mb: 6144, total_mb: 8192 }],
  ['/api/audio-queue/items', { items: [] }],
  ['/api/audio-queue/state', { paused: false }],
  ['/api/songs', { songs: [] }],
  ['/api/songs/recent', { songs: [] }],
  ['/api/studio-drafts/playlist', { document: null }],
  ['/api/workflows/jobs', { jobs: [] }],
  ['/api/settings/env', { values: {}, restartKeys: [] }],
  ['/api/settings/gpus', { gpus: [] }],
  ['/api/settings/engine-workers', { workers: [], busy: false }],
  ['/api/updates/status', {
    channel: 'release',
    installed: { kind: 'unknown', version: 'Preview', tag: null, sha: null, variant: null, buildInfo: null, buildInfoProblem: null, git: null },
    status: 'never-checked',
    latest: { release: null, master: null },
    gitRelation: 'unknown',
    notesUrl: null, lastSuccessAt: null, lastAttemptAt: null, stale: false, error: null,
  }],
]);

export default defineConfig({
  ...baseConfig,
  server: {
    ...baseConfig.server,
    host: '127.0.0.1',
    port: 4174,
    strictPort: true,
    proxy: {},
  },
  plugins: [
    ...baseConfig.plugins,
    {
      name: 'hotstep-skin-preview',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const { pathname } = new URL(req.url, 'http://localhost');
          if (!pathname.startsWith('/api/')) return next();

          // The host publishes its current model selection on startup. Discard it.
          if (req.method === 'PUT' && pathname === '/api/training/active-models') {
            req.resume();
            res.writeHead(200, { 'Content-Type': 'application/json' }).end('{}');
            return;
          }

          if (req.method === 'GET' && pathname === '/api/logs') {
            res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
            res.write(': Skin preview\n\n');
            return;
          }

          const fixture = req.method === 'GET' && responses.get(pathname);
          res.writeHead(fixture ? 200 : 501, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(fixture || { error: 'This endpoint is unavailable in the skin preview.' }));
        });
      },
    },
  ],
});
