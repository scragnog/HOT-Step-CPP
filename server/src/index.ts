// index.ts — HOT-Step CPP Server
//
// Express server that:
// 1. Serves the React frontend (pre-built static files in production)
// 2. Manages the SQLite database (songs, playlists, users)
// 3. Orchestrates generation via ace-server HTTP API
// 4. Optionally spawns ace-server as a managed child process

import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

import { config, PROJECT_ROOT, PORTABLE_MODE } from './config.js';
import { initLogger, closeLogger } from './services/logger.js';
import { initDb, closeDb } from './db/database.js';
// lireekDb is now part of the unified hotstep.db — no separate init needed
import authRoutes from './routes/auth.js';
import songRoutes from './routes/songs.js';
import generateRoutes from './routes/generate.js';
import modelRoutes from './routes/models.js';
import healthRoutes from './routes/health.js';
import shutdownRoutes from './routes/shutdown.js';
import masteringRoutes from './routes/mastering.js';
import downloadRoutes from './routes/download.js';
import adapterRoutes from './routes/adapters.js';
import logsRoutes from './routes/logs.js';
import lireekRoutes from './routes/lireek.js';
import vstRoutes from './routes/vst.js';
import analyzeRoutes from './routes/analyze.js';
import uploadRoutes from './routes/upload.js';
import supersepRoutes from './routes/supersep.js';
import settingsRoutes from './routes/settings.js';
import modelManagerRoutes from './routes/modelManager.js';
import stemStudioRoutes from './routes/stemStudio.js';
import assistantRoutes from './routes/assistant.js';
import pluginRoutes from './routes/plugins.js';
import inspireRoutes from './routes/inspire.js';
import coverArtRoutes from './routes/coverArt.js';
import seedsRoutes from './routes/seeds.js';
import profilesRoutes from './routes/profiles.js';
import songBuilderRoutes from './routes/songBuilder.js';
import midiStudioRoutes from './routes/midiStudio.js';
import trainingRoutes from './routes/training.js';
import backendsRoutes from './routes/backends.js';
import audioRoutes from './routes/audio.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Initialize file-based logging BEFORE any console output
const logDir = initLogger();

console.log(`
╔══════════════════════════════════════════╗
║         HOT-Step 9000 ⚡ CPP            ║
║    High-Performance Music Generation     ║
╚══════════════════════════════════════════╝
`);
console.log(`[Logger] Session logs: ${logDir}`);

// Initialize databases
initDb();
// lireek tables are created in initDb() — no separate init

// Create Express app
const app = express();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/songs', songRoutes);
app.use('/api/generate', generateRoutes);
app.use('/api/models', modelRoutes);
app.use('/api/health', healthRoutes);
app.use('/api/shutdown', shutdownRoutes);
app.use('/api/mastering', masteringRoutes);
app.use('/api/download', downloadRoutes);
app.use('/api/adapters', adapterRoutes);
app.use('/api/logs', logsRoutes);
app.use('/api/lireek', lireekRoutes);
app.use('/api/vst', vstRoutes);
app.use('/api/analyze', analyzeRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/supersep', supersepRoutes);
app.use('/api/settings', settingsRoutes);
app.use('/api/model-manager', modelManagerRoutes);
app.use('/api/stem-studio', stemStudioRoutes);
app.use('/api/assistant', assistantRoutes);
app.use('/api/plugins', pluginRoutes);
app.use('/api/inspire', inspireRoutes);
app.use('/api/cover-art', coverArtRoutes);
app.use('/api/seeds', seedsRoutes);
app.use('/api/profiles', profilesRoutes);
app.use('/api/builder', songBuilderRoutes);
app.use('/api/midi-studio', midiStudioRoutes);
app.use('/api/training', trainingRoutes);
app.use('/api/audio', audioRoutes);
// Mounted at '/api' (not '/api/backends') — the router spells its own full
// sub-paths (/backends, /backends/active, /capabilities) per the plan's
// top-level /api/capabilities path (docs/plans/multi-backend-architecture.md §4.2).
app.use('/api', backendsRoutes);

// Serve audio files from data/audio/
app.use('/audio', express.static(config.data.audioDir, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.mp3')) {
      res.setHeader('Content-Type', 'audio/mpeg');
    } else if (filePath.endsWith('.wav')) {
      res.setHeader('Content-Type', 'audio/wav');
    }
  },
}));

// Serve reference audio files from data/references/
const refsDir = path.join(config.data.dir, 'references');
fs.mkdirSync(refsDir, { recursive: true });
app.use('/references', express.static(refsDir, {
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.mp3')) {
      res.setHeader('Content-Type', 'audio/mpeg');
    } else if (filePath.endsWith('.wav')) {
      res.setHeader('Content-Type', 'audio/wav');
    } else if (filePath.endsWith('.flac')) {
      res.setHeader('Content-Type', 'audio/flac');
    }
  },
}));
// Serve React frontend (production only — in dev, Vite handles this)
const uiDistPath = path.join(PROJECT_ROOT, 'ui', 'dist');
if (fs.existsSync(uiDistPath)) {
  // Assets with content hashes get long cache; index.html always revalidates
  app.use(express.static(uiDistPath, {
    setHeaders: (res, filePath) => {
      if (filePath.endsWith('.html')) {
        res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
      } else {
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
      }
    },
  }));
  // SPA fallback: serve index.html for all unmatched routes
  app.get('/{*splat}', (_req, res) => {
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.sendFile(path.join(uiDistPath, 'index.html'));
  });
  console.log(`[Server] Serving UI from ${uiDistPath}`);
} else {
  console.log('[Server] No UI build found — run "npm run build" in ui/ for production');
  console.log('[Server] For development, run Vite dev server separately');
}

// The ace-server child lifecycle (spawn, log fan-out, crash-respawn limiter,
// deliberate stop/restart) lives in services/aceEngineProcess.ts so that the
// training preprocess job can borrow the GPU. §4.1 of the preprocess plan.
import { setEngineReady } from './engineState.js';
import { restoreMm3Selection } from './services/backends/minimax/index.js';
import { aceClient } from './services/aceClient.js';
import { startAceServer, stopAceServer } from './services/aceEngineProcess.js';
import { killActiveChildren } from './services/training/labelingQueue.js';

// ── Required runtime DLL bootstrap ──────────────────────────────────
// On first launch, the CUDA engine variant needs cuBLAS DLLs that aren't
// in the release ZIP (they're ~530 MB). Download them from HuggingFace
// before starting ace-server, with clear progress and error messages.

import { modelDownloadService } from './services/modelDownloadService.js';

/** Detect CUDA major version from engine build marker */
function detectCudaMajorVersion(): number {
  try {
    const versionFile = path.join(path.dirname(config.aceServer.exe), '.cuda-version');
    if (fs.existsSync(versionFile)) {
      return parseInt(fs.readFileSync(versionFile, 'utf-8').trim(), 10);
    }
  } catch {}
  return 13; // Default: assume CUDA 13 (latest release)
}

/** IDs of registry files that must exist before engine start (CUDA only) */
function getRequiredRuntimeIds(): string[] {
  const cudaMajor = detectCudaMajorVersion();
  if (cudaMajor <= 12) {
    return ['cuda-rt-cublas-12', 'cuda-rt-cublaslt-12', 'cuda-rt-cudart-12'];
  }
  return ['cuda-rt-cublas', 'cuda-rt-cublaslt', 'cuda-rt-cudart'];
}

async function ensureRequiredRuntime(): Promise<{ ok: boolean; missing: string[] }> {
  const engineDir = path.dirname(config.aceServer.exe);
  const registry = JSON.parse(
    fs.readFileSync(path.join(__dirname, 'data', 'model-registry.json'), 'utf-8')
  );

  const missing: Array<{ id: string; filename: string }> = [];
  const REQUIRED_RUNTIME_IDS = getRequiredRuntimeIds();
  for (const id of REQUIRED_RUNTIME_IDS) {
    const file = registry.files.find((f: any) => f.id === id);
    if (!file) continue;
    if (!fs.existsSync(path.join(engineDir, file.filename))) {
      missing.push({ id, filename: file.filename });
    }
  }

  if (missing.length === 0) return { ok: true, missing: [] };

  console.log('');
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║  First-launch setup: downloading GPU runtime libraries  ║');
  console.log('╚══════════════════════════════════════════════════════════╝');
  console.log('');
  console.log(`  Missing: ${missing.map(m => m.filename).join(', ')}`);
  console.log('  Source:  HuggingFace (scragnog/HOT-Step-CPP-SuperSep)');
  console.log('');

  // Start all downloads
  const jobIds: string[] = [];
  for (const m of missing) {
    const jobId = modelDownloadService.startDownload(m.id);
    jobIds.push(jobId);
    console.log(`  ⬇ Queued: ${m.filename}`);
  }
  console.log('');

  // Wait for all downloads to complete, logging progress
  let lastProgressLog = 0;
  await new Promise<void>((resolve) => {
    const check = () => {
      const jobs = modelDownloadService.getJobs();
      const active = jobs.filter(j => jobIds.includes(j.jobId));
      const allDone = active.every(j => j.status === 'completed' || j.status === 'failed');

      // Log progress every 2 seconds
      const now = Date.now();
      if (now - lastProgressLog > 2000) {
        lastProgressLog = now;
        for (const j of active) {
          if (j.status === 'downloading' && j.totalBytes > 0) {
            const pct = Math.round((j.bytesDownloaded / j.totalBytes) * 100);
            const mb = Math.round(j.bytesDownloaded / 1024 / 1024);
            const totalMb = Math.round(j.totalBytes / 1024 / 1024);
            const speedMb = (j.speed / 1024 / 1024).toFixed(1);
            console.log(`  ⬇ ${j.filename}: ${mb}/${totalMb} MB (${pct}%) — ${speedMb} MB/s`);
          }
        }
      }

      if (allDone) {
        const failed = active.filter(j => j.status === 'failed');
        if (failed.length > 0) {
          console.log('');
          console.log('╔══════════════════════════════════════════════════════════╗');
          console.log('║  ⚠  GPU Runtime Download Failed                         ║');
          console.log('╠══════════════════════════════════════════════════════════╣');
          for (const f of failed) {
            console.log(`║  ✗ ${f.filename}`);
            if (f.error) console.log(`║    Error: ${f.error}`);
          }
          console.log('║                                                          ║');
          console.log('║  The engine will start on CPU only (much slower).        ║');
          console.log('║                                                          ║');
          console.log('║  To fix:                                                 ║');
          console.log('║  1. Settings → Model Manager → CUDA Runtime → Download  ║');
          console.log('║  2. Or restart the app with internet access              ║');
          console.log('╚══════════════════════════════════════════════════════════╝');
          console.log('');
        } else {
          console.log('');
          console.log('  ✓ GPU runtime downloaded successfully!');
          console.log('');
        }
        resolve();
      } else {
        setTimeout(check, 500);
      }
    };
    check();
  });

  // Re-check which files are actually present
  const stillMissing: string[] = [];
  for (const m of missing) {
    if (!fs.existsSync(path.join(engineDir, m.filename))) {
      stillMissing.push(m.filename);
    }
  }

  return { ok: stillMissing.length === 0, missing: stillMissing };
}

// Bootstrap: download required DLLs (portable only), then start engine
// In dev/build-from-source mode, CUDA DLLs are in the system PATH via the
// toolkit install — no need to download them into the engine directory.
(async () => {
  let cudaReady = true;

  if (PORTABLE_MODE && process.platform === 'win32') {
    // CUDA runtime DLLs are only needed for CUDA builds — skip for Vulkan/CPU
    const variantFile = path.join(path.dirname(config.aceServer.exe), '.variant');
    const variant = fs.existsSync(variantFile)
      ? fs.readFileSync(variantFile, 'utf-8').trim()
      : 'cuda'; // Assume CUDA if no marker (pre-v1.1 builds)

    if (variant === 'cuda') {
      try {
        setEngineReady(false, 'Downloading CUDA runtime...');
        const result = await ensureRequiredRuntime();
        cudaReady = result.ok;
        if (!cudaReady) {
          console.error(`[Server] CUDA runtime incomplete — missing: ${result.missing.join(', ')}`);
          console.error('[Server] Engine will start but GPU acceleration will not be available.');
        }
      } catch (err: any) {
        console.error('[Server] Runtime bootstrap failed:', err.message);
        cudaReady = false;
      }
    } else {
      console.log(`[Server] Build variant: ${variant} — skipping CUDA runtime download`);
    }
  }

  setEngineReady(false, cudaReady ? 'Starting engine...' : 'Starting engine (CPU only — CUDA runtime missing)...');
  startAceServer();
  setEngineReady(true, cudaReady ? 'Ready' : 'Ready (CPU only — GPU runtime missing)');

  // MiniMax-Music3 holds its chosen quant in engine memory, so a restart
  // reverts to the best-first default (f16) and silently discards the user's
  // pick. Put the persisted choice back before the first generation can run.
  // Fire-and-forget: a failed restore leaves the engine on its default, which
  // still generates — it must never block request serving.
  void restoreMm3Selection().catch(err => {
    console.warn('[Server] MM3 model restore failed:', err?.message || err);
  });

  // Fire-and-forget warm-on-startup: once the engine /health is up, POST /warm
  // with the configured DiT + VAE + adapter so the first user /synth skips the
  // cold-start. Gated on keepLoaded (the engine evicts instantly under STRICT,
  // making warm pointless) and a configured warmDit. Off by default since
  // keepLoaded is off. Failures only log — they never block request serving.
  if (config.aceServer.warmOnStartup && config.aceServer.keepLoaded && config.aceServer.warmDit) {
    void warmEngineOnStartup();
  } else if (config.aceServer.warmOnStartup && config.aceServer.warmDit && !config.aceServer.keepLoaded) {
    console.log('[Server] warm-on-startup skipped: keep-loaded is off (engine would evict immediately)');
  }
})();

/** Poll engine /health until reachable (or 90s), then POST /warm with the
 *  configured DiT + VAE + adapter. The warm is itself an async engine job; we
 *  kick it off without awaiting, so the wrapper stays free to accept requests
 *  while the LoKr deltas are copied to VRAM. Any /synth that arrives mid-warm
 *  queues behind it and gets the same hot cache for free. */
async function warmEngineOnStartup(): Promise<void> {
  const deadline = Date.now() + 90_000;
  let healthy = false;
  while (Date.now() < deadline) {
    if (await aceClient.isReachable()) { healthy = true; break; }
    await new Promise(r => setTimeout(r, 1000));
  }
  if (!healthy) {
    console.warn('[Server] warm-on-startup: engine /health never came up in 90s — skipping warm');
    return;
  }
  const cfg = config.aceServer;
  const req: { dit: string; vae?: string; adapter?: string; adapter_scale?: number } = { dit: cfg.warmDit };
  if (cfg.warmVae) req.vae = cfg.warmVae;
  if (cfg.warmAdapter) {
    req.adapter = cfg.warmAdapter;
    if (Number.isFinite(cfg.warmAdapterScale)) req.adapter_scale = cfg.warmAdapterScale;
  }
  try {
    const jobId = await aceClient.warm(req, true);
    console.log(`[Server] warm-on-startup: posted /warm dit=${cfg.warmDit}${cfg.warmAdapter ? ` adapter=${cfg.warmAdapter}` : ''} job=${jobId}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.warn(`[Server] warm-on-startup: /warm failed (will warm on first user request instead): ${msg}`);
  }
}

// Start Express server
const server = app.listen(config.server.port, config.server.host, () => {
  console.log(`[Server] Listening on http://localhost:${config.server.port}`);
  console.log(`[Server] ace-server URL: ${config.aceServer.url}`);
  console.log(`[Server] Data directory: ${config.data.dir}`);
  console.log('');
  console.log(`  🎵 Open http://localhost:${config.server.port} in your browser`);
  console.log('');
});

// Graceful shutdown
let isShuttingDown = false;
function shutdown() {
  if (isShuttingDown) return;
  isShuttingDown = true;

  console.log('\n[Server] Shutting down...');

  // Kill any spawned training child (ace-train) FIRST. It is not detached and
  // not in a job object, so Node exiting without this leaves a GPU-resident
  // process (~3.2 GB) behind that competes with the engine the relaunched
  // server starts — routine with tsx watch's SIGTERM during dev.
  try { killActiveChildren(); } catch (err) { console.error('[Server] killActiveChildren failed:', err); }

  // Kill ace-server child process — tree kill on Windows, SIGTERM elsewhere.
  // `suspend: false` — a shutdown is not a preprocess suspension, so
  // /api/generate must not answer with the "paused for training" message
  // during the 1 s exit window.
  // Fire-and-forget: the 1 s process.exit delay below covers the wait.
  void stopAceServer('Server shutting down', undefined, { suspend: false });

  // Close HTTP server
  server.close(() => {
    console.log('[Server] HTTP server closed');
  });

  // Close DB and logger
  closeDb();
  closeLogger();
  console.log('[Server] Goodbye!');

  // Force exit after a short delay to let response flush
  setTimeout(() => {
    process.exit(0);
  }, 1000);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('uncaughtException', (err) => {
  console.error('[Server] Uncaught exception:', err);
});
process.on('unhandledRejection', (err) => {
  console.error('[Server] Unhandled rejection:', err);
});

