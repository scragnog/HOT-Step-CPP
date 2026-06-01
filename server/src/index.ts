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
import { spawn, ChildProcess } from 'child_process';
import { execSync } from 'child_process';

import { config, PROJECT_ROOT, PORTABLE_MODE } from './config.js';
import { initLogger, logEngine, closeLogger } from './services/logger.js';
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
import logsRoutes, { pushLog } from './routes/logs.js';
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

// Start ace-server as a child process if configured
let aceProcess: ChildProcess | null = null;

import { setEngineReady } from './engineState.js';

// Crash-count limiter: prevent infinite respawn on fatal errors (missing DLLs, etc.)
let crashCount = 0;
let firstCrashTime = 0;
const MAX_CRASHES = 3;
const CRASH_WINDOW_MS = 30_000; // 30 seconds

function startAceServer(): ChildProcess | null {
  const exe = config.aceServer.exe;
  if (!exe || !fs.existsSync(exe)) {
    console.log(`[Server] ace-server not found at: ${exe}`);
    console.log('[Server] Start ace-server manually, or set ACESTEPCPP_EXE in .env');
    return null;
  }

  const args = [
    '--models', config.aceServer.models,
    '--host', config.aceServer.host,
    '--port', String(config.aceServer.port),
  ];

  // Add adapters dir if it exists
  if (config.aceServer.adapters && fs.existsSync(config.aceServer.adapters)) {
    args.push('--adapters', config.aceServer.adapters);
  }

  // Add noise profile if available
  if (config.aceServer.noiseProfile && fs.existsSync(config.aceServer.noiseProfile)) {
    args.push('--noise-profile', config.aceServer.noiseProfile);
    console.log(`[Server] Noise profile: ${config.aceServer.noiseProfile}`);
  }

  // Add draft LM for speculative decoding (if available)
  if (config.aceServer.draftLm && fs.existsSync(config.aceServer.draftLm)) {
    args.push('--draft-lm', config.aceServer.draftLm);
    console.log(`[Server] Draft LM: ${path.basename(config.aceServer.draftLm)}`);
  }

  // VAE tiling parameters (resolves Vulkan pinned memory allocation failures)
  if (config.aceServer.vaeChunk) {
    args.push('--vae-chunk', String(config.aceServer.vaeChunk));
  }
  if (config.aceServer.vaeOverlap) {
    args.push('--vae-overlap', String(config.aceServer.vaeOverlap));
  }

  // Add ONNX model directory for ORT/TRT VAE (if it exists and contains .onnx files)
  if (config.aceServer.onnxDir && fs.existsSync(config.aceServer.onnxDir)) {
    const hasOnnx = fs.readdirSync(config.aceServer.onnxDir).some(f => f.endsWith('.onnx'));
    if (hasOnnx) {
      args.push('--onnx-dir', config.aceServer.onnxDir);
      console.log(`[Server] ONNX models: ${config.aceServer.onnxDir}`);
    }
  }

  console.log(`[Server] Starting ace-server: ${path.basename(exe)}`);
  console.log(`[Server] Models: ${config.aceServer.models}`);
  console.log(`[Server] Port: ${config.aceServer.port}`);

  // Inject TensorRT libs into PATH if available (so ORT can load nvinfer_10.dll)
  // IMPORTANT: On Windows, process.env is a case-insensitive Proxy, but spreading
  // it to a plain object creates case-sensitive keys. The key is typically 'Path'
  // not 'PATH', so we must find the actual key to avoid creating a shadowing duplicate.
  let spawnOpts: { stdio: any; env?: NodeJS.ProcessEnv } = {
    stdio: ['ignore', 'pipe', 'pipe'] as any,
  };
  if (config.aceServer.trtLibs && fs.existsSync(config.aceServer.trtLibs)) {
    const env = { ...process.env };
    // Find the actual PATH key (case-insensitive on Windows)
    const pathKey = Object.keys(env).find(k => k.toUpperCase() === 'PATH') || 'PATH';
    env[pathKey] = config.aceServer.trtLibs + ';' + (env[pathKey] || '');

    // Also inject TRT-LLM Executor libs if available (tensorrt_llm.dll + plugin)
    // exe is at engine/build/Release/ace-server.exe → up 3 to engine/
    const trtllmLibs = path.join(path.dirname(config.aceServer.exe), '..', '..', 'trtllm-libs');
    if (fs.existsSync(trtllmLibs)) {
      env[pathKey] = trtllmLibs + ';' + env[pathKey];
      console.log(`[Server] TRT-LLM libs: ${trtllmLibs}`);
    }

    spawnOpts.env = env;
    console.log(`[Server] TensorRT libs: ${config.aceServer.trtLibs}`);
  }

  const child = spawn(exe, args, spawnOpts);

  // Filter repetitive GGML noise from console output (still written to ace_engine.log via logEngine)
  const isNoise = (line: string) =>
    line.includes('CUDA graph warmup') || line.includes('CUDA Graph id') || line.includes('ggml_backend_cuda_graph_compute');

  child.stdout?.on('data', (data: Buffer) => {
    const lines = data.toString().split('\n').filter(Boolean);
    for (const line of lines) {
      if (!isNoise(line)) console.log(`[ace-server] ${line}`);
      logEngine(line);
      pushLog(line, 'engine');
    }
  });

  child.stderr?.on('data', (data: Buffer) => {
    const lines = data.toString().split('\n').filter(Boolean);
    for (const line of lines) {
      if (!isNoise(line)) console.log(`[ace-server] ${line}`);
      logEngine(line);
      pushLog(line, 'engine');
    }
  });

  child.on('exit', (code, signal) => {
    if (signal !== 'SIGTERM' && signal !== 'SIGINT' && code !== 0) {
      console.error(`[ace-server] Process exited with code ${code}, signal ${signal}`);

      // Crash-count limiter: reset window if enough time has passed
      const now = Date.now();
      if (now - firstCrashTime > CRASH_WINDOW_MS) {
        crashCount = 0;
        firstCrashTime = now;
      }
      crashCount++;

      if (crashCount >= MAX_CRASHES) {
        console.error(`[ace-server] Crashed ${MAX_CRASHES} times within ${CRASH_WINDOW_MS / 1000}s — giving up.`);
        console.error('[ace-server] This usually means a required DLL is missing from the engine/ directory.');
        console.error('[ace-server] Check the error above, or try re-extracting the release zip.');
        setEngineReady(false, `Engine crashed ${MAX_CRASHES} times — check logs for missing DLLs`);
        return;
      }

      console.log(`[ace-server] Restarting in 3 seconds... (crash ${crashCount}/${MAX_CRASHES})`);
      setTimeout(() => {
        aceProcess = startAceServer();
      }, 3000);
    }
  });

  child.on('error', (err) => {
    console.error(`[ace-server] Failed to start: ${err.message}`);
  });

  return child;
}

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
  aceProcess = startAceServer();
  setEngineReady(true, cudaReady ? 'Ready' : 'Ready (CPU only — GPU runtime missing)');
})();

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

  // Kill ace-server child process — use taskkill on Windows for proper tree kill
  if (aceProcess && !aceProcess.killed && aceProcess.pid) {
    console.log('[Server] Stopping ace-server...');
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /PID ${aceProcess.pid} /T /F`, { stdio: 'ignore' });
      } else {
        aceProcess.kill('SIGTERM');
      }
    } catch {
      // Process may already be dead
    }
  }

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

