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

import { config } from './config.js';
import { initLogger, logEngine, closeLogger } from './services/logger.js';
import { initDb, closeDb } from './db/database.js';
import authRoutes from './routes/auth.js';
import songRoutes from './routes/songs.js';
import generateRoutes from './routes/generate.js';
import modelRoutes from './routes/models.js';
import healthRoutes from './routes/health.js';
import shutdownRoutes from './routes/shutdown.js';

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

// Initialize database
initDb();

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

// Serve React frontend (production only — in dev, Vite handles this)
const uiDistPath = path.resolve(__dirname, '../../ui/dist');
if (fs.existsSync(uiDistPath)) {
  app.use(express.static(uiDistPath));
  // SPA fallback: serve index.html for all unmatched routes
  app.get('/{*splat}', (_req, res) => {
    res.sendFile(path.join(uiDistPath, 'index.html'));
  });
  console.log(`[Server] Serving UI from ${uiDistPath}`);
} else {
  console.log('[Server] No UI build found — run "npm run build" in ui/ for production');
  console.log('[Server] For development, run Vite dev server separately');
}

// Start ace-server as a child process if configured
let aceProcess: ChildProcess | null = null;

function startAceServer(): ChildProcess | null {
  const exe = config.aceServer.exe;
  if (!exe || !fs.existsSync(exe)) {
    console.log(`[Server] ace-server.exe not found at: ${exe}`);
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

  console.log(`[Server] Starting ace-server: ${path.basename(exe)}`);
  console.log(`[Server] Models: ${config.aceServer.models}`);
  console.log(`[Server] Port: ${config.aceServer.port}`);

  const child = spawn(exe, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  child.stdout?.on('data', (data: Buffer) => {
    const lines = data.toString().split('\n').filter(Boolean);
    for (const line of lines) {
      console.log(`[ace-server] ${line}`);
      logEngine(line);
    }
  });

  child.stderr?.on('data', (data: Buffer) => {
    const lines = data.toString().split('\n').filter(Boolean);
    for (const line of lines) {
      console.log(`[ace-server] ${line}`);
      logEngine(line);
    }
  });

  child.on('exit', (code, signal) => {
    if (signal !== 'SIGTERM' && signal !== 'SIGINT' && code !== 0) {
      console.error(`[ace-server] Process exited with code ${code}, signal ${signal}`);
      console.log('[ace-server] Restarting in 3 seconds...');
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

// Start ace-server
aceProcess = startAceServer();

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

