// vst.ts — VST3 Post-Processing routes
//
// Endpoints:
//   GET    /api/vst/scan              — Scan for installed VST3 plugins
//   GET    /api/vst/chain             — Get current chain config
//   PUT    /api/vst/chain             — Update chain config
//   POST   /api/vst/gui               — Launch plugin GUI (native window)
//   POST   /api/vst/process           — Process audio through the full chain

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { execFile, spawn, ChildProcess } from 'child_process';
import { promisify } from 'util';
import { config } from '../config.js';

const execFileAsync = promisify(execFile);
const router = Router();

// ── Types ───────────────────────────────────────────────────

export interface VstPlugin {
  name: string;
  vendor: string;
  version: string;
  path: string;
  uid: string;
  subcategories: string;
}

export interface ChainEntry {
  uid: string;
  name: string;
  vendor: string;
  path: string;       // .vst3 module path
  enabled: boolean;
  statePath: string;   // .vststate file path (may not exist yet)
}

interface ChainConfig {
  plugins: ChainEntry[];
}

// ── Helpers ─────────────────────────────────────────────────

function ensureDirs(): void {
  fs.mkdirSync(config.vst.statesDir, { recursive: true });
  fs.mkdirSync(path.dirname(config.vst.chainFile), { recursive: true });
}

function loadChain(): ChainConfig {
  ensureDirs();
  try {
    if (fs.existsSync(config.vst.chainFile)) {
      const raw = fs.readFileSync(config.vst.chainFile, 'utf-8');
      return JSON.parse(raw) as ChainConfig;
    }
  } catch (err) {
    console.error('[VST] Failed to load chain config:', err);
  }
  return { plugins: [] };
}

function saveChain(chain: ChainConfig): void {
  ensureDirs();
  fs.writeFileSync(config.vst.chainFile, JSON.stringify(chain, null, 2), 'utf-8');
}

function statePathForPlugin(uid: string): string {
  return path.join(config.vst.statesDir, `${uid}.vststate`);
}

// Cached scan results (scanning 40 plugins takes ~2-3 seconds)
let cachedPlugins: VstPlugin[] | null = null;

// ── GET /scan — Scan for installed VST3 plugins ─────────────

router.get('/scan', async (_req, res) => {
  try {
    const exe = config.vst.exe;
    if (!fs.existsSync(exe)) {
      res.status(503).json({
        error: `vst-host.exe not found at ${exe}`,
        hint: 'Rebuild the engine with: engine/build.cmd',
      });
      return;
    }

    console.log('[VST] Scanning for plugins...');
    const { stdout, stderr } = await execFileAsync(exe, ['--scan'], {
      timeout: 30_000,
      maxBuffer: 1024 * 1024, // 1MB should be plenty for JSON
    });

    if (stderr) {
      for (const line of stderr.split('\n')) {
        if (line.trim()) console.log(`[VST] ${line.trim()}`);
      }
    }

    if (!stdout || stdout.trim().length === 0) {
      cachedPlugins = [];
      res.json({ plugins: [] });
      return;
    }

    const plugins: VstPlugin[] = JSON.parse(stdout);
    cachedPlugins = plugins;
    console.log(`[VST] Found ${plugins.length} plugin(s)`);
    res.json({ plugins });
  } catch (err: any) {
    console.error('[VST] Scan failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── GET /chain — Get current chain config ───────────────────

router.get('/chain', (_req, res) => {
  const chain = loadChain();
  res.json(chain);
});

// ── PUT /chain — Update chain config ────────────────────────

router.put('/chain', (req, res) => {
  const { plugins } = req.body as ChainConfig;
  if (!Array.isArray(plugins)) {
    res.status(400).json({ error: 'plugins array required' });
    return;
  }

  // Validate and ensure state paths
  const validated: ChainEntry[] = plugins.map(p => ({
    uid: p.uid,
    name: p.name,
    vendor: p.vendor || '',
    path: p.path,
    enabled: p.enabled !== false,
    statePath: p.statePath || statePathForPlugin(p.uid),
  }));

  const chain: ChainConfig = { plugins: validated };
  saveChain(chain);
  console.log(`[VST] Chain updated: ${validated.length} plugin(s), ${validated.filter(p => p.enabled).length} enabled`);
  res.json(chain);
});

// ── POST /gui — Launch plugin GUI ───────────────────────────

router.post('/gui', (req, res) => {
  const { pluginPath, uid } = req.body;
  if (!pluginPath) {
    res.status(400).json({ error: 'pluginPath required' });
    return;
  }

  const exe = config.vst.exe;
  if (!fs.existsSync(exe)) {
    res.status(503).json({ error: 'vst-host.exe not found' });
    return;
  }

  const statePath = uid ? statePathForPlugin(uid) : '';
  const args = ['--gui', '--plugin', pluginPath];
  if (statePath) {
    args.push('--state', statePath);
  }

  console.log(`[VST] Launching GUI: ${path.basename(pluginPath)}`);

  // Spawn detached — the GUI process lives independently
  const child = spawn(exe, args, {
    detached: true,
    stdio: 'ignore',
  });
  child.unref();

  res.json({ ok: true, pid: child.pid });
});

// ── POST /process — Process audio through the VST chain ─────

router.post('/process', async (req, res) => {
  const { inputPath, outputPath } = req.body;
  if (!inputPath || !outputPath) {
    res.status(400).json({ error: 'inputPath and outputPath required' });
    return;
  }

  const exe = config.vst.exe;
  if (!fs.existsSync(exe)) {
    res.status(503).json({ error: 'vst-host.exe not found' });
    return;
  }

  if (!fs.existsSync(inputPath)) {
    res.status(404).json({ error: `Input file not found: ${inputPath}` });
    return;
  }

  try {
    const chain = loadChain();
    const enabled = chain.plugins.filter(p => p.enabled);

    if (enabled.length === 0) {
      // No plugins enabled — just copy
      fs.copyFileSync(inputPath, outputPath);
      res.json({ ok: true, skipped: true });
      return;
    }

    // Write temporary chain JSON for vst-host.exe
    const tempChainFile = path.join(config.vst.statesDir, `_temp_chain_${Date.now()}.json`);
    const chainData = {
      plugins: enabled.map(p => ({
        path: p.path,
        state: fs.existsSync(p.statePath) ? p.statePath : '',
        enabled: true,
      })),
    };
    fs.writeFileSync(tempChainFile, JSON.stringify(chainData), 'utf-8');

    console.log(`[VST] Processing through ${enabled.length} plugin(s):`);
    for (const p of enabled) {
      console.log(`[VST]   → ${p.name} (${p.vendor})`);
    }

    const startTime = Date.now();
    const { stderr } = await execFileAsync(exe, [
      '--process-chain',
      '--chain', tempChainFile,
      '--input', inputPath,
      '--output', outputPath,
    ], { timeout: 300_000 }); // 5 min timeout

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    if (stderr) {
      for (const line of stderr.split('\n')) {
        if (line.trim()) console.log(`[VST] ${line.trim()}`);
      }
    }

    // Clean up temp chain file
    try { fs.unlinkSync(tempChainFile); } catch {}

    console.log(`[VST] Processing complete in ${elapsed}s → ${path.basename(outputPath)}`);
    res.json({ ok: true, elapsed: parseFloat(elapsed) });
  } catch (err: any) {
    console.error('[VST] Process failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Monitor — Real-time playback through VST chain ──────────

let monitorProcess: ChildProcess | null = null;
const monitorControlFile = () => path.join(config.vst.statesDir, 'monitor_control.json');

function writeMonitorControl(data: Record<string, unknown>): void {
  fs.writeFileSync(monitorControlFile(), JSON.stringify(data), 'utf-8');
}

function isMonitorAlive(): boolean {
  if (!monitorProcess) return false;
  try {
    // Sending signal 0 tests if the process is alive (throws if dead)
    process.kill(monitorProcess.pid!, 0);
    return true;
  } catch {
    monitorProcess = null;
    return false;
  }
}

// POST /monitor/start — Start real-time monitoring
router.post('/monitor/start', (req, res) => {
  const { trackPath } = req.body;
  if (!trackPath) {
    res.status(400).json({ error: 'trackPath required' });
    return;
  }

  // Resolve to absolute path
  let absTrackPath: string;
  if (path.isAbsolute(trackPath)) {
    absTrackPath = trackPath;
  } else {
    // Relative to audio dir (e.g. "/audio/uuid.wav" → data/audio/uuid.wav)
    const filename = path.basename(trackPath);
    absTrackPath = path.join(config.data.audioDir, filename);
  }

  if (!fs.existsSync(absTrackPath)) {
    res.status(404).json({ error: `Track not found: ${absTrackPath}` });
    return;
  }

  const exe = config.vst.exe;
  if (!fs.existsSync(exe)) {
    res.status(503).json({ error: 'vst-host.exe not found' });
    return;
  }

  // Kill existing monitor if running
  if (isMonitorAlive()) {
    writeMonitorControl({ action: 'stop' });
    setTimeout(() => {
      try { monitorProcess?.kill(); } catch {}
      monitorProcess = null;
    }, 1000);
  }

  // Write temp chain JSON for the monitor (uses statePath for state files)
  const chain = loadChain();
  const enabled = chain.plugins.filter(p => p.enabled);
  if (enabled.length === 0) {
    res.status(400).json({ error: 'No enabled plugins in chain' });
    return;
  }

  const tempChainFile = path.join(config.vst.statesDir, '_monitor_chain.json');
  const chainData = {
    plugins: enabled.map(p => ({
      path: p.path,
      state: fs.existsSync(p.statePath) ? p.statePath : '',
      enabled: true,
    })),
  };
  fs.writeFileSync(tempChainFile, JSON.stringify(chainData), 'utf-8');

  // Write initial control file
  writeMonitorControl({ track: absTrackPath, action: 'play' });

  console.log(`[VST] Starting monitor: ${enabled.length} plugin(s), track=${path.basename(absTrackPath)}`);

  const child = spawn(exe, [
    '--monitor',
    '--chain', tempChainFile,
    '--input', absTrackPath,
    '--control', monitorControlFile(),
  ], {
    stdio: ['ignore', 'ignore', 'pipe'],
  });

  monitorProcess = child;

  // Log stderr
  child.stderr?.on('data', (data: Buffer) => {
    for (const line of data.toString().split('\n')) {
      if (line.trim()) console.log(`[VST] ${line.trim()}`);
    }
  });

  child.on('exit', (code) => {
    console.log(`[VST] Monitor exited (code ${code})`);
    monitorProcess = null;
    // Clean up temp chain
    try { fs.unlinkSync(tempChainFile); } catch {}
  });

  res.json({ ok: true, pid: child.pid, plugins: enabled.length });
});

// POST /monitor/stop — Stop monitoring
router.post('/monitor/stop', (_req, res) => {
  if (!isMonitorAlive()) {
    res.json({ ok: true, wasRunning: false });
    return;
  }
  writeMonitorControl({ action: 'stop' });
  // Give it a moment to save state gracefully, then force-kill
  setTimeout(() => {
    if (isMonitorAlive()) {
      try { monitorProcess?.kill(); } catch {}
      monitorProcess = null;
    }
  }, 3000);
  res.json({ ok: true, wasRunning: true });
});

// POST /monitor/switch — Switch to a different track
router.post('/monitor/switch', (req, res) => {
  const { trackPath } = req.body;
  if (!trackPath) {
    res.status(400).json({ error: 'trackPath required' });
    return;
  }
  if (!isMonitorAlive()) {
    res.status(400).json({ error: 'Monitor is not running' });
    return;
  }

  let absTrackPath: string;
  if (path.isAbsolute(trackPath)) {
    absTrackPath = trackPath;
  } else {
    const filename = path.basename(trackPath);
    absTrackPath = path.join(config.data.audioDir, filename);
  }

  if (!fs.existsSync(absTrackPath)) {
    res.status(404).json({ error: `Track not found: ${absTrackPath}` });
    return;
  }

  console.log(`[VST] Monitor switching track → ${path.basename(absTrackPath)}`);
  writeMonitorControl({ track: absTrackPath, action: 'play' });
  res.json({ ok: true });
});

// GET /monitor/status — Is the monitor running?
router.get('/monitor/status', (_req, res) => {
  res.json({ running: isMonitorAlive(), pid: monitorProcess?.pid || null });
});


/**
 * Apply the VST chain to a WAV file in-place.
 * Returns true if processing was applied, false if skipped.
 */
export async function applyVstChain(wavPath: string): Promise<boolean> {
  const chain = loadChain();
  const enabled = chain.plugins.filter(p => p.enabled);

  if (enabled.length === 0) return false;

  const exe = config.vst.exe;
  if (!fs.existsSync(exe)) {
    console.warn('[VST] vst-host.exe not found, skipping chain');
    return false;
  }

  // Process in-place: output to temp, then replace
  const tempOut = wavPath + '.vst_processed.wav';
  const tempChain = path.join(config.vst.statesDir, `_chain_${Date.now()}.json`);

  try {
    ensureDirs();

    const chainData = {
      plugins: enabled.map(p => ({
        path: p.path,
        state: fs.existsSync(p.statePath) ? p.statePath : '',
        enabled: true,
      })),
    };
    fs.writeFileSync(tempChain, JSON.stringify(chainData), 'utf-8');

    console.log(`[VST] Applying chain (${enabled.length} plugins) to ${path.basename(wavPath)}`);

    await execFileAsync(exe, [
      '--process-chain',
      '--chain', tempChain,
      '--input', wavPath,
      '--output', tempOut,
    ], { timeout: 300_000 });

    // Replace original with processed
    fs.copyFileSync(tempOut, wavPath);
    return true;
  } catch (err: any) {
    console.error('[VST] Chain processing failed:', err.message);
    return false;
  } finally {
    try { fs.unlinkSync(tempOut); } catch {}
    try { fs.unlinkSync(tempChain); } catch {}
  }
}

export default router;
