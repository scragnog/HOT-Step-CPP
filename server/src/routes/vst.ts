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
import type {
  VstPlugin, ChainEntry, ChainConfig, ScanPluginsResponse, GetChainResponse, UpdateChainResponse,
  GuiResponse, ProcessResponse, MonitorStartResponse, MonitorStopResponse, MonitorSwitchResponse,
  MonitorStatusResponse, MonitorSeekResponse, MonitorPauseResponse, MonitorResumeResponse, MonitorRestartResponse,
} from '../contracts/vst.js';
import { createControlWriter, createSerializer, releaseMonitor } from './vstControl.js';
import { getSessionDir } from '../services/logger.js';
import { maintenance } from '../services/updates/maintenance.js';

const execFileAsync = promisify(execFile);
// macOS objc runtime prints 'Class X is implemented in both A and B' for every
// plugin that bundles a duplicate framework, and forwarding those verbatim grew
// node_console.log past 100 MB on one machine (#125). Not ours, not logged.
const isObjcNoise = (line: string): boolean => /^objc\[\d+\]: Class .* is implemented in both/.test(line.trim());

const router = Router();

// ── vst-host lifecycle log (logs/<session>/vst_host.log) ────
// Spawn, exit and the child's own stdout/stderr ([lifecycle] lines for plugin
// load/unload, editor open/close, sample-rate change, seek, device start/stop).
function vstLog(line: string): void {
  const dir = getSessionDir();
  if (!dir) return;
  try {
    fs.appendFileSync(path.join(dir, 'vst_host.log'), `${new Date().toISOString()} ${line}\n`);
  } catch { /* logging must never break VST */ }
}

function attachVstLog(child: ChildProcess, label: string, args: string[]): void {
  vstLog(`[${label}] spawn pid=${child.pid ?? '?'} args=${JSON.stringify(args)}`);
  const pipe = (stream: NodeJS.ReadableStream | null | undefined, tag: string) => {
    stream?.on('data', (data: Buffer) => {
      for (const line of data.toString().split(/\r?\n/)) {
        if (line.trim() && !isObjcNoise(line)) vstLog(`[${label} ${child.pid} ${tag}] ${line.trim()}`);
      }
    });
  };
  pipe(child.stdout, 'out');
  pipe(child.stderr, 'err');
  child.on('exit', (code, signal) => {
    vstLog(`[${label}] exit pid=${child.pid ?? '?'} code=${code} signal=${signal ?? 'none'}`);
  });
  child.on('error', (err) => vstLog(`[${label}] spawn error: ${err.message}`));
}

// VstPlugin, ChainEntry and ChainConfig are now contracts/vst.ts's shared
// definitions — this file used to declare its own copy, next to api.ts's.

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
        if (line.trim() && !isObjcNoise(line)) console.log(`[VST] ${line.trim()}`);
      }
      vstLog(`[process-chain] ${stderr.trim()}`);
    }

    if (!stdout || stdout.trim().length === 0) {
      cachedPlugins = [];
      res.json({ plugins: [] } satisfies ScanPluginsResponse);
      return;
    }

    const plugins: VstPlugin[] = JSON.parse(stdout);
    cachedPlugins = plugins;
    console.log(`[VST] Found ${plugins.length} plugin(s)`);
    res.json({ plugins } satisfies ScanPluginsResponse);
  } catch (err: any) {
    console.error('[VST] Scan failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── GET /chain — Get current chain config ───────────────────

router.get('/chain', (_req, res) => {
  const chain = loadChain();
  res.json(chain satisfies GetChainResponse);
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
  res.json(chain satisfies UpdateChainResponse);
});

const guiProcs = new Map<string, ChildProcess>();

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

  // One editor process per chain entry: a second click must not spawn a
  // duplicate window that reopens after the user closes the first one.
  const existing = uid ? guiProcs.get(uid) : undefined;
  if (existing && existing.exitCode === null && !existing.killed) {
    res.json({ ok: true, pid: existing.pid, alreadyOpen: true });
    return;
  }

  console.log(`[VST] Launching GUI: ${path.basename(pluginPath)}`);

  // Spawn detached — the GUI process lives independently
  const child = spawn(exe, args, {
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  child.unref();
  attachVstLog(child, 'gui', args);
  if (uid) {
    guiProcs.set(uid, child);
    const clear = () => { if (guiProcs.get(uid) === child) guiProcs.delete(uid); };
    child.on('exit', clear);
    child.on('error', clear);
  }

  res.json({ ok: true, pid: child.pid } satisfies GuiResponse);
});

// ── GET /gui/open — uids whose editor process is still running ──

router.get('/gui/open', (_req, res) => {
  res.json({ uids: [...guiProcs.keys()] });
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
      res.json({ ok: true, skipped: true } satisfies ProcessResponse);
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
        if (line.trim() && !isObjcNoise(line)) console.log(`[VST] ${line.trim()}`);
      }
      vstLog(`[process-chain] ${stderr.trim()}`);
    }

    // Clean up temp chain file
    try { fs.unlinkSync(tempChainFile); } catch {}

    console.log(`[VST] Processing complete in ${elapsed}s → ${path.basename(outputPath)}`);
    res.json({ ok: true, elapsed: parseFloat(elapsed) } satisfies ProcessResponse);
  } catch (err: any) {
    console.error('[VST] Process failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Monitor — Real-time playback through VST chain ──────────

let monitorProcess: ChildProcess | null = null;
let monitorPaused = false;
let monitorCurrentTrack = '';  // track path currently loaded in monitor
const monitorControlFile = () => path.join(config.vst.statesDir, 'monitor_control.json');
const monitorStatusFile  = () => path.join(config.vst.statesDir, 'monitor_status.json');

// Atomic, seq-stamped, merging writes (see vstControl.ts). Pass reset=true only
// when starting a fresh monitor process.
const monitorControl = createControlWriter(monitorControlFile);
function writeMonitorControl(data: Record<string, unknown>, reset = false): void {
  monitorControl.write(data, reset);
}

// Start, restart and stop each await the old host's exit. Run them one at a
// time so two overlapping requests cannot both resume and spawn a host that
// shares the control and status files with another.
const monitorTransition = createSerializer();

// An open plugin window or the monitor can stay up for hours, so they hold an
// update only, never the shared busy registry (hash queue, data-folder move).
maintenance.addBlocker(() => [
  ...(guiProcs.size ? [`${guiProcs.size} VST plugin window(s) open`] : []),
  ...(isMonitorAlive() ? ['the VST monitor is running'] : []),
]);

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

/** Ask the given monitor child to stop, wait for it to exit, then force-kill
 *  it if it has not exited within timeoutMs. Targets the captured child, never
 *  the shared monitorProcess variable, so a newer monitor is never killed. */
function stopMonitorChild(child: ChildProcess | null, timeoutMs: number): Promise<void> {
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise(resolve => {
    let done = false;
    const finish = () => { if (!done) { done = true; clearTimeout(timer); resolve(); } };
    const timer = setTimeout(() => {
      try { child.kill(); } catch {}
      setTimeout(finish, 500);
    }, timeoutMs);
    child.once('exit', finish);
    writeMonitorControl({ action: 'stop' });
  });
}

/** Resolve a trackPath (from the UI) to an absolute filesystem path.
 *  Handles: "/audio/uuid.wav", "http://localhost:3001/audio/uuid.wav",
 *  "D:\\...\\uuid.wav", and bare "uuid.wav". */
function resolveTrackPath(trackPath: string): string {
  // Strip URL origin if present
  let p = trackPath;
  try {
    const url = new URL(p);
    p = url.pathname; // "/audio/uuid.wav"
  } catch { /* not a full URL */ }

  // If it's a Windows absolute path with drive letter, keep it
  if (/^[A-Za-z]:[\\/]/.test(p)) return p;

  // Otherwise extract just the filename and resolve from audioDir
  const filename = path.basename(p);
  return path.join(config.data.audioDir, filename);
}

// POST /monitor/start — Start real-time monitoring
router.post('/monitor/start', async (req, res) => {
  const { trackPath } = req.body;
  if (!trackPath) {
    res.status(400).json({ error: 'trackPath required' });
    return;
  }

  const absTrackPath = resolveTrackPath(trackPath);

  if (!fs.existsSync(absTrackPath)) {
    res.status(404).json({ error: `Track not found: ${absTrackPath}` });
    return;
  }

  const exe = config.vst.exe;
  if (!fs.existsSync(exe)) {
    res.status(503).json({ error: 'vst-host.exe not found' });
    return;
  }

  await monitorTransition(async () => {
    // Stop the existing monitor and wait for it to exit before spawning, so the
    // new control file cannot overwrite its pending stop.
    if (isMonitorAlive()) {
      const old = monitorProcess;
      await stopMonitorChild(old, 3000);
      if (monitorProcess === old) monitorProcess = null;
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
    writeMonitorControl({ track: absTrackPath, action: 'play' }, true);
    monitorPaused = false;
    monitorCurrentTrack = absTrackPath;

    console.log(`[VST] Starting monitor: ${enabled.length} plugin(s), track=${path.basename(absTrackPath)}`);

    const child = spawn(exe, [
      '--monitor',
      '--chain', tempChainFile,
      '--input', absTrackPath,
      '--control', monitorControlFile(),
      '--status', monitorStatusFile(),
    ], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    monitorProcess = child;
    attachVstLog(child, 'monitor', child.spawnargs.slice(1));

    // Log stderr
    child.stderr?.on('data', (data: Buffer) => {
      for (const line of data.toString().split('\n')) {
        if (line.trim() && !isObjcNoise(line)) console.log(`[VST] ${line.trim()}`);
      }
    });

    child.on('exit', (code) => {
      console.log(`[VST] Monitor exited (code ${code})`);
      monitorProcess = releaseMonitor(child, monitorProcess, tempChainFile);
    });

    res.json({ ok: true, pid: child.pid, plugins: enabled.length } satisfies MonitorStartResponse);
  });
});

// POST /monitor/stop — Stop monitoring
router.post('/monitor/stop', async (_req, res) => {
  await monitorTransition(async () => {
    if (!isMonitorAlive()) {
      res.json({ ok: true, wasRunning: false } satisfies MonitorStopResponse);
      return;
    }
    monitorPaused = false;
    monitorCurrentTrack = '';
    // Give it a moment to save state gracefully, then force-kill the captured
    // child (not whatever monitorProcess points to by then). Waiting for the exit
    // keeps a start queued behind this stop from overlapping the old host.
    const old = monitorProcess;
    await stopMonitorChild(old, 3000);
    if (monitorProcess === old) monitorProcess = null;
    res.json({ ok: true, wasRunning: true } satisfies MonitorStopResponse);
  });
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

  const absTrackPath = resolveTrackPath(trackPath);

  if (!fs.existsSync(absTrackPath)) {
    res.status(404).json({ error: `Track not found: ${absTrackPath}` });
    return;
  }

  console.log(`[VST] Monitor switching track → ${path.basename(absTrackPath)}`);
  writeMonitorControl({ track: absTrackPath, action: 'play' });
  res.json({ ok: true } satisfies MonitorSwitchResponse);
});

// GET /monitor/status — Is the monitor running? Returns position too.
router.get('/monitor/status', (_req, res) => {
  const running = isMonitorAlive();
  let position = 0;
  let duration = 0;
  if (running) {
    try {
      const statusPath = monitorStatusFile();
      if (fs.existsSync(statusPath)) {
        const raw = fs.readFileSync(statusPath, 'utf-8');
        const data = JSON.parse(raw);
        position = data.position || 0;
        duration = data.duration || 0;
      }
    } catch { /* ignore parse errors */ }
  }
  res.json({ running, paused: monitorPaused, pid: monitorProcess?.pid || null, position, duration } satisfies MonitorStatusResponse);
});

// POST /monitor/seek — Seek to a position in seconds
router.post('/monitor/seek', (req, res) => {
  const { position } = req.body;
  if (typeof position !== 'number') {
    res.status(400).json({ error: 'position (number) required' });
    return;
  }
  if (!isMonitorAlive()) {
    res.status(400).json({ error: 'Monitor is not running' });
    return;
  }

  // Merge the seek into the current control state (keeps any pending command)
  writeMonitorControl({ seek: position });

  res.json({ ok: true, position } satisfies MonitorSeekResponse);
});


// POST /monitor/pause — pause playback
router.post('/monitor/pause', (_req, res) => {
  if (!isMonitorAlive()) {
    res.status(400).json({ error: 'Monitor is not running' });
    return;
  }
  monitorPaused = true;
  writeMonitorControl({ action: 'pause' });
  res.json({ ok: true } satisfies MonitorPauseResponse);
});

// POST /monitor/resume — resume playback
router.post('/monitor/resume', (_req, res) => {
  if (!isMonitorAlive()) {
    res.status(400).json({ error: 'Monitor is not running' });
    return;
  }
  monitorPaused = false;
  writeMonitorControl({ action: 'play' });
  res.json({ ok: true } satisfies MonitorResumeResponse);
});

// POST /monitor/restart — kill and restart monitor, reloading state files from disk
// This is how GUI parameter changes take effect in the monitor.
router.post('/monitor/restart', async (_req, res) => {
  await monitorTransition(async () => {
    const exe = config.vst.exe;
    const track = monitorCurrentTrack;

    if (!track || !fs.existsSync(track)) {
      res.status(400).json({ error: 'No track loaded in monitor' });
      return;
    }
    if (!fs.existsSync(exe)) {
      res.status(503).json({ error: 'vst-host.exe not found' });
      return;
    }

    // Gracefully stop the existing process: send stop, wait for it to exit
    // (it saves plugin state on the way out), force-kill only after a timeout.
    if (isMonitorAlive()) {
      const old = monitorProcess;
      await stopMonitorChild(old, 3000);
      if (monitorProcess === old) monitorProcess = null;
    }
    monitorPaused = false;

    // Reload chain — picks up any GUI state file changes
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
    writeMonitorControl({ track, action: 'play' }, true);
    monitorCurrentTrack = track;

    console.log(`[VST] Restarting monitor: ${enabled.length} plugin(s), track=${path.basename(track)}`);

    const child = spawn(exe, [
      '--monitor',
      '--chain', tempChainFile,
      '--input', track,
      '--control', monitorControlFile(),
      '--status', monitorStatusFile(),
    ], { stdio: ['ignore', 'pipe', 'pipe'] });

    monitorProcess = child;
    attachVstLog(child, 'monitor', child.spawnargs.slice(1));

    child.stderr?.on('data', (data: Buffer) => {
      for (const line of data.toString().split('\n')) {
        if (line.trim() && !isObjcNoise(line)) console.log(`[VST] ${line.trim()}`);
      }
    });

    child.on('exit', (code) => {
      console.log(`[VST] Monitor exited (code ${code})`);
      monitorProcess = releaseMonitor(child, monitorProcess, tempChainFile);
    });

    res.json({ ok: true, pid: child.pid, plugins: enabled.length } satisfies MonitorRestartResponse);
  });
});


/**
 * Apply the VST chain to a WAV file in-place.
 * Returns true if processing was applied, false if skipped.
 */
/** True if the VST chain would actually process audio — at least one enabled
 *  plugin and a vst-host binary to run it.
 *
 *  The post-processing chain needs this *before* it reaches the VST stage: a
 *  stage that has audio-modifying work after it must hand its peaks on
 *  untouched rather than limiting or clipping them, and when the downstream
 *  stage is the user's own VST chain, that chain owns the ceiling entirely.
 *  See docs/plans/2026-08-28-post-processing-gain-staging.md. */
export function vstChainActive(): boolean {
  try {
    return loadChain().plugins.some(p => p.enabled) && fs.existsSync(config.vst.exe);
  } catch {
    return false;
  }
}

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
  const tempChain = path.join(config.vst.statesDir, `_chain_${Date.now()}_${Math.random().toString(36).slice(2, 8)}.json`);  // several chains can run at once

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
