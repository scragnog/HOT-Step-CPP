// aceEngineProcess.ts — ace-server child process lifecycle
//
// Relocated verbatim from index.ts (spawn args, log fan-out, crash-count
// limiter and respawn timer are unchanged) so that other services can stop and
// restart the engine, not just the bootstrap.
//
// The one behavioural addition is `suspended`: a deliberate stop sets it BEFORE
// the kill, and the 'exit' handler returns immediately while it is set, so a
// planned shutdown can never trip the crash-respawn logic. Training preprocess
// jobs use this to own the GPU for the duration of a run.
//
// Spec: docs/plans/2026-07-27-preprocess-implementation.md §4.1 (P26)

import fs from 'fs';
import path from 'path';
import { spawn, execSync, ChildProcess } from 'child_process';

import { config } from '../config.js';
import { logEngine } from './logger.js';
import { pushLog } from '../routes/logs.js';
import { setEngineReady } from '../engineState.js';
import { aceClient } from './aceClient.js';

/** The live child, or null when nothing is running. */
let aceProcess: ChildProcess | null = null;

/** True while a deliberate stop is in effect — the exit handler must not respawn. */
let suspended = false;

/**
 * Pending crash-respawn timer, and the lifecycle epoch it was scheduled in.
 *
 * Both are load-bearing. A crash schedules a respawn 3 s out; if a deliberate
 * stop *and* a restart both land inside that window (a preprocess job whose
 * ace-train fails fast — bad --dit name is sub-second), `suspended` is already
 * back to false when the timer fires and it spawns a SECOND, untracked engine
 * that nothing will ever kill. So every stop/restart cancels the timer AND
 * bumps the epoch, and the timer refuses to fire for a stale epoch.
 */
let respawnTimer: NodeJS.Timeout | null = null;
let lifecycleEpoch = 0;

/** Cancel any pending crash-respawn and invalidate one already in flight. */
function cancelPendingRespawn(): void {
  lifecycleEpoch++;
  if (respawnTimer) {
    clearTimeout(respawnTimer);
    respawnTimer = null;
  }
}

// Crash-count limiter: prevent infinite respawn on fatal errors (missing DLLs, etc.)
let crashCount = 0;
let firstCrashTime = 0;
const MAX_CRASHES = 3;
const CRASH_WINDOW_MS = 30_000; // 30 seconds

/** How long restartAceServer() waits for /health after a respawn. */
const RESTART_HEALTH_TIMEOUT_MS = 90_000;

// ── Post-restart restore hooks ──────────────────────────────────────────────
//
// Some engine state is HELD IN THE ENGINE PROCESS and nowhere else, so a
// respawn silently drops it. MiniMax-Music3's model selection is the one that
// bites: the engine forgets which quant of each role was chosen and falls back
// to best-first, which is f16 — and an MM3 render on an f16 stack is garbled.
// A training run stops and restarts the engine several times, so without this
// the user comes back from training on models they never picked.
//
// Registered from server/src/index.ts rather than imported here: this module is
// imported BY the backends, so reaching into them would close an import cycle.
// Hooks run after /health answers, before this function resolves, so a caller
// that awaits the restart can rely on the restore having happened rather than
// racing a capabilities poll. Failures are logged, never thrown — a failed
// restore leaves the engine on its default, which still generates.
type EngineRestoreHook = () => void | Promise<void>;
const restoreHooks: EngineRestoreHook[] = [];

export function onEngineRestarted(fn: EngineRestoreHook): void {
  restoreHooks.push(fn);
}

async function runRestoreHooks(): Promise<void> {
  for (const fn of restoreHooks) {
    try {
      await fn();
    } catch (err: any) {
      console.warn('[Server] engine restore hook failed:', err?.message || err);
    }
  }
}

/**
 * Spawn ace-server. Returns null when the binary is missing.
 * Also assigns the module-level current process.
 */
export function startAceServer(): ChildProcess | null {
  const exe = config.aceServer.exe;
  if (!exe || !fs.existsSync(exe)) {
    console.log(`[Server] ace-server not found at: ${exe}`);
    console.log('[Server] Start ace-server manually, or set ACESTEPCPP_EXE in .env');
    aceProcess = null;
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

  // --keep-loaded: flips the engine's ModelStore to EVICT_NEVER so the ~17 s
  // LoKr precompute (and the DiT/VAE load) only happens once per combo instead
  // of every /synth. Default OFF (VRAM trade-off) — toggle in Settings →
  // Environment → "Keep models in VRAM" (ACESTEPCPP_KEEP_LOADED), restart-required.
  if (config.aceServer.keepLoaded) {
    args.push('--keep-loaded');
    console.log('[Server] --keep-loaded: DiT + adapter stay resident across requests');
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
  // and CUDA_VISIBLE_DEVICES for GPU selection.
  // IMPORTANT: On Windows, process.env is a case-insensitive Proxy, but spreading
  // it to a plain object creates case-sensitive keys. The key is typically 'Path'
  // not 'PATH', so we must find the actual key to avoid creating a shadowing duplicate.
  const spawnOpts: { stdio: any; env?: NodeJS.ProcessEnv } = {
    stdio: ['ignore', 'pipe', 'pipe'] as any,
  };

  const needsCustomEnv = (config.aceServer.trtLibs && fs.existsSync(config.aceServer.trtLibs))
    || config.aceServer.cudaVisibleDevices;

  if (needsCustomEnv) {
    const env = { ...process.env };

    // GPU device selection (e.g. "0", "1", "0,1")
    if (config.aceServer.cudaVisibleDevices) {
      env.CUDA_VISIBLE_DEVICES = config.aceServer.cudaVisibleDevices;
      console.log(`[Server] GPU selection: CUDA_VISIBLE_DEVICES=${config.aceServer.cudaVisibleDevices}`);
    }

    if (config.aceServer.trtLibs && fs.existsSync(config.aceServer.trtLibs)) {
      // Find the actual PATH key (case-insensitive on Windows)
      const pathKey = Object.keys(env).find(k => k.toUpperCase() === 'PATH') || 'PATH';
      const pathSep = process.platform === 'win32' ? ';' : ':';
      env[pathKey] = config.aceServer.trtLibs + pathSep + (env[pathKey] || '');

      // Also inject TRT-LLM Executor libs if available (tensorrt_llm.dll + plugin)
      // exe is at engine/build/Release/ace-server.exe → up 3 to engine/
      const trtllmLibs = path.join(path.dirname(config.aceServer.exe), '..', '..', 'trtllm-libs');
      if (fs.existsSync(trtllmLibs)) {
        env[pathKey] = trtllmLibs + pathSep + env[pathKey];
        console.log(`[Server] TRT-LLM libs: ${trtllmLibs}`);
      }

      console.log(`[Server] TensorRT libs: ${config.aceServer.trtLibs}`);
    }

    spawnOpts.env = env;
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

  const spawnEpoch = lifecycleEpoch;

  child.on('exit', (code, signal) => {
    if (aceProcess === child) aceProcess = null;   // never hand out a dead child
    if (suspended) return;   // deliberate stop — never respawn
    if (spawnEpoch !== lifecycleEpoch) return;     // superseded by a stop/restart
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
      if (respawnTimer) clearTimeout(respawnTimer);
      respawnTimer = setTimeout(() => {
        respawnTimer = null;
        // A stop OR a restart that landed inside the 3 s window must win, or
        // this spawns a duplicate engine that nothing tracks.
        if (suspended || spawnEpoch !== lifecycleEpoch) return;
        startAceServer();
      }, 3000);
    }
  });

  child.on('error', (err) => {
    console.error(`[ace-server] Failed to start: ${err.message}`);
  });

  aceProcess = child;
  return child;
}

/** The live child, or null. */
export function getAceProcess(): ChildProcess | null {
  return aceProcess;
}

/** True while a deliberate stop is in effect (something else owns the GPU). */
export function isEngineSuspended(): boolean {
  return suspended;
}

/**
 * Stop ace-server and wait for the process to actually exit.
 *
 * `suspended` is set FIRST so the exit handler never respawns; it stays set
 * until restartAceServer() clears it. Resolves when the 'exit' event fires, or
 * after `timeoutMs`. A no-op when there is no live child.
 */
export function stopAceServer(reason = 'Engine stopped', timeoutMs = 15_000,
                              opts: { suspend?: boolean } = {}): Promise<boolean> {
  // A shutdown is not a suspension: leaving `suspended` set makes
  // POST /api/generate answer with the training-preprocess message during the
  // Ctrl-C window, which is misleading in the logs.
  if (opts.suspend !== false) suspended = true;
  cancelPendingRespawn();
  setEngineReady(false, reason);

  const child = aceProcess;
  if (!child || !child.pid || child.exitCode !== null || child.signalCode !== null) {
    aceProcess = null;
    return Promise.resolve(true);
  }

  console.log(`[Server] Stopping ace-server (pid ${child.pid}) — ${reason}`);

  const kill = (force: boolean) => {
    try {
      if (process.platform === 'win32') {
        // Tree kill — the engine can hold GPU contexts in worker threads.
        execSync(`taskkill /PID ${child.pid} /T /F`, { stdio: 'ignore' });
      } else {
        child.kill(force ? 'SIGKILL' : 'SIGTERM');
      }
    } catch {
      // Process may already be dead — the 'exit' listener or the timer resolves us.
    }
  };

  return new Promise<boolean>((resolve) => {
    let settled = false;
    let escalated = false;

    const onExit = () => finish(true);

    const finish = (exited: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.removeListener('exit', onExit);   // or the child leaks a listener for its lifetime
      // Only forget the child once it has ACTUALLY exited. Nulling it on a
      // timeout makes the caller's restart spawn a second engine alongside a
      // still-live one, which then cannot bind :8085 and burns the crash budget.
      if (exited && aceProcess === child) aceProcess = null;
      resolve(exited);
    };

    const onTimeout = () => {
      if (!escalated && process.platform !== 'win32') {
        // SIGTERM can be ignored by an engine wedged in a CUDA/Vulkan sync.
        escalated = true;
        console.warn('[Server] ace-server ignored SIGTERM — escalating to SIGKILL');
        kill(true);
        timer.refresh();
        return;
      }
      console.warn(`[Server] ace-server did not exit within ${timeoutMs} ms — it may still be running`);
      finish(false);
    };
    const timer: NodeJS.Timeout = setTimeout(onTimeout, timeoutMs);

    child.once('exit', onExit);
    kill(false);
  });
}

/**
 * Clear `suspended`, respawn ace-server and poll /health until it answers.
 * Resolves true when the engine came back, false on timeout.
 */
export async function restartAceServer(): Promise<boolean> {
  // Order matters: kill any crash-respawn already scheduled BEFORE clearing
  // `suspended`, or the orphaned timer spawns a duplicate engine 3 s from now.
  cancelPendingRespawn();
  suspended = false;
  setEngineReady(false, 'Restarting engine...');

  if (aceProcess && aceProcess.exitCode === null && aceProcess.signalCode === null) {
    // A stop that timed out left the old engine alive; a second spawn would
    // just fail to bind :8085 and burn the crash budget.
    console.warn('[Server] restartAceServer: an engine child is still live — not spawning a second one');
    const ok = await aceClient.isReachable();
    setEngineReady(ok, ok ? 'Ready' : 'Engine did not come back — restart the app');
    if (ok) await runRestoreHooks();
    return ok;
  }

  const child = startAceServer();
  if (!child) {
    setEngineReady(false, 'Engine did not come back — restart the app');
    return false;
  }

  const deadline = Date.now() + RESTART_HEALTH_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await aceClient.isReachable()) {
      setEngineReady(true, 'Ready');
      console.log('[Server] ace-server is back up');
      await runRestoreHooks();
      return true;
    }
    await new Promise(r => setTimeout(r, 1000));
  }

  console.error('[Server] ace-server did not answer /health within 90 s after a restart');
  setEngineReady(false, 'Engine did not come back — restart the app');
  return false;
}
