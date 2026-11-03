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
// Multi-GPU (GH #13): the lifecycle is one instance per engine worker
// (engineWorkers.ts). The primary, w0, is the engine every export below has
// always driven; secondaries only exist under a non-empty opt-in and are
// started, stopped and restarted through the *EngineWorker(s) functions.
//
// Spec: docs/plans/2026-07-27-preprocess-implementation.md §4.1 (P26)

import fs from 'fs';
import path from 'path';
import { spawn, execSync, ChildProcess } from 'child_process';

import { config, USER_PLUGINS_DIR } from '../config.js';
import { logEngine } from './logger.js';
import { pushLog } from '../routes/logs.js';
import { setEngineReady, setEngineNoModels, backendLineFeeder, resetEngineDevice,
         engineBackendFromLine, noteEngineBackendLine } from '../engineState.js';
import { aceClient } from './aceClient.js';
import { buildGpuEnv, listGpusSync } from './gpuDevices.js';
import { currentWorker, engineWorkers, inWorkerScope, primaryWorker, workerById,
         type EngineWorker } from './engineWorkers.js';
import { gpuLaneBusy, gpuLaneDepth, holdGpuLane } from './generation/gpuLane.js';

/**
 * While the engine waits on an empty models folder, a model copied in by hand
 * (not through the Model Manager) must start it too. The engine scans only
 * top-level *.gguf, so look for one that has stopped growing (mtime older than
 * 3 s, so a copy in progress does not trigger an early start).
 */
const MODELS_WATCH_MS = 5_000;

// Crash-count limiter: prevent infinite respawn on fatal errors (missing DLLs, etc.)
const MAX_CRASHES = 3;
const CRASH_WINDOW_MS = 30_000; // 30 seconds

/**
 * Bumped on every unexpected engine exit. A job in flight when it changes is
 * gone — the respawned engine never had it — so pollers fail fast instead of
 * retrying a dead socket until the wall-clock timeout (#179). Per worker: the
 * one the caller is bound to (the primary outside any worker scope).
 */
export function engineCrashSeq(): number {
  return currentWorker().crashSeq;
}

/** Line ace-server prints before exiting 1 on an empty models folder (hot-step-server.cpp). */
const NO_MODELS_LINE = "ERROR: no models found";

/** How long restartAceServer() waits for /health after a respawn. */
const RESTART_HEALTH_TIMEOUT_MS = 90_000;
/** Poll interval of the watcher that outlives that timeout. */
const LATE_READY_POLL_MS = 2_000;

/** Test seam: swap the spawn and shrink the timings. Production never touches it. */
export const testHooks = {
  spawn: spawn as typeof spawn,
  listGpus: listGpusSync,
  restartHealthTimeoutMs: RESTART_HEALTH_TIMEOUT_MS,
  lateReadyPollMs: LATE_READY_POLL_MS,
};

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
//
// Hooks run inside the restarted worker's scope, so their engine calls reach
// that worker and no other.
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

interface EngineProcess {
  start(): ChildProcess | null;
  stop(reason?: string, timeoutMs?: number, opts?: { suspend?: boolean }): Promise<boolean>;
  restart(): Promise<boolean>;
  retryIfGaveUp(): void;
  process(): ChildProcess | null;
  suspended(): boolean;
}

const processes = new Map<EngineWorker, EngineProcess>();

function engineProcess(worker: EngineWorker): EngineProcess {
  let p = processes.get(worker);
  if (!p) { p = createEngineProcess(worker); processes.set(worker, p); }
  return p;
}

function createEngineProcess(worker: EngineWorker): EngineProcess {
  // Primary output is unchanged; a secondary names itself in every line.
  const tag = worker.primary ? '[ace-server]' : `[ace-server:${worker.id}]`;
  const srv = worker.primary ? '[Server]' : `[Server] ${worker.id}`;

  /** Readiness: the primary keeps driving the global engine state it always has. */
  const setReady = (ready: boolean, status: string) => {
    if (worker.primary) { setEngineReady(ready, status); return; }   // w0 reads the globals
    worker.ready = ready;
    worker.status = status;
  };
  const setNoModels = (v: boolean) => { if (worker.primary) setEngineNoModels(v); };
  const isReachable = () => inWorkerScope(worker, () => aceClient.isReachable());
  const restore = () => inWorkerScope(worker, runRestoreHooks);

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

  let modelsWatch: NodeJS.Timeout | null = null;

  function stopWatchingForModels(): void {
    if (modelsWatch) { clearInterval(modelsWatch); modelsWatch = null; }
  }

  function watchForModels(): void {
    stopWatchingForModels();
    modelsWatch = setInterval(() => {
      try {
        const dir = config.aceServer.models;
        const found = fs.readdirSync(dir).some(n => n.toLowerCase().endsWith('.gguf')
          && Date.now() - fs.statSync(path.join(dir, n)).mtimeMs > 3000);
        if (found) retryIfGaveUp();
      } catch { /* folder missing or unreadable: keep waiting */ }
    }, MODELS_WATCH_MS);
    modelsWatch.unref();
  }

  /** Cancel any pending crash-respawn and invalidate one already in flight. */
  function cancelPendingRespawn(): void {
    lifecycleEpoch++;
    stopWatchingForModels();
    if (respawnTimer) {
      clearTimeout(respawnTimer);
      respawnTimer = null;
    }
  }

  let crashCount = 0;
  let firstCrashTime = 0;
  /** Set when the crash limiter gives up; cleared by the next restart. */
  let gaveUp = false;

  /**
   * Spawn ace-server. Returns null when the binary is missing.
   * Also assigns the current process.
   */
  function start(): ChildProcess | null {
    const exe = config.aceServer.exe;
    if (!exe || !fs.existsSync(exe)) {
      console.log(`${srv} ace-server not found at: ${exe}`);
      console.log(`${srv} Start ace-server manually, or set ACESTEPCPP_EXE in .env`);
      // A crash respawn lands here when the binary has vanished (an unmounted
      // volume, #179). Nothing else will ever retry, so say so.
      setReady(false, `ace-server not found at ${exe}`);
      aceProcess = null;
      worker.pid = null;
      return null;
    }

    const args = [
      '--models', config.aceServer.models,
      '--host', config.aceServer.host,
      '--port', String(worker.port),
    ];

    // Add adapters dir if it exists
    if (config.aceServer.adapters && fs.existsSync(config.aceServer.adapters)) {
      args.push('--adapters', config.aceServer.adapters);
    }

    // Release only: the data folder's plugins/ overlay, scanned after the shipped plugins.
    if (USER_PLUGINS_DIR) args.push('--user-plugins', USER_PLUGINS_DIR);

    // --keep-loaded: flips the engine's ModelStore to EVICT_NEVER so the ~17 s
    // LoKr precompute (and the DiT/VAE load) only happens once per combo instead
    // of every /synth. Default OFF (VRAM trade-off) — toggle in Settings →
    // Environment → "Keep models in VRAM" (ACESTEPCPP_KEEP_LOADED), restart-required.
    if (config.aceServer.keepLoaded) {
      args.push('--keep-loaded');
      console.log(`${srv} --keep-loaded: DiT + adapter stay resident across requests`);
    }

    // Add noise profile if available
    if (config.aceServer.noiseProfile && fs.existsSync(config.aceServer.noiseProfile)) {
      args.push('--noise-profile', config.aceServer.noiseProfile);
      console.log(`${srv} Noise profile: ${config.aceServer.noiseProfile}`);
    }

    // Add draft LM for speculative decoding (if available)
    if (config.aceServer.draftLm && fs.existsSync(config.aceServer.draftLm)) {
      args.push('--draft-lm', config.aceServer.draftLm);
      console.log(`${srv} Draft LM: ${path.basename(config.aceServer.draftLm)}`);
    }

    // VAE tiling parameters (resolves Vulkan pinned memory allocation failures)
    if (config.aceServer.vaeChunk) {
      args.push('--vae-chunk', String(config.aceServer.vaeChunk));
    }
    if (config.aceServer.vaeOverlap) {
      args.push('--vae-overlap', String(config.aceServer.vaeOverlap));
    }


    console.log(`${srv} Starting ace-server: ${path.basename(exe)}`);
    console.log(`${srv} Models: ${config.aceServer.models}`);
    console.log(`${srv} Port: ${worker.port}`);

    // Inject TensorRT libs into PATH if available.
    // and CUDA_VISIBLE_DEVICES for GPU selection.
    // IMPORTANT: On Windows, process.env is a case-insensitive Proxy, but spreading
    // it to a plain object creates case-sensitive keys. The key is typically 'Path'
    // not 'PATH', so we must find the actual key to avoid creating a shadowing duplicate.
    //
    // The env is now ALWAYS rebuilt, because the GPU decision includes "unset
    // CUDA_VISIBLE_DEVICES". Inheriting process.env was not neutral: the Settings
    // UI writes `CUDA_VISIBLE_DEVICES=` for Auto, dotenv turns that into an empty
    // string, and an empty CUDA_VISIBLE_DEVICES hides EVERY GPU — the engine
    // silently fell back to CPU.
    //
    // An opted-in worker is pinned to its own card by UUID instead; a card
    // that has vanished since boot leaves that worker down, not the server.
    let built: ReturnType<typeof buildGpuEnv>;
    try {
      built = buildGpuEnv(process.env, worker.gpuUuid ?? undefined, testHooks.listGpus());
    } catch (err: any) {
      console.error(`${srv} ${err?.message || err}`);
      setReady(false, `GPU ${worker.gpuUuid} not found`);
      aceProcess = null;
      worker.pid = null;
      return null;
    }
    const { env, selection: gpu } = built;
    const spawnOpts: { stdio: any; env: NodeJS.ProcessEnv } = {
      stdio: ['ignore', 'pipe', 'pipe'] as any,
      env,
    };
    console.log(worker.gpuUuid ? gpu.log.replace('[Training]', srv) : gpu.log);

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
        console.log(`${srv} TRT-LLM libs: ${trtllmLibs}`);
      }

      console.log(`${srv} TensorRT libs: ${config.aceServer.trtLibs}`);
    }

    if (worker.primary) resetEngineDevice();   // w0 reads the global
    else worker.device = null;
    const child = testHooks.spawn(exe, args, spawnOpts);

    // Filter repetitive GGML noise from console output (still written to ace_engine.log via logEngine)
    const isNoise = (line: string) =>
      line.includes('CUDA graph warmup') || line.includes('CUDA Graph id') || line.includes('ggml_backend_cuda_graph_compute');

    const noteDevice = (line: string) => {
      const device = engineBackendFromLine(line);
      if (!device) return;
      if (worker.primary) noteEngineBackendLine(line);   // w0 reads the global
      else worker.device = device;
    };
    const logWorker = worker.primary ? undefined : worker.id;
    const feedOut = backendLineFeeder(noteDevice);
    const feedErr = backendLineFeeder(noteDevice);
    child.stdout?.on('data', (data: Buffer) => {
      feedOut(data.toString());
      const lines = data.toString().split('\n').filter(Boolean);
      for (const line of lines) {
        if (!isNoise(line)) console.log(`${tag} ${line}`);
        logEngine(line, worker.logFile);
        pushLog(line, 'engine', logWorker);
      }
    });

    child.stderr?.on('data', (data: Buffer) => {
      feedErr(data.toString());
      // Chunks are not lines: match on a rolling tail so a split marker still counts.
      errTail = (errTail + data.toString()).slice(-256);
      if (errTail.includes(NO_MODELS_LINE)) sawNoModels = true;
      const lines = data.toString().split('\n').filter(Boolean);
      for (const line of lines) {
        if (!isNoise(line)) console.log(`${tag} ${line}`);
        logEngine(line, worker.logFile);
        pushLog(line, 'engine', logWorker);
      }
    });

    const spawnEpoch = lifecycleEpoch;
    let sawNoModels = false;
    let errTail = '';

    child.on('exit', (code, signal) => {
      if (aceProcess === child) { aceProcess = null; worker.pid = null; }   // never hand out a dead child
      // 'exit' can precede the last stderr chunk; classify only after it drains.
      const err = child.stderr;
      if (err && err.readableEnded === false && !err.destroyed) {
        let done = false;
        const go = () => { if (done) return; done = true; clearTimeout(t); onExited(code, signal); };
        const t = setTimeout(go, 500);
        err.once('close', go);
        err.once('end', go);
      } else {
        onExited(code, signal);
      }
    });

    const onExited = (code: number | null, signal: NodeJS.Signals | null) => {
      if (suspended) return;   // deliberate stop — never respawn
      if (spawnEpoch !== lifecycleEpoch) return;     // superseded by a stop/restart
      if (sawNoModels && code === 1) {
        // Empty models folder: a first run, not a crash. No restart budget spent;
        // retryIfGaveUp() brings the engine up once a download lands.
        console.log(`${tag} No models installed — waiting for a download in the Model Manager.`);
        setNoModels(true);
        setReady(false, 'No models installed — download one in the Model Manager');
        gaveUp = true;
        watchForModels();
        return;
      }
      if (signal !== 'SIGTERM' && signal !== 'SIGINT' && code !== 0) {
        console.error(`${tag} Process exited with code ${code}, signal ${signal}`);
        worker.crashSeq++;

        // Crash-count limiter: reset window if enough time has passed
        const now = Date.now();
        if (now - firstCrashTime > CRASH_WINDOW_MS) {
          crashCount = 0;
          firstCrashTime = now;
        }
        crashCount++;

        if (crashCount >= MAX_CRASHES) {
          console.error(`${tag} Crashed ${MAX_CRASHES} times within ${CRASH_WINDOW_MS / 1000}s — giving up.`);
          console.error(`${tag} Check the error above: usually no models downloaded yet, or a DLL missing from the engine/ directory.`);
          console.error(`${tag} The engine starts again by itself once a model download finishes.`);
          setReady(false, `Engine crashed ${MAX_CRASHES} times — download a model in the Model Manager, or check the logs`);
          gaveUp = true;
          return;
        }

        console.log(`${tag} Restarting in 3 seconds... (crash ${crashCount}/${MAX_CRASHES})`);
        // Down until the respawn answers /health: nothing may dispatch to a dead engine.
        setReady(false, 'Engine crashed — restarting...');
        if (respawnTimer) clearTimeout(respawnTimer);
        respawnTimer = setTimeout(() => {
          respawnTimer = null;
          // A stop OR a restart that landed inside the 3 s window must win, or
          // this spawns a duplicate engine that nothing tracks.
          if (suspended || spawnEpoch !== lifecycleEpoch) return;
          // Same health probe and restore hooks as restart(), without resetting
          // the crash limiter: the new process forgot the user's selections.
          const respawned = start();
          if (respawned) void lateReadyWatch(respawned, lifecycleEpoch, 'after a crash');
        }, 3000);
      }
    };

    child.on('error', (err) => {
      console.error(`${tag} Failed to start: ${err.message}`);
    });

    aceProcess = child;
    worker.pid = child.pid ?? null;
    return child;
  }

  /**
   * Stop ace-server and wait for the process to actually exit.
   *
   * `suspended` is set FIRST so the exit handler never respawns; it stays set
   * until restart() clears it. Resolves when the 'exit' event fires, or
   * after `timeoutMs`. A no-op when there is no live child.
   */
  function stop(reason = 'Engine stopped', timeoutMs = 15_000,
                opts: { suspend?: boolean } = {}): Promise<boolean> {
    // A shutdown is not a suspension: leaving `suspended` set makes
    // POST /api/generate answer with the training-preprocess message during the
    // Ctrl-C window, which is misleading in the logs.
    if (opts.suspend !== false) suspended = true;
    cancelPendingRespawn();
    setReady(false, reason);

    const child = aceProcess;
    if (!child || !child.pid || child.exitCode !== null || child.signalCode !== null) {
      aceProcess = null;
      worker.pid = null;
      return Promise.resolve(true);
    }

    console.log(`${srv} Stopping ace-server (pid ${child.pid}) — ${reason}`);

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
        // still-live one, which then cannot bind its port and burns the crash budget.
        if (exited && aceProcess === child) { aceProcess = null; worker.pid = null; }
        resolve(exited);
      };

      const onTimeout = () => {
        if (!escalated && process.platform !== 'win32') {
          // SIGTERM can be ignored by an engine wedged in a CUDA/Vulkan sync.
          escalated = true;
          console.warn(`${srv} ace-server ignored SIGTERM — escalating to SIGKILL`);
          kill(true);
          timer.refresh();
          return;
        }
        console.warn(`${srv} ace-server did not exit within ${timeoutMs} ms — it may still be running`);
        finish(false);
      };
      const timer: NodeJS.Timeout = setTimeout(onTimeout, timeoutMs);

      child.once('exit', onExit);
      kill(false);
    });
  }

  /**
   * Respawn the engine if the crash limiter gave up. A first launch with an empty
   * models folder exits at the model scan three times before the user has had a
   * chance to download anything; this brings it back once they have.
   */
  function retryIfGaveUp(): void {
    if (!gaveUp || suspended) return;
    console.log(`${srv} Models changed since the engine gave up — starting it again`);
    void restart();
  }

  /**
   * Clear `suspended`, respawn ace-server and poll /health until it answers.
   * Resolves true when the engine came back, false on timeout.
   */
  async function restart(): Promise<boolean> {
    // Order matters: kill any crash-respawn already scheduled BEFORE clearing
    // `suspended`, or the orphaned timer spawns a duplicate engine 3 s from now.
    cancelPendingRespawn();
    suspended = false;
    gaveUp = false;
    crashCount = 0;
    setNoModels(false);
    setReady(false, 'Restarting engine...');
    const epoch = lifecycleEpoch;

    if (aceProcess && aceProcess.exitCode === null && aceProcess.signalCode === null) {
      // A stop that timed out left the old engine alive; a second spawn would
      // just fail to bind its port and burn the crash budget.
      console.warn(`${srv} restartAceServer: an engine child is still live — not spawning a second one`);
      const live = aceProcess;
      if (await isReachable()) return restoreThenReady(live, epoch);
      setReady(false, 'Engine did not come back — restart the app');
      return false;
    }

    const child = start();
    if (!child) {
      setReady(false, 'Engine did not come back — restart the app');
      return false;
    }

    const deadline = Date.now() + testHooks.restartHealthTimeoutMs;
    while (Date.now() < deadline) {
      if (await isReachable()) {
        console.log(`${srv} ace-server is back up`);
        return restoreThenReady(child, epoch);
      }
      await new Promise(r => setTimeout(r, 1000));
    }

    console.error(`${srv} ace-server did not answer /health within 90 s after a restart`);
    setReady(false, 'Engine did not come back — restart the app');
    // The verdict above is not final while the child lives: a cold first boot on
    // fresh models can out-run 90 s, and engineReady stays false (every backend
    // reads 'down', so the model lists stay empty) until something flips it.
    void lateReadyWatch(child, lifecycleEpoch);
    return false;
  }

  /**
   * Replay the persisted selections, then report ready. Ready goes last so no job
   * dispatches to an engine still on its defaults: a render holding the engine
   * mutex makes the restore hooks skip as stale, and YuE2 has no later reconcile.
   * A stop, restart or crash during the replay wins; its verdict stands.
   */
  async function restoreThenReady(child: ChildProcess, epoch: number): Promise<boolean> {
    await restore();
    if (aceProcess !== child || epoch !== lifecycleEpoch || suspended
        || child.exitCode !== null || child.signalCode !== null) return false;
    setReady(true, 'Ready');
    return true;
  }

  /** Keep polling a live child until /health answers; then restore and flip ready. */
  async function lateReadyWatch(child: ChildProcess, epoch: number, when = 'after the restart timeout'): Promise<void> {
    while (aceProcess === child && epoch === lifecycleEpoch && !suspended
           && child.exitCode === null && child.signalCode === null) {
      await new Promise(r => setTimeout(r, testHooks.lateReadyPollMs));
      if (aceProcess !== child || epoch !== lifecycleEpoch || suspended) return;
      const up = await isReachable();
      // A stop or restart can land while the request is in flight; its stale
      // success must not mark a stopped engine ready.
      if (aceProcess !== child || epoch !== lifecycleEpoch || suspended
          || child.exitCode !== null || child.signalCode !== null) return;
      if (up) {
        console.log(`${srv} ace-server answered /health ${when} — marking ready`);
        await restoreThenReady(child, epoch);
        return;
      }
    }
  }

  return {
    start, stop, restart, retryIfGaveUp,
    process: () => aceProcess,
    suspended: () => suspended,
  };
}

const primary = () => engineProcess(primaryWorker());

/**
 * Spawn the primary ace-server (w0). Returns null when the binary is missing.
 */
export function startAceServer(): ChildProcess | null {
  return primary().start();
}

/** The primary's live child, or null. */
export function getAceProcess(): ChildProcess | null {
  return primary().process();
}

/** True while a deliberate stop of the primary is in effect (something else owns the GPU). */
export function isEngineSuspended(): boolean {
  return primary().suspended();
}

/** Stop the primary ace-server and wait for it to exit. See stop() above. */
export function stopAceServer(reason = 'Engine stopped', timeoutMs = 15_000,
                              opts: { suspend?: boolean } = {}): Promise<boolean> {
  return primary().stop(reason, timeoutMs, opts);
}

/** Respawn every worker whose crash limiter gave up (a model download landed). */
export function retryIfGaveUp(): void {
  for (const w of engineWorkers()) {
    const p = processes.get(w);
    if (p) p.retryIfGaveUp();
  }
}

/** Clear `suspended`, respawn the primary and poll /health until it answers. */
export function restartAceServer(): Promise<boolean> {
  return primary().restart();
}

/**
 * Bring up the secondary workers of a non-empty opt-in (the primary is
 * startAceServer()'s). Each comes up through restart(): spawn, wait for
 * /health, run the restore hooks in its own scope. A no-op with one worker.
 */
export function startEngineWorkers(): Promise<boolean[]> {
  return Promise.all(engineWorkers().filter(w => !w.primary).map(w => engineProcess(w).restart()));
}

/** Stop the secondary workers (shutdown). The primary stays stopAceServer()'s. */
export function stopEngineWorkers(reason = 'Engine stopped', timeoutMs = 15_000,
                                  opts: { suspend?: boolean } = {}): Promise<boolean[]> {
  return Promise.all(engineWorkers().filter(w => !w.primary).map(w => engineProcess(w).stop(reason, timeoutMs, opts)));
}

/** Stop every worker (training, maintenance, shutdown). True when all of them exited. */
export async function stopAllEngines(reason = 'Engine stopped', timeoutMs = 15_000,
                                     opts: { suspend?: boolean } = {}): Promise<boolean> {
  const exited = await Promise.all(engineWorkers().map(w => engineProcess(w).stop(reason, timeoutMs, opts)));
  return exited.every(Boolean);
}

/** Lane holds taken by suspendEngines(), handed back in order by resumeEngines(). */
const trainingHolds: Array<() => void> = [];

/**
 * Training owns every card (exclusive in v1): wait for all GPU lane work,
 * including a YuE2 render still draining after its early lane release, then
 * hold the whole lane and stop every worker, so no job starts on another card
 * mid-run. resumeEngines() undoes both. Rejects with LaneResetError if the
 * queue is reset while it waits; nothing has been stopped then.
 */
export async function suspendEngines(reason: string): Promise<boolean> {
  if (gpuLaneBusy() || gpuLaneDepth() > 0) pushLog(`[Engine] ${reason}: waiting for running GPU work to finish`);
  trainingHolds.push(await holdGpuLane(reason));
  return stopAllEngines(reason);
}

/** Restart every worker, then hand the lane back. Answers for the primary, like restartAceServer(). */
export async function resumeEngines(): Promise<boolean> {
  try {
    const [primaryUp] = await Promise.all(engineWorkers().map(w => engineProcess(w).restart()));
    return primaryUp;
  } finally {
    trainingHolds.shift()?.();
  }
}

/**
 * Restart one worker on request. Idle only: refused while any call is bound to
 * it (runWithWorker), so a request never kills a render in flight. Crash
 * respawns do not go through here and are not gated.
 */
export async function restartEngineWorker(id: string): Promise<boolean> {
  const worker = workerById(id);
  if (!worker) throw new Error(`no engine worker ${id}`);
  if (worker.busy > 0) throw new Error(`engine worker ${id} is busy; restart it when idle`);
  const p = engineProcess(worker);
  await p.stop(`Restarting ${worker.id}`, 15_000, { suspend: false });
  return p.restart();
}
