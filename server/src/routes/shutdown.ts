// shutdown.ts — Graceful shutdown and restart endpoints
//
// POST /api/shutdown — stops ace-server, Vite (dev only) and the Node process tree
// POST /api/restart  — stops and relaunches (writes marker for the loop wrapper)
//
// SAFETY: we only kill processes we own — the ace-server child (via
// stopAceServer), the dev Vite server on :3000 (only when dev.bat set
// HOT_STEP_DEV), and our own parent on Windows (tsx / tsx watch).

import { Router } from 'express';
import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { PROJECT_ROOT, PORTABLE_MODE } from '../config.js';
import { killActiveChildren } from '../services/training/labelingQueue.js';
import { stopAceServer } from '../services/aceEngineProcess.js';

const router = Router();

/** Reap spawned training children (ace-train + its ffmpeg) before we exit. */
function killTrainingChildren(): void {
  try { killActiveChildren(); } catch (err) { console.error('[Shutdown] killActiveChildren failed:', err); }
}

/** Kill the Vite dev server by port. Only when dev.bat started it (HOT_STEP_DEV),
 *  or a LAUNCH.bat user with something unrelated on :3000 loses it on Quit. */
function killVite(): void {
  if (process.platform !== 'win32' || !process.env.HOT_STEP_DEV) return;
  try {
    const output = execSync(
      `netstat -ano | findstr ":3000" | findstr "LISTENING"`,
      { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }
    );
    const pids = new Set<string>();
    for (const line of output.split('\n')) {
      const pid = line.trim().split(/\s+/).pop();
      if (pid && /^\d+$/.test(pid) && pid !== '0') pids.add(pid);
    }
    for (const pid of pids) {
      try {
        execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
        console.log(`[Shutdown] Killed Vite PID ${pid} (port 3000)`);
      } catch { /* already dead */ }
    }
  } catch { /* nothing on :3000 */ }
}

/** Kill our own process tree from outside (Windows).
 *  Chain: cmd.exe → npx → tsx [watch] → node (us). A plain process.exit()
 *  leaves `tsx watch` alive waiting for a file change, so the restart loop
 *  never sees us exit and Quit leaves a Node process behind. Killing the
 *  parent with /T takes the whole tree; cmd.exe /c then finishes its command.
 *  Uses process.ppid — the old `wmic` lookup no longer exists on Windows 11
 *  24H2+, which silently broke both Restart and Quit.
 *  On macOS/Linux, process.exit() is enough: launch.sh uses exec. */
function killSelf(): void {
  if (process.platform !== 'win32') return;
  const parentPid = process.ppid;
  if (!parentPid) return;
  console.log(`[Shutdown] Killing parent PID ${parentPid} (our process tree)`);
  // Delay so the HTTP response has flushed. Do NOT use `cmd /c ping` as the
  // sleep: PING.EXE can hang forever (observed 2026-07-17).
  setTimeout(() => {
    try {
      spawn('taskkill', ['/PID', String(parentPid), '/T', '/F'], {
        detached: true, stdio: 'ignore', windowsHide: true,
      }).unref();
    } catch { /* fallback: process.exit below still runs */ }
  }, 500);
}

async function stopEverything(restart: boolean): Promise<void> {
  killTrainingChildren();
  // Through the owning module so the exit handler knows this was deliberate —
  // the old port-based taskkill made it log "crash 1/3, restarting in 3 s".
  await stopAceServer(restart ? 'Server restarting' : 'Server shutting down', 15_000, { suspend: false });
  if (!restart) killVite();
  // Portable restart: HOT-Step.bat runs node directly, so our parent IS the
  // restart loop — exiting is enough there. Every other case needs the tree kill.
  if (!restart || !PORTABLE_MODE) killSelf();
  setTimeout(() => {
    console.log(restart ? '[Server] Exiting for restart.' : '[Server] Exiting.');
    process.exit(0);
  }, 1000);
}

// POST /api/shutdown — terminate everything
router.post('/', (_req, res) => {
  console.log('[Server] Shutdown requested via API');
  res.json({ success: true, message: 'Shutting down...' });
  setTimeout(() => void stopEverything(false), 300);
});

// POST /api/restart — restart server (loop wrapper relaunches)
router.post('/restart', (_req, res) => {
  console.log('[Server] Restart requested via API');
  const markerPath = path.join(PROJECT_ROOT, '.restart-requested');
  try {
    fs.writeFileSync(markerPath, new Date().toISOString(), 'utf8');
    console.log(`[Server] Wrote restart marker: ${markerPath}`);
  } catch (err: any) {
    console.error(`[Server] Failed to write restart marker: ${err.message}`);
  }
  res.json({ success: true, message: 'Restarting...' });
  setTimeout(() => void stopEverything(true), 300);
});

export default router;
