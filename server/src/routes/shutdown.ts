// shutdown.ts — Graceful shutdown and restart endpoints
//
// POST /api/shutdown — gracefully stops Node server + ace-server child
// POST /api/restart  — stops and relaunches (writes marker for loop wrapper)
// Platform-aware: uses taskkill on Windows, targeted SIGTERM on macOS/Linux.
//
// SAFETY: We only kill processes we own (our PID and our child ace-server).
// We NEVER kill by port on macOS — that can destroy unrelated services.
// On Windows, port-based kill is used for ace-server because we don't
// have the child PID available in this module.

import { Router } from 'express';
import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { PROJECT_ROOT, PORTABLE_MODE } from '../config.js';

const router = Router();

/** Kill the ace-server child process safely (cross-platform). */
function killAceServer(): void {
  try {
    if (process.platform === 'win32') {
      // Windows: netstat + taskkill (port-based, needed because we don't have the child PID here)
      const output = execSync(
        `netstat -ano | findstr ":8085" | findstr "LISTENING"`,
        { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }
      );
      const pids = new Set<string>();
      for (const line of output.split('\n')) {
        const parts = line.trim().split(/\s+/);
        const pid = parts[parts.length - 1];
        if (pid && /^\d+$/.test(pid) && pid !== '0') {
          pids.add(pid);
        }
      }
      for (const pid of pids) {
        try {
          execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
          console.log(`[Shutdown] Killed ace-server PID ${pid}`);
        } catch {
          // Process may already be dead
        }
      }
    } else {
      // macOS/Linux: find ace-server processes that are children of us
      try {
        const output = execSync(
          `pgrep -P ${process.pid} -f ace-server 2>/dev/null || true`,
          { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }
        ).trim();
        if (output) {
          for (const pid of output.split('\n').filter(Boolean)) {
            try {
              process.kill(parseInt(pid, 10), 'SIGTERM');
              console.log(`[Shutdown] Sent SIGTERM to ace-server child PID ${pid}`);
            } catch {
              // Already dead
            }
          }
        }
      } catch {
        // No matching processes
      }
    }
  } catch {
    // No process found — that's fine
  }
}

/** Kill the Vite dev server by port (Windows only, used during full shutdown). */
function killVite(): void {
  if (process.platform !== 'win32') return; // macOS: Vite isn't our child in production
  try {
    const output = execSync(
      `netstat -ano | findstr ":3000" | findstr "LISTENING"`,
      { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }
    );
    const pids = new Set<string>();
    for (const line of output.split('\n')) {
      const parts = line.trim().split(/\s+/);
      const pid = parts[parts.length - 1];
      if (pid && /^\d+$/.test(pid) && pid !== '0') {
        pids.add(pid);
      }
    }
    for (const pid of pids) {
      try {
        execSync(`taskkill /PID ${pid} /T /F`, { stdio: 'ignore' });
        console.log(`[Shutdown] Killed Vite PID ${pid} (port 3000)`);
      } catch {
        // Process may already be dead
      }
    }
  } catch {
    // No process found on port 3000 — that's fine
  }
}

/** Kill our own process tree from outside (Windows).
 *  Chain: cmd.exe → npx → tsx watch → node (us)
 *  Killing the parent tsx/npx with /T kills everything, and
 *  cmd.exe /c exits because its command finished.
 *  On macOS/Linux, process.exit() is sufficient because launch.sh
 *  uses exec (replaces shell with node, no orphan parents). */
function killSelf(): void {
  if (process.platform !== 'win32') return; // macOS doesn't need this
  try {
    const output = execSync(
      `wmic process where processid=${process.pid} get parentprocessid /value`,
      { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }
    ).trim();
    const match = output.match(/ParentProcessId=(\d+)/i);
    if (match) {
      const parentPid = match[1];
      console.log(`[Shutdown] Killing parent PID ${parentPid} (our process tree)`);

      // Spawn taskkill directly after a Node-side delay. Do NOT use the
      // `cmd /c ping -n 2 ... & taskkill` sleep idiom: ping can hang forever
      // (observed 2026-07-17 — hung PING.EXE processes meant taskkill never
      // ran, tsx watch survived, and the restart-loop marker was never
      // consumed, leaving the server dead after an in-app restart).
      setTimeout(() => {
        try {
          const killer = spawn('taskkill', ['/PID', parentPid, '/T', '/F'], {
            detached: true,
            stdio: 'ignore',
            windowsHide: true,
          });
          killer.unref();
        } catch {
          // Fallback: our own process.exit still runs
        }
      }, 700);
    }
  } catch {
    // Fallback: just exit
  }
}

// POST /api/shutdown — terminate everything gracefully
router.post('/', (_req, res) => {
  console.log('[Server] Shutdown requested via API');
  res.json({ success: true, message: 'Shutting down...' });

  setTimeout(() => {
    console.log('[Server] Shutting down...');
    killAceServer();
    killVite();

    // On Windows: kill our process tree from outside (needed for dev-rebuild workflow)
    // On macOS: process.exit() is sufficient
    killSelf();

    // Fallback exit
    setTimeout(() => {
      console.log('[Server] Exiting.');
      process.exit(0);
    }, 1000);
  }, 300);
});

// POST /api/restart — restart server (loop wrapper relaunches)
router.post('/restart', (_req, res) => {
  console.log('[Server] Restart requested via API');

  // Write marker file so the loop wrapper (launch.bat / launch.sh)
  // knows to re-launch instead of exiting
  const markerPath = path.join(PROJECT_ROOT, '.restart-requested');
  try {
    fs.writeFileSync(markerPath, new Date().toISOString(), 'utf8');
    console.log(`[Server] Wrote restart marker: ${markerPath}`);
  } catch (err: any) {
    console.error(`[Server] Failed to write restart marker: ${err.message}`);
  }

  res.json({ success: true, message: 'Restarting...' });

  setTimeout(() => {
    console.log('[Server] Restarting — stopping ace-server and self...');
    killAceServer();

    // Do NOT kill Vite (port 3000) — leave it running for dev mode

    // In portable mode, the bat file has a restart loop that checks
    // .restart-requested after node exits — just exit cleanly.
    // In dev mode, kill our process tree so tsx watch relaunches us.
    if (!PORTABLE_MODE) {
      killSelf();
    }

    // Exit — portable bat loop will relaunch, dev tsx watch will relaunch
    setTimeout(() => {
      console.log('[Server] Exiting for restart.');
      process.exit(0);
    }, 1000);
  }, 300);
});

export default router;
