// shutdown.ts — Graceful shutdown and restart endpoints
//
// POST /api/shutdown — gracefully stops Node server + ace-server child
// POST /api/restart  — stops and relaunches (writes marker for loop wrapper)
// Platform-aware: uses taskkill on Windows, targeted SIGTERM on macOS/Linux.
//
// SAFETY: We only kill processes we own (our PID and our child ace-server).
// We NEVER kill by port or process group — that can destroy unrelated services.

import { Router } from 'express';
import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { PROJECT_ROOT } from '../config.js';

const router = Router();

/** Kill the ace-server child process safely.
 *  We track it by PID from the spawn in index.ts, not by port scanning. */
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

// POST /api/shutdown — terminate everything gracefully
router.post('/', (_req, res) => {
  console.log('[Server] Shutdown requested via API');
  res.json({ success: true, message: 'Shutting down...' });

  setTimeout(() => {
    console.log('[Server] Shutting down...');
    killAceServer();

    // Give ace-server a moment to die, then exit ourselves
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

    // Exit cleanly — the launch wrapper script will restart us
    setTimeout(() => {
      console.log('[Server] Exiting for restart.');
      process.exit(0);
    }, 1000);
  }, 300);
});

export default router;
