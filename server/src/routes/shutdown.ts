// shutdown.ts — Graceful shutdown and restart endpoints
//
// POST /api/shutdown — kills ALL processes on our ports (Node, Vite, ace-server)
// POST /api/restart  — kills Node + ace-server, writes marker file for loop restart
// Uses port-based taskkill on Windows, like HOT-Step 9000.

import { Router } from 'express';
import { execSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Project root — two levels up from server/src/routes/ */
const PROJECT_ROOT = path.resolve(__dirname, '../../..');

const router = Router();

/** Kill all processes listening on a given port (Windows) */
function killPort(port: number): void {
  try {
    const output = execSync(
      `netstat -ano | findstr ":${port}" | findstr "LISTENING"`,
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
        console.log(`[Shutdown] Killed PID ${pid} (port ${port})`);
      } catch {
        // Process may already be dead
      }
    }
  } catch {
    // No process found on this port — that's fine
  }
}

/** Walk up the process tree from our PID and kill the top-level parent.
 *  Chain: cmd.exe → npx → tsx watch → node (us)
 *  Killing the parent tsx/npx with /T kills everything, and
 *  cmd.exe /c exits because its command finished. */
function killSelf(): void {
  try {
    // Find our parent PID (tsx watch or npx)
    const output = execSync(
      `wmic process where processid=${process.pid} get parentprocessid /value`,
      { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }
    ).trim();
    const match = output.match(/ParentProcessId=(\d+)/i);
    if (match) {
      const parentPid = match[1];
      console.log(`[Shutdown] Killing parent PID ${parentPid} (our process tree)`);

      // Spawn detached killer to kill parent after we start exiting
      const killer = spawn('cmd.exe', [
        '/c', `ping -n 2 127.0.0.1 > nul & taskkill /PID ${parentPid} /T /F`
      ], {
        detached: true,
        stdio: 'ignore',
        windowsHide: true,
      });
      killer.unref();
    }
  } catch {
    // Fallback: just exit
  }
}

// POST /api/shutdown — terminate everything
router.post('/', (_req, res) => {
  console.log('[Server] Shutdown requested via API');
  res.json({ success: true, message: 'Shutting down...' });

  // Give the response time to flush, then kill everything
  setTimeout(() => {
    console.log('[Server] Killing all processes...');

    if (process.platform === 'win32') {
      // Kill ace-server and Vite by port
      killPort(8085);
      killPort(3000);

      // Kill our own process tree from outside
      killSelf();
    }

    // Fallback in case killSelf doesn't work
    setTimeout(() => process.exit(0), 3000);
  }, 300);
});

// POST /api/restart — restart server (loop wrapper relaunches)
router.post('/restart', (_req, res) => {
  console.log('[Server] Restart requested via API');

  // Write marker file so the loop wrapper (launch.bat / restart-loop.cmd)
  // knows to re-launch instead of exiting
  const markerPath = path.join(PROJECT_ROOT, '.restart-requested');
  try {
    fs.writeFileSync(markerPath, new Date().toISOString(), 'utf8');
    console.log(`[Server] Wrote restart marker: ${markerPath}`);
  } catch (err: any) {
    console.error(`[Server] Failed to write restart marker: ${err.message}`);
  }

  res.json({ success: true, message: 'Restarting...' });

  // Give the response time to flush, then kill server + ace-server (but NOT Vite)
  setTimeout(() => {
    console.log('[Server] Restarting — killing ace-server and self...');

    if (process.platform === 'win32') {
      // Kill ace-server by port — it will be re-spawned on restart
      killPort(8085);

      // Do NOT kill Vite (port 3000) — leave it running for dev mode

      // Kill our own process tree from outside
      killSelf();
    }

    // Fallback in case killSelf doesn't work
    setTimeout(() => process.exit(0), 3000);
  }, 300);
});

export default router;

