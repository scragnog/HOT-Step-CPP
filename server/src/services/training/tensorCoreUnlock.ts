import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { listGpus } from '../gpuDevices.js';

export type TensorCoreUnlock = 'on' | 'off' | 'unknown';

interface BootLog {
  bootMs: number;
  writtenMs: number;
  handoff: string;
}

export function tensorCoreUnlockFromLog({ bootMs, writtenMs, handoff }: BootLog): TensorCoreUnlock {
  const flags = /HANDOFF-OUT:.*\bga\/ada:ss0_unlock=(true|false) ss1_unlock=(true|false) plm_open=(true|false)\b/.exec(handoff);
  if (!flags || !Number.isFinite(bootMs) || !Number.isFinite(writtenMs) || bootMs <= 0 || writtenMs <= 0) {
    return 'unknown';
  }
  // The USB loader hands off just before Windows boots. Allow for the
  // filesystem timestamp resolution and boot-time clock corrections.
  if (writtenMs < bootMs - 30_000) return 'off';
  if (writtenMs > bootMs + 30_000) return 'unknown';
  if (flags.slice(1).some(value => value === 'false')) return 'off';
  return /\breadout1=0x00000000\b/i.test(handoff) ? 'on' : 'unknown';
}

// Enumerate only local disks. USB sticks may be reported as fixed disks.
// Pick the most recent recognized log and never send its contents to callers.
const bootLogProbe = String.raw`
$boot = (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
$disks = Get-CimInstance Win32_LogicalDisk -Filter 'DriveType = 2 OR DriveType = 3' -ErrorAction Stop
$logs = $disks | ForEach-Object {
  Get-Item -LiteralPath ($_.DeviceID + '\loader.log') -ErrorAction SilentlyContinue
} | Sort-Object LastWriteTimeUtc -Descending
foreach ($log in $logs) {
  $handoff = Get-Content -LiteralPath $log.FullName -Tail 100 -ErrorAction SilentlyContinue |
    Where-Object { $_ -match 'HANDOFF-OUT:.*ga/ada:ss0_unlock=' } | Select-Object -Last 1
  if ($handoff) {
    @{
      bootMs = ([DateTimeOffset]$boot).ToUnixTimeMilliseconds()
      writtenMs = ([DateTimeOffset]$log.LastWriteTimeUtc).ToUnixTimeMilliseconds()
      handoff = [string]$handoff
    } | ConvertTo-Json -Compress
    break
  }
}
`;

async function probe(): Promise<TensorCoreUnlock> {
  try {
    const gpus = await listGpus();
    // The boot log does not identify the GPU in a way we can match to a
    // trainer device. Report a state only on the single-card setup verified.
    if (gpus.length !== 1 || gpus[0].name !== 'NVIDIA GeForce RTX 4090') return 'unknown';
    const { stdout } = await promisify(execFile)('powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', bootLogProbe],
      { timeout: 4000, windowsHide: true });
    return tensorCoreUnlockFromLog(JSON.parse(stdout) as BootLog);
  } catch {
    return 'unknown';
  }
}

let cached: { checkedAt: number; result: Promise<TensorCoreUnlock> } | undefined;

export function tensorCoreUnlockStatus(): Promise<TensorCoreUnlock> {
  if (process.platform !== 'win32') return Promise.resolve('unknown');
  const now = Date.now();
  if (!cached || now - cached.checkedAt >= 30_000) {
    cached = { checkedAt: now, result: probe() };
  }
  return cached.result;
}
