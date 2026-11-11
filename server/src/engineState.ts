// engineState.ts — Shared engine readiness state
//
// Extracted to avoid circular imports between index.ts and route modules.
// index.ts sets these values; routes read them.

/** True only once the engine's own /health has answered (boot, restart or crash respawn). */
export let engineReady = false;

/** Human-readable status for what the engine is doing before it's ready. */
export let engineBootStatus = 'Initializing...';

/** True when ace-server exited because the models folder is empty (a first run, not a crash). */
export let engineNoModels = false;

export function setEngineNoModels(v: boolean) {
  engineNoModels = v;
}

/** True when the engine cannot come up without help: binary missing, spawn
 *  failed, or the crash limiter gave up. Cleared by the next spawn. */
export let engineFailed = false;

export function setEngineFailed(v: boolean) {
  engineFailed = v;
}

/** Update the engine boot status (called from index.ts bootstrap) */
export function setEngineReady(ready: boolean, status: string) {
  engineReady = ready;
  engineBootStatus = status;
}

/** Raw ggml device name ("CUDA0", "Vulkan0", "CPU") from the engine's last "[Load] <label> backend:" line; null until a model has loaded. */
export let engineDevice: string | null = null;

/** The device named by one engine "[Load] <label> backend: <device>" line, or null. */
export function engineBackendFromLine(line: string): string | null {
  const m = /^\[Load\] .+? backend: (\S+)/.exec(line);
  return m ? m[1] : null;
}

/** Record the device from one engine log line. Returns true if the line was a backend line. */
export function noteEngineBackendLine(line: string): boolean {
  const device = engineBackendFromLine(line);
  if (!device) return false;
  engineDevice = device;
  return true;
}

/** Forget the device on every respawn: the new process has not loaded anything yet. */
export function resetEngineDevice() {
  engineDevice = null;
}

/** Coarse backend family from a ggml device name; null for an unknown name. */
export function backendFamily(device: string | null): 'cuda' | 'vulkan' | 'metal' | 'hip' | 'cpu' | null {
  if (!device) return null;
  const d = device.toLowerCase();
  if (d.startsWith('cuda')) return 'cuda';
  if (d.startsWith('vulkan')) return 'vulkan';
  if (d.startsWith('metal') || d.startsWith('mtl')) return 'metal';
  if (d.startsWith('rocm') || d.startsWith('hip')) return 'hip';
  if (d.startsWith('cpu')) return 'cpu';
  return null;
}

/** Per-stream feeder for raw chunks: holds back a trailing partial line so a marker split across chunks still parses. */
export function backendLineFeeder(note: (line: string) => unknown = noteEngineBackendLine): (chunk: string) => void {
  let rest = '';
  return (chunk) => {
    const lines = (rest + chunk).split('\n');
    rest = lines.pop() ?? '';
    for (const l of lines) note(l);
  };
}
