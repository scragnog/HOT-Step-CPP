// gpuDevices.ts — NVIDIA GPU enumeration and engine device selection
//
// Why this exists (issue #153): the Settings GPU picker listed devices by
// nvidia-smi index and wrote that index to CUDA_VISIBLE_DEVICES. Those two
// numbers are NOT the same enumeration:
//
//   - nvidia-smi numbers GPUs in PCI bus order.
//   - CUDA numbers them with CUDA_DEVICE_ORDER, which defaults to
//     FASTEST_FIRST — the driver's own guess at which card is quickest.
//
// On a mixed rig (RTX 5060 Ti 16 GB + RTX 4060 8 GB) those orders came out as
// exact reverses, so picking the 16 GB card in the UI handed the engine the
// 8 GB one. "Auto" was no better: the engine calls ggml_backend_init_best(),
// which takes the first GPU device — CUDA0 — whatever that happens to be.
//
// The fix is to stop passing indices to the driver at all. nvidia-smi also
// reports each GPU's UUID, and CUDA_VISIBLE_DEVICES accepts "GPU-<uuid>"
// strings, which are order-independent by construction. So:
//
//   - Auto with >1 GPU  → the UUID of the card with the most total VRAM.
//   - Manual            → the UUID of the card the user picked (an old
//                         index-shaped value is mapped through nvidia-smi's
//                         index, which is the index the picker showed).
//   - No nvidia-smi     → pass the raw value and force CUDA_DEVICE_ORDER=
//                         PCI_BUS_ID so indices at least mean what nvidia-smi
//                         (and therefore the picker) said.
//
// Everything here degrades to "do nothing" when nvidia-smi is missing, which
// is the AMD / Intel / Apple / CPU-only case.

import { execFile, execFileSync } from 'child_process';
import { promisify } from 'util';
import { config } from '../config.js';

const execFileAsync = promisify(execFile);

export interface NvidiaGpu {
  /** nvidia-smi index — PCI bus order, NOT the CUDA device index. */
  index: number;
  /** "GPU-xxxxxxxx-...." — stable, order-independent, accepted by CUDA_VISIBLE_DEVICES. */
  uuid: string;
  name: string;
  memoryMB: number;
}

const SMI_ARGS = [
  '--query-gpu=index,uuid,name,memory.total',
  '--format=csv,noheader,nounits',
];

/** GPUs do not appear or vanish while the app runs, so one probe is enough. */
let cache: NvidiaGpu[] | null = null;

function parseSmi(stdout: string): NvidiaGpu[] {
  return stdout
    .trim()
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean)
    .map((line) => {
      // index, uuid, name, memory.total — name is taken as "everything in the
      // middle" so a comma in a product name cannot shift the other fields.
      const parts = line.split(',').map((s) => s.trim());
      if (parts.length < 4) return null;
      const index = parseInt(parts[0], 10);
      const memoryMB = parseInt(parts[parts.length - 1], 10);
      if (!Number.isFinite(index)) return null;
      return {
        index,
        uuid: parts[1],
        name: parts.slice(2, -1).join(', '),
        memoryMB: Number.isFinite(memoryMB) ? memoryMB : 0,
      } as NvidiaGpu;
    })
    .filter((g): g is NvidiaGpu => g !== null);
}

/** Probe nvidia-smi without blocking the event loop. Empty on any failure. */
export async function listGpus(): Promise<NvidiaGpu[]> {
  if (cache) return cache;
  try {
    const { stdout } = await execFileAsync('nvidia-smi', SMI_ARGS, { timeout: 5000 });
    cache = parseSmi(stdout);
  } catch {
    cache = [];
  }
  return cache;
}

/**
 * Same probe, synchronously — startAceServer() is sync and the answer has to
 * be in hand before the child's env is built. Costs a few hundred ms once per
 * process, then serves the cache.
 */
export function listGpusSync(): NvidiaGpu[] {
  if (cache) return cache;
  try {
    const stdout = execFileSync('nvidia-smi', SMI_ARGS, {
      timeout: 5000,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    cache = parseSmi(stdout);
  } catch {
    cache = [];
  }
  return cache;
}

export interface GpuSelection {
  /**
   * What to set CUDA_VISIBLE_DEVICES to for the engine child, or null to leave
   * the variable UNSET (which is not the same as empty — an empty
   * CUDA_VISIBLE_DEVICES hides every GPU and drops the engine to CPU).
   */
  visibleDevices: string | null;
  /** True when indices survive into the child and therefore need PCI ordering. */
  forcePciOrder: boolean;
  /** One line for the startup log: what was chosen and why. */
  log: string;
}

const gb = (mb: number) => (mb / 1024).toFixed(mb >= 10 * 1024 ? 0 : 1);

/** A value CUDA can resolve without any index ordering (UUID or MIG id). */
function isDeviceId(value: string): boolean {
  return /^(GPU-|MIG-)/i.test(value);
}

/**
 * Decide the engine child's CUDA_VISIBLE_DEVICES.
 *
 * `setting` is config.aceServer.cudaVisibleDevices — whatever the Settings UI
 * or the user's .env put there. Empty means "Auto".
 */
export function resolveGpuSelection(setting: string): GpuSelection {
  const raw = (setting || '').trim();
  const gpus = listGpusSync();

  // ── Auto ────────────────────────────────────────────────────────────────
  if (!raw) {
    if (gpus.length < 2) {
      return {
        visibleDevices: null,
        forcePciOrder: gpus.length > 0,
        log: gpus.length
          ? `[Server] GPU: auto — one CUDA device (${gpus[0].name}), leaving selection to the engine`
          : '[Server] GPU: auto — nvidia-smi reported no CUDA devices, leaving selection to the engine',
      };
    }
    // Most total VRAM wins; ties go to the lower PCI index. The engine is
    // VRAM-bound long before it is FLOP-bound, and the driver's FASTEST_FIRST
    // guess ignores memory entirely — which is how a 16 GB card lost to an
    // 8 GB one on the reporter's machine.
    const best = gpus.reduce((a, b) =>
      b.memoryMB > a.memoryMB || (b.memoryMB === a.memoryMB && b.index < a.index) ? b : a);
    return {
      visibleDevices: best.uuid,
      forcePciOrder: true,
      log: `[Server] GPU: auto — picked GPU ${best.index} ${best.name} (${gb(best.memoryMB)} GB), `
         + `the largest of ${gpus.length}; CUDA_VISIBLE_DEVICES=${best.uuid}`,
    };
  }

  // ── Manual: already an order-independent device id ──────────────────────
  if (raw.split(',').every((p) => isDeviceId(p.trim()))) {
    const named = raw
      .split(',')
      .map((p) => gpus.find((g) => g.uuid.toLowerCase() === p.trim().toLowerCase()))
      .filter((g): g is NvidiaGpu => !!g)
      .map((g) => `GPU ${g.index} ${g.name} (${gb(g.memoryMB)} GB)`)
      .join(', ');
    return {
      visibleDevices: raw,
      forcePciOrder: false,
      log: `[Server] GPU: manual — ${named || 'device id from settings'}; CUDA_VISIBLE_DEVICES=${raw}`,
    };
  }

  // ── Manual: index-shaped (Settings UI before this fix, or a hand-edited
  //    .env). Translate through nvidia-smi, because that index is the one the
  //    picker displayed — and it is NOT the index CUDA would have used. ─────
  const parts = raw.split(',').map((p) => p.trim()).filter(Boolean);
  if (parts.every((p) => /^\d+$/.test(p)) && gpus.length > 0) {
    const matched = parts.map((p) => gpus.find((g) => g.index === parseInt(p, 10)));
    if (matched.every((g): g is NvidiaGpu => !!g)) {
      const uuids = matched.map((g) => g.uuid).join(',');
      const named = matched.map((g) => `GPU ${g.index} ${g.name} (${gb(g.memoryMB)} GB)`).join(', ');
      return {
        visibleDevices: uuids,
        forcePciOrder: false,
        log: `[Server] GPU: manual — index ${raw} is ${named}; passing its UUID `
           + `(CUDA_VISIBLE_DEVICES=${uuids}) so driver ordering cannot swap it`,
      };
    }
  }

  // ── Manual: anything else — respect it verbatim, and pin the ordering so an
  //    index means what nvidia-smi says it means. ──────────────────────────
  return {
    visibleDevices: raw,
    forcePciOrder: true,
    log: `[Server] GPU: manual — using CUDA_VISIBLE_DEVICES=${raw} as given `
       + '(could not resolve it to a GPU UUID; forcing CUDA_DEVICE_ORDER=PCI_BUS_ID)',
  };
}

/** Total VRAM (MB) of the card the engine and trainers run on, or null when
 *  nvidia-smi cannot say (no NVIDIA GPU, or a selection it cannot resolve). */
export function selectedGpuMemoryMB(): number | null {
  const gpus = listGpusSync();
  const pick = resolveGpuSelection(config.aceServer.cudaVisibleDevices).visibleDevices;
  if (!pick) return gpus.length === 1 ? gpus[0].memoryMB : null;
  const gpu = gpus.find((g) => g.uuid.toLowerCase() === pick.split(',')[0].trim().toLowerCase());
  return gpu ? gpu.memoryMB : null;
}

/**
 * A child env with the GPU decision applied: CUDA_VISIBLE_DEVICES unset (in
 * any letter case Windows stored it) and then set only when the decision
 * names a device, plus CUDA_DEVICE_ORDER=PCI_BUS_ID when indices survive and
 * the user has not pinned an order themselves. Used for ace-server and every
 * ace-train child, so training lands on the same card as generation (#153).
 */
export function buildGpuEnv(base: NodeJS.ProcessEnv = process.env): { env: NodeJS.ProcessEnv; selection: GpuSelection } {
  const env: NodeJS.ProcessEnv = { ...base };
  // config.aceServer.cudaVisibleDevices tracks Settings edits live; process.env
  // only holds what .env said at boot.
  const selection = resolveGpuSelection(config.aceServer.cudaVisibleDevices);
  for (const k of Object.keys(env)) {
    if (k.toUpperCase() === 'CUDA_VISIBLE_DEVICES') delete env[k];
  }
  if (selection.visibleDevices) env.CUDA_VISIBLE_DEVICES = selection.visibleDevices;
  const hasDeviceOrder = Object.keys(env).some((k) => k.toUpperCase() === 'CUDA_DEVICE_ORDER');
  if (selection.forcePciOrder && !hasDeviceOrder) env.CUDA_DEVICE_ORDER = 'PCI_BUS_ID';
  return { env, selection };
}
