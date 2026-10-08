// contracts/settings.ts — wire shapes for /api/settings (server/src/routes/settings.ts).

import type { NvidiaGpu } from '../services/gpuDevices.js';

export interface EnvResponse {
  /** Exposed .env keys only; unset keys are backfilled with their resolved default. */
  values: Record<string, string>;
  /** Keys that need a restart to take effect — the UI badges them. */
  restartKeys: string[];
}

export interface EnvUpdateResponse {
  updated: string[];
  restartRequired: boolean;
}

/** `services/gpuDevices.ts` already names this shape; re-exported under the
 *  name GET /settings/gpus docs it by, rather than a second definition. */
export type GpuInfo = NvidiaGpu;

export interface GpusResponse { gpus: GpuInfo[] }
