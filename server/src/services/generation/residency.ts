// generation/residency.ts — queued demand accounting (step 5)
//
// Demand accounting is a leaf. evictOtherFamilies() loads the backend
// registry lazily: lane callers such as the YuE2 takes and the training
// runner sit inside the registry's own import graph (a backend module reaches
// them), so a static import here would be a cycle.

import { awaitBackendRelease, trackBackendRelease } from './backendRelease.js';

type Family = string;

/** Jobs queued or running per backend family. The selector is intentionally
 * not consulted: demand belongs to the job's captured family. */
const demand = new Map<Family, number>();

export function noteEnqueued(family: Family): void {
  demand.set(family, (demand.get(family) ?? 0) + 1);
}

/** Decrement once from the outer runOnGpuLane promise's finally. Never allow a
 * bookkeeping mismatch to create negative demand. Later eviction logic will
 * use the zero transition to enqueue a stale-safe release. */
export function noteFinished(family: Family): void {
  const next = (demand.get(family) ?? 0) - 1;
  if (next > 0) demand.set(family, next);
  else demand.delete(family);
}

export function familyDemand(family: Family): number {
  return demand.get(family) ?? 0;
}

// Future compatible exports from §2.3 are intentionally not implemented in
// this step: activeFamily(), holdEngine(), engineHeld(), enqueueEviction(),
// require the real GPU lane and registry injection.

/** What an eviction needs from a backend: its id, and how to free it. */
type Evictable = { id: Family; releaseVram?: () => Promise<void> };
/** Test hook: the backends an eviction consults. A test of the lane must not
 *  reach a real engine over HTTP just because another family is registered. */
let evictables: (() => Evictable[]) | null = null;
export function _setEvictionBackends(lister: (() => Evictable[]) | null): void { evictables = lister; }

/** Free every other family's VRAM and wait until each one confirms, before
 *  `family` loads weights on the lane. Both run on the caller's engine worker
 *  (backendRelease.ts), so a job on one card never evicts or waits on another. Run on every lane task that loads a
 *  family, not only on a family change: a per-job backendOverride switches
 *  family without a selector change, and a YuE2 score preview under
 *  keep-loaded leaves its LM resident without being a generation. Releasing a
 *  family with nothing loaded is a cheap no-op. */
export async function evictOtherFamilies(family: string): Promise<void> {
  const others: Evictable[] = evictables
    ? evictables()
    : (await import('../backends/registry.js')).listBackends();
  for (const other of others) {
    if (other.id === family || !other.releaseVram) continue;
    trackBackendRelease(other.id, () => other.releaseVram!()).catch(err => {
      console.warn(`[Residency] releaseVram not confirmed for '${other.id}' before '${family}':`, err?.message || err);
    });
  }
  await awaitBackendRelease();
}
