// generation/gpuLane.ts — the single-GPU serialization lane
//
// The C++ engine owns one GPU and one global log pub/sub with no job tagging,
// so two pieces of engine work running at once both thrash VRAM and leak each
// other's progress lines. Everything heavy therefore goes through here and
// runs strictly one at a time, in submission order.
//
// This used to be private to routes/generate.ts, where only generations could
// use it. Post-processing re-runs (SuperSep split, StableStep/SA3, PP-VAE
// re-encode) are just as GPU-hungry as a generation and have to share the same
// lane, not race it.

type Task = () => void;

const pending: Task[] = [];
let running = false;

/** True while a lane task is executing. */
export function gpuLaneBusy(): boolean {
  return running;
}

/** How many tasks are waiting behind the running one. */
export function gpuLaneDepth(): number {
  return pending.length;
}

/**
 * Drop every waiting task and mark the lane free. Backs the "reset queue"
 * escape hatch, so it deliberately lies about a task that is still running —
 * the caller has already cancelled the engine job it was waiting on. Returns
 * how many waiting tasks were dropped.
 */
export function resetGpuLane(): number {
  const drained = pending.length;
  pending.length = 0;
  running = false;
  return drained;
}

/**
 * Run `fn` on the GPU lane, waiting for any earlier task to finish first.
 * Resolves/rejects with `fn`'s own outcome — a rejection releases the lane
 * exactly like a success, so one failed task cannot wedge the queue.
 */
export function runOnGpuLane<T>(fn: () => Promise<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const execute = () => {
      running = true;
      fn().then(resolve, reject).finally(() => {
        running = false;
        const next = pending.shift();
        if (next) next();
      });
    };

    if (running) pending.push(execute);
    else execute();
  });
}
