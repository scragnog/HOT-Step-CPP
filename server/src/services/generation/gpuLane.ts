// One GPU owner at a time, including while reset work drains.

export interface LaneLease {
  readonly id: number;
  readonly generation: number;
  readonly label: string;
  readonly family?: string;
  readonly draining: boolean;
  isCurrent(): boolean;
}

export class LaneResetError extends Error {
  constructor() {
    super('Queue reset by user');
    this.name = 'LaneResetError';
  }
}

interface Pending {
  label: string;
  family?: string;
  start(lease: LaneLease): void;
  reject(error: Error): void;
}

const pending: Pending[] = [];
let generation = 0;
let nextLeaseId = 1;
let current: LaneLease | null = null;

export function gpuLaneBusy(): boolean { return current !== null; }
export function gpuLaneDepth(): number { return pending.length; }
export function gpuLaneOwner(): LaneLease | null { return current; }
/** The family of the task next in line, if any. */
export function gpuLaneNextFamily(): string | undefined { return pending[0]?.family; }

/** Hand the lane to the next task while this one carries on without the
 *  GPU (a YuE2 job whose render the engine runs on its own lane). The lease
 *  stops being current; its eventual completion releases nothing. */
export function releaseGpuLane(lease: LaneLease): boolean {
  if (current !== lease) return false;
  current = null;
  pump();
  return true;
}

/** Invalidates callbacks and rejects queued tasks. The current task retains
 * ownership until its promise settles, including any external post-processing.
 * There is no timer release or release based only on an engine restart. */
export function resetGpuLane(): number {
  generation++;
  const drained = pending.splice(0);
  for (const task of drained) task.reject(new LaneResetError());
  return drained.length;
}

function pump(): void {
  if (current) return;
  const task = pending.shift();
  if (!task) return;
  const lease: LaneLease = {
    id: nextLeaseId++, generation, label: task.label, family: task.family,
    get draining() { return lease.generation !== generation; },
    isCurrent: () => current === lease && lease.generation === generation,
  };
  Object.freeze(lease);
  current = lease;
  task.start(lease);
}

function finish(lease: LaneLease, settle: () => void): void {
  // An old completion can never release another owner's lease.
  if (current === lease) current = null;
  settle();
  pump();
}

export function runOnGpuLane<T>(
  fn: (lease: LaneLease) => Promise<T>,
  opts: { label?: string; family?: string } = {},
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    pending.push({
      label: opts.label ?? 'gpu task', family: opts.family, reject,
      start(lease) {
        Promise.resolve().then(() => {
          // Reset may have invalidated the lease before its first callback.
          if (!lease.isCurrent()) throw new LaneResetError();
          return fn(lease);
        }).then(
          value => finish(lease, () => resolve(value)),
          error => finish(lease, () => reject(error)),
        );
      },
    });
    pump();
  });
}
