// paramHistory.ts — the host's undo history for generation parameters. Pure logic: the
// writes go through the `apply` callback, so a restore uses the same store setters as an
// edit (and the same hs-* keys). In memory only, 100 steps, drags coalesced. The singleton
// that talks to the real stores is in paramStore.ts.

export interface ParamChange { id: string; backend: string; before: unknown; after: unknown }
export interface HistoryStep { label: string; at: number; changes: ParamChange[] }
export interface HistorySnapshot {
  canUndo: boolean; canRedo: boolean;
  /** Oldest first. Steps at index >= `cursor` are undone and can be redone. */
  steps: ReadonlyArray<Pick<HistoryStep, 'label' | 'at'>>;
  cursor: number;
}

export const HISTORY_LIMIT = 100;
/** Edits to one control closer together than this are one step (a drag, a held key). */
export const COALESCE_MS = 600;

const same = (a: unknown, b: unknown) => a === b || JSON.stringify(a) === JSON.stringify(b);

export function createHistory(apply: (c: ParamChange, value: unknown) => void, now: () => number = Date.now) {
  let steps: HistoryStep[] = [];
  let cursor = 0;
  let applying = false;
  let tx: { label: string; changes: ParamChange[] } | null = null;
  let snap: HistorySnapshot = { canUndo: false, canRedo: false, steps: [], cursor: 0 };
  const listeners = new Set<() => void>();
  const emit = () => {
    snap = { canUndo: cursor > 0, canRedo: cursor < steps.length, steps: steps.map(({ label, at }) => ({ label, at })), cursor };
    listeners.forEach(l => l());
  };

  const push = (label: string, changes: ParamChange[]) => {
    const t = now();
    const last = cursor > 0 && cursor === steps.length ? steps[cursor - 1] : null;
    const one = changes.length === 1 ? changes[0] : null;
    if (one && last && last.changes.length === 1 && last.changes[0].id === one.id && last.changes[0].backend === one.backend && t - last.at < COALESCE_MS) {
      last.changes[0].after = one.after;
      last.at = t;
      if (same(last.changes[0].before, last.changes[0].after)) { steps.pop(); cursor--; } // dragged back to where it began
    } else {
      steps = steps.slice(0, cursor);
      steps.push({ label, at: t, changes });
      if (steps.length > HISTORY_LIMIT) steps.shift();
      cursor = steps.length;
    }
    emit();
  };

  return {
    /** Call after the write has happened. A no-op change and a restore in progress are ignored. */
    record(label: string, change: ParamChange): void {
      if (applying || same(change.before, change.after)) return;
      if (tx) {
        const prior = tx.changes.find(c => c.id === change.id && c.backend === change.backend);
        if (prior) prior.after = change.after; else tx.changes.push({ ...change });
        return;
      }
      push(label, [{ ...change }]);
    },
    /** Everything `fn` records becomes one step. */
    transaction<T>(label: string, fn: () => T): T {
      if (tx) return fn(); // nested: the outer one owns the step
      tx = { label, changes: [] };
      try { return fn(); } finally {
        const done = tx; tx = null;
        const changes = done.changes.filter(c => !same(c.before, c.after));
        if (changes.length) push(done.label, changes);
      }
    },
    undo(): void {
      if (cursor === 0) return;
      const step = steps[--cursor];
      applying = true;
      try { for (const c of [...step.changes].reverse()) apply(c, c.before); } finally { applying = false; }
      emit();
    },
    redo(): void {
      if (cursor >= steps.length) return;
      const step = steps[cursor++];
      applying = true;
      try { for (const c of step.changes) apply(c, c.after); } finally { applying = false; }
      emit();
    },
    clear(): void { steps = []; cursor = 0; emit(); },
    subscribe(l: () => void): () => void { listeners.add(l); return () => { listeners.delete(l); }; },
    getSnapshot: (): HistorySnapshot => snap,
  };
}
