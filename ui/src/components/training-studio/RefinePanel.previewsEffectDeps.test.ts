// RefinePanel.previewsEffectDeps.test.ts — a ladder handed off from Review
// (its id already selected) must retry reading previews once the local run
// list lands, not read the worker once with this machine's own namespaced
// run id and stay empty forever. No UI test runner is wired up for this
// project; run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/training-studio/RefinePanel.previewsEffectDeps.test.ts)
import assert from 'node:assert/strict';
import test from 'node:test';
import { previewsEffectDeps } from './RefinePanel.tsx';

test('indexedLocally flipping with everything else unchanged still yields a different deps tuple', () => {
  const before = previewsEffectDeps('ds1', 'remote:LivingRoom:job1', 'done', false);
  const after = previewsEffectDeps('ds1', 'remote:LivingRoom:job1', 'done', true);
  assert.notDeepEqual(before, after);
  assert.equal(before.some((v, i) => v !== after[i]), true);
});

test('identical inputs yield shallow-equal deps (a real effect would not retrigger)', () => {
  const a = previewsEffectDeps('ds1', 'run1', 'done', true);
  const b = previewsEffectDeps('ds1', 'run1', 'done', true);
  assert.deepEqual(a, b);
  assert.equal(a.every((v, i) => v === b[i]), true);
});

// A minimal stand-in for React's effect scheduler: runs the effect again
// only when its deps (from previewsEffectDeps) are not shallow-equal to the
// last run, tearing down the previous run first — the same contract
// RefinePanel.tsx's own useEffect calls rely on.
function runEffect(
  deps: readonly unknown[], prevDeps: readonly unknown[] | undefined, prevCleanup: (() => void) | undefined,
  effect: () => (() => void) | undefined,
): { cleanup: (() => void) | undefined; ran: boolean } {
  if (prevDeps && deps.length === prevDeps.length && deps.every((v, i) => v === prevDeps[i])) return { cleanup: prevCleanup, ran: false };
  prevCleanup?.();
  return { cleanup: effect(), ran: true };
}

test('a ladder opened before the local run list lands retries once ownership resolves, and a late worker response never overwrites the local one', async () => {
  // Mirrors RefinePanel.tsx: readPreviews(worker|local) resolves async; the
  // worker call here is deliberately the slower of the two, as it would be
  // if it even reached a real network — the bug this guards against is that
  // first, stale call racing past the retry and winning.
  const calls: string[] = [];
  let previews: number | null = null;
  const readPreviews = (indexedLocally: boolean) => new Promise<number>(resolve => {
    calls.push(indexedLocally ? 'local' : 'worker');
    setTimeout(() => resolve(indexedLocally ? 1 : 0), indexedLocally ? 0 : 20);
  });

  let prevDeps: readonly unknown[] | undefined;
  let cleanup: (() => void) | undefined;
  const mount = (indexedLocally: boolean) => {
    const deps = previewsEffectDeps('ds1', 'remote:LivingRoom:job1', 'done', indexedLocally);
    const result = runEffect(deps, prevDeps, cleanup, () => {
      let cancelled = false;
      void readPreviews(indexedLocally).then(n => { if (!cancelled) previews = n; });
      return () => { cancelled = true; };
    });
    prevDeps = deps; cleanup = result.cleanup;
    return result.ran;
  };

  assert.equal(mount(false), true); // first render: runs=[] yet, reads the worker
  assert.equal(mount(false), false); // an unrelated re-render with nothing changed must not refire
  assert.equal(mount(true), true); // the local run list has landed: must retrigger

  await new Promise(resolve => setTimeout(resolve, 30)); // let both the fast local and the slow stale worker call settle
  assert.deepEqual(calls, ['worker', 'local']);
  assert.equal(previews, 1, 'the local read must win even though the stale worker call resolves later');
});
