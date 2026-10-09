// Pure decision logic for audio queue ownership — no storage, no network,
// no browser globals, so it can be unit-tested without a DOM. The stateful
// wiring (localStorage, migrateAudioQueue, the banner's own effect) lives in
// audioGenQueueStore.ts and AudioQueueMigrationBanner.tsx.

export type QueueOwner = 'browser' | 'migrating' | 'server';

export type OwnerDecisionAction = 'none' | 'set-server' | 'migrate' | 'retry';

/** What to do with a browser's audio queue ownership on load.
 *  - `'server'`/`'browser'` are terminal: already decided, nothing to do.
 *  - `'migrating'` means an attempt was already made and may have reached
 *    the server — per the rule that once an import has started the browser
 *    executor never runs again on its own, this always retries (replaying
 *    the persisted payload is idempotent), never reverts to `'browser'`.
 *  - A missing/unrecognised key (never decided): no pending work goes
 *    straight to server-owned, pending work migrates fresh. */
export function decideQueueOwnerAction(raw: string | null, hasPending: boolean, rollbackRequested = false): OwnerDecisionAction {
  if (raw === 'server' || raw === 'browser') return 'none';
  // A failed rollback also leaves 'migrating'. Replaying the import on load
  // would quietly undo the user's rollback: leave it fenced for the banner.
  if (raw === 'migrating') return rollbackRequested ? 'none' : 'retry';
  return hasPending ? 'migrate' : 'set-server';
}

/** Pure visibility rule for AudioQueueMigrationBanner: silent/successful
 *  ownership (server-owned, or browser-owned with nothing pending) never
 *  shows anything. The only state needing the user's attention is a stuck
 *  `'migrating'` — an attempt that's already failed once (the silent retry
 *  on load didn't resolve it), where only a manual Retry or an explicit
 *  rollback can move it forward. */
export function shouldShowMigrationBanner(hasToken: boolean, ready: boolean, owner: QueueOwner): boolean {
  return hasToken && ready && owner === 'migrating';
}
