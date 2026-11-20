// Pure decision logic for audio queue ownership — no storage, no network,
// no browser globals, so it can be unit-tested without a DOM. The stateful
// wiring (localStorage, migrateAudioQueue, the banner's own effect) lives in
// audioGenQueueStore.ts and AudioQueueMigrationBanner.tsx.

export type QueueOwner = 'browser' | 'migrating' | 'server';

export type OwnerDecisionAction = 'none' | 'set-server' | 'migrate' | 'retry';

/** What to do with a browser's audio queue ownership on load.
 *  - `'server'` is terminal: already decided, nothing to do.
 *  - `'migrating'` means an attempt was already made and may have reached
 *    the server, so it always retries (replaying the persisted payload is
 *    idempotent). A rollback left unfinished by v1.4.x retries too.
 *  - Anything else is never decided. That includes `'browser'`, which only a
 *    v1.4.x rollback wrote: the browser no longer runs a queue, so its items
 *    would otherwise wait forever. No pending work goes straight to
 *    server-owned, pending work migrates fresh. */
export function decideQueueOwnerAction(raw: string | null, hasPending: boolean): OwnerDecisionAction {
  if (raw === 'server') return 'none';
  if (raw === 'migrating') return 'retry';
  return hasPending ? 'migrate' : 'set-server';
}

/** Pure visibility rule for AudioQueueMigrationBanner: silent/successful
 *  ownership never shows anything. The only state needing the user's
 *  attention is a stuck `'migrating'` — an attempt that's already failed once
 *  (the silent retry on load didn't resolve it), where only a manual Retry
 *  can move it forward. */
export function shouldShowMigrationBanner(hasToken: boolean, ready: boolean, owner: QueueOwner): boolean {
  return hasToken && ready && owner === 'migrating';
}
