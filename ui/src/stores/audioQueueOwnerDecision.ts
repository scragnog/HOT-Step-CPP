// Pure decision logic for audio queue ownership — no storage, no network,
// no browser globals, so it can be unit-tested without a DOM. The stateful
// wiring (localStorage, migrateAudioQueue, the banner's own effect) lives in
// audioGenQueueStore.ts and AudioQueueMigrationBanner.tsx.

export type QueueOwner = 'browser' | 'migrating' | 'server';

export type OwnerDecisionAction = 'none' | 'set-server' | 'migrate';

/** What to do with a browser's audio queue ownership on load.
 *  - `'server'`/`'browser'` are terminal: already decided, nothing to do.
 *  - Anything else — a missing/unrecognised key (never decided) OR
 *    `'migrating'` (an attempt interrupted by reload; whoever was running it
 *    is gone, so it's safe to retry) — resolves fresh: no pending work goes
 *    straight to server-owned, pending work migrates. */
export function decideQueueOwnerAction(raw: string | null, hasPending: boolean): OwnerDecisionAction {
  if (raw === 'server' || raw === 'browser') return 'none';
  return hasPending ? 'migrate' : 'set-server';
}

/** Pure visibility rule for AudioQueueMigrationBanner: silent/successful
 *  ownership (server-owned, or browser-owned with nothing pending) never
 *  shows anything. This only appears when the automatic migration actually
 *  failed and left pending work stuck browser-side. */
export function shouldShowMigrationBanner(hasToken: boolean, ready: boolean, owner: QueueOwner, count: number): boolean {
  return hasToken && ready && owner === 'browser' && count > 0;
}
