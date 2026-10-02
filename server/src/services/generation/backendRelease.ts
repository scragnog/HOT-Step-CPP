/**
 * Unconfirmed VRAM evictions from backend switches (#204).
 *
 * A backend switch answers at once and evicts the outgoing family in the
 * background; no render may load weights while any such eviction is still
 * unconfirmed. Every family's eviction is tracked on its own, so a later
 * switch can neither hide an earlier failure nor be skipped by a wait that
 * started before it.
 */

interface Eviction { release: () => Promise<void>; done: Promise<void> }

const pending = new Map<string, Eviction>();

/** Start evicting `family` (the outgoing backend of a switch). A second switch
 *  away from the same family queues behind the first. */
export function trackBackendRelease(family: string, release: () => Promise<void>): Promise<void> {
  const prior = pending.get(family)?.done.catch(() => {}) ?? Promise.resolve();
  const done = prior.then(release);
  done.catch(() => {});  // observed by awaitBackendRelease; never an unhandled rejection
  pending.set(family, { release, done });
  return done;
}

/** Resolve once no eviction is unconfirmed. A failed eviction is tried once
 *  more (the engine may have finished it after the client gave up); if that
 *  fails too, this throws naming the family and leaves it tracked, so the
 *  next render tries again rather than loading next to it. Evictions started
 *  while this waits are waited for as well. */
export async function awaitBackendRelease(): Promise<void> {
  while (pending.size > 0) {
    const [family, ev] = pending.entries().next().value as [string, Eviction];
    try {
      await ev.done;
    } catch {
      try {
        await ev.release();
      } catch (err: any) {
        throw new Error(`The previous backend (${family}) has not confirmed it freed its GPU memory, so this `
          + `render was not started: ${err?.message || err}`);
      }
    }
    // A newer eviction of the same family may have replaced this one meanwhile.
    if (pending.get(family) === ev) pending.delete(family);
  }
}

/** Test hook. */
export function _resetBackendReleases(): void { pending.clear(); }
