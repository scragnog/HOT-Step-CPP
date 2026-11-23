/**
 * App-level window events.
 *
 * The stores and panels talk to each other through `window` CustomEvents
 * (song created, song renamed, restarting, post-processing finished). Senders
 * and listeners had drifted on whether `detail` was always present, so a
 * listener reading `e.detail.songId` could throw on a sender that omitted it.
 * emit keeps the shape in one place.
 */
export function emit<T = undefined>(name: string, detail?: T): void {
  window.dispatchEvent(detail === undefined ? new CustomEvent(name) : new CustomEvent(name, { detail }));
}
