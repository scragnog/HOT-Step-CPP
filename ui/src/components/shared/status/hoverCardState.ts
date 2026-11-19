// hoverCardState.ts — who has a hover card open, shared by every card on the page.
//
// A flanking card portals to <body>, outside the panel it describes, so reaching
// for it fires that panel's mouseleave. Panels consult `hoverCardIsOpen()` and
// defer closing while a card is up, the same way they already defer while the
// window has lost focus.
//
// Only one card may be up at a time. Moving the pointer from one explained
// button to the next used to leave the first card inside its 320ms grace period
// while the second opened: two cards, one of them stale, which is the ghost
// people were seeing between neighbouring buttons. A card that opens closes
// whatever else was open, in the same frame, so there is never a second one.
//
// This is a module of its own rather than state inside ParamLabel because that
// file must only export a component for React Fast Refresh: a module-level
// function export made the rule complain, and editing the file reset the shared
// count. The state belongs beside the rule that reads it, not in the component.

let liveCards = 0;
let closeOpen: (() => void) | null = null;

/** Is any hover card open anywhere on the page? */
export function hoverCardIsOpen(): boolean {
  return liveCards > 0;
}

/** A card opened (true) or unmounted while open (false). */
export function noteCardOpenState(open: boolean): void {
  liveCards = Math.max(0, liveCards + (open ? 1 : -1));
}

/**
 * Claim the one-card slot, closing whichever card held it.
 *
 * Called before the new card opens, so the previous one is already gone in the
 * frame the pointer enters the next control.
 */
export function claimOpenCard(close: () => void): void {
  if (closeOpen && closeOpen !== close) closeOpen();
  closeOpen = close;
}

/** Give the slot back if this card still holds it. */
export function releaseOpenCard(close: () => void): void {
  if (closeOpen === close) closeOpen = null;
}
