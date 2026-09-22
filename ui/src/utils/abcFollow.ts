// Keep the playing note visible inside the score's own scroll box.
//
// scrollIntoView() scrolls EVERY scrollable ancestor, the page included, so a
// playing lead sheet yanked the browser back to it on every note and the user
// could not scroll anywhere else. This moves only `box`, and only when the
// note has left it.
export function followNoteInBox(note: Element | undefined, box: HTMLElement | null): void {
  if (!note || !box) return;
  const n = note.getBoundingClientRect();
  const b = box.getBoundingClientRect();
  const margin = 24;
  if (n.top < b.top + margin) box.scrollTop -= (b.top + margin) - n.top;
  else if (n.bottom > b.bottom - margin) box.scrollTop += n.bottom - (b.bottom - margin);
}
