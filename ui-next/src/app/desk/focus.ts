// Where focus goes when a step takes away what had it.

/** After a removal, focus moves to the neighbouring tile instead of falling to the page (A-18). */
export function nextFocusAfter(id: string): string {
  const tiles = [...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')].map(t => t.dataset.m!).filter(m => m !== 'lyrics');
  const i = tiles.indexOf(id), n = tiles[i + 1] ?? tiles[i - 1];
  return n ? `.tile[data-m="${n}"] .grip` : '#tiers [aria-checked="true"]';
}
