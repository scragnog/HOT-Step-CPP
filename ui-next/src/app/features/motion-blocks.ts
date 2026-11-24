// What pushes the content under it moves it smoothly (the concept's grow and blockIn/blockOut, 19-text-boxes.js and
// 18-motion.js, F-09, F-28): the prompt box growing or shrinking by more than a line and a half, a suggestion under it,
// Song info and the save row opening and closing. The concept animates their height, which lays the page out on every
// frame; here the page is laid out once, at its new size, and what follows the block slides from where it stood
// (transform), while the frame that holds the block is revealed down to its new edge (clip-path) and the block fades
// in. A block that closes leaves its copy where it stood for the moment it fades (--t-fast, --e-in). Under reduced
// motion nothing moves: a block fades in or out over 120ms and the box simply changes. Each element is watched with a
// ResizeObserver, so the change is seen after layout and before the page is painted; no component is drawn again.

import { booted, easing, mtime, reduced } from '../kit/motion.ts';

const BOX = '#prompt', BLOCKS = '#suggest, #songInfo, #saveRow';
/** The card a block stands in: revealed down to its new edge while what follows slides (a side panel runs the window's
 *  height, so its foot is not an edge that moves: nothing is clipped there). */
const FRAME = '.prompt-card';
const last = new WeakMap<Element, { h: number; copy: HTMLElement | null; r: DOMRect | null }>();

/** What follows `el` down the page, up to the centre column or the side panel: later siblings at each level, those in
 *  the flow (a bar that sticks or floats is not carried). */
function followers(el: Element): HTMLElement[] {
  const out: HTMLElement[] = [];
  for (let at: Element | null = el; at && !at.matches('#centre, .side, body'); at = at.parentElement)
    for (let n = at.nextElementSibling; n; n = n.nextElementSibling) {
      if (!(n instanceof HTMLElement) || !n.getClientRects().length) continue;
      const pos = getComputedStyle(n).position; if (pos === 'sticky' || pos === 'fixed' || pos === 'absolute') continue;
      out.push(n);
    }
  return out;
}
/** What follows slides from `dy` above where it stands now, and the frame shows down to its new edge as it goes. */
function carry(el: HTMLElement, dy: number, time: string, ease: string): void {
  const o = { duration: mtime(time), easing: easing(ease) };
  followers(el).forEach(n => n.animate([{ transform: `translateY(${-dy}px)` }, { transform: 'none' }], o));
  const f = el.closest<HTMLElement>(FRAME);
  if (f && dy > 0) f.animate([{ clipPath: `inset(-40px -40px ${dy}px -40px)` }, { clipPath: 'inset(-40px -40px -40px -40px)' }], o);
}

function blockIn(el: HTMLElement, h: number): void {
  if (reduced()) { el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 120 }); return; }
  el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: mtime('--t-base'), easing: easing('--e-out') });
  carry(el, h, '--t-base', '--e-out');
}
function blockOut(el: HTMLElement, copy: HTMLElement, r: DOMRect): void {
  /* the copy stands where the block stood, out of the flow, over what slides up into its place */
  copy.style.cssText += ';position:absolute;margin:0;left:0;top:0;pointer-events:none';
  el.after(copy);
  const c = copy.getBoundingClientRect(); copy.style.left = `${r.left - c.left}px`; copy.style.top = `${r.top - c.top}px`; copy.style.width = `${r.width}px`;
  const calm = reduced();
  copy.animate([{ opacity: 1 }, { opacity: 0 }], { duration: calm ? 120 : mtime('--t-fast'), easing: easing('--e-in'), fill: 'forwards' }).finished.then(() => copy.remove(), () => copy.remove());
  if (!calm) carry(el, -r.height, '--t-fast', '--e-in');
}
/** A copy of a block as it stands, to fade where it stood once it closes: inert, hidden, without ids. */
function copyOf(el: HTMLElement): HTMLElement {
  const c = el.cloneNode(true) as HTMLElement;
  [c, ...c.querySelectorAll<HTMLElement>('[id]')].forEach(x => x.removeAttribute('id'));
  c.removeAttribute('hidden'); c.inert = true; c.setAttribute('aria-hidden', 'true'); c.classList.add('block-out');
  return c;
}

const ro = new ResizeObserver(es => es.forEach(e => {
  const el = e.target as HTMLElement, h = el.offsetHeight, was = last.get(el), block = el.matches(BLOCKS);
  last.set(el, { h, copy: block && h ? copyOf(el) : was?.copy ?? null, r: h ? el.getBoundingClientRect() : was?.r ?? null });
  if (!was || !booted() || document.documentElement.classList.contains('tier-morph') || !el.isConnected) return;
  if (!block) {
    const line = +getComputedStyle(el).lineHeight.replace('px', '') || 20;
    if (!reduced() && was.h && Math.abs(h - was.h) > line * 1.5) carry(el, h - was.h, '--t-base', '--e-out');
    return;
  }
  if (!was.h && h) blockIn(el, h);
  else if (was.h && !h && was.copy && was.r) blockOut(el, was.copy, was.r);
}));
const seen = new WeakSet<Element>();
/** The box and the blocks are drawn by their parts and kept; they are found as they appear (looked for in what was
 *  added, not in the whole page again). */
const SEL = `${BOX}, ${BLOCKS}`;
const take = (el: Element) => { if (!seen.has(el)) { seen.add(el); ro.observe(el); } };
new MutationObserver(rs => { for (const r of rs) for (const n of r.addedNodes) if (n instanceof Element) { if (n.matches(SEL)) take(n); n.querySelectorAll(SEL).forEach(take); } }).observe(document.documentElement, { childList: true, subtree: true });
document.querySelectorAll(SEL).forEach(take);
