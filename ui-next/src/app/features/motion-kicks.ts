// The small motions the concept plays on a change of state, each read from the markup React draws, so no part or shell
// carries them (the concept's 18-motion.js, 23-dropdown.js, 31-side-panels.js):
// - an LED switch turned on lights at once and its glow decays (M-14, `.kick`): only when it changes, never on a redraw;
// - a language stamp chosen lands with a thunk (`.thunk`);
// - a side panel opened or folded fades its contents in (--t-slow);
// - Generate's label and the saved state's word cross-fade when their words change (textIn);
// - the decorative loops rest when nothing has happened for 12 seconds and no song renders (`body.calm`), and pick up
//   on any input.
// Opacity, a class or a CSS animation only; nothing here re-renders a component. Drawing only.

import { booted, easing, mtime, reduced } from '../kit/motion.ts';

/** A class that plays its CSS animation once, from the start, and is taken off after `ms`. */
function kick(el: Element, cls: string, ms: number): void {
  el.classList.remove(cls); void (el as HTMLElement).offsetWidth; el.classList.add(cls);
  window.setTimeout(() => el.classList.remove(cls), ms);
}
/** Text that just changed fades in. */
const textIn = (el: Element) => { if (!reduced()) el.animate([{ opacity: 0 }, { opacity: 1 }], { duration: mtime('--t-fast'), easing: easing('--e-out') }); };
const fadeIn = (els: Iterable<Element>) => { for (const x of els) x.animate([{ opacity: 0 }, { opacity: 1 }], { duration: mtime('--t-slow'), easing: easing('--e-out') }); };

/* an LED switch that came on (aria-pressed is written only when it changes) */
new MutationObserver(ms => {
  if (!booted() || reduced()) return;
  for (const m of ms) { const el = m.target as HTMLElement; if (el.matches('[data-ledtog]') && el.getAttribute('aria-pressed') === 'true' && m.oldValue !== 'true') kick(el, 'kick', 400); }
}).observe(document.documentElement, { subtree: true, attributes: true, attributeFilter: ['aria-pressed'], attributeOldValue: true });

/* a side panel opened or folded: what it now shows fades in */
const panels = new MutationObserver(ms => {
  if (!booted() || reduced()) return;
  for (const m of ms) {
    const el = m.target as HTMLElement, was = (m.oldValue ?? '').split(' ').includes('collapsed'), now = el.classList.contains('collapsed');
    if (was !== now) fadeIn(el.querySelectorAll(now ? ':scope > .rail-btn' : ':scope > .side-head, :scope > .side-body, :scope > .tabs, :scope > .save-row'));
  }
});
/* Generate's label and the saved state's word: the bars that hold them are watched, not the page */
const TEXT_IN = '#goLabel, #saveState > span';
const texts = new MutationObserver(ms => {
  if (!booted()) return;
  const seen = new Set<Element>();
  for (const m of ms) { const el = (m.target.nodeType === 1 ? m.target : m.target.parentElement) as Element | null, t = el?.closest(TEXT_IN); if (t) seen.add(t); }
  seen.forEach(textIn);
});
const watched = new WeakSet<Element>();
const attach = () => {
  document.querySelectorAll('#leftPanel, #rightPanel').forEach(p => { if (!watched.has(p)) { watched.add(p); panels.observe(p, { attributes: true, attributeFilter: ['class'], attributeOldValue: true }); } });
  document.querySelectorAll('.gobar, .deskbar').forEach(b => { if (!watched.has(b)) { watched.add(b); texts.observe(b, { subtree: true, characterData: true, childList: true }); } });
};
/* the panels and bars are drawn once and kept; they are found when the page has drawn them, and again on a click (a
   page drawn anew, the Library's stand-in and back) */
const find = () => { if (!booted()) { requestAnimationFrame(find); return; } attach(); };
requestAnimationFrame(find); addEventListener('pointerdown', attach, { passive: true, capture: true });

/* a stamp chosen from its list lands on the paper (the concept's thunk, 23-dropdown.js) */
document.addEventListener('click', e => {
  if (!(e.target as Element).closest?.('#dd .opt-stamp')) return;
  const w = document.querySelector('.stamp-wrap:has(> button[aria-expanded="true"])');
  if (w) requestAnimationFrame(() => kick(w, 'thunk', mtime('--t-slow') + 40));
}, true);

/* the decorative loops rest when nothing has happened for a while, unless a song is rendering */
let calmT = 0;
function stir(): void {
  document.body.classList.remove('calm'); clearTimeout(calmT);
  calmT = window.setTimeout(() => { if (document.body.classList.contains('running')) stir(); else document.body.classList.add('calm'); }, 12000);
}
for (const ev of ['pointermove', 'pointerdown', 'keydown', 'wheel', 'scroll', 'focusin']) addEventListener(ev, stir, { passive: true, capture: true });
stir();
