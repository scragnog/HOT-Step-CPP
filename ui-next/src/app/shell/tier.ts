// A tier change from the view (the concept's setTier, deskAnchor and ringCtls, 31-shell.js, 11-render.js): the desk
// shows more or less of itself, the place the user was reading stays where it was on screen (Q-14, RC-11), and the
// controls a link named are rung on arrival.
//
// It is one View Transition in which every part that exists in both tiers is one object throughout (Q-04, Q-13):
// - a tile, the prompt card and the lyric sheet are frames: the frame grows or shrinks from its old box to its new one,
//   drawn by the transition itself, and what is on it is never scaled: the old face, at its own size and pinned to the
//   frame's top-left, hands over to the new one, clipped by the frame. A tile's header is the same in both tiers and
//   does not fade; its remove button slides to its new corner;
// - what only one tier has (a Lab module, Quick's heading, the hidden-modules note) fades out or in where it stands;
// - the bars never fade: their contents slide where the column puts them, and they stay above the tiles moving under
//   them; the tier switch does not take part (its thumb has its own slide);
// - the rest of the page is shown in its new state at once.
// The choreography is the concept's stylesheet (35-motion.css, html.tier-morph); this names the parts and gives each
// its class. React draws the new tier inside the transition's update (flushSync), and the desk is laid out there, so
// what the transition captures is the page as it will stay. Under reduced motion, or without View Transitions, the
// desk's own transition runs (features/motion.ts: nothing moves, what arrives fades in).

import { flushSync } from 'react-dom';
import type { Tier } from '../../definitions/types.ts';
import { deskModel } from '../../derive.ts';
import { LEAD, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { deskFold } from '../store/fold.ts';
import { mtime, reduced } from '../kit/motion.ts';
import { deferPack, flushPack } from '../desk/layout.ts';

const ANCHORS = '#bento > .tile[data-m], .lyr-card';
const anchorKey = (el: Element) => (el as HTMLElement).dataset.m ? `#bento > .tile[data-m="${(el as HTMLElement).dataset.m}"]` : '.lyr-card';
/* where the pointer last was over the page (none once it left the window): the tile under it is the one being read */
let lastPointer: [number, number] | null = null;
addEventListener('pointermove', e => { lastPointer = [e.clientX, e.clientY]; }, { passive: true, capture: true });
document.documentElement.addEventListener('pointerleave', () => { lastPointer = null; }, { passive: true });
/** The place the user is reading, taken before the change: the tile focus is in, else the one under the pointer, else
 *  the one nearest under the bars; the function puts it back where it was on screen (nothing at the top of the page:
 *  there is nothing to keep). */
function deskAnchor(): () => void {
  if (scrollY < 1) return () => {};
  const line = Math.max(document.querySelector('.top')?.getBoundingClientRect().bottom ?? 0, document.querySelector('.deskbar')?.getBoundingClientRect().bottom ?? 0);
  const under = lastPointer ? document.elementFromPoint(...lastPointer)?.closest(ANCHORS) ?? null : null;
  let el: Element | null = document.activeElement?.closest?.(ANCHORS) ?? under, best = Infinity;
  if (!el) document.querySelectorAll(ANCHORS).forEach(t => { const r = t.getBoundingClientRect(); if (!r.height || r.bottom <= line + 8 || r.top >= innerHeight) return; const d = Math.abs(r.top - line); if (d < best) { best = d; el = t; } });
  if (!el) return () => {};
  const k = anchorKey(el), top = (el as Element).getBoundingClientRect().top;
  return () => { const n = document.querySelector(k); if (!n) return; const d = n.getBoundingClientRect().top - top; if (Math.abs(d) > 0.5) scrollTo(scrollX, scrollY + d); };
}

type VT = { finished: Promise<void>; ready: Promise<void>; updateCallbackDone: Promise<void> };
const doc = document as Document & { startViewTransition?: (f: () => void) => VT };
/** View Transitions with their classes: the tier change is choreographed only where both are there. */
const VT_OK = !!doc.startViewTransition && !!window.CSS?.supports?.('view-transition-class: a');

/* the tiles only one of the two tiers shows, and whether the old state is being named */
let only = new Set<string>(), naming: 'old' | 'new' = 'old';
const frame = (el: HTMLElement) => 'frame' + (el.dataset.m === LEAD?.id ? ' keep' : '') + (el.classList.contains('source') ? ' source' : '') + (only.has(el.dataset.m!) ? (naming === 'old' ? ' gone' : ' fresh') : '');
const m = (el: Element) => (el as HTMLElement).dataset.m ?? '';
const tileOf = (el: Element) => m(el.closest('.tile')!);
/** [selector, the name of each match, its view-transition-class] (the concept's VT_PARTS, the same names and classes) */
const PARTS: [string, (el: HTMLElement) => string, string | ((el: HTMLElement) => string)][] = [
  ['.top', () => 'vt-top', 'still'],
  ['.deskbar', () => 'vt-bar', 'band'],
  ['.deskbar > .panel-toggles', () => 'vt-toggles', 'slide'],
  ['#tiers', () => 'vt-tiers', 'still'],
  ['.deskbar-end', () => 'vt-deskend', 'slide end'],
  ['#ask', () => 'vt-ask', 'only'],
  ['.prompt-card', () => 'vt-prompt', 'frame keep prompt'],
  ['.lyr-card', () => 'vt-lyrics', 'frame keep'],
  ['.lyr-card > .head', () => 'vt-h-lyrics', 'head'],
  ['#hiddenNote', () => 'vt-hnote', 'swap'],
  ['#bento > .tile[data-m]', el => 'vt-' + m(el), frame],
  ['#bento > .tile[data-m] > .tile-head', el => 'vt-h-' + m(el.parentElement!), 'head'],
  ['#bento > .tile[data-m] > .tile-head > .x', el => 'vt-x-' + tileOf(el), 'slide'],
  ['#bento > .tile[data-m] > .tile-head > .lbl', el => 'vt-l-' + tileOf(el), 'only'],
  ['#bento > :is(.desk-empty, .q-empty)', () => 'vt-empty', 'swap'],
  ['#bento > :not(.tile, .desk-empty, .q-empty)', el => 'vt-b-' + (el.classList[0] || 'x'), 'swap'],
  ['.gobar', () => 'vt-gobar', 'band'],
  ['.gobar > [id]', el => 'vt-g-' + el.id, el => (el.id === 'meter' ? 'swap' : 'slide')],
  ['.latest', () => 'vt-latest', 'swap'],
];
/** Names every part for the transition (on), or takes every name off. A part not drawn has no name; a name is given
 *  once. Whether each is drawn is read before any is named: a name restyles its part, and a read after it would have
 *  the page restyled again, once a part. */
function nameForMorph(on: boolean): void {
  const used = new Set<string>();
  const parts = PARTS.flatMap(([sel, name, cls]) => [...document.querySelectorAll<HTMLElement>(sel)].map(el => ({ el, name, cls, drawn: on && el.getClientRects().length > 0 })));
  for (const { el, name, cls, drawn } of parts) {
    const n = drawn ? name(el) : '';
    if (!n || used.has(n)) { el.style.removeProperty('view-transition-name'); el.style.removeProperty('view-transition-class'); continue; }
    used.add(n); el.style.setProperty('view-transition-name', n); el.style.setProperty('view-transition-class', typeof cls === 'function' ? cls(el) : cls);
  }
}

let seq = 0;
const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), select, textarea, [tabindex="0"]';
const ringFor = () => (reduced() ? 1200 : mtime('--t-slow') * 4);

/** Changes the tier. `focus`: focus the chosen tier's button (Q-16); `ring`: ring the controls a link named, focus going
 *  to the first (Q-28); `reveal`: ring the tiles the new tier brings, the first brought into view (Q-15); `instant`: the
 *  tier is in force when this returns, for what the caller does next in it (Lab's offer adds its module there), so the
 *  desk's own transition carries the change instead of the View Transition, whose update waits for the next frame. */
export function setTierFromView(t: Tier, o: { ring?: string[]; focus?: boolean; reveal?: boolean; instant?: boolean } = {}): void {
  const s = useDesk.getState();
  if (s.desk.tier === t) { if (o.focus) document.querySelector<HTMLElement>(`#tiers [data-tier="${t}"]`)?.focus({ preventScroll: true }); return; }
  const was = new Set(deskModel(ix, s.desk, s.mods, deskFold.get()).tiles), keep = deskAnchor();
  /* drawn and laid out now, so the anchor is put back before the page is painted; the desk is laid out once, after the
     parts' own effects have run */
  const go = () => { deferPack(() => flushSync(() => useDesk.getState().setTier(t))); flushPack(); keep(); };
  const after = () => {
    if (o.focus) document.querySelector<HTMLElement>(`#tiers [data-tier="${t}"]`)?.focus({ preventScroll: true });
    if (o.reveal) { const st = useDesk.getState(); deskModel(ix, st.desk, st.mods, deskFold.get()).tiles.filter(id => !was.has(id)).forEach((id, i) => ringTile(id, i > 0)); }
    if (o.ring?.length) ringCtls(o.ring);
  };
  if (VT_OK && !o.instant && !reduced() && !document.hidden) {
    const me = ++seq, root = document.documentElement;
    const now = new Set([...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')].map(m)), next = new Set(deskModel(ix, { ...s.desk, tier: t }, s.mods, deskFold.get()).tiles);
    only = new Set([...now, ...next].filter(x => now.has(x) !== next.has(x))); naming = 'old';
    nameForMorph(true); root.classList.add('tier-morph');
    const vt = doc.startViewTransition!(() => { go(); naming = 'new'; nameForMorph(true); after(); });
    /* a change started while another runs skips the first (Q-21): only the last one clears up */
    const clear = () => { if (me !== seq) return; nameForMorph(false); root.classList.remove('tier-morph'); };
    vt.finished.then(clear, clear); vt.ready.catch(() => {}); vt.updateCallbackDone.catch(() => {});
    return;
  }
  go();
  /* again once the tiles' own sizes have settled the layout (a re-pack on the next frame) */
  requestAnimationFrame(() => { keep(); after(); });
}

/** A tile the tier brought: the "it is here" ring; the first is brought into view (the concept's flash). */
function ringTile(id: string, stay: boolean): void {
  requestAnimationFrame(() => {
    const el = document.querySelector<HTMLElement>(`#bento > .tile[data-m="${id}"]`); if (!el) return;
    el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    window.setTimeout(() => el.classList.remove('flash'), reduced() ? 1200 : mtime('--t-slow') * 2);
    if (!stay) el.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
  });
}

/** The controls an "Also set" line named: each rung, the first brought into view; a file's field rings its drop zone
 *  (the fix round's Q-10), and focus goes into it when the link that named them is gone with the tier (Q-28). */
function ringCtls(keys: string[]): void {
  requestAnimationFrame(() => {
    const els = keys.map(k => document.querySelector<HTMLElement>(k.startsWith('[') ? `#bento ${k}` : `#bento [data-ctl="${k}"], #bento [data-bind="${k}"]`)).filter((x): x is HTMLElement => !!x).map(el => el.closest<HTMLElement>('label.drop') ?? el);
    els.forEach(el => { el.classList.remove('ring-here'); void el.offsetWidth; el.classList.add('ring-here'); window.setTimeout(() => el.classList.remove('ring-here'), ringFor()); });
    els[0]?.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
    const a = document.activeElement, lost = !a || a === document.body || !a.isConnected, f = els[0] && (els[0].matches(FOCUSABLE) ? els[0] : els[0].querySelector<HTMLElement>(FOCUSABLE));
    if (lost && f) f.focus({ preventScroll: true });
  });
}
