// A tier change from the view (the concept's setTier, deskAnchor and ringCtls, 31-shell.js, 11-render.js): the desk
// shows more or less of itself, the place the user was reading stays where it was on screen (Q-14, RC-11), and the
// controls a link named are rung on arrival.
//
// Every part that exists in both tiers is one object throughout (Q-04, Q-13); the motion is drawn by hand (morph.ts):
// - a tile, the prompt card and the lyric sheet are frames: the frame grows or shrinks from its old box to its new one,
//   and what is on it is never scaled: the old face goes, the new one, at its own size and pinned to the frame's
//   top-left, comes in, clipped by the frame. A tile's header is the same in both tiers and does not fade; its remove
//   button slides to its new corner;
// - what only one tier has (a Lab module, Quick's heading, the hidden-modules note) fades out or in where it stands;
// - the bars never fade: their contents slide where the column puts them, and they stay above the tiles moving under
//   them; the tier switch does not take part (its thumb has its own slide);
// - the rest of the page is shown in its new state at once.
// The choreography is the concept's (35-motion.css, html.tier-morph; morph.ts); this names the parts and gives each its
// class. The click is answered at once: the switch shows the chosen tier and the old faces start to go in the click's
// own frame; the new tier is drawn in the next task, and the desk laid out there, then everything moves from where it
// stood. Under reduced motion the desk's own transition runs (features/motion.ts: nothing moves, what arrives fades in).

import { useSyncExternalStore } from 'react';
import { flushSync } from 'react-dom';
import type { Tier } from '../../definitions/types.ts';
import { deskModel } from '../../derive.ts';
import { LEAD, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { deskFold } from '../store/fold.ts';
import { mtime, reduced } from '../kit/motion.ts';
import { deferPack, flushPack } from '../desk/layout.ts';
import { drawHeld, drawInView, hold, leaving, planStage, settle, type Stage } from '../desk/stage.ts';
import { heldTiles } from '../render/tileTier.ts';
import { TALLY, tally } from '../tally.ts';
import { morphEnd, morphIn, morphOut, type Part } from './morph.ts';

const ANCHORS = '#bento > .tile[data-m], .lyr-card';
const anchorKey = (el: Element) => (el as HTMLElement).dataset.m ? `#bento > .tile[data-m="${(el as HTMLElement).dataset.m}"]` : '.lyr-card';
/* where the pointer last was over the page (none once it left the window): the tile under it is the one being read */
let lastPointer: [number, number] | null = null;
addEventListener('pointermove', e => { lastPointer = [e.clientX, e.clientY]; }, { passive: true, capture: true });
document.documentElement.addEventListener('pointerleave', () => { lastPointer = null; }, { passive: true });
/** The place the user is reading, taken before the change: the tile focus is in, else the one under the pointer, else
 *  the one nearest under the bars; `put` puts it back where it was on screen (nothing at the top of the page: there is
 *  nothing to keep), and `reader` says which tile it is and where its top stands now. */
function deskAnchor(): { put: () => void; reader: { key: string; top: number } | null } {
  const none = { put: () => {}, reader: null };
  if (scrollY < 1) return none;
  const line = Math.max(document.querySelector('.top')?.getBoundingClientRect().bottom ?? 0, document.querySelector('.deskbar')?.getBoundingClientRect().bottom ?? 0);
  const under = lastPointer ? document.elementFromPoint(...lastPointer)?.closest(ANCHORS) ?? null : null;
  let el: Element | null = document.activeElement?.closest?.(ANCHORS) ?? under, best = Infinity;
  if (!el) document.querySelectorAll(ANCHORS).forEach(t => { const r = t.getBoundingClientRect(); if (!r.height || r.bottom <= line + 8 || r.top >= innerHeight) return; const d = Math.abs(r.top - line); if (d < best) { best = d; el = t; } });
  if (!el) return none;
  const k = anchorKey(el), top = (el as Element).getBoundingClientRect().top, id = (el as HTMLElement).dataset.m;
  return { put: () => { const n = document.querySelector(k); if (!n) return; const d = n.getBoundingClientRect().top - top; if (Math.abs(d) > 0.5) scrollTo(scrollX, scrollY + d); }, reader: id ? { key: id, top } : null };
}

/* the tiles only one of the two tiers shows, which side is being named, and the tiles left out (held by a staged change:
   out of view, they take no part) */
let only = new Set<string>(), side: 'old' | 'new' = 'old', left = new Set<string>();
const tileIdOf = (el: Element) => el.closest<HTMLElement>('#bento > .tile[data-m]')?.dataset.m ?? '';
const frame = (el: HTMLElement) => 'frame' + (el.dataset.m === LEAD?.id ? ' keep' : '') + (el.classList.contains('source') ? ' source' : '') + (only.has(el.dataset.m!) ? (side === 'old' ? ' gone' : ' fresh') : '');
const m = (el: Element) => (el as HTMLElement).dataset.m ?? '';
const tileOf = (el: Element) => m(el.closest('.tile')!);
/** [selector, the name of each match, its class] (the concept's VT_PARTS, the same names and classes) */
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
/** Every part on the page now, with its name and class; a name is taken once, and a tile left out has none. */
function collect(): Part[] {
  const used = new Set<string>(), out: Part[] = [];
  for (const [sel, name, cls] of PARTS) for (const el of document.querySelectorAll<HTMLElement>(sel)) {
    const n = left.has(tileIdOf(el)) ? '' : name(el); if (!n || used.has(n)) continue;
    used.add(n); out.push({ name: n, el, cls: typeof cls === 'function' ? cls(el) : cls });
  }
  return out;
}
/** Which tiles only one of the desk now and the desk at `t` shows. */
function onlyOne(t: Tier): Set<string> {
  const s = useDesk.getState(), now = new Set([...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')].map(m)), next = new Set(deskModel(ix, { ...s.desk, tier: t }, s.mods, deskFold.get()).tiles);
  return new Set([...now, ...next].filter(x => now.has(x) !== next.has(x)));
}

/* the tier the switch shows from the click on, while the desk is still to be drawn at it */
let going: Tier | null = null;
const goers = new Set<() => void>();
const setGoing = (t: Tier | null) => { if (going === t) return; going = t; goers.forEach(f => f()); };
/** The tier the switch shows: the one chosen, from the moment it is chosen. */
export const useShownTier = (tier: Tier): Tier => useSyncExternalStore(f => { goers.add(f); return () => goers.delete(f); }, () => going) ?? tier;

let seq = 0, timer = 0, frameId = 0, inDraw = false;
/* the place the user is reading, taken as the change is chosen (before anything about it moves) */
let anchorAt: ReturnType<typeof deskAnchor> | null = null;
/* what the change's own drawing (before anything moves) counted: the views' models asked, and the tiles measured (the
   test layer's counters, tools/behaviour/scenarios/40-tier.mjs; nothing in other builds) */
const SYNC = ['model', 'model.changed', 'measure.tiles'];
const counts = () => SYNC.map(k => TALLY[k] ?? 0);
const counted = (was: number[]) => SYNC.forEach((k, i) => tally('tier.sync.' + k, (TALLY[k] ?? 0) - was[i]!));
const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), select, textarea, [tabindex="0"]';
const ringFor = () => (reduced() ? 1200 : mtime('--t-slow') * 4);
const focusTier = (t: Tier) => document.querySelector<HTMLElement>(`#tiers [data-tier="${t}"]`)?.focus({ preventScroll: true });
/** A change chosen but not drawn yet is called off, and a motion that runs ends where it was going. */
function callOff(): void {
  cancelAnimationFrame(frameId); clearTimeout(timer); frameId = timer = 0; seq++; anchorAt = null;
  morphEnd(); document.documentElement.classList.remove('tier-morph');
}

type Opts = { ring?: string[]; focus?: boolean; reveal?: boolean; instant?: boolean };
/** Changes the tier. `focus`: focus the chosen tier's button (Q-16); `ring`: ring the controls a link named, focus going
 *  to the first (Q-28); `reveal`: ring the tiles the new tier brings, the first brought into view (Q-15); `instant`: the
 *  tier is in force when this returns, for what the caller does next in it (Lab's offer adds its module there), and the
 *  desk's own transition carries the change. A change chosen while another is still to be drawn replaces it; one chosen
 *  while another moves ends that one where it was going (Q-21). */
export function setTierFromView(t: Tier, o: Opts = {}): void {
  const s = useDesk.getState(), drawing = !!(frameId || timer);
  if (drawing && t === s.desk.tier) { callOff(); flushSync(() => setGoing(null)); if (o.focus) focusTier(t); return; }
  if ((going ?? s.desk.tier) === t) { if (o.focus) focusTier(t); return; }
  if (o.instant || reduced() || document.hidden) { callOff(); drawNow(t, o); return; }
  const root = document.documentElement;
  if (!drawing) { morphEnd(); anchorAt = deskAnchor(); only = onlyOne(t); side = 'old'; left = new Set(); morphOut(collect()); root.classList.add('tier-morph'); }
  /* the switch shows the choice in the click's own frame; the new tier is drawn once that frame is on screen */
  flushSync(() => setGoing(t)); if (o.focus) focusTier(t);
  const me = ++seq;
  cancelAnimationFrame(frameId); clearTimeout(timer);
  frameId = requestAnimationFrame(() => { frameId = 0; timer = window.setTimeout(() => { timer = 0; if (me === seq) drawMoving(t, o, me); }); });
}

/** The desk drawn at `t` and laid out, the reader's place put back before the page is painted (the desk laid out once,
 *  after the parts' own effects have run); a staged change holds the tiles out of view (desk/stage.ts). */
function drawAt(t: Tier, st: Stage | null, keep: () => void): void {
  if (!st) settle();
  deferPack(() => flushSync(() => { if (st) hold(st); setGoing(null); useDesk.getState().setTier(t); }));
  flushPack(); keep();
}
/** What the change does once the tier is drawn: focus, the tiles it brought rung, the controls a link named rung. */
function after(t: Tier, o: Opts, was: Set<string>): void {
  if (o.focus) focusTier(t);
  if (o.reveal) { const st = useDesk.getState(); deskModel(ix, st.desk, st.mods, deskFold.get()).tiles.filter(id => !was.has(id)).forEach((id, i) => ringTile(id, i > 0)); }
  if (o.ring?.length) ringCtls(o.ring);
}
const tilesNow = () => { const s = useDesk.getState(); return new Set(deskModel(ix, s.desk, s.mods, deskFold.get()).tiles); };

function drawMoving(t: Tier, o: Opts, me: number): void {
  leaving();
  const was = tilesNow(), anchor = anchorAt ?? deskAnchor(), c = counts();
  anchorAt = null;
  only = onlyOne(t);
  /* staged only when nothing is to be rung in the tiles */
  const st = !o.ring && !o.reveal ? planStage(t, anchor.reader) : null;
  inDraw = true; drawAt(t, st, anchor.put); inDraw = false; if (st) drawInView();
  left = new Set(heldTiles()); side = 'new';
  const parts = collect(); after(t, o, was); counted(c);
  morphIn(parts).then(() => { if (me === seq) document.documentElement.classList.remove('tier-morph'); });
  /* once the motion has started, the tiles a staged change held are drawn */
  requestAnimationFrame(() => { anchor.put(); drawHeld(); });
}

function drawNow(t: Tier, o: Opts): void {
  leaving();
  const was = tilesNow(), anchor = deskAnchor();
  inDraw = true; drawAt(t, null, anchor.put); inDraw = false;
  /* again once the tiles' own sizes have settled the layout (a re-pack on the next frame) */
  requestAnimationFrame(() => { anchor.put(); after(t, o, was); });
}

/* the desk's make-up changed some other way while a change was chosen or moving (the tier set directly, a module added,
   the engine switched): that change is called off, its motion ended where it was going */
useDesk.subscribe((s, p) => {
  if (inDraw || !(frameId || timer || document.documentElement.classList.contains('tier-morph'))) return;
  if (s.desk.tier !== p.desk.tier || s.mods !== p.mods || s.board !== p.board || s.desk.engine !== p.desk.engine) { callOff(); setGoing(null); }
});

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
