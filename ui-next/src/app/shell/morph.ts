// The tier change's motion, drawn by hand (docs/09, "Motion"; TC-2). It was a View Transition, which froze the page
// from the click until the new tier was drawn, laid out and captured (150-250 ms on a full desk, and a stall of the GPU
// rasterising the captures on the first change). Here the old boxes are read, the change is drawn, the new boxes are
// read, and the difference is played with the Web Animations API, on the page's own elements: nothing is captured, the
// page answers the click at once, and the motion runs while the rest of the page is live.
//
// The choreography is the one the View Transition had (the concept's html.tier-morph, 35-motion.css), by the same
// classes (shell/tier.ts PARTS):
// - frame: a tile, the prompt card, the lyric sheet. The element stands at its new box; it is moved back to its old
//   place (transform) and clipped to the frame of the moment (clip-path, from its old size to its new, so what is on it
//   is never scaled and stays pinned to the frame's top-left); its surface, line, radius and shadow are drawn for the
//   moment by its ::before ([data-morph], 35-motion.css), sized with the frame, under its content. The old face (every
//   child but the header) is gone in the first 60 ms, before the change is drawn (`out`); the new comes in from 60 to
//   190 ms. `keep` (the prompt card, the lyric sheet, the lead tile): the new face is shown at once.
// - gone / fresh: a tile only one tier has fades where it stands (out in 100 ms; in over 160 ms after 80).
// - head, slide: moved from their old place to the new at their own size (`end`: by their right edge); only: the same,
//   its old fading in 60 ms, its new coming in over 120 ms after 160; swap: moved and clipped as a frame, without a
//   surface of its own, old out in 60 ms, new in from 60 to 190 ms; band: its surface (::before) goes from the old box
//   to the new, under contents that are parts of their own; still: nothing moves.
// A part nested in another is moved by what its own place changed less what its parent's did (one easing, one
// duration: the two add up to the part's own path). A part out of view before and after takes no part.

import { easing, mtime } from '../kit/motion.ts';

export interface Part { name: string; el: HTMLElement; cls: string }
interface Took { r: DOMRect; cls: string }

const has = (cls: string, k: string) => cls.split(' ').includes(k);
const isFrame = (cls: string) => has(cls, 'frame');
/** What of a frame is its face (all but its header, which is a part of its own); of any other part, the part. */
const faces = (p: Part): HTMLElement[] => (isFrame(p.cls) ? [...p.el.children].filter((c): c is HTMLElement => c instanceof HTMLElement && !c.matches('.tile-head, .head')) : [p.el]);
const near = (r: DOMRect) => r.height > 0 && r.bottom > -innerHeight / 3 && r.top < innerHeight * 4 / 3;

let took = new Map<string, Took>(), outs: Animation[] = [], runs: Animation[] = [], marked: HTMLElement[] = [], placed: HTMLElement[] = [];
const fade = (el: HTMLElement, from: number, to: number, ms: number, delay: number, ease: string, into: Animation[]) => {
  into.push(el.animate([{ opacity: from }, { opacity: to }], { duration: ms, delay, easing: easing(ease), fill: 'both' }));
};

/** Before the change: where every part stands, and the old faces set fading (they are gone before the new tier is drawn
 *  in most changes; what is left of them goes with the old tier). */
export function morphOut(parts: Part[]): void {
  const rs = parts.map(p => p.el.getBoundingClientRect());
  took = new Map(parts.map((p, i) => [p.name, { r: rs[i]!, cls: p.cls }]));
  parts.forEach((p, i) => {
    if (!near(rs[i]!) || has(p.cls, 'keep')) return;
    if (has(p.cls, 'gone')) fade(p.el, 1, 0, 100, 0, '--e-in', outs);
    else if (isFrame(p.cls) || has(p.cls, 'swap') || has(p.cls, 'only')) faces(p).forEach(f => fade(f, 1, 0, 60, 0, '--e-out', outs));
  });
}

/** After the change is drawn and laid out: every part moves from where it stood to where it stands. Resolves when the
 *  motion is over (or another change cut it short). */
export function morphIn(parts: Part[]): Promise<void> {
  const rs = parts.map(p => p.el.getBoundingClientRect());
  const looks = parts.map(p => (isFrame(p.cls) || has(p.cls, 'band') ? lookOf(p.el) : null));
  outs.forEach(a => a.cancel()); outs = [];
  shown = [];
  const T = mtime('--t-move'), E = easing('--e-out'), moved = new Map<HTMLElement, [number, number]>();
  const move = (el: HTMLElement, dx: number, dy: number) => {
    let a = el.parentElement; while (a && !moved.has(a)) a = a.parentElement;
    const [px, py] = a ? moved.get(a)! : [0, 0]; moved.set(el, [dx, dy]);
    if (Math.abs(dx - px) < 0.5 && Math.abs(dy - py) < 0.5) return;
    runs.push(el.animate([{ transform: `translate(${dx - px}px, ${dy - py}px)` }, { transform: 'none' }], { duration: T, easing: E, fill: 'both' }));
  };
  const clip = (el: HTMLElement, o: DOMRect, r: DOMRect, round: string) => runs.push(el.animate(
    [{ clipPath: `inset(0px ${r.width - o.width}px ${r.height - o.height}px 0px${round})` }, { clipPath: `inset(0px 0px 0px 0px${round})` }], { duration: T, easing: E, fill: 'both' }));
  parts.forEach((p, i) => {
    const r = rs[i]!, o = took.get(p.name)?.r, c = p.cls;
    if (has(c, 'still') || !(near(r) || (o && near(o)))) return;
    shown.push(p);
    if (!o || has(c, 'fresh')) {
      /* only in the new tier: comes in where it stands */
      if (isFrame(c) || has(c, 'head') || has(c, 'slide')) fade(p.el, 0, 1, 160, 80, '--e-out', runs);
      else if (has(c, 'only')) fade(p.el, 0, 1, 120, 160, '--e-out', runs);
      else if (has(c, 'swap')) fade(p.el, 0, 1, 130, 60, '--e-out', runs);
      return;
    }
    if (has(c, 'band')) { surface(p.el, looks[i]!, o, r, true, T, E); return; }
    move(p.el, has(c, 'end') ? o.right - r.right : o.left - r.left, o.top - r.top);
    if (isFrame(c)) { surface(p.el, looks[i]!, o, r, false, T, E); clip(p.el, o, r, looks[i]!.round); }
    else if (has(c, 'swap')) clip(p.el, o, r, '');
    if (has(c, 'keep')) return;
    if (isFrame(c) || has(c, 'swap')) faces(p).forEach(f => fade(f, 0, 1, 130, 60, '--e-out', runs));
    else if (has(c, 'only')) fade(p.el, 0, 1, 120, 160, '--e-out', runs);
  });
  const mine = runs;
  return Promise.all(mine.map(a => a.finished.catch(() => {}))).then(() => { if (mine === runs) morphEnd(); });
}

/** Ends the motion where it is going: every animation dropped, every surface taken off (the next change, or the end). */
export function morphEnd(): void {
  outs.forEach(a => a.cancel()); runs.forEach(a => a.cancel()); outs = []; runs = [];
  marked.forEach(el => el.removeAttribute('data-morph')); placed.forEach(el => el.style.removeProperty('position'));
  marked = []; placed = []; took = new Map(); shown = [];
}

/* the parts moving, for the test layer (tools/motion/frames.mjs reads each one's box and faces step by step) */
let shown: Part[] = [];
if (import.meta.env.MODE === 'test') (globalThis as { __morph?: unknown }).__morph = () => shown.map(p => ({ name: p.name, el: p.el, cls: p.cls, faces: faces(p), surface: p.el.hasAttribute('data-morph') }));

/* a frame's or a band's own surface, read from its style as the tier draws it: colour, line (each side's border, drawn
   inside), radius, shadow */
interface Look { bg: string; shadow: string; radius: string; round: string; bl: number; bt: number; placed: boolean }
function lookOf(el: HTMLElement): Look {
  const s = getComputedStyle(el);
  const sides: [string, string, (w: string) => string][] = [[s.borderTopWidth, s.borderTopColor, w => `0 ${w}`], [s.borderBottomWidth, s.borderBottomColor, w => `0 -${w}`], [s.borderLeftWidth, s.borderLeftColor, w => `${w} 0`], [s.borderRightWidth, s.borderRightColor, w => `-${w} 0`]];
  const lines = sides.filter(([w]) => w !== '0px').map(([w, c, at]) => `inset ${at(w)} 0 0 ${c}`);
  const shadow = [...lines, ...(s.boxShadow === 'none' ? [] : [s.boxShadow])].join(', ') || 'none';
  const radius = s.borderRadius, round = radius && radius !== '0px' ? ` round ${radius}` : '';
  return { bg: s.backgroundColor, shadow, radius, round, bl: el.clientLeft, bt: el.clientTop, placed: s.position !== 'static' };
}
/** The element's surface drawn by its ::before for the moment (its own taken off), going from the old box to the new:
 *  a frame's moves with its element (moved back to its old place), a band's moves itself (the element stays). */
function surface(el: HTMLElement, k: Look, o: DOMRect, r: DOMRect, own: boolean, T: number, E: string): void {
  el.setAttribute('data-morph', ''); marked.push(el); if (!k.placed) { el.style.position = 'relative'; placed.push(el); }
  const box = (x: number, y: number, w: number, h: number) => ({ left: `${x - k.bl}px`, top: `${y - k.bt}px`, width: `${w}px`, height: `${h}px`, background: k.bg, borderRadius: k.radius, boxShadow: k.shadow });
  const [fx, fy] = own ? [o.left - r.left, o.top - r.top] : [0, 0];
  runs.push(el.animate([box(fx, fy, o.width, o.height), box(0, 0, r.width, r.height)], { duration: T, easing: E, fill: 'both', pseudoElement: '::before' }));
}
