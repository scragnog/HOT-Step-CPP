// How an overlay leaves (the concept's ovOut, 16-overlays.js, M-03): each overlay enters with a CSS animation on the
// motion tokens (the concept's stylesheets), and closing stays at once, so `el.hidden` always means what it says; what
// the user sees leaving is a copy of the overlay put back where it stood for the length of the exit (about 70% of its
// entrance, on --e-in). The copy is inert, hidden from screen readers and carries no ids, so no id is on the page twice
// while it plays (as a drag's preview, deviation 39). With reduced motion every exit is a short fade. Any overlay may
// use it: call `ovOut(el, kind)` just before hiding `el`. Drawing only.

import { motionVar, mtime, reduced } from '../parts/motion.ts';

export type ExitKind = 'drop' | 'drawer' | 'fade' | 'toast' | 'sheet' | 'modal';

const KF: { [k in ExitKind]: (up: boolean) => Keyframe[] } = {
  drop: up => [{ opacity: 1, translate: '0 0', scale: '1' }, { opacity: 0, translate: up ? '0 4px' : '0 -4px', scale: '.98' }],
  drawer: () => [{ opacity: 1, translate: '0 0' }, { opacity: 0, translate: '24px 0' }],
  fade: () => [{ opacity: 1 }, { opacity: 0 }],
  toast: () => [{ opacity: 1, translate: '0 0' }, { opacity: 0, translate: '0 4px' }],
  sheet: () => [{ opacity: 1, translate: '0 0' }, { opacity: 0, translate: '0 16px' }],
  modal: () => [{ opacity: 1 }, { opacity: 0 }],
};

/** Plays `el`'s exit from where it stands; call it just before `el` is hidden. */
export function ovOut(el: HTMLElement | null, kind: ExitKind = 'drop'): void {
  if (!el || el.hidden || !el.isConnected) return;
  const r = el.getBoundingClientRect(); if (!r.width || !r.height) return;
  const g = el.cloneNode(true) as HTMLElement, st = el.scrollTop;
  g.classList.add('ov-ghost'); g.setAttribute('aria-hidden', 'true'); g.inert = true; g.removeAttribute('role');
  for (const x of [g, ...g.querySelectorAll('[id]')]) x.removeAttribute('id');
  /* fixed and absolute overlays keep their place; the copy goes right after the original */
  el.after(g); g.scrollTop = st;
  const calm = reduced(), dur = calm ? 120 : mtime(kind === 'drawer' ? '--t-base' : '--t-fast');
  const kf = calm ? KF.fade(false) : KF[kind](el.classList.contains('up'));
  g.animate(kf, { duration: dur, easing: motionVar('--e-in'), fill: 'forwards' }).finished.then(() => g.remove(), () => g.remove());
}
