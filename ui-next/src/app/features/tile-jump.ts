// Moving between modules from the keyboard (the concept's S-33, 33-tile-jump.js): F6 or Alt+Down goes to the next tile,
// Shift+F6 or Alt+Up to the one before, in reading order; focus lands on the tile's grip, so a full Lab desk is a dozen
// presses from end to end instead of hundreds of Tabs. From outside the desk the first press enters it. Not while a
// layer (the picker, a popover) is open.

import { isTextField, onKey } from '../register/keys.ts';
import { useDesk } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';
import { topLayer } from './layers.ts';

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;
const stops = () => [...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')].filter(t => t.getClientRects().length);

function jump(dir: 1 | -1): boolean {
  const ts = stops(); if (!ts.length) return false;
  const a = document.activeElement, at = ts.findIndex(t => t.contains(a));
  let i: number;
  if (at >= 0) i = at + dir;
  else { const after = ts.findIndex(t => a && a !== document.body && !!(a.compareDocumentPosition(t) & Node.DOCUMENT_POSITION_FOLLOWING)); i = dir > 0 ? (after < 0 ? ts.length : after) : (after < 0 ? ts.length : after) - 1; }
  if (i < 0 || i >= ts.length) { useDesk.getState().say(ui(dir > 0 ? 'jump.last' : 'jump.first')); return true; }
  const t = ts[i]!, g = t.querySelector<HTMLElement>('.tile-head .grip') ?? t.querySelector<HTMLElement>('button, [tabindex="0"], input, textarea');
  g?.focus({ preventScroll: true }); t.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
  useDesk.getState().say(t.querySelector('.tile-head h3, .tile-head')?.textContent?.trim() ?? '');
  return true;
}

onKey({
  id: 'tile-jump', order: 30,
  when: e => !e.defaultPrevented && !topLayer() && ((e.key === 'F6' && !e.ctrlKey && !e.altKey && !e.metaKey)
    || (e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey && (e.key === 'ArrowDown' || e.key === 'ArrowUp') && !isTextField(e.target as Element))),
  run: e => { if (jump(e.key === 'ArrowUp' || (e.key === 'F6' && e.shiftKey) ? -1 : 1)) e.preventDefault(); },
});
