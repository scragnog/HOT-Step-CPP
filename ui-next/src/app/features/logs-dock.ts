// Opening and closing the engine log, and its dock (the concept's 34-logs.js toggleLogs, dockLogs; the view is
// logs.tsx): while it is open the page has the log's height of room at its foot (`--logs-h`, html.logs-open: the go
// bar and the toast stand on it, 30-logs.css, shell/placeToast.ts); closing it plays its exit and, when focus was in
// it, gives focus back to the VRAM meter. Escape closes it last of all.

import { flushSync } from 'react-dom';
import { onKey } from '../register/keys.ts';
import { placeToast } from '../shell/placeToast.ts';
import { useNotesLogs } from '../store/notes-logs.ts';
import { ovOut } from './overlay-exit.ts';

export const logsDock = (): HTMLElement | null => document.getElementById('logs');
export const vramMeter = (): HTMLElement | null => document.getElementById('engMeter');

/** The page makes room at its foot for the docked log, and the go bar and the toast stand on it. */
export function dockLogs(): void {
  const l = logsDock(), h = !l || l.hidden ? 0 : Math.round(l.getBoundingClientRect().height);
  document.documentElement.style.setProperty('--logs-h', h + 'px');
  document.documentElement.classList.toggle('logs-open', !!h);
  placeToast();
}

/** Opens the log, closes it, or (no argument) toggles it; `refocus`: focus inside it goes back to the meter. */
export function toggleLogs(open?: boolean, refocus = true): void {
  const l = logsDock(); if (!l) return;
  const want = open ?? l.hidden;
  if (want && l.hidden) {
    flushSync(() => useNotesLogs.setState({ logs: true }));
    document.getElementById('logsClose')?.focus({ preventScroll: true });
  } else if (!want && !l.hidden) {
    const inside = l.contains(document.activeElement);
    ovOut(l, 'sheet');
    flushSync(() => useNotesLogs.setState({ logs: false }));
    if (inside && refocus) vramMeter()?.focus({ preventScroll: true });
  }
}

const ddOpen = () => !!document.querySelector('#dd:not([hidden])');
/* Escape closes the log last of all: after a list, any layer, a drawer, the Modules panel over the desk and the save
   row have had theirs, the engine panel among them (16-overlays.js) */
onKey({ id: 'logs-escape', order: 90, when: e => e.key === 'Escape' && !e.defaultPrevented && !ddOpen() && useNotesLogs.getState().logs, run: () => toggleLogs(false) });
