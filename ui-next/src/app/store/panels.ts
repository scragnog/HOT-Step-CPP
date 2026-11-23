// Which side panels stand open: the view's own state, nothing about a value. The user's choice for roomy windows is
// kept between visits; below 1420px the Modules panel opens over the desk (rOpen), and in Lab below 1600px the
// Workflows panel starts on its rail (lOpen); neither of those is kept (31-side-panels.js panelKey).

import { create } from 'zustand';
import type { Tier } from '../../definitions/types.ts';

const KEY = 'hotstep.create.panels';
const read = (): { left: boolean; right: boolean } => {
  try { const o = JSON.parse(localStorage.getItem(KEY) ?? '{}') as { left?: unknown; right?: unknown }; return { left: o.left !== false, right: o.right !== false }; } catch { return { left: true, right: true }; }
};

export interface Panels { left: boolean; right: boolean; rOpen: boolean; lOpen: boolean }
export const usePanelStore = create<Panels>()(() => ({ ...read(), rOpen: false, lOpen: false }));
usePanelStore.subscribe(s => { try { localStorage.setItem(KEY, JSON.stringify({ left: s.left, right: s.right })); } catch { /* not kept */ } });

/** Which of the four a side's open state is, at a window width and tier. */
export const panelKey = (side: 'left' | 'right', w: number, tier: Tier): keyof Panels =>
  w <= 1180 ? side : side === 'right' && w < 1420 ? 'rOpen' : side === 'left' && ((tier === 'lab' && w < 1600) || w < 1240) ? 'lOpen' : side;
