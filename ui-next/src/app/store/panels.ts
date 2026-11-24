// Which side panels stand open: the view's own state, nothing about a value. The user's choice for roomy windows is
// kept between visits; below 1420px the Modules panel opens over the desk (rOpen), and in Lab below 1600px the
// Workflows panel starts on its rail (lOpen); neither of those is kept (31-side-panels.js panelKey).

import { create } from 'zustand';
import type { Tier } from '../../definitions/types.ts';
import { readPref, writePref } from '../kit/storage.ts';

const KEY = 'hotstep.create.panels';
const read = (): { left: boolean; right: boolean } => readPref(KEY, { left: true, right: true }, v => { const o = (v && typeof v === 'object' ? v : {}) as { left?: unknown; right?: unknown }; return { left: o.left !== false, right: o.right !== false }; });

export interface Panels {
  left: boolean; right: boolean; rOpen: boolean; lOpen: boolean;
  /** At 1180px and below both panels are drawers over the desk: the one open, if any (31-side-panels.js). */
  drawer: 'left' | 'right' | null;
}
export const usePanelStore = create<Panels>()(() => ({ ...read(), rOpen: false, lOpen: false, drawer: null }));
/** The window is narrow enough for drawers. */
export const drawerMode = (): boolean => matchMedia('(max-width: 1180px)').matches;
usePanelStore.subscribe(s => writePref(KEY, { left: s.left, right: s.right }));

/** Which of the four a side's open state is, at a window width and tier. */
export const panelKey = (side: 'left' | 'right', w: number, tier: Tier): 'left' | 'right' | 'rOpen' | 'lOpen' =>
  w <= 1180 ? side : side === 'right' && w < 1420 ? 'rOpen' : side === 'left' && ((tier === 'lab' && w < 1600) || w < 1240) ? 'lOpen' : side;
