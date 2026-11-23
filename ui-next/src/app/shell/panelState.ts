// The side panels' state as the view reads it, and the panel's ids.

import { useEffect, useLayoutEffect, useState } from 'react';
import type { Definition } from '../../definitions/types.ts';
import { useDesk } from '../store/desk.ts';
import { panelKey, usePanelStore } from '../store/panels.ts';

/** The concept's ids in the panel's markup: a pedal is pd:<id>, a side unit ex:<id>. */
export const itemId = (d: Definition) => (d.kind === 'pedal' ? `pd:${d.id}` : d.kind === 'unit' ? `ex:${d.id}` : d.id);
/** Which panels stand open (store/panels.ts), at the window's width and the tier; and the Modules panel, when it opens
 *  over the desk, closes on a press outside it or on Escape, focus going back to its rail button (F-27). */
export function usePanels() {
  const [w, setW] = useState(innerWidth), tier = useDesk(s => s.desk.tier), st = usePanelStore();
  useLayoutEffect(() => { const f = () => setW(innerWidth); addEventListener('resize', f); return () => removeEventListener('resize', f); }, []);
  const get = (side: 'left' | 'right') => st[panelKey(side, w, tier)];
  const set = (side: 'left' | 'right', v: boolean) => usePanelStore.setState({ [panelKey(side, w, tier)]: v });
  const over = panelKey('right', w, tier) === 'rOpen' && get('right');
  useEffect(() => {
    if (!over) return;
    const press = (e: PointerEvent) => { if (!(e.target as Element).closest?.('#rightPanel, .pop, .menu, .dd, .toast')) usePanelStore.setState({ rOpen: false }); };
    const key = (e: globalThis.KeyboardEvent) => { if (e.key === 'Escape') { e.preventDefault(); usePanelStore.setState({ rOpen: false }); requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-expand="right"]')?.focus({ preventScroll: true })); } };
    document.addEventListener('pointerdown', press); document.addEventListener('keydown', key);
    return () => { document.removeEventListener('pointerdown', press); document.removeEventListener('keydown', key); };
  }, [over]);
  return { L: get('left'), R: get('right'), over, set };
}

