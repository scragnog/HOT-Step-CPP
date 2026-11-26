// The drawers' state, as the view changes it: at 1180px and below a panel opens over the desk as a drawer (its button
// in the desk bar), and closes on the scrim, Escape, its collapse button or once something is added from it
// (31-side-panels.js). Which one is open is view state (store/panels.ts); the panels draw it.

import { usePanelStore } from '../store/panels.ts';

const buttonOf = (side: 'left' | 'right') => document.querySelector<HTMLElement>(`[data-open="${side}"]`);

/** Opens or closes a drawer from its button; an opened drawer takes focus (its first button, else the panel). */
export function toggleDrawer(side: 'left' | 'right'): void {
  const now = usePanelStore.getState().drawer, to = now === side ? null : side;
  usePanelStore.setState({ drawer: to });
  if (to) requestAnimationFrame(() => {
    const p = document.getElementById(to === 'left' ? 'leftPanel' : 'rightPanel');
    (p?.querySelector<HTMLElement>('.side-head button, .side-body [tabindex="0"], input') ?? p)?.focus({ preventScroll: true });
  });
}

/** Closes the open drawer; `refocus` sends focus back to its button. */
export function closeDrawer(refocus = false): void {
  const was = usePanelStore.getState().drawer; if (!was) return;
  usePanelStore.setState({ drawer: null });
  if (refocus) requestAnimationFrame(() => buttonOf(was)?.focus({ preventScroll: true }));
}
