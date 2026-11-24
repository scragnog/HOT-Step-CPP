// Which page shows (the concept's 31-screens.js, RC-01): Create, or the page that stands in for every other one (the
// skeleton's #other), chosen from the top bar's navigation, the brand and the links that name a page (data-goto).
// View state only. features/screens.tsx draws the stand-in.

import { create } from 'zustand';

/** The top bar's pages: each one's name (its data-nav) and its word's id. */
export const NAV = [['create', 'create'], ['Library', 'library'], ['Studios', 'studios'], ['Training', 'training'], ['Settings', 'settings']] as const;
export const pageWord = (name: string): string => `nav.${NAV.find(([n]) => n === name)?.[1] ?? 'create'}`;

export const useScreen = create<{ name: string }>(() => ({ name: 'create' }));
/** Create is the page showing: its keys (register/keys.ts `create`) do nothing on another page. */
export const onCreate = (): boolean => useScreen.getState().name === 'create';

/** Shows a page: Create, or the stand-in for `name`; the top bar's button for it is the current one. */
export function showScreen(name: string): void {
  useScreen.setState({ name });
  const app = document.getElementById('app'), other = document.getElementById('other');
  if (app) app.hidden = name !== 'create';
  if (other) other.hidden = name === 'create';
}
