// The Create page. The document's skeleton is the concept's (index.html: the skip link, the top bar, #app, the other
// pages' stand-in and the overlays' containers, in the concept's order), so its stylesheets apply unchanged; React is
// rooted in #app and draws into the top bar and the overlays through portals. docs/09-architecture.md.

import { useEffect, useLayoutEffect } from 'react';
import { createPortal } from 'react-dom';
import { useDesk } from './store/desk.ts';
import { ui, useSay } from './i18n/index.ts';
import { TopBar } from './shell/TopBar.tsx';
import { Centre } from './shell/Centre.tsx';
import { ModulesPanel, WorkflowsPanel } from './shell/Panels.tsx';
import { usePanels } from './shell/panelState.ts';
import { LiveRegion, Toast } from './shell/Toast.tsx';
import { closeDrawer } from './shell/drawers.ts';
import { usePanelStore, drawerMode } from './store/panels.ts';
import { dispatchKey } from './register/keys.ts';
import { Slot } from './register/Slot.tsx';
import { service } from './register/services.ts';

/** Focus that a step left nowhere goes where the step said: the first of its places that is on the page (A-18). A leaf
 *  of its own, so a focus request draws nothing else again. */
function FocusRequests() {
  const focus = useDesk(s => s.focus);
  useEffect(() => {
    if (!focus) return;
    requestAnimationFrame(() => {
      const el = focus.map(sel => document.querySelector<HTMLElement>(sel)).find(x => x && x.getClientRects().length);
      el?.focus({ preventScroll: true }); useDesk.getState().setFocus(null);
    });
  }, [focus]);
  return null;
}

export function App() {
  const say = useSay(), panels = usePanels(), drawer = usePanelStore(s => s.drawer);
  /* #app is React's root: its own classes say which panels stand open (31-side-panels.js paintPanels) */
  useLayoutEffect(() => {
    const app = document.getElementById('app')!;
    app.classList.toggle('no-left', !panels.L); app.classList.toggle('no-right', !panels.R || panels.over); app.classList.toggle('r-over', panels.over);
  });
  /* what the server tells the page: its lists and the plugin host's count (the models and their selection are the engine
     panel's: features/engine-panel.tsx) */
  useEffect(() => {
    const s = service('lists');
    void Promise.all([s.lists(), s.vstChain()]).then(([lists, chain]) => useDesk.getState().told({ lists, vstPlugins: chain.plugins.length }));
  }, []);
  /* the page's keys: each feature registers its own (register/keys.ts, src/app/features) */
  useEffect(() => {
    addEventListener('keydown', dispatchKey);
    return () => removeEventListener('keydown', dispatchKey);
  }, []);
  /* the drawers' scrim, a sibling of #app in the skeleton: shown while a drawer is open, a press on it closes it */
  useEffect(() => {
    const scrim = document.getElementById('drawerScrim'); if (!scrim) return;
    scrim.hidden = !drawer;
    const close = () => closeDrawer(false);
    scrim.addEventListener('click', close);
    return () => scrim.removeEventListener('click', close);
  }, [drawer]);
  /* a drawer left open when the window widens past drawers is closed */
  useEffect(() => { const f = () => { if (!drawerMode()) closeDrawer(false); }; addEventListener('resize', f); return () => removeEventListener('resize', f); }, []);
  const header = document.querySelector('header.top'), skip = document.querySelector('a.skip');
  return (
    <>
      {header ? createPortal(<TopBar />, header) : null}
      {skip ? createPortal(say(ui('skip')), skip) : null}
      <WorkflowsPanel open={panels.L} set={v => panels.set('left', v)} />
      <Centre />
      <ModulesPanel open={panels.R} set={v => panels.set('right', v)} />
      <Toast />
      {/* the overlays the packages fill (register/slots.ts 'overlays'): each portals into its container in the skeleton */}
      <Slot name="overlays" />
      <LiveRegion />
      <FocusRequests />
    </>
  );
}
