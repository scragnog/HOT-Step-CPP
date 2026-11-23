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
import { setTierFromView } from './shell/tier.ts';
import type { Engine } from './services/engine.ts';

const isTextField = (el: Element | null) => !!el && (el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && !/^(checkbox|radio|button|submit|range|file|color)$/i.test((el as HTMLInputElement).type)) || (el as HTMLElement).isContentEditable);

export function App({ engine }: { engine: Engine }) {
  const say = useSay(), panels = usePanels(), focus = useDesk(s => s.focus);
  /* #app is React's root: its own classes say which panels stand open (31-side-panels.js paintPanels) */
  useLayoutEffect(() => {
    const app = document.getElementById('app')!;
    app.classList.toggle('no-left', !panels.L); app.classList.toggle('no-right', !panels.R || panels.over); app.classList.toggle('r-over', panels.over);
  });
  /* what the engine tells the page: its lists, the selected models, the plugin host's count */
  useEffect(() => {
    void Promise.all([engine.lists(), engine.vst.chain(), engine.models.list(useDesk.getState().desk.engine)]).then(([lists, chain, models]) =>
      useDesk.getState().told({ lists, selection: models.selection, vstPlugins: chain.plugins.length }));
  }, [engine]);
  /* focus that a step left nowhere goes where the step said */
  useEffect(() => {
    if (!focus) return;
    requestAnimationFrame(() => { document.querySelector<HTMLElement>(focus)?.focus({ preventScroll: true }); useDesk.getState().setFocus(null); });
  }, [focus]);
  /* the page's keys: Ctrl+Z and Ctrl+Shift+Z or Ctrl+Y (a text field keeps them while it has typing to take back),
     Alt+1, 2, 3 choose a tier */
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const s = useDesk.getState();
      if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'z' && !isTextField(document.activeElement)) { e.preventDefault(); if (e.shiftKey) s.runRedo(); else s.runUndo(); return; }
      if ((e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'y' && !isTextField(document.activeElement)) { e.preventDefault(); s.runRedo(); return; }
      if (e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey && /^Digit[123]$/.test(e.code)) { e.preventDefault(); setTierFromView((['quick', 'studio', 'lab'] as const)[+e.code.slice(5) - 1]!); }
    };
    addEventListener('keydown', key);
    return () => removeEventListener('keydown', key);
  }, []);
  const drawers = (side: 'left' | 'right') => {
    const p = document.getElementById(side === 'left' ? 'leftPanel' : 'rightPanel')!, scrim = document.getElementById('drawerScrim')!;
    p.classList.toggle('open'); scrim.hidden = !document.querySelector('.side.open');
    scrim.onclick = () => { document.querySelectorAll('.side.open').forEach(x => x.classList.remove('open')); scrim.hidden = true; };
  };
  const header = document.querySelector('header.top'), skip = document.querySelector('a.skip');
  return (
    <>
      {header ? createPortal(<TopBar />, header) : null}
      {skip ? createPortal(say(ui('skip')), skip) : null}
      <WorkflowsPanel open={panels.L} set={v => panels.set('left', v)} />
      <Centre drawers={drawers} />
      <ModulesPanel open={panels.R} set={v => panels.set('right', v)} />
      <Toast />
      <LiveRegion />
    </>
  );
}
