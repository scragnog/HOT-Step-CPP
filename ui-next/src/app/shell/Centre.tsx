// The centre column (create/20-centre.html): the desk bar (the drawers' buttons, the tier switch, the save state,
// Undo and Redo), the prompt card's place (a seam for the prompt package), the hidden-modules note, the desk, the go
// bar's frame and the Latest frame (seams for the generation and player packages).

import { useLayoutEffect, useRef, type KeyboardEvent } from 'react';
import type { Tier } from '../../definitions/types.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Desk, HiddenNote } from '../desk/Desk.tsx';
import { setTierFromView } from './tier.ts';
import { toggleDrawer } from './drawers.ts';
import { usePanelStore } from '../store/panels.ts';
import { Slot } from '../register/Slot.tsx';

const TIERS: Tier[] = ['quick', 'studio', 'lab'];

function Tiers() {
  const say = useSay(), tier = useDesk(s => s.desk.tier), el = useRef<HTMLDivElement>(null);
  /* the thumb slides under the chosen tier; its place is measured (31-shell.js placeTierThumb) */
  useLayoutEffect(() => {
    const t = el.current, b = t?.querySelector<HTMLElement>('[aria-checked="true"]'); if (!t || !b || !b.offsetWidth) return;
    t.style.setProperty('--tx', b.offsetLeft + 'px'); t.style.setProperty('--tw', b.offsetWidth + 'px');
    if (!t.classList.contains('ready')) requestAnimationFrame(() => t.classList.add('ready'));
  });
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = TIERS.indexOf(tier), j = ({ ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: 2 } as Record<string, number>)[e.key];
    if (j === undefined) return; e.preventDefault();
    const to = TIERS[Math.max(0, Math.min(2, j))]!; if (to !== tier) setTierFromView(to, { focus: true });
  };
  return (
    <div className="tiers" id="tiers" role="radiogroup" aria-label={say(ui('tier'))} ref={el} onKeyDown={key}>
      {TIERS.map((t, i) => (
        <button key={t} data-tier={t} role="radio" aria-checked={t === tier} tabIndex={t === tier ? 0 : -1} title={say(ui('tier.title', { d: ui(`tier.${t}.d`), n: i + 1 }))} aria-keyshortcuts={`Alt+${i + 1}`}
          onClick={() => setTierFromView(t, { focus: true })}>{say({ id: `tier.${t}` })}</button>
      ))}
    </div>
  );
}

function UndoRedo() {
  const say = useSay(), last = useDesk(s => s.undo[s.undo.length - 1]), next = useDesk(s => s.redo[s.redo.length - 1]);
  return (
    <>
      <button className="small-btn undo-btn" id="undoBtn" type="button" hidden={!last} title={last ? say(ui('undo.title', { label: last.words.label, put: last.words.put })) : undefined} aria-label={last ? say(ui('undo.label', { label: last.words.label })) : undefined} onClick={() => useDesk.getState().runUndo()}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 14 4 9l5-5" /><path d="M4 9h11a5 5 0 0 1 0 10h-3" /></svg>{say(ui('undo'))} <span>{last ? say(last.words.label) : ''}</span>
      </button>
      <button className="small-btn redo-btn" id="redoBtn" type="button" hidden={!next} aria-keyshortcuts="Control+Shift+Z Control+Y" title={next ? say(ui('redo.title', { label: next.words.label })) : undefined} aria-label={next ? say(ui('redo.label', { label: next.words.label })) : undefined} onClick={() => useDesk.getState().runRedo()}>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m15 14 5-5-5-5" /><path d="M20 9H9a5 5 0 0 0 0 10h3" /></svg><span className="redo-t">{say(ui('redo'))}</span>
      </button>
    </>
  );
}

export function Centre() {
  const say = useSay(), tier = useDesk(s => s.desk.tier), drawer = usePanelStore(s => s.drawer);
  return (
    <main className={`centre t-${tier}`} id="centre">
      <div className="deskbar">
        <h1 className="sr-only" id="deskH1" hidden={tier === 'quick'}>{say(ui('desk.h1'))}</h1>
        <div className="panel-toggles">
          <button className="small-btn" data-open="left" aria-controls="leftPanel" aria-expanded={drawer === 'left'} onClick={() => toggleDrawer('left')}>{say(ui('panel.workflows'))}</button>
          <button className="small-btn" data-open="right" aria-controls="rightPanel" aria-expanded={drawer === 'right'} onClick={() => toggleDrawer('right')}>{say(ui('panel.modules'))}</button>
        </div>
        <Tiers />
        <div className="deskbar-end">
          <Slot name="centre.saveState" />
          <UndoRedo />
        </div>
      </div>
      {/* the prompt card's place: the prompt package draws the card and the Quick lyric sheet here */}
      <div className="quickcol" id="quickcol">
        <h1 className="ask" id="ask" hidden={tier !== 'quick'}>{say(ui('ask'))}</h1>
        <div id="quickLyrics" />
      </div>
      <HiddenNote />
      <div className="deskwrap" id="deskwrap"><Desk /></div>
      {/* the go bar's frame: the generation package fills the meter, Drafts and Generate */}
      <div className="gobar">
        <div className="meter" id="meter" />
        <Slot name="centre.drafts" />
        <button className="go" id="go" title={say(ui('generate.title'))} aria-keyshortcuts="Control+Enter"><i className="go-fill" aria-hidden="true" /><svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4v16l13-8z" /></svg><span id="goLabel">{say(ui('generate'))}</span></button>
      </div>
      <section className="latest" aria-label={say(ui('latest.songs'))}><div className="res-head"><h2>{say(ui('latest'))}</h2><span className="lbl" id="songCount" /><button className="linkbtn" data-goto="Library">{say(ui('latest.library'))}</button></div><div className="songgrid" id="latest" /></section>
    </main>
  );
}
