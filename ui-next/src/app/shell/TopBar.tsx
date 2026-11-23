// The top bar (the concept's shell/10-topbar.html and 31-topbar.js): the brand, the main navigation (stubs), the
// engine button with its mark and the VRAM meter's frame, the theme switch, the notes button. The engine panel it
// opens holds the five engines here; its models are the models package's (docs/09, "The plan").

import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { ENGINES } from '../../definitions/engines.ts';
import type { EngineId } from '../../definitions/types.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { ENGINE_COLOUR, logoOf, CONCEPT_ID } from './engines.ts';
import { Theme } from './Theme.tsx';

const NAV = ['create', 'Library', 'Studios', 'Training', 'Settings'] as const;
const EM_N = 32, CARD_GB = 31.8;

export function TopBar() {
  const say = useSay(), engine = useDesk(s => s.desk.engine), [open, setOpen] = useState(false);
  const name = ENGINES.find(e => e.id === engine)!.name;
  return (
    <div className="top-in">
      <button className="brand" type="button" data-goto="create" aria-label={say(ui('brand.label'))} title={say(ui('brand.title'))}>HOT<b>-</b>Step</button>
      <nav className="main" aria-label={say(ui('nav'))}>
        {NAV.map(n => <button key={n} data-nav={n} aria-current={n === 'create' ? 'page' : undefined}>{say(ui(`nav.${n.toLowerCase()}`))}</button>)}
      </nav>
      <div className="top-right">
        <div className="engine" id="engine" style={{ '--ec': ENGINE_COLOUR[engine] } as CSSProperties}>
          <button className="eng-btn" id="engBtn" type="button" aria-haspopup="dialog" aria-expanded={open} aria-controls="engPanel" aria-label={say(ui('engine.label', { engine: name }))} onClick={() => setOpen(o => !o)}>
            <span className="eng-glyph" id="engGlyph" aria-hidden="true" dangerouslySetInnerHTML={{ __html: logoOf(engine, 22) }} />
            <span className="eng-txt"><small>{say(ui('engine'))}</small><b id="engName">{name}</b></span>
            <span className="eng-caret" aria-hidden="true"><svg className="ico" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 10l5 5 5-5" /></svg></span>
          </button>
          <button className="eng-meter" id="engMeter" type="button" aria-expanded="false" aria-describedby="emRead" title={say(ui('vram.title'))}>
            <span className="em-cap" aria-hidden="true">{say(ui('vram'))}</span>
            <span className="em-bar" id="emBar" aria-hidden="true">{Array.from({ length: EM_N }, (_, i) => <i key={i} style={{ '--i': i } as CSSProperties} />)}<u className="em-peak" />{[8, 16, 24].map(g => <em key={g} style={{ left: `${g / CARD_GB * 100}%` }} />)}</span>
            <span className="em-read" id="emRead"><b id="emUse">0.0</b><small><span className="em-of">{say(ui('vram.of'))}</span>{say(ui('vram.gb'))}</small></span>
          </button>
        </div>
        <Theme />
        <button className="pill-btn" id="notesBtn">{say(ui('notes'))}</button>
      </div>
      {open ? <EnginePanel close={(refocus: boolean) => { setOpen(false); if (refocus) document.getElementById('engBtn')?.focus({ preventScroll: true }); }} /> : null}
    </div>
  );
}

/** The engine panel's cards: a radio group, the arrows move focus, Enter or a click chooses (F-16); Escape or a click
 *  outside closes it and focus goes back to its button. */
function EnginePanel({ close }: { close: (refocus: boolean) => void }) {
  const say = useSay(), engine = useDesk(s => s.desk.engine), el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const pn = document.getElementById('engPanel'); if (!pn) return;
    const r = document.getElementById('engBtn')!.getBoundingClientRect(), top = Math.max(r.bottom, document.querySelector('.top')!.getBoundingClientRect().bottom) + 6;
    pn.hidden = false; pn.style.setProperty('--ec', ENGINE_COLOUR[engine]);
    const w = pn.offsetWidth; pn.style.top = top + 'px'; pn.style.left = Math.max(12, Math.min(innerWidth - w - 12, r.left + r.width - w + 120)) + 'px'; pn.style.maxHeight = Math.max(160, innerHeight - top - 12) + 'px';
    pn.querySelector<HTMLElement>('[aria-checked="true"]')?.focus();
    const out = (e: PointerEvent) => { if (!pn.contains(e.target as Node) && !document.getElementById('engBtn')!.contains(e.target as Node)) close(false); };
    document.addEventListener('pointerdown', out, true);
    return () => { document.removeEventListener('pointerdown', out, true); pn.hidden = true; };
  }, [engine, close]);
  const choose = (e: EngineId) => useDesk.getState().switchEngine(e);
  const key = (ev: React.KeyboardEvent<HTMLDivElement>) => {
    const cards = [...el.current!.querySelectorAll<HTMLElement>('.ep-card')], i = cards.indexOf(document.activeElement as HTMLElement);
    const to = ({ ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1, Home: 0, End: cards.length - 1 } as Record<string, number>)[ev.key];
    if (to !== undefined) { ev.preventDefault(); cards[(to + cards.length) % cards.length]?.focus(); }
    else if (ev.key === 'Escape') { ev.preventDefault(); close(true); }
  };
  const pn = document.getElementById('engPanel');
  if (!pn) return null;
  return createPortalInto(pn,
    <div ref={el} onKeyDown={key}>
      <div className="ep-head"><b>{say(ui('engine'))}</b></div>
      <div className="ep-cards" role="radiogroup" aria-label={say(ui('engine'))}>
        {ENGINES.map(e => (
          <button key={e.id} type="button" className="ep-card" role="radio" data-backend={CONCEPT_ID[e.id]} aria-checked={e.id === engine} tabIndex={e.id === engine ? 0 : -1} style={{ '--c': ENGINE_COLOUR[e.id] } as CSSProperties} onClick={() => choose(e.id)}>
            <span className="ep-scope" dangerouslySetInnerHTML={{ __html: logoOf(e.id, 40) }} /><b>{e.name}</b>
          </button>
        ))}
      </div>
    </div>);
}

import { createPortal } from 'react-dom';
const createPortalInto = (el: Element, node: React.ReactNode) => createPortal(node, el);
