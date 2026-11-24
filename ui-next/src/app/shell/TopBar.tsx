// The top bar (the concept's shell/10-topbar.html, 31-topbar.js and 34-engine-panel.js): the brand, the main
// navigation (stubs), the engine button with its mark, the VRAM meter, the theme switch, the notes button; and the
// engine panel the button opens: a card for each engine (its tagline, output rate, songs per Generate, a badge while it
// is in testing) and, under them, the engine's models (the engine.panel slot, features/engine-panel.tsx). What the cards
// and the meter say is the core's (derive/vram.ts enginePanelModel, meterModel).

import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { ENGINES } from '../../definitions/engines.ts';
import type { EngineId } from '../../definitions/types.ts';
import { enginePanelModel, meterEase, meterLeds, meterModel, meterTarget } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useModel } from '../render/useModel.ts';
import { Slot } from '../register/Slot.tsx';
import { ui, useSay } from '../i18n/index.ts';
import { ENGINE_COLOUR, logoOf, CONCEPT_ID } from './engines.ts';
import { Theme } from './Theme.tsx';
import { NAV, useScreen } from './screens.ts';
import { layerClosed, layerOpen } from '../features/layers.ts';
import { useGen } from '../store/generate.ts';

/* the main navigation: each page's hook in the concept's markup (data-nav) and its words' id */
const EM_N = 32;
const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function TopBar() {
  const say = useSay(), engine = useDesk(s => s.desk.engine), [open, setOpen] = useState(false), live = useGen(s => s.runs.length > 0), screen = useScreen(s => s.name);
  const name = ENGINES.find(e => e.id === engine)!.name;
  /* the panel is named whether it is open or not, as the concept's markup names it */
  useEffect(() => { document.getElementById('engPanel')?.setAttribute('aria-label', say(ui('engine.panel'))); }, [say]);
  return (
    <div className="top-in">
      {/* the brand is the one word the page does not translate */}
      {/* eslint-disable-next-line no-restricted-syntax */}
      <button className="brand" type="button" data-goto="create" aria-label={say(ui('brand.label'))} title={say(ui('brand.title'))}>HOT<b>-</b>Step</button>
      <nav className="main" aria-label={say(ui('nav'))}>
        {/* a press shows that page (features/screens.tsx: Create, or the stand-in for the others: RC-01) */}
        {NAV.map(([n, id]) => <button key={n} data-nav={n} aria-current={n === screen ? 'page' : undefined}>{say(ui(`nav.${id}`))}</button>)}
      </nav>
      <div className="top-right">
        <div className={`engine${live ? ' lg-live' : ''}`} id="engine" style={{ '--ec': ENGINE_COLOUR[engine] } as CSSProperties}>
          <button className="eng-btn" id="engBtn" type="button" aria-haspopup="dialog" aria-expanded={open} aria-controls="engPanel" aria-label={say(ui('engine.label', { engine: name }))} onClick={() => setOpen(o => !o)}>
            <span className="eng-glyph" id="engGlyph" aria-hidden="true" dangerouslySetInnerHTML={{ __html: logoOf(engine, 22, 'glyph') }} />
            <span className="eng-txt"><small>{say(ui('engine'))}</small><b id="engName">{name}</b></span>
            <span className="eng-caret" aria-hidden="true"><svg className="ico" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 10l5 5 5-5" /></svg></span>
          </button>
          <Meter />
        </div>
        <Theme />
        <button className="pill-btn" id="notesBtn">{say(ui('notes'))}</button>
      </div>
      {open ? <EnginePanel close={(refocus: boolean) => { setOpen(false); if (refocus) document.getElementById('engBtn')?.focus({ preventScroll: true }); }} /> : null}
    </div>
  );
}

/** The VRAM meter: LED segments like a loudness meter, lit up to what is loaded, outlined up to what the next Generate
 *  needs, with its peak line; a fast attack and a slow release (34-engine-panel.js stepMeter). Its click opens the engine
 *  log, whose package draws it. */
function Meter() {
  /* the run that renders (the head of the queue) holds the card: while it runs the needle swings toward what it needs,
     and when it is another engine's, the meter shows what that engine holds (B-34, package 6) */
  const head = useGen(s => s.runs[0]), run = head?.spec.desk ?? null, busy = run !== null;
  const say = useSay(), m = useModel(s => meterModel(ix, s.desk, run), `vram-meter:${head?.id ?? ''}`), [cur, setCur] = useState(0);
  useEffect(() => {
    let raf = 0, at = cur;
    const step = () => { raf = 0; const target = meterTarget(m.use, m.est, busy, Math.random); at = meterEase(at, target, reduced()); setCur(at); if (busy || Math.abs(target - at) > 0.05) raf = requestAnimationFrame(step); };
    if (busy || Math.abs(m.use - at) > 0.05 || at !== cur) raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [m.use, m.est, busy]); // eslint-disable-line react-hooks/exhaustive-deps -- it starts from where the needle is
  const leds = meterLeds(cur, m.est, m.card, EM_N);
  return (
    <button className={`eng-meter${m.over ? ' over' : ''}${m.tight ? ' tight' : ''}`} id="engMeter" type="button" aria-expanded="false" aria-describedby="emRead" aria-label={say(m.aria)} title={say(m.title)}>
      <span className="em-cap" aria-hidden="true">{say(ui('vram'))}</span>
      <span className="em-bar" id="emBar" aria-hidden="true">
        {leds.map((c, i) => <i key={i} className={c || undefined} style={{ '--i': i } as CSSProperties} />)}
        <u className="em-peak" style={{ left: `${m.peak * 100}%` }} />
        {m.ticks.map(t => <em key={t} style={{ left: `${t * 100}%` }} />)}
      </span>
      <span className="em-read" id="emRead"><b id="emUse">{m.read}</b><small><span className="em-of">{say(m.of)}</span>{say(ui('vram.gb'))}</small></span>
    </button>
  );
}

/** The engine panel: the engines are a radio group, the arrows move focus between them and Enter, Space or a click
 *  chooses (a switch rebuilds the desk, so it does not follow focus: F-16); Escape or a click outside closes it and focus
 *  goes back to its button. It hangs 6px under the top bar, capped to the room under it (B-21, F-20); on a narrow window
 *  the engines are a strip that scrolls sideways, the chosen one brought into view. A model's list (#dd) is the panel's
 *  own layer: a click in it leaves the panel open, and its Escape closes the list only. */
function EnginePanel({ close }: { close: (refocus: boolean) => void }) {
  const say = useSay(), engine = useDesk(s => s.desk.engine), el = useRef<HTMLElement | null>(document.getElementById('engPanel'));
  const m = useModel(s => enginePanelModel(ix, s.desk.tier, s.desk.engine), 'engine-panel');
  /* the card the arrows moved to holds the Tab stop until the engine changes */
  const [roving, setRoving] = useState<{ on: EngineId; at: EngineId } | null>(null);
  useEffect(() => {
    const pn = document.getElementById('engPanel'); if (!pn) return;
    const place = () => {
      const r = document.getElementById('engBtn')!.getBoundingClientRect(), top = Math.max(r.bottom, document.querySelector('.top')!.getBoundingClientRect().bottom) + 6;
      pn.style.maxHeight = ''; const w = pn.offsetWidth;
      pn.style.top = top + 'px'; pn.style.left = Math.max(12, Math.min(innerWidth - w - 12, r.left + r.width - w + 120)) + 'px'; pn.style.maxHeight = Math.max(160, innerHeight - top - 12) + 'px';
    };
    pn.hidden = false; pn.style.setProperty('--ec', ENGINE_COLOUR[engine]); place();
    const cs = pn.querySelector<HTMLElement>('.ep-cards'), sel = pn.querySelector<HTMLElement>('.ep-card[aria-checked="true"]');
    if (cs && sel && cs.scrollWidth > cs.clientWidth) cs.scrollLeft = sel.offsetLeft - cs.offsetLeft - 8;
    if (pn.contains(document.activeElement) || document.activeElement === document.body || document.activeElement?.id === 'engBtn') sel?.focus();
    const out = (e: PointerEvent) => { const t = e.target as Element; if (t.isConnected && !t.closest('#engPanel, #engBtn, #dd')) close(false); };
    document.addEventListener('pointerdown', out, true); addEventListener('resize', place);
    return () => { document.removeEventListener('pointerdown', out, true); removeEventListener('resize', place); pn.hidden = true; };
  }, [engine, close]);
  /* the panel is one of the page's layers (features/layers.ts): Escape closes it, the picker opening closes it (RC-06),
     no shortcut acts under it but the page's Undo (deviation 30) */
  const closeNow = useRef(close); useEffect(() => { closeNow.current = close; });
  useEffect(() => {
    const pn = document.getElementById('engPanel'); if (!pn) return;
    layerOpen(pn, { trigger: document.getElementById('engBtn'), undo: true, close: () => closeNow.current(true) });
    return () => layerClosed(pn, false);
  }, []);
  const choose = (e: EngineId) => useDesk.getState().switchEngine(e);
  /* the panel's own keys, on the panel (a model's list is drawn elsewhere, #dd, and keeps its own) */
  const key = (ev: KeyboardEvent) => {
    const pn = el.current; if (ev.defaultPrevented || !pn) return;
    const card = (ev.target as Element).closest<HTMLElement>('.ep-card'); if (!card) return;
    const cards = [...pn.querySelectorAll<HTMLElement>('.ep-card')], i = cards.indexOf(card);
    const to = ({ ArrowRight: i + 1, ArrowDown: i + 1, ArrowLeft: i - 1, ArrowUp: i - 1 } as Record<string, number>)[ev.key];
    if (to === undefined) return;
    ev.preventDefault(); const n = cards[(to + cards.length) % cards.length]!;
    setRoving({ on: engine, at: m.cards[cards.indexOf(n)]!.id }); n.focus();
  };
  const keyNow = useRef(key); useEffect(() => { keyNow.current = key; });
  useEffect(() => {
    const pn = el.current; if (!pn) return;
    const k = (e: KeyboardEvent) => keyNow.current(e);
    pn.addEventListener('keydown', k); return () => pn.removeEventListener('keydown', k);
  }, []);
  const pn = document.getElementById('engPanel');
  if (!pn) return null;
  const tab = roving?.on === engine ? roving.at : engine;
  return createPortalInto(pn,
    <>
      <div className="ep-head"><b>{say(ui('engine'))}</b></div>
      <div className="ep-cards" role="radiogroup" aria-label={say(ui('engine'))}>
        {m.cards.map(c => (
          <button key={c.id} type="button" className="ep-card" role="radio" data-backend={CONCEPT_ID[c.id]} aria-checked={c.id === engine} tabIndex={c.id === tab ? 0 : -1} style={{ '--c': ENGINE_COLOUR[c.id] } as CSSProperties} onClick={() => choose(c.id)}>
            <span className="ep-scope" dangerouslySetInnerHTML={{ __html: logoOf(c.id, 40, 'card') + (c.testing ? `<i class="ep-badge">${escHtml(say(c.testing))}</i>` : '') }} />
            <b>{c.name}</b><small>{say(c.tag[0])}<br />{say(c.tag[1])}</small>
            <span className="ep-specs"><i title={say(c.rateTip)}>{say(c.rate)}</i><i title={say(c.takesTip)}>{say(c.takes)}</i></span>
          </button>
        ))}
      </div>
      <Slot name="engine.panel" />
    </>);
}

const createPortalInto = (el: Element, node: ReactNode) => createPortal(node, el);
/** A word put into the mark's markup (the testing badge stands inside the scope with the mark). */
const escHtml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
