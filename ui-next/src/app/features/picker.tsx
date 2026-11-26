// The module picker's view (F9; the concept's 33-picker.js and 12-picker.html): the full-screen wall of every module,
// pedal and side unit as its drawing, in two columns (the modules; the board's pedals and side units), with a filter, a
// category bar, the count of what it shows and an info line that says what the keys do and what is under the pointer
// or the keys. A click, Enter or Space adds; a Lab module outside Lab offers Lab in the info line; one already on the
// desk closes the picker and rings it; a drawing can be dragged onto the desk (dnd.tsx). Its keys, opening and closing
// are picker-keys.ts; what each drawing is and how it matches is the core's (src/derive/picker.ts pickerModel).

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { engineName } from '../../definitions/engines.ts';
import type { Definition } from '../../definitions/types.ts';
import { categoryWord, defDesc, defName, groupOf, msg, PICKER_CATS, PICKER_LEFT, PICKER_RIGHT, pickerCountMsg, pickerModel, type PickerCat, type PickerItem, type PickerModel } from '../../derive.ts';
import { defOf, OFFERED } from '../model/index.ts';
import { fill } from '../register/slots.ts';
import { useDesk } from '../store/desk.ts';
import { usePicker } from '../store/overlays.ts';
import { ui, useSay, sayNow } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { thumbBare, thumbOf, bareSize } from '../thumbs/index.ts';
import { itemId } from '../shell/panelState.ts';
import { useEdgeFade } from '../shell/edgeFade.ts';
import { labAdd } from './adding.ts';
import { activate, closePicker, defOfItem, setQuery } from './picker-keys.ts';
import { reduced } from '../kit/motion.ts';

const host = () => document.getElementById('pickerWrap');
const coarse = () => matchMedia('(pointer: coarse)').matches;

/* ---------------------------------------------------------------- the wall */

const SIDES = { L: PICKER_LEFT, R: PICKER_RIGHT } as const;

function Tags({ it }: { it: PickerItem }) {
  const say = useSay(), engine = useDesk(s => s.desk.engine);
  if (!(it.on || it.lab || it.notHere)) return null;
  return (
    <span className="ctags">
      {it.on && it.notHere ? <i className="ct park" title={say(ui('card.parked.title', { engine: engineName(engine) }))}>{say(ui('card.parked'))}</i>
        : it.on ? <i className="ct in" title={say(ui('card.onDesk.title'))}><Icon name="check" size={11} />{say(ui('card.onDesk'))}</i> : null}
      {it.lab ? <i className="ct lab" title={say(ui('card.labTag.title'))}>{say(ui('card.labTag'))}</i> : null}
      {it.notHere && it.tag ? <i className="ct no" title={say(it.says)}>{say(it.tag.words)}</i> : null}
    </span>
  );
}

function Snapshot({ it, d, rove, q }: { it: PickerItem; d: Definition; rove: boolean; q: string }) {
  const say = useSay(), engine = useDesk(s => s.desk.engine), id = itemId(d);
  const size = d.kind === 'pedal' ? bareSize(d) : { w: 176, h: 112 };
  const art = d.kind === 'pedal' ? thumbBare(d, engine, it.thumbName) : thumbOf(d, engine, it.thumbName, 'pk');
  const cls = ['fl-it', it.pedal ? 'bare' : '', size.w > 120 ? 'u2' : '', it.locked ? 'locked' : '', it.on ? 'in' : '', it.lab ? 'lab' : '', it.notHere ? 'no' : '',
    it.gone ? 'gone' : '', it.dim ? 'dim' : '', it.nomatch ? 'nomatch' : '', it.rank === 3 ? 'weak' : ''].filter(Boolean).join(' ');
  return (
    <div className={cls} {...(it.locked ? {} : { draggable: true, 'data-drag': id })} data-pkid={id} tabIndex={rove ? 0 : -1} role="button"
      aria-label={say(msg('card.label', { name: it.name, tip: it.says }))} title={it.rank === 3 ? say(ui('picker.weak', { q: q.trim() })) : ''}
      style={{ '--w': `${size.w}px`, '--h': `${size.h}px`, ...(it.rank > 1 ? { order: it.rank } : {}) } as CSSProperties}>
      <div className="box" aria-hidden="true" dangerouslySetInnerHTML={{ __html: art }} />
      <span className="fl-cap"><span className="fl-name">{say(it.name)}</span><Tags it={it} /></span>
    </div>
  );
}

function Info({ m, byId }: { m: PickerModel; byId: Map<string, PickerItem> }) {
  const say = useSay(), info = usePicker(s => s.info), desk = useDesk(s => s.desk), engine = desk.engine;
  const first = m.order[0], at = info.kind === 'first' ? (first ? { kind: 'item' as const, id: itemId(defOf(first)!) } : { kind: 'hint' as const }) : info;
  if (at.kind === 'hint') {
    const fit = m.fit ? byId.get(itemId(defOf(m.fit)!)) : undefined;
    const words = coarse() ? (fit ? ui('picker.hint.tapFirst', { name: fit.name }) : ui('picker.hint.tap')) : fit ? ui('picker.hint.first', { name: fit.name }) : ui('picker.hint');
    return <span className="hint">{say(words)}</span>;
  }
  const it = byId.get(at.id), d = defOfItem(at.id); if (!it || !d) return null;
  if (at.kind === 'added') {
    return <><b>{say(ui('picker.added', { name: it.name }))}</b><span>{say(ui(at.lab ? 'picker.added.lab' : coarse() ? 'picker.added.tap' : 'picker.added.more'))}</span></>;
  }
  return (
    <>
      <b>{say(it.name)}</b><span>{say(defDesc(d, desk))}</span>
      {it.locked ? <button type="button" className="small-btn primary" data-labadd={at.id} onClick={() => labAdd(d.id)}>{say(ui('picker.labAdd'))}</button>
        : it.on ? <em>{say(ui(coarse() ? 'picker.onDesk.tap' : 'picker.onDesk'))}</em>
        : it.notHere ? <em className="warn">{say(ui('picker.notUsed', { engine: engineName(engine) }))}</em> : null}
    </>
  );
}

type More = { side: 'L' | 'R'; n: number; top: number; left: number };

function Wall() {
  const say = useSay(), st = usePicker(), desk = useDesk(s => s.desk);
  const left = useRef<HTMLDivElement>(null), right = useRef<HTMLDivElement>(null);
  useEdgeFade(left); useEdgeFade(right);
  const m = pickerModel(OFFERED, desk, st.q, st.cat, d => ({ name: sayNow(defName(d, desk)), desc: sayNow(defDesc(d, desk)), group: sayNow(categoryWord(groupOf(d))) }));
  const byId = new Map(m.items.map(x => [itemId(defOf(x.id)!), x]));
  /* the one Tab stop: the drawing the keys stand on, else the first shown (the first match while a filter is typed) */
  const shownIds = m.order.map(id => itemId(defOf(id)!)), stop = st.rove && shownIds.includes(st.rove) ? st.rove : shownIds[0] ?? null;
  const target = m.target ? itemId(defOf(m.target)!) : null;
  const group = (g: (typeof PICKER_CATS)[number]) => {
    const n = m.groups.find(x => x.group === g)?.n ?? 0, ds = OFFERED.filter(d => byId.get(itemId(d))!.pgroup === g);
    return (
      <section key={g} className="fl-grp" data-grp={g} aria-label={say(categoryWord(g))} hidden={!n}>
        <h3 className="fl-gh">{say(categoryWord(g))}{g === 'Pedals' ? <small>{say(ui('picker.chainOrder'))}</small> : null}<i className="fl-gn">{n}</i></h3>
        <div className="fl-row">{ds.map(d => { const id = itemId(d); return <Snapshot key={id} d={d} it={byId.get(id)!} rove={id === stop} q={st.q} />; })}</div>
      </section>
    );
  };
  const none = (side: 'L' | 'R') => { const n = SIDES[side].reduce((a, g) => a + (m.groups.find(x => x.group === g)?.n ?? 0), 0); return !n && st.q.trim() ? <p className="fl-empty">{say(ui(side === 'L' ? 'picker.none.left' : 'picker.none.right', { q: st.q }))}</p> : null; };
  /* a column taller than the window scrolls by itself and says at its foot how many of its drawings are below (F-05) */
  const [more, setMore] = useState<More[]>([]);
  useLayoutEffect(() => {
    const w = host(); if (!w) return;
    const measure = () => {
      const wr = w.getBoundingClientRect(), out: More[] = [];
      ([['L', left.current], ['R', right.current]] as const).forEach(([side, wall]) => {
        if (!wall) return; const cr = wall.getBoundingClientRect();
        const n = [...wall.querySelectorAll<HTMLElement>('.fl-it')].filter(el => el.offsetParent && !el.classList.contains('dim') && el.getBoundingClientRect().bottom > cr.bottom + 2).length;
        if (n) out.push({ side, n, top: cr.bottom - wr.top - 34, left: cr.left + cr.width / 2 - wr.left });
      });
      setMore(was => (JSON.stringify(was) === JSON.stringify(out) ? was : out));
    };
    measure();
    const f = () => requestAnimationFrame(measure), l = left.current, r = right.current, first = requestAnimationFrame(measure);
    l?.addEventListener('scroll', f, { passive: true }); r?.addEventListener('scroll', f, { passive: true }); addEventListener('resize', f);
    return () => { cancelAnimationFrame(first); l?.removeEventListener('scroll', f); r?.removeEventListener('scroll', f); removeEventListener('resize', f); };
  });
  /* the drawing Enter will add is ringed while focus stays in the filter (F-29) */
  useLayoutEffect(() => {
    document.querySelectorAll('#pickerWrap .fl-it.target').forEach(x => x.classList.remove('target'));
    if (target) document.querySelector(`#pickerWrap .fl-it[data-pkid="${CSS.escape(target)}"]`)?.classList.add('target');
  });
  return (
    <>
      <label className="fl-search">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></svg>
        <input id="pkSearch" placeholder={say(ui('panel.filter'))} autoComplete="off" spellCheck="false" aria-label={say(ui('panel.filter'))} value={st.q} onChange={e => setQuery(e.currentTarget.value)} />
        <span id="pkCount" className="mono">{say(pickerCountMsg(m.count))}</span>
      </label>
      <button className="fl-close" id="pkClose" type="button" aria-label={say(ui('picker.close'))} onClick={() => closePicker()}><Icon name="x" size={15} /></button>
      <div className="fl-cols" id="pkCols">
        <section className="fl-col"><h2>{say(ui('panel.modules'))}</h2><div className="fl-wall" id="pkLeft" role="group" aria-label={say(ui('panel.modules'))} ref={left}>{PICKER_LEFT.map(group)}{none('L')}</div></section>
        <section className="fl-col"><h2>{say(ui('picker.board'))}</h2><div className="fl-wall" id="pkRight" role="group" aria-label={say(ui('picker.boardGroup'))} ref={right}>{PICKER_RIGHT.map(group)}{none('R')}</div></section>
      </div>
      <div className="fl-info" id="pkInfo" aria-live="polite"><Info m={m} byId={byId} /></div>
      <nav className="fl-bar" id="pkBar" aria-label={say(ui('picker.bar'))}>
        {PICKER_CATS.map(c => <button key={c} type="button" className={c === 'All' ? 'all' : undefined} data-pkcat={c} aria-pressed={st.cat === c} onClick={() => usePicker.setState({ cat: c as PickerCat })}>{say(c === 'All' ? ui('picker.all') : categoryWord(c))}</button>)}
      </nav>
      {more.map(x => (
        <button key={x.side} id={`pkMore${x.side}`} type="button" className="fl-more" aria-label={say(ui(x.side === 'L' ? 'picker.more.left' : 'picker.more.right', { n: x.n }))} style={{ top: x.top + 'px', left: x.left + 'px' }}
          onClick={() => { const wall = x.side === 'L' ? left.current : right.current; wall?.scrollBy({ top: wall.clientHeight * 0.7, behavior: reduced() ? 'auto' : 'smooth' }); }}>{say(ui('picker.more', { n: x.n }))}</button>
      ))}
    </>
  );
}

export function Picker() {
  const say = useSay(), open = usePicker(s => s.open), closing = usePicker(s => s.closing), peek = usePicker(s => s.peek);
  useLayoutEffect(() => {
    const el = host(); if (!el) return;
    el.hidden = !open; el.classList.toggle('closing', closing); el.classList.toggle('peek', peek);
    el.setAttribute('aria-label', say(ui('picker.label')));
  }, [open, closing, peek, say]);
  /* the pointer says what it points at; leaving a drawing gives the line back to the one the keys stand on (unless it
     says what was just added); a click on the backdrop closes the picker */
  useEffect(() => {
    const el = host(); if (!el) return;
    const over = (e: MouseEvent) => {
      const it = (e.target as Element).closest?.<HTMLElement>('.fl-it'), s = usePicker.getState();
      if (it && !(s.info.kind === 'item' && s.info.id === it.dataset.pkid) && !(document.activeElement?.id === 'pkSearch' && s.q.trim())) usePicker.setState({ info: { kind: 'item', id: it.dataset.pkid! } });
    };
    const out = (e: MouseEvent) => {
      const s = usePicker.getState();
      if (!(e.target as Element).closest?.('.fl-it') || (e.relatedTarget as Element | null)?.closest?.('.fl-it, #pkInfo') || s.info.kind === 'added') return;
      const f = document.activeElement?.closest?.<HTMLElement>('.fl-it');
      usePicker.setState({ info: f ? { kind: 'item', id: f.dataset.pkid! } : { kind: 'hint' } });
    };
    const focus = (e: FocusEvent) => { const it = (e.target as Element).closest?.<HTMLElement>('.fl-it'); if (it) usePicker.setState({ rove: it.dataset.pkid!, info: { kind: 'item', id: it.dataset.pkid! } }); };
    const click = (e: MouseEvent) => {
      const t = e.target as Element, it = t.closest?.<HTMLElement>('.fl-it');
      if (it) { activate(it.dataset.pkid!); return; }
      if (t.matches('#pickerWrap, .fl-cols, .fl-col, .fl-wall, .fl-col h2, .fl-grp, .fl-row')) closePicker();
    };
    el.addEventListener('mouseover', over); el.addEventListener('mouseout', out); el.addEventListener('focusin', focus); el.addEventListener('click', click);
    return () => { el.removeEventListener('mouseover', over); el.removeEventListener('mouseout', out); el.removeEventListener('focusin', focus); el.removeEventListener('click', click); };
  }, []);
  const el = host();
  if (!el) return null;
  return createPortal(open ? <Wall /> : <Closed />, el);
}
fill('overlays', Picker);

/** The closed picker holds its frame, as the concept's page does (12-picker.html): the filter, the close button, the
 *  two columns, the info line and the bar, empty; the wall is drawn only while it is open. */
function Closed() {
  const say = useSay();
  return (
    <>
      <label className="fl-search">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></svg>
        <input id="pkSearch" placeholder={say(ui('panel.filter'))} autoComplete="off" spellCheck="false" aria-label={say(ui('panel.filter'))} defaultValue="" />
        <span id="pkCount" className="mono" />
      </label>
      <button className="fl-close" id="pkClose" type="button" aria-label={say(ui('picker.close'))}><Icon name="x" size={15} /></button>
      <div className="fl-cols" id="pkCols">
        <section className="fl-col"><h2>{say(ui('panel.modules'))}</h2><div className="fl-wall" id="pkLeft" role="group" aria-label={say(ui('panel.modules'))} /></section>
        <section className="fl-col"><h2>{say(ui('picker.board'))}</h2><div className="fl-wall" id="pkRight" role="group" aria-label={say(ui('picker.boardGroup'))} /></section>
      </div>
      <div className="fl-info" id="pkInfo" aria-live="polite" />
      <nav className="fl-bar" id="pkBar" aria-label={say(ui('picker.bar'))} />
    </>
  );
}
