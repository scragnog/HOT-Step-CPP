// The two side panels (10-workflows-panel.html, 30-modules-panel.html, 31-side-panels.js, 32-browser.js): columns on
// roomy windows, a rail when collapsed, an overlay or a rail on smaller ones. The Workflows panel is a shell here (its
// list is the workflows package's); the Modules panel offers every module, pedal and side unit as a card with its
// drawing, drawn from the definitions: a click, Enter or Space adds it, a Lab module outside Lab offers Lab.

import { useRef, useState, type KeyboardEvent } from 'react';
import { engineName } from '../../definitions/engines.ts';
import type { Definition } from '../../definitions/types.ts';
import { cardGroups, cardModel, categoryWord, defName, groupOf, tileName, usedHere, type CardModel } from '../../derive.ts';
import { OFFERED } from '../model/index.ts';
import { useModel } from '../render/useModel.ts';
import { Slot } from '../register/Slot.tsx';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { thumbOf } from '../thumbs/index.ts';
import { setTierFromView } from './tier.ts';
import { itemId } from './panelState.ts';
import { closeDrawer } from './drawers.ts';
import { useEdgeFade } from './edgeFade.ts';
import { drawerMode, usePanelStore } from '../store/panels.ts';

/** A panel's own classes and, while it is an open drawer, its dialog role (C-13). */
function useSide(side: 'left' | 'right', open: boolean) {
  const drawer = usePanelStore(s => s.drawer) === side;
  return { className: `side ${side}${open ? '' : ' collapsed'}${drawer ? ' open' : ''}`, ...(drawer && drawerMode() ? { role: 'dialog', 'aria-modal': true } : {}) };
}
/** The collapse button: in a drawer it closes the drawer, else it puts the panel on its rail. */
const collapse = (side: 'left' | 'right', set: (v: boolean) => void) => {
  if (usePanelStore.getState().drawer === side) { closeDrawer(true); return; }
  set(false); requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-expand="${side}"]`)?.focus({ preventScroll: true }));
};

/* the groups of the panel: the modules' categories (CATEGORIES in types.ts), then the pedals, then the side units; a
   group with nothing to offer is not drawn (cardGroups) */
const GROUPS = cardGroups();

export function WorkflowsPanel({ open, set }: { open: boolean; set: (v: boolean) => void }) {
  const say = useSay();
  return (
    <aside {...useSide('left', open)} id="leftPanel" aria-label={say(ui('panel.workflows'))}>
      <button className="rail-btn" data-expand="left" onClick={() => { set(true); requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-collapse="left"]')?.focus({ preventScroll: true })); }}>{say(ui('panel.workflows'))}</button>
      <div className="side-head"><h2>{say(ui('panel.workflows'))}</h2>
        <Slot name="panel.workflows.head" />
        <button className="icon-btn" data-collapse="left" aria-label={say(ui('panel.collapse'))} title={say(ui('panel.collapse.title'))} onClick={() => collapse('left', set)}><Icon name="chevL" size={15} /></button>
      </div>
      <Slot name="panel.workflows.save" />
      <div className="side-body" id="presetList"><Slot name="panel.workflows" /></div>
    </aside>
  );
}

export function ModulesPanel({ open, set }: { open: boolean; set: (v: boolean) => void }) {
  const say = useSay(), engine = useModel(s => s.desk.engine, 'engine'), [q, setQ] = useState(''), [rove, setRove] = useState<string | null>(null);
  const list = useRef<HTMLDivElement>(null), body = useRef<HTMLDivElement>(null);
  useEdgeFade(body);
  /* every card as the core sees it on the desk; the filter matches the words the page shows, in its language */
  const cards = useModel(s => OFFERED.map(d => cardModel(d, s.desk, q, { name: say(defName(d, s.desk)), desc: say({ id: `create.${d.id}.desc`, en: d.desc }), group: say(categoryWord(groupOf(d))) })), 'cards:' + q);
  const byId = new Map(cards.map(c => [c.id, c]));
  const groups = GROUPS.map(g => [g, OFFERED.filter(d => groupOf(d) === g && byId.get(d.id)!.rank).sort((a, b) => byId.get(a.id)!.rank - byId.get(b.id)!.rank)] as const).filter(([, ds]) => ds.length);
  const first = groups[0]?.[1][0], stop = rove && OFFERED.some(d => itemId(d) === rove) ? rove : first ? itemId(first) : null;
  /* adding from the panel keeps focus on the card and says what arrived (F-25); a module the engine does not use waits
     parked, and its toast carries Undo */
  const add = (d: Definition) => {
    const s = useDesk.getState(), r = s.add(d.id), name = tileName(d, s.desk);
    if (r === 'lab') { s.showToast({ text: ui('labOffer.toast', { name }), action: { label: ui('labOffer.go'), run: () => { setTierFromView('lab'); useDesk.getState().add(d.id); } } }); return; }
    if (r !== 'added') return;
    /* adding from a drawer closes it (31-side-panels.js) */
    closeDrawer(false);
    const after = useDesk.getState(), entry = after.undo[after.undo.length - 1];
    after.say(ui('added', { name }));
    if (!usedHere(d, after.desk.engine) && entry) after.showToast({ text: ui('added.notUsed', { name, engine: engineName(after.desk.engine) }), action: { label: ui('undo'), run: () => useDesk.getState().runUndo(entry) } });
  };
  const keys = (e: KeyboardEvent<HTMLDivElement>, d: Definition) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (!e.repeat) add(d); return; }
    const cs = [...list.current!.querySelectorAll<HTMLElement>('.bcard')], cur = e.currentTarget;
    const dir = ({ ArrowRight: [1, 0], ArrowLeft: [-1, 0], ArrowDown: [0, 1], ArrowUp: [0, -1] } as Record<string, [number, number]>)[e.key];
    let to: HTMLElement | null = null;
    if (e.key === 'Home') to = cs[0] ?? null; else if (e.key === 'End') to = cs[cs.length - 1] ?? null;
    else if (dir) { const a = cur.getBoundingClientRect(), cx = a.left + a.width / 2, cy = a.top + a.height / 2; let best = Infinity;
      cs.forEach(c => { if (c === cur) return; const r = c.getBoundingClientRect(), x = r.left + r.width / 2 - cx, y = r.top + r.height / 2 - cy, along = dir[0] ? x * dir[0] : y * dir[1], across = dir[0] ? Math.abs(y) : Math.abs(x); if (along <= 4) return; const s = along + across * 3; if (s < best) { best = s; to = c; } }); }
    else return;
    e.preventDefault(); if (to) { setRove((to as HTMLElement).dataset.additem!); (to as HTMLElement).focus(); (to as HTMLElement).scrollIntoView({ block: 'nearest' }); }
  };
  const card = (d: Definition) => {
    const c: CardModel = byId.get(d.id)!, id = itemId(d), engineWords = engineName(engine);
    return (
      <div key={id} className={`bcard ${c.on ? 'in' : ''} ${c.locked ? 'locked' : ''} ${c.notHere ? 'no' : ''} ${c.rank === 3 ? 'weak' : ''}`} {...(c.locked ? {} : { draggable: true, 'data-drag': id })} data-additem={id} role="button" tabIndex={id === stop ? 0 : -1}
        title={say(c.tip)} aria-label={say(c.tip)} onClick={() => add(d)} onKeyDown={e => keys(e, d)} onFocus={() => setRove(id)}>
        <span className="bthumb" aria-hidden="true" dangerouslySetInnerHTML={{ __html: thumbOf(d, engine, c.thumbName) }} />
        <b>{say(c.name)}</b>
        {c.on || c.locked || c.notHere || c.lab ? <span className="ctags">
          {c.on && c.notHere ? <i className="ct park" title={say(ui('card.parked.title', { engine: engineWords }))}>{say(ui('card.parked'))}</i> : c.on ? <i className="ct in" title={say(ui('card.onDesk.title'))}><Icon name="check" size={11} />{say(ui('card.onDesk'))}</i> : null}
          {c.lab ? <i className="ct lab" title={say(ui('card.labTag.title'))}>{say(ui('card.labTag'))}</i> : null}
          {c.notHere && c.tag ? <i className="ct no" title={say(c.why)}>{say(c.tag.words)}</i> : null}
        </span> : null}
      </div>
    );
  };
  return (
    <aside {...useSide('right', open)} id="rightPanel" aria-label={say(ui('panel.modules'))}>
      <button className="rail-btn" data-expand="right" onClick={() => { set(true); requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-collapse="right"]')?.focus({ preventScroll: true })); }}>{say(ui('panel.modules'))}</button>
      <div className="side-head"><h2>{say(ui('panel.modules'))}</h2>
        <button className="small-btn" data-openpicker="" title={say(ui('panel.picker.title'))}>{say(ui('panel.picker'))} <kbd>F9</kbd></button>
        <button className="icon-btn" data-collapse="right" aria-label={say(ui('panel.collapse.modules'))} title={say(ui('panel.collapse.title'))} onClick={() => collapse('right', set)}><Icon name="chevR" size={15} /></button>
      </div>
      <div className="side-body" id="browser" ref={body}>
        <input id="brSearch" className="br-search" placeholder={say(ui('panel.filter'))} aria-label={say(ui('panel.filter'))} aria-controls="brList" value={q} onChange={e => setQ(e.currentTarget.value)}
          onKeyDown={e => { if (e.key === 'Escape') { if (q) { e.preventDefault(); e.stopPropagation(); setQ(''); } else e.currentTarget.blur(); } }} />
        <div id="brList" ref={list}>{groups.map(([c, ds]) => [<div key={c + ':h'} className="br-group">{say(categoryWord(c))}</div>, <div key={c} className="bgrid">{ds.map(card)}</div>])}</div>
      </div>
    </aside>
  );
}
