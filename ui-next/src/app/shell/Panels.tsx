// The two side panels (10-workflows-panel.html, 30-modules-panel.html, 31-side-panels.js, 32-browser.js): columns on
// roomy windows, a rail when collapsed, an overlay or a rail on smaller ones. The Workflows panel is a shell here (its
// list is the workflows package's); the Modules panel offers every module, pedal and side unit as a card with its
// drawing, drawn from the definitions: a click, Enter or Space adds it, a Lab module outside Lab offers Lab.

import { useRef, useState, type KeyboardEvent } from 'react';
import { ENGINES } from '../../definitions/engines.ts';
import type { Definition, ModuleDef } from '../../definitions/types.ts';
import { defName, tileName } from '../../derive.ts';
import { OFFERED, defOf } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../parts/icons.tsx';
import { thumbOf } from '../thumbs/index.ts';
import { setTierFromView } from './tier.ts';
import { itemId } from './panelState.ts';

/* the groups of the panel: the modules' categories (Category in types.ts), then the pedals, then the side units; a group
   with nothing to offer is not drawn */
const CATS = ['Compose', 'Engine', 'Chain', 'Words', 'Pedals', 'Side units'] as const;
const catOf = (d: Definition) => (d.kind === 'pedal' ? 'Pedals' : d.kind === 'unit' ? 'Side units' : (d as ModuleDef).category);
const SHORT: Record<string, string> = { ace: 'ACE-Step', 'minimax-m3': 'MiniMax', yue2: 'YuE2', sa3: 'SA3', heartmula: 'HeartMuLa' };

export function WorkflowsPanel({ open, set }: { open: boolean; set: (v: boolean) => void }) {
  const say = useSay();
  return (
    <aside className={`side left${open ? '' : ' collapsed'}`} id="leftPanel" aria-label={say(ui('panel.workflows'))}>
      <button className="rail-btn" data-expand="left" onClick={() => { set(true); requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-collapse="left"]')?.focus({ preventScroll: true })); }}>{say(ui('panel.workflows'))}</button>
      <div className="side-head"><h2>{say(ui('panel.workflows'))}</h2>
        <button className="small-btn" id="saveBtn" aria-controls="saveRow" aria-expanded="false"><Icon name="plus" size={13} />{say(ui('panel.save'))}</button>
        <button className="icon-btn" data-collapse="left" aria-label={say(ui('panel.collapse'))} title={say(ui('panel.collapse.title'))} onClick={() => { set(false); requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-expand="left"]')?.focus({ preventScroll: true })); }}><Icon name="chevL" size={15} /></button>
      </div>
      <div className="side-body" id="presetList" />
    </aside>
  );
}

export function ModulesPanel({ open, set }: { open: boolean; set: (v: boolean) => void }) {
  const say = useSay(), desk = useDesk(s => s.desk), [q, setQ] = useState(''), [rove, setRove] = useState<string | null>(null);
  const list = useRef<HTMLDivElement>(null);
  const engine = ENGINES.find(e => e.id === desk.engine)!.name;
  const query = q.toLowerCase().trim();
  const rank = (d: Definition) => { if (!query) return 1; const nm = defName(d, desk).en!.toLowerCase(); return nm.startsWith(query) || nm.split(/[\s&-]+/).some(w => w.startsWith(query)) ? 1 : nm.includes(query) ? 2 : query.length >= 3 && new RegExp('(^|[^a-z0-9])' + query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).test((d.desc + ' ' + catOf(d)).toLowerCase()) ? 3 : 0; };
  const groups = CATS.map(c => [c, OFFERED.filter(d => catOf(d) === c && rank(d)).sort((a, b) => rank(a) - rank(b))] as const).filter(([, ds]) => ds.length);
  const first = groups[0]?.[1][0], stop = rove && OFFERED.some(d => itemId(d) === rove) ? rove : first ? itemId(first) : null;
  /* adding from the panel keeps focus on the card and says what arrived (F-25); a module the engine does not use waits
     parked, and its toast carries Undo */
  const add = (d: Definition) => {
    const s = useDesk.getState(), r = s.add(d.id), name = tileName(d, s.desk);
    if (r === 'lab') { s.showToast({ text: ui('labOffer.toast', { name }), action: { label: ui('labOffer.go'), run: () => { setTierFromView('lab'); useDesk.getState().add(d.id); } } }); return; }
    if (r !== 'added') return;
    const after = useDesk.getState(), entry = after.undo[after.undo.length - 1];
    after.say(ui('added', { name }));
    if (!d.engines.includes(after.desk.engine) && entry) after.showToast({ text: ui('added.notUsed', { name, engine }), action: { label: ui('undo'), run: () => useDesk.getState().runUndo(entry) } });
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
    const id = itemId(d), name = tileName(d, desk), locked = d.kind === 'module' && !!d.labUnder && desk.tier !== 'lab', no = !d.engines.includes(desk.engine), on = desk.on.has(d.id), weak = rank(d) === 3;
    const engines = { list: d.engines.map(e => ENGINES.find(x => x.id === e)!.name), join: 'and' as const };
    const tip = no ? ui(locked ? 'card.notUsedLab' : 'card.notUsed', { engine, engines }) : locked ? ui('card.lab') : on ? ui('card.on') : { id: `create.${d.id}.desc`, en: d.desc };
    const tag = d.engines.length > 2 ? say(ui('card.engines', { n: d.engines.length })) : d.engines.map(e => SHORT[e]).join(' · ');
    return (
      <div key={id} className={`bcard ${on ? 'in' : ''} ${locked ? 'locked' : ''} ${no ? 'no' : ''} ${weak ? 'weak' : ''}`} {...(locked ? {} : { draggable: true, 'data-drag': id })} data-additem={id} role="button" tabIndex={id === stop ? 0 : -1}
        title={say(ui('card.label', { name, tip }))} aria-label={say(ui('card.label', { name, tip }))} onClick={() => add(d)} onKeyDown={e => keys(e, d)} onFocus={() => setRove(id)}>
        <span className="bthumb" aria-hidden="true" dangerouslySetInnerHTML={{ __html: thumbOf(d, desk.engine, name.en ?? '') }} />
        <b>{say(name)}</b>
        {on || locked || no || (d.kind === 'module' && d.labUnder) ? <span className="ctags">
          {on && no ? <i className="ct park" title={say(ui('card.parked.title', { engine }))}>{say(ui('card.parked'))}</i> : on ? <i className="ct in" title={say(ui('card.onDesk.title'))}><Icon name="check" size={11} />{say(ui('card.onDesk'))}</i> : null}
          {d.kind === 'module' && d.labUnder ? <i className="ct lab" title={say(ui('card.labTag.title'))}>{say(ui('card.labTag'))}</i> : null}
          {no ? <i className="ct no" title={say(tip)}>{tag}</i> : null}
        </span> : null}
      </div>
    );
  };
  return (
    <aside className={`side right${open ? '' : ' collapsed'}`} id="rightPanel" aria-label={say(ui('panel.modules'))}>
      <button className="rail-btn" data-expand="right" onClick={() => { set(true); requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-collapse="right"]')?.focus({ preventScroll: true })); }}>{say(ui('panel.modules'))}</button>
      <div className="side-head"><h2>{say(ui('panel.modules'))}</h2>
        <button className="small-btn" data-openpicker="" title={say(ui('panel.picker.title'))}>{say(ui('panel.picker'))} <kbd>F9</kbd></button>
        <button className="icon-btn" data-collapse="right" aria-label={say(ui('panel.collapse.modules'))} title={say(ui('panel.collapse.title'))} onClick={() => { set(false); requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-expand="right"]')?.focus({ preventScroll: true })); }}><Icon name="chevR" size={15} /></button>
      </div>
      <div className="side-body" id="browser">
        <input id="brSearch" className="br-search" placeholder={say(ui('panel.filter'))} aria-label={say(ui('panel.filter'))} aria-controls="brList" value={q} onChange={e => setQ(e.currentTarget.value)}
          onKeyDown={e => { if (e.key === 'Escape') { if (q) { e.preventDefault(); e.stopPropagation(); setQ(''); } else e.currentTarget.blur(); } }} />
        <div id="brList" ref={list}>{groups.map(([c, ds]) => [<div key={c + ':h'} className="br-group">{say(ui(`cat.${c}`))}</div>, <div key={c} className="bgrid">{ds.map(card)}</div>])}</div>
      </div>
    </aside>
  );
}
void defOf;
