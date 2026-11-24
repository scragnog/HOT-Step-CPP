// The desk (the concept's renderDesk, 30-desk.js): the lyric sheet's place at its head (a slot the lyrics package
// fills), the tiles of the modules on the desk in the user's order, the empty desk, and the parked tray; then the tiles
// are placed by the layout. Which tiles show, what the empty desk offers and says, and what waits parked are the
// core's (deskModel). The desk redraws when its make-up changes, not when a value does; the layout runs again when the
// make-up or a tile's shape changes, or a tile's content changes size (layout.ts packSoon).

import { useEffect, useLayoutEffect, useState } from 'react';
import type { ModuleDef } from '../../definitions/types.ts';
import { deskModel, tileModel } from '../../derive.ts';
import { defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { Tile } from '../render/Tile.tsx';
import { useModel } from '../render/useModel.ts';
import { Slot } from '../register/Slot.tsx';
import { setTierFromView } from '../shell/tier.ts';
import { packSoon, renderPack } from './layout.ts';
import { ring } from './ring.ts';

export function HiddenNote() {
  const say = useSay(), note = useModel(s => deskModel(ix, s.desk, s.mods).hidden, 'hidden');
  return (
    <div className="hidden-note" id="hiddenNote" hidden={!note}>
      {note ? <><span>{say(note)}</span> <button className="small-btn" data-tier="lab" onClick={() => setTierFromView('lab')}>{say(ui('hidden.show'))}</button></> : null}
    </div>
  );
}

/* a starter adds its module and focus lands on it (Q-16) */
const addStarter = (id: string) => { if (useDesk.getState().add(id).did === 'added') useDesk.getState().setFocus(`.tile[data-m="${id}"] .grip`); };

export function Desk() {
  const say = useSay(), m = useModel(s => deskModel(ix, s.desk, s.mods), 'desk');
  /* the window's width is part of the desk's make-up: a resize lays the desk out afresh */
  const [width, setWidth] = useState(innerWidth);
  useEffect(() => { const f = () => setWidth(innerWidth); addEventListener('resize', f); return () => removeEventListener('resize', f); }, []);
  /* each tile's shape: its width class and attributes, and the lines outside its body (the engine's note, the "Also
     set" line): a tile that becomes a strip, widens or grows a line is laid out again (its body's own changes of size
     are the ResizeObserver's below) */
  const shape = useModel(s => m.tiles.map(id => { const t = tileModel(ix, defOf(id) as ModuleDef, s.desk); return JSON.stringify([t.cls, t.attrs, !!t.unsup, !!t.also]); }).join('|'), 'shape:' + m.tiles.join());
  const make = `${m.make}|${width}`;
  useLayoutEffect(() => renderPack(make), [make, shape]);
  /* a tile whose content changes size, or the window, re-places the desk */
  useEffect(() => {
    const g = document.getElementById('bento'); if (!g) return;
    const now = () => { const s = useDesk.getState(); return `${deskModel(ix, s.desk, s.mods).make}|${innerWidth}`; };
    const ro = new ResizeObserver(() => packSoon(now, true));
    const watch = () => g.querySelectorAll('.tile-body').forEach(b => ro.observe(b));
    watch(); const mo = new MutationObserver(watch); mo.observe(g, { childList: true });
    const rs = () => packSoon(now, false);
    addEventListener('resize', rs);
    return () => { ro.disconnect(); mo.disconnect(); removeEventListener('resize', rs); };
  }, []);
  /* adding what is already there rings it */
  const flash = useDesk(x => x.flash);
  useEffect(() => { if (flash) ring(flash.id, flash.ring); }, [flash]);
  const tag = (t: { words: Parameters<typeof say>[0]; tip: Parameters<typeof say>[0] } | null) => (t ? <small className="pc-eng" title={say(t.tip)}>{say(t.words)}</small> : null);
  const starters = m.empty?.starters.map(d => <button key={d.id} className="sbtn" data-additem={d.id} onClick={() => addStarter(d.id)}>+ <b>{say(d.name)}</b></button>);
  return (
    <div className="bento" id="bento">
      {m.tier !== 'quick' ? (m.lyricsHere ? <Slot name="desk.lead" /> : <div className="w4 deskrow"><Slot name="desk.lead" /></div>) : null}
      {m.tiles.map(id => <Tile key={id} def={defOf(id) as ModuleDef} />)}
      {m.empty?.quick ? (
        <div className="q-empty"><span>{`${say(m.empty.says ?? '')} `}<span aria-hidden="true">→</span></span><span className="de-acts">{starters}</span></div>
      ) : null}
      {m.empty && !m.empty.quick ? (
        <div className="desk-empty"><span className="de-art" aria-hidden="true"><i /><i /><i /><i /></span><div><h3>{say(ui('empty.title'))}</h3></div>
          <div className="de-acts"><button className="sbtn" data-openpicker="">{say(ui('empty.picker'))} <kbd>F9</kbd></button>{starters}</div></div>
      ) : null}
      {m.parked ? (
        <div className="parked"><span className="lbl">{say(m.parked.says)}</span>
          {m.parked.modules.map(d => <span key={d.id} className="pchip" data-pm={d.id}>{say(d.name)}{tag(d.tag)}<button className="x" data-remove={d.id} aria-label={say(ui('tile.remove', { name: d.name }))} onClick={() => useDesk.getState().remove(d.id)}><Icon name="x" size={12} /></button></span>)}
          {m.parked.board.map(d => <span key={d.id} className="pchip" data-pp={d.id}>{say(d.name)}<small>{say(d.what)}</small>{tag(d.tag)}<button className="x" data-pedalx={d.id} aria-label={say(ui('tile.remove', { name: d.name }))} onClick={() => useDesk.getState().remove(d.id)}><Icon name="x" size={12} /></button></span>)}
        </div>
      ) : null}
    </div>
  );
}
