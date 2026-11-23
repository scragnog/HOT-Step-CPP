// The desk (the concept's renderDesk, 30-desk.js): the lyric sheet's place at its head (a seam the lyrics package
// fills), the tiles of the modules on the desk in the user's order (the tier hides Lab modules; a module the engine
// does not use waits parked), the empty desk, and the parked tray; then the tiles are placed by the layout. The note
// above the desk names the Lab modules a tier hides.

import { useEffect, useLayoutEffect } from 'react';
import type { ModuleDef } from '../../definitions/types.ts';
import { ENGINES } from '../../definitions/engines.ts';
import { defName, tileName, hiddenLabModules, parkedOn, tileAtTier, usedHere } from '../../derive.ts';
import { defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../parts/icons.tsx';
import { Tile } from '../render/Tile.tsx';
import { setTierFromView } from '../shell/tier.ts';
import { deskMake, packSoon, renderPack } from './layout.ts';
import { DEFINITIONS } from '../../definitions/index.ts';

const LYRICS = DEFINITIONS.find(d => d.id === 'lyrics')!;
const STARTERS = ['tempo', 'sampler', 'seed', 'takes'];
const SHORT: Record<string, string> = { ace: 'ACE-Step', 'minimax-m3': 'MiniMax', yue2: 'YuE2', sa3: 'SA3', heartmula: 'HeartMuLa' };

/** The lyric sheet's place on the desk: the lyrics package draws it here (a tile at Studio and Lab, the card under the
 *  prompt at Quick). Until then it is empty, and the desk lays out as the concept's does without it. */
export function LyricsSeam() { return null; }

export function HiddenNote() {
  const say = useSay(), desk = useDesk(s => s.desk);
  const note = hiddenLabModules(ix, desk);
  return (
    <div className="hidden-note" id="hiddenNote" hidden={!note}>
      {note ? <><span>{say(note)}</span> <button className="small-btn" data-tier="lab" onClick={() => setTierFromView('lab')}>{say(ui('hidden.show'))}</button></> : null}
    </div>
  );
}

export function Desk() {
  const say = useSay(), s = useDesk();
  const { desk, mods } = s;
  const defs = mods.map(id => defOf(id)).filter((d): d is ModuleDef => d?.kind === 'module');
  const live = defs.filter(d => tileAtTier(d, desk) && usedHere(d, desk.engine));
  const parked = parkedOn(ix, desk);
  const starters = STARTERS.map(id => defOf(id)).filter((d): d is ModuleDef => !!d && d.kind === 'module' && usedHere(d, desk.engine) && !desk.on.has(d.id)).slice(0, 3);
  const make = deskMake(desk.tier, desk.engine, mods, false);
  useLayoutEffect(() => renderPack(make));
  /* a tile whose content changes size, or the window, re-places the desk */
  useEffect(() => {
    const g = document.getElementById('bento'); if (!g) return;
    const ro = new ResizeObserver(() => packSoon(() => deskMake(useDesk.getState().desk.tier, useDesk.getState().desk.engine, useDesk.getState().mods, false), true));
    const watch = () => g.querySelectorAll('.tile-body').forEach(b => ro.observe(b));
    watch(); const mo = new MutationObserver(watch); mo.observe(g, { childList: true });
    const rs = () => packSoon(() => deskMake(useDesk.getState().desk.tier, useDesk.getState().desk.engine, useDesk.getState().mods, false), false);
    addEventListener('resize', rs);
    return () => { ro.disconnect(); mo.disconnect(); removeEventListener('resize', rs); };
  }, []);
  const add = (id: string) => useDesk.getState().add(id);
  const lyricsHere = LYRICS.engines.includes(desk.engine);
  const engineName = ENGINES.find(e => e.id === desk.engine)!.name;
  const parkTag = (d: (typeof parked.modules)[number]) => d.engines.length < ENGINES.length
    ? <small className="pc-eng" title={say(ui('parked.worksOn', { engines: { list: d.engines.map(e => ENGINES.find(x => x.id === e)!.name), join: 'and' } }))}>{d.engines.length > 2 ? say(ui('parked.engines', { n: d.engines.length })) : d.engines.map(e => SHORT[e]).join(' · ')}</small> : null;
  return (
    <div className="bento" id="bento">
      {desk.tier !== 'quick' ? (lyricsHere ? <LyricsSeam /> : <div className="w4 deskrow"><LyricsSeam /></div>) : null}
      {live.map(d => <Tile key={d.id} def={d} />)}
      {!live.length && desk.tier === 'quick' ? (
        <div className="q-empty"><span>{`${say(ui(desk.engine === 'sa3' || desk.engine === 'heartmula' ? 'qempty.length' : 'qempty.tempo'))} `}<span aria-hidden="true">→</span></span><span className="de-acts">{starters.map(d => <button key={d.id} className="sbtn" data-additem={d.id} onClick={() => add(d.id)}>+ <b>{say(tileName(d, desk))}</b></button>)}</span></div>
      ) : null}
      {!live.length && desk.tier !== 'quick' ? (
        <div className="desk-empty"><span className="de-art" aria-hidden="true"><i /><i /><i /><i /></span><div><h3>{say(ui('empty.title'))}</h3></div>
          <div className="de-acts"><button className="sbtn" data-openpicker="">{say(ui('empty.picker'))} <kbd>F9</kbd></button>{starters.map(d => <button key={d.id} className="sbtn" data-additem={d.id} onClick={() => add(d.id)}>+ <b>{say(tileName(d, desk))}</b></button>)}</div></div>
      ) : null}
      {parked.says ? (
        <div className="parked"><span className="lbl">{say(parked.says)}</span>
          {parked.modules.map(d => <span key={d.id} className="pchip" data-pm={d.id}>{say(tileName(d, desk))}{parkTag(d)}<button className="x" data-remove={d.id} aria-label={say(ui('tile.remove', { name: tileName(d, desk) }))} onClick={() => useDesk.getState().remove(d.id)}><Icon name="x" size={12} /></button></span>)}
          {parked.board.map(d => <span key={d.id} className="pchip" data-pp={d.id}>{d.name}<small>{say(ui(d.kind === 'unit' ? 'parked.unit' : 'parked.pedal'))}</small>{parkTag(d)}<button className="x" data-pedalx={d.id} aria-label={say(ui('pedal.take', { name: defName(d, desk) }))} onClick={() => useDesk.getState().remove(d.id)}><Icon name="x" size={12} /></button></span>)}
        </div>
      ) : null}
    </div>
  );
  void engineName;
}
