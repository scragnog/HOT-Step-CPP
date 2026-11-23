// The Pedalboard's body (the concept's post face, paintBoard and pedalHTML, 27-pedalboard.js): the head with the board
// module's face (its chain switch) and the count, then the board: the cables, the pedals in chain order, and the side
// units. Every pedal and unit is drawn from what the core says of it (pedalModel: its state, its words, its body laid
// out from its parameters, at the tier); each laid-out control is a node like any other, drawn by its part on the
// pedal's surface. No pedal or unit has a component of its own; each redraws only when its own model changes.

import { memo, useEffect, useLayoutEffect, useMemo, useRef, type CSSProperties } from 'react';
import type { Definition, ModuleDef, Node, PedalDef, UnitDef } from '../../definitions/types.ts';
import { boardModel, nodeKeys, pedalModel, type CellModel } from '../../derive.ts';
import { defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { Face, NodeView } from '../render/Face.tsx';
import { useModel } from '../render/useModel.ts';
import { setSwitch } from '../render/bind.ts';
import { fixCables, snapPedals, watchBoard } from './cables.ts';

/** One laid-out cell: a knob, a choice, a text field or a switch, bound to its parameter as any node is, or one of
 *  the pedal's own nodes. A choice or a text stands in an LCD selector with its caps label after it (or, on a unit,
 *  before it). */
function LaidOut({ def, cell, keys, unit }: { def: Definition; cell: CellModel; keys: Map<Node, string>; unit?: boolean }) {
  const say = useSay(), surface = def.kind === 'unit' ? 'unit' : 'pedal';
  const node = useMemo(() => cell.node ?? ({ part: cell.part, param: cell.param } as Node), [cell.node, cell.part, cell.param]);
  if (cell.part === 'plate' || (!cell.param && !cell.node)) return null;
  const view = <NodeView def={def} node={node} nodeKey={cell.node ? keys.get(cell.node) ?? '' : ''} keys={keys} surface={surface} cell={cell.words} />;
  if (cell.part === 'node' || cell.part === 'knob' || cell.part === 'ledButton') return view;
  if (unit) return <div className="lcdsel"><span className="kl">{say(cell.label)}</span>{view}</div>;
  return (
    <div className={`lcdsel${cell.beside ? ' psw-rev' : ''}`} style={cell.beside && !cell.shown ? { visibility: 'hidden' } : undefined}>
      {view}
      <span className={`kl${cell.text ? ' cdot-lcd' : ''}`} data-ctl={cell.text ? cell.param : undefined}>{say(cell.label)}</span>
    </div>
  );
}

const Pedal = memo(function Pedal({ def }: { def: PedalDef }) {
  const say = useSay(), keys = useMemo(() => nodeKeys(def), [def]), m = useModel(s => pedalModel(ix, def, s.desk), def.id);
  const cell = (c: CellModel, i: number) => <LaidOut key={(c.param ?? '') + (c.beside ?? '') + i} def={def} cell={c} keys={keys} />;
  const lab = m.tier === 'lab', why = m.plates[0]?.plate;
  return (
    <div className={m.cls} data-pid={def.id} style={{ '--h': m.hue } as CSSProperties} title={m.tip ? say(m.tip) : undefined}>
      <b>{say(m.name)}</b>
      {m.tier === 'studio' ? <div className="pmid">{m.mid.map(cell)}</div> : null}
      {lab ? <>
        {m.knobs.length ? <div className="pknobs" style={{ '--cols': m.cols } as CSSProperties}>
          {m.knobs.map(cell)}
          {m.plates.map((c, i) => <div key={c.param} className="kn kn-empty" aria-hidden="true" {...(!i && why ? { 'data-blank': say(why), style: { '--span': Math.min(m.plates.length, m.cols) } as CSSProperties } : {})} />)}
        </div> : null}
        {m.sels.length ? <div className="psels">{m.sels.map(cell)}</div> : null}
        {m.nodes.map(cell)}
        {m.switches.length ? <div className="psw">{m.switches.map(cell)}</div> : null}
      </> : null}
      <span className="led" title={m.idle ? say(m.idle) : undefined} />
      {m.foot ? <button className="foot" data-foot={m.foot.key} aria-pressed={m.foot.on} aria-label={say(m.foot.label)} onClick={() => setFoot(def, m.foot?.key, !m.foot!.on)} />
        : <span className="always">{say(ui('pedal.always'))}</span>}
      <button className="px tab" data-pedalx={def.id} aria-label={say(ui('pedal.take', { name: m.full }))} title={say(ui('pedal.take.title'))} onClick={() => useDesk.getState().remove(def.id, pedalFocusAfter(def.id))}><Icon name="x" size={12} /></button>
    </div>
  );
});

const Unit = memo(function Unit({ def }: { def: UnitDef }) {
  const say = useSay(), keys = useMemo(() => nodeKeys(def), [def]), m = useModel(s => pedalModel(ix, def, s.desk), def.id);
  return (
    <div className={m.cls} data-xid={def.id} title={m.gated ? say(ui('unit.gated')) : undefined}>
      <b>{say(m.name)}{m.waits ? <small className="u-wait">{say(ui('unit.wait'))}</small> : null}</b>
      <div className="u-ctls">
        {m.nodes.map((c, i) => <LaidOut key={i} def={def} cell={c} keys={keys} unit />)}
        {m.knobs.map(c => <LaidOut key={c.param} def={def} cell={c} keys={keys} unit />)}
        {m.switches.map(c => <LaidOut key={c.param} def={def} cell={c} keys={keys} unit />)}
      </div>
      <div className="u-lip">
        <button className="u-pow" data-foot={m.foot?.key} aria-pressed={!!m.foot?.on} aria-label={say(m.foot?.label)} title={m.foot?.title ? say(m.foot.title) : undefined} onClick={() => setFoot(def, m.foot?.key, !m.foot?.on)}>
          <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M8 2v6" /><path d="M4.6 4.4a5 5 0 1 0 6.8 0" /></svg>
        </button>
        <button className="u-x" data-pedalx={def.id} aria-label={say(ui('pedal.take', { name: m.name }))} title={say(ui('pedal.take.title'))} onClick={() => useDesk.getState().remove(def.id, pedalFocusAfter(def.id))}><Icon name="x" size={12} /></button>
      </div>
    </div>
  );
});

/** A foot switch or a unit's power: its key set as any control sets it, one step named by the core. */
const setFoot = (def: PedalDef | UnitDef, key: string | undefined, on: boolean) => { if (key) setSwitch(def, key, on); };

/** After a pedal or unit leaves, focus goes to the next one on the board, else the one before, else the board's grip:
 *  places tried in that order (A-18). */
function pedalFocusAfter(id: string): string[] {
  const s = useDesk.getState(), ring = boardModel(ix, s.desk, s.board), all = [...ring.pedals, ...ring.units], i = all.indexOf(id);
  const sel = (x: string) => (defOf(x)?.kind === 'unit' ? [`[data-xid="${x}"] .u-pow`] : [`[data-pid="${x}"] .foot`, `[data-pid="${x}"] .px`]);
  return [all[i + 1], all[i - 1]].filter((x): x is string => !!x).flatMap(sel).concat('[data-m="post"] .grip');
}

export function Board({ def }: { def: ModuleDef }) {
  const say = useSay(), b = useModel(s => boardModel(ix, s.desk, s.board), 'board'), tier = useModel(s => s.desk.tier, 'tier');
  const boardEl = useRef<HTMLDivElement>(null);
  /* the board's geometry is measured again when what stands on it changes, and when it resizes (watchBoard) */
  const shape = [tier, b.pedals.join(), b.units.join(), b.chainOn].join('|');
  useLayoutEffect(() => { snapPedals(boardEl.current, tier === 'lab'); requestAnimationFrame(() => fixCables(boardEl.current)); }, [shape, tier]);
  useEffect(() => watchBoard(boardEl.current, () => tier === 'lab'), [tier]);
  return (
    <>
      <div className="board-head"><Face def={def} surface="tile" /><span data-ppcount="">{say(b.count)}</span></div>
      <div ref={boardEl} className={`board${b.chainOn ? '' : ' bypassed'}${b.pedals.length ? '' : ' empty'}`}>
        <svg className="cables" aria-hidden="true" />
        <div className="pedals" data-pedals="">
          {b.pedals.length ? b.pedals.map(id => <Pedal key={id} def={defOf(id) as PedalDef} />) : <span className="board-empty">{say(ui('board.empty'))}</span>}
        </div>
        <div className="units" data-units="">
          {b.units.length ? <><span className="units-label">{say(ui('board.units'))}</span>{b.units.map(id => <Unit key={id} def={defOf(id) as UnitDef} />)}</> : null}
        </div>
      </div>
    </>
  );
}
