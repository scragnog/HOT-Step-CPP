// The Pedalboard's body (the concept's post face, paintBoard and pedalHTML, 27-pedalboard.js): the head with the board
// module's face (its chain switch) and the count, then the board: the cables, the pedals in chain order, and the side
// units. Every pedal and unit is drawn from its definition: laid out from its parameters by the core (boardNodes), its
// own face nodes by the generic renderer. No pedal or unit has a component of its own.

import { useEffect, useLayoutEffect, useMemo, useRef, type CSSProperties } from 'react';
import type { Definition, ModuleDef, Node, PedalDef, UnitDef } from '../../definitions/types.ts';
import { boardCount, boardNodes, defName, idleOf, nodeKeys, paramWords, usedHere, valueOf, type Cell as LaidCell } from '../../derive.ts';
import { defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../parts/icons.tsx';
import { PART_VIEWS } from '../parts/registry.ts';
import { Face, NodeView } from '../render/Face.tsx';
import { bind } from '../render/bind.ts';
import { fixCables, snapPedals, watchBoard } from './cables.ts';

const HOW = { id: 'part.knob.howTo', en: 'Drag up or down (hold Shift for fine steps), scroll, or use the arrow keys. Double-click to reset.' };

/** One laid-out cell: a knob, a choice, a text field or a switch, bound to its parameter as any node is. */
function LaidOut({ def, cell, keys }: { def: Definition; cell: LaidCell; keys: Map<Node, string> }) {
  const desk = useDesk(s => s.desk), say = useSay();
  if (cell.part === 'node') return cell.node ? <NodeView def={def} node={cell.node} keys={keys} desk={desk} /> : null;
  if (cell.part === 'plate' || !cell.param) return null;
  const node = { part: cell.part, param: cell.param } as Node, View = PART_VIEWS[cell.part];
  if (!View) return null;
  const props = bind(def, node, desk), w = paramWords(ix, cell.param, desk), f = w.face;
  if (cell.part === 'knob') {
    const full = w.label, label = f.short ? w.short! : w.label;
    return <View {...props} def={def} nodeKey="" cell={{ small: true, bare: !!f.unit && f.unit !== '%', label, aria: full, title: ui('pedal.tip', { a: full, b: HOW }) }} />;
  }
  if (cell.part === 'ledButton') return <View {...props} def={def} nodeKey="" cell={{ small: true }} />;
  /* a choice or text: an LCD selector with its caps label after it */
  const text = ix.params.get(cell.param)!.p.type === 'text';
  return (
    <div className={`lcdsel${cell.beside ? ' psw-rev' : ''}`} style={cell.beside && !cell.shown ? { visibility: 'hidden' } : undefined}>
      <View {...props} def={def} nodeKey="" cell={{}} />
      <span className={`kl${text ? ' cdot-lcd' : ''}`} data-ctl={text ? cell.param : undefined}>{say(w.label)}</span>
    </div>
  );
}

function Pedal({ def, chainOn }: { def: PedalDef; chainOn: boolean }) {
  const say = useSay(), desk = useDesk(s => s.desk), keys = useMemo(() => nodeKeys(def), [def]);
  const { cells } = boardNodes(ix, def, desk);
  const lab = desk.tier === 'lab', on = def.on ? valueOf(ix, def.on, desk) === true : true;
  const idle = idleOf(ix, def, desk), name = def.short && !lab ? { id: `create.${def.id}.short`, en: def.short } : defName(def, desk), full = defName(def, desk);
  const tip = [def.short && !lab ? say(full) : '', idle ? say(idle.says) : ''].filter(Boolean).join('. ');
  const knobs = cells.filter(c => c.part === 'knob' && !c.beside && c.shown), plates = cells.filter(c => c.part === 'plate');
  const sels = cells.filter(c => (c.part === 'pick' || c.part === 'textField' || c.part === 'chips') && !c.beside);
  const nodes = cells.filter(c => c.part === 'node'), switches = cells.filter(c => (c.part === 'ledButton' && c.param !== def.on) || !!c.beside);
  const units = lab && knobs.length >= 3 ? 2 : 1;
  const n = knobs.length + plates.length, cols = n <= 5 ? n : Math.ceil(n / 2), why = plates[0]?.plate;
  let mid: React.ReactNode = null;
  if (desk.tier === 'studio') {
    const main = cells.find(c => c.part === 'knob');
    mid = <div className="pmid">{main?.shown ? <LaidOut def={def} cell={main} keys={keys} /> : nodes.map((c, i) => <LaidOut key={i} def={def} cell={c} keys={keys} />)}</div>;
  }
  if (lab) mid = (
    <>
      {knobs.length ? <div className="pknobs" style={{ '--cols': cols } as CSSProperties}>
        {knobs.map(c => <LaidOut key={c.param} def={def} cell={c} keys={keys} />)}
        {plates.map((c, i) => <div key={c.param} className="kn kn-empty" aria-hidden="true" {...(!i && why ? { 'data-blank': say(why), style: { '--span': Math.min(plates.length, cols) } as CSSProperties } : {})} />)}
      </div> : null}
      {sels.length ? <div className="psels">{sels.map(c => <LaidOut key={c.param} def={def} cell={c} keys={keys} />)}</div> : null}
      {nodes.map((c, i) => <LaidOut key={i} def={def} cell={c} keys={keys} />)}
      {switches.length ? <div className="psw">{switches.map(c => <LaidOut key={c.param + (c.beside ?? '')} def={def} cell={c} keys={keys} />)}</div> : null}
    </>
  );
  const foot = def.on ? bind(def, { part: 'ledButton', param: def.on } as Node, desk).params[def.on] : undefined;
  return (
    <div className={`pedal u${units} ${on && chainOn ? '' : 'off'} ${def.on && on && !chainOn ? 'held' : ''} ${idle ? 'idle' : ''} ${def.on && !on ? 'byp' : ''} ${def.on ? '' : 'neutral'}`} data-pid={def.id} style={{ '--h': def.hue } as CSSProperties} title={tip || undefined}>
      <b>{say(name)}</b>{mid}
      <span className="led" title={idle ? say(idle.says) : undefined} />
      {foot ? <button className="foot" data-foot={def.on!} aria-pressed={on} aria-label={say(ui(on ? 'pedal.off' : 'pedal.on', { name: full }))} onClick={() => foot.set(!on)} />
        : <span className="always">{say(ui('pedal.always'))}</span>}
      <button className="px tab" data-pedalx={def.id} aria-label={say(ui('pedal.take', { name: full }))} title={say(ui('pedal.take.title'))} onClick={() => useDesk.getState().remove(def.id, pedalFocusAfter(def.id))}><Icon name="x" size={12} /></button>
    </div>
  );
}

function Unit({ def, chainOn }: { def: UnitDef; chainOn: boolean }) {
  const say = useSay(), desk = useDesk(s => s.desk), keys = useMemo(() => nodeKeys(def), [def]);
  const { cells } = boardNodes(ix, def, desk), on = valueOf(ix, def.on, desk) === true, name = defName(def, desk);
  const lab = desk.tier === 'lab';
  const objs = cells.filter(c => c.part === 'node'), knobs = lab ? cells.filter(c => c.part === 'knob' && c.shown) : [], rest = lab ? cells.filter(c => c.part !== 'node' && c.part !== 'knob' && c.part !== 'plate' && c.param !== def.on).sort((a, b) => def.params.findIndex(p => p.key === a.param) - def.params.findIndex(p => p.key === b.param)) : [];
  const power = bind(def, { part: 'ledButton', param: def.on } as Node, desk).params[def.on];
  return (
    <div className={`pedal xunit ${def.wide ? 'u-wide' : ''} ${on && (!def.gated || chainOn) ? '' : 'off'}`} data-xid={def.id} title={def.gated ? say(ui('unit.gated')) : undefined}>
      <b>{say(name)}{def.gated && !chainOn ? <small className="u-wait">{say(ui('unit.wait'))}</small> : null}</b>
      <div className="u-ctls">
        {objs.map((c, i) => <LaidOut key={i} def={def} cell={c} keys={keys} />)}
        {knobs.map(c => <LaidOut key={c.param} def={def} cell={c} keys={keys} />)}
        {rest.map(c => <UnitCtl key={c.param} def={def} cell={c} keys={keys} />)}
      </div>
      <div className="u-lip">
        <button className="u-pow" data-foot={def.on} aria-pressed={on} aria-label={say(ui(on ? 'pedal.off' : 'pedal.on', { name }))} title={say(ui(on ? 'unit.off' : 'unit.on'))} onClick={() => power?.set(!on)}>
          <svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M8 2v6" /><path d="M4.6 4.4a5 5 0 1 0 6.8 0" /></svg>
        </button>
        <button className="u-x" data-pedalx={def.id} aria-label={say(ui('pedal.take', { name }))} title={say(ui('pedal.take.title'))} onClick={() => useDesk.getState().remove(def.id, pedalFocusAfter(def.id))}><Icon name="x" size={12} /></button>
      </div>
    </div>
  );
}

/** A unit's control at Lab, laid out as the concept's rackCtl lays it out: a choice of up to four as chips, a longer
 *  choice as an LCD selector with its label first, a switch as an LED button, text as an LCD input. */
function UnitCtl({ def, cell, keys }: { def: Definition; cell: LaidCell; keys: Map<Node, string> }) {
  const desk = useDesk(s => s.desk), say = useSay();
  if (!cell.param) return null;
  if (cell.part === 'ledButton') return <LaidOut def={def} cell={cell} keys={keys} />;
  const node = { part: cell.part, param: cell.param } as Node, View = PART_VIEWS[cell.part as keyof typeof PART_VIEWS];
  if (!View) return null;
  const props = bind(def, node, desk), w = paramWords(ix, cell.param, desk);
  return <div className="lcdsel"><span className="kl">{say(w.label)}</span><View {...props} def={def} nodeKey="" cell={{}} /></div>;
}

/** After a pedal or unit leaves, focus goes to the next one on the board, else the one before, else the board's grip. */
function pedalFocusAfter(id: string): string {
  const s = useDesk.getState(), ring = s.board.filter(x => { const d = defOf(x); return d && usedHere(d, s.desk.engine); }), i = ring.indexOf(id);
  const sel = (x: string) => (defOf(x)?.kind === 'unit' ? `[data-xid="${x}"] .u-pow` : `[data-pid="${x}"] .foot, [data-pid="${x}"] .px`);
  return [ring[i + 1], ring[i - 1]].filter((x): x is string => !!x).map(sel).concat('[data-m="post"] .grip').join(', ');
}

export function Board({ def }: { def: ModuleDef }) {
  const say = useSay(), s = useDesk(), { desk, board } = s;
  const chain = def.params.find(p => p.type === 'boolean')?.key ?? null, chainOn = chain ? valueOf(ix, chain, desk) === true : true;
  const pedals = board.map(id => defOf(id)).filter((d): d is PedalDef => d?.kind === 'pedal' && usedHere(d, desk.engine));
  const units = board.map(id => defOf(id)).filter((d): d is UnitDef => d?.kind === 'unit' && usedHere(d, desk.engine));
  const boardEl = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { snapPedals(boardEl.current, desk.tier === 'lab'); requestAnimationFrame(() => fixCables(boardEl.current)); });
  useEffect(() => watchBoard(boardEl.current, () => desk.tier === 'lab'), [desk.tier]);
  return (
    <>
      <div className="board-head"><Face def={def} /><span data-ppcount="">{say(boardCount(ix, desk, chain))}</span></div>
      <div ref={boardEl} className={`board${chainOn ? '' : ' bypassed'}${pedals.length ? '' : ' empty'}`}>
        <svg className="cables" aria-hidden="true" />
        <div className="pedals" data-pedals="">
          {pedals.length ? pedals.map(d => <Pedal key={d.id} def={d} chainOn={chainOn} />) : <span className="board-empty">{say(ui('board.empty'))}</span>}
        </div>
        <div className="units" data-units="">
          {units.length ? <><span className="units-label">{say(ui('board.units'))}</span>{units.map(d => <Unit key={d.id} def={d} chainOn={chainOn} />)}</> : null}
        </div>
      </div>
    </>
  );
}
