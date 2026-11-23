// The tile shell (the concept's tileHTML, 30-tile.js): its header with the grip, the group and the name and the
// remove button; the note when the engine does not use it; its body, which is its face drawn by the generic renderer,
// or the one-line strip it becomes when it has nothing to show; and the "Also set" line at its foot. Drawn from the
// definition alone: a module needs no component of its own. The module that holds the board draws the board in its
// body after its face.

import type { KeyboardEvent } from 'react';
import type { ModuleDef } from '../../definitions/types.ts';
import { alsoLinkWords, alsoSet, defName, stripOf, stripWords, tileName, usedHere } from '../../derive.ts';
import { defOf, ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../parts/icons.tsx';
import { doAction } from '../parts/doAction.ts';
import { setTierFromView } from '../shell/tier.ts';
import { Face } from './Face.tsx';
import { Board } from '../board/Board.tsx';
import { moveTile } from '../desk/move.ts';
import { nextFocusAfter } from '../desk/focus.ts';
import { ENGINES } from '../../definitions/engines.ts';

export function Tile({ def }: { def: ModuleDef }) {
  const say = useSay(), desk = useDesk(s => s.desk);
  const unsup = !usedHere(def, desk.engine), strip = unsup ? null : stripOf(ix, def, desk);
  const size = { ...def.size, ...(def.per?.find(l => l.engines.includes(desk.engine))?.size ?? {}) };
  let w = desk.tier === 'quick' ? (size.quickW === 4 ? 'w4' : size.quickW === 2 ? 'w2' : '') : size.w === 2 ? 'w2' : size.w === 4 ? 'w4' : '';
  if (size.foldAt3 && desk.tier !== 'quick') w += ' fold3';
  if (strip) w = size.stripKeepsWidth ? `${w} is-strip` : 'is-strip';
  const name = tileName(def, desk), parent = def.labUnder ? defOf(def.labUnder) : undefined;
  const also = strip || unsup ? null : alsoSet(ix, def, desk);
  const attrs: Record<string, string> = {};
  if (size.grow && !strip && !/w4/.test(w)) attrs['data-grow'] = '';
  if (size.fold && /w2/.test(w)) attrs['data-fold'] = '';
  if (size.stretch && !unsup && !strip) attrs['data-stretch'] = String(size.stretch);
  if (strip && !unsup) attrs['data-span'] = '';
  const grip = (e: KeyboardEvent<HTMLButtonElement>) => {
    const dir = ({ ArrowLeft: 'left', ArrowUp: 'up', ArrowRight: 'right', ArrowDown: 'down' } as const)[e.key as 'ArrowLeft'];
    if (!dir) return; e.preventDefault(); moveTile(def.id, dir);
  };
  const s = strip ? stripWords(ix, `create.${def.id}.strip`, strip, desk) : null;
  return (
    <article className={`tile ${w} ${def.labUnder ? 'labmod' : ''} ${unsup ? 'unsup' : ''}`} data-m={def.id} {...attrs}>
      <header className="tile-head" draggable="true" data-drag={`move:${def.id}`}>
        <button type="button" className="grip" data-move={def.id} aria-label={say(ui('tile.move', { name }))} title={say(ui('tile.move.title'))} onKeyDown={grip}><Icon name="grip" size={16} /></button>
        <span className="grp">{say(def.labUnder && parent ? ui('tile.group.lab', { parent: defName(parent, desk) }) : ui(`tile.cat.${def.category}`))}</span>
        <h3>{say(name)}</h3>
        <button className="x" data-remove={def.id} aria-label={say(ui('tile.remove', { name }))} onClick={() => useDesk.getState().remove(def.id, nextFocusAfter(def.id))}><Icon name="x" size={14} /></button>
      </header>
      {unsup ? <div className="unsup-note"><b>{say(ui('tile.unsup', { engine: ENGINES.find(e => e.id === desk.engine)!.name }))}</b>{say(ui('tile.unsup.more'))}</div> : null}
      <div className="tile-body">
        {s ? (
          <div className="mod-strip"><span className="ms-why">{say(s.says)}</span>{s.actions.length ? <span className="ms-act">{s.actions.map(({ action, label, tip }, i) => <button key={i} className="sbtn" title={tip ? say(tip) : undefined} onClick={() => doAction(def, action)}>{say(label)}</button>)}</span> : null}</div>
        ) : def.board ? <Board def={def} /> : <Face def={def} />}
      </div>
      {also ? <p className="q-also"><span>{say(also.text)}</span> <button type="button" className="linkbtn" data-tier={also.link} data-ring={also.keys.join(' ')} onClick={() => setTierFromView(also.link, { ring: also.keys })}>{say(alsoLinkWords(also))}</button></p> : null}
    </article>
  );
}
