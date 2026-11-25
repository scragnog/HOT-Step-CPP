// The tile shell (the concept's tileHTML, 30-tile.js): its header with the grip, the group and the name and the
// remove button; the note when the engine does not use it; its body, which is its face drawn by the generic renderer,
// or the one-line strip it becomes when it has nothing to show; and the "Also set" line at its foot. What it shows is
// the core's (tileModel), from the definition alone: a module needs no component of its own. The module that holds
// the board draws the board in its body after its face. A tile redraws only when its own model changes. It is drawn at
// the desk's tier, or at the one a staged tier change holds it at (render/tileTier.ts); a tile the change brings is
// drawn as its shell until its turn.

import { memo, useLayoutEffect, useMemo, type KeyboardEvent } from 'react';
import type { ModuleDef } from '../../definitions/types.ts';
import { tileModel } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { Icon } from '../kit/icons.tsx';
import { setTierFromView } from '../shell/tier.ts';
import { Face } from './Face.tsx';
import { useModel } from './useModel.ts';
import { TierScope, dropScope, setScope, useHold, type Scope } from './tileTier.ts';
import { runStrip } from './bind.ts';
import { Board } from '../board/Board.tsx';
import { moveTile } from '../desk/move.ts';
import { nextFocusAfter } from '../desk/focus.ts';

const DIRS = { ArrowLeft: 'left', ArrowUp: 'up', ArrowRight: 'right', ArrowDown: 'down' } as const;

export const Tile = memo(function Tile({ def }: { def: ModuleDef }) {
  const say = useSay(), held = useHold(def.id), deskTier = useDesk(s => s.desk.tier), tier = held && held !== 'later' ? held : deskTier;
  const scope = useMemo<Scope>(() => ({ tier, models: new Set() }), [tier]);
  useLayoutEffect(() => { setScope(def.id, scope); return () => dropScope(def.id, scope); }, [def.id, scope]);
  const m = useModel(s => tileModel(ix, def, s.desk), def.id, scope), quick = tier === 'quick';
  const grip = (e: KeyboardEvent<HTMLButtonElement>) => { const dir = DIRS[e.key as keyof typeof DIRS]; if (!dir) return; e.preventDefault(); moveTile(def.id, dir); };
  return (
    <article className={m.cls} data-m={def.id} {...m.attrs}>
      <header className="tile-head" draggable="true" data-drag={`move:${def.id}`}>
        <button type="button" className="grip" data-move={def.id} aria-label={say(ui('tile.move', { name: m.name }))} title={say(ui('tile.move.title'))} onKeyDown={grip}><Icon name="grip" size={16} /></button>
        <span className="grp">{say(m.group)}</span>
        <h3>{say(m.name)}</h3>
        <button className="x" data-remove={def.id} aria-label={say(ui('tile.remove', { name: m.name }))} onClick={() => useDesk.getState().remove(def.id, nextFocusAfter(def.id))}><Icon name="x" size={14} /></button>
      </header>
      {m.unsup ? <div className="unsup-note"><b>{say(m.unsup)}</b>{say(ui('tile.unsup.more'))}</div> : null}
      <div className="tile-body"><TierScope.Provider value={scope}>
        {held === 'later' ? null : m.strip ? (
          <div className="mod-strip"><span className="ms-why">{say(m.strip.says)}{m.strip.held ? <span className="tk-held">{say(m.strip.held)}</span> : null}</span>{m.strip.actions.length ? <span className="ms-act">{m.strip.actions.map((a, i) => <button key={i} className="sbtn" title={a.tip ? say(a.tip) : undefined} onClick={() => runStrip(def, i)}>{say(a.label)}</button>)}</span> : null}</div>
        ) : m.board ? <Board def={def} /> : <Face def={def} surface={quick ? 'quick' : 'tile'} />}
      </TierScope.Provider></div>
      {m.also ? <p className="q-also"><span>{say(m.also.text)}</span> <button type="button" className="linkbtn" data-tier={m.also.link} data-ring={m.also.ring} onClick={() => setTierFromView(m.also!.link, { ring: m.also!.ring.split(' ') })}>{say(m.also.linkWords)}</button></p> : null}
    </article>
  );
});
