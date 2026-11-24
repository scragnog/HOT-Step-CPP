// A tile's moves as a menu (the concept's B-23, 30-desk.js): a tap or click on a tile's grip offers Move to the top,
// Move earlier, Move later and Move to the end in the popover, so a tile can be placed without a drag or the arrow
// keys (a finger cannot start the desk's drag and drop). Each move goes to the next place in the order that looks
// different (a change of width alone is not a move); one that cannot is offered disabled. One undo step each, spoken.

import { defName } from '../../derive.ts';
import { defOf } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui, useSay } from '../i18n/index.ts';
import { deskSlots, lastMeasure } from '../desk/layout.ts';
import { closePop, openPop } from './pop.ts';

type Place = { k: number; x: number; y: number; w: number; h: number };
const DIRS = ['top', 'earlier', 'later', 'end'] as const;
type Dir = (typeof DIRS)[number];
const same = (a: Place, b: Place) => Math.abs(a.x - b.x) < 2 && Math.abs(a.y - b.y) < 2;
const tilesNow = () => [...document.querySelectorAll<HTMLElement>('#bento > .tile[data-m]')].filter(t => t.dataset.m !== 'lyrics' && !t.classList.contains('moving')).map(t => t.dataset.m!);

/** Where each move of tile `m` would take it: the place, or null where it cannot go. */
function targets(m: string): Record<Dir, Place | null> & { list: string[] } {
  const g = document.getElementById('bento'), it = lastMeasure()?.items.find(x => x.key === m), all = tilesNow(), list = all.filter(x => x !== m);
  const none = { top: null, earlier: null, later: null, end: null, list };
  if (!g || !it) return none;
  const slots = deskSlots(g, m, it), now = slots.find(s => s.k === all.indexOf(m)); if (!now) return none;
  const moved = slots.filter(s => s.k !== now.k && !same(s, now)), up = moved.filter(s => s.k < now.k), down = moved.filter(s => s.k > now.k);
  return {
    top: up.sort((a, b) => a.k - b.k)[0] ?? null, earlier: [...up].sort((a, b) => b.k - a.k)[0] ?? null,
    later: down.sort((a, b) => a.k - b.k)[0] ?? null, end: [...down].sort((a, b) => b.k - a.k)[0] ?? null, list,
  };
}

function moveTo(m: string, dir: Dir): void {
  const t = targets(m), to = t[dir], d = defOf(m); if (!to || !d) return;
  const store = useDesk.getState(), name = defName(d, store.desk);
  store.move(m, to.k < t.list.length ? t.list[to.k]! : null, `.tile[data-m="${m}"] .grip`);
  requestAnimationFrame(() => {
    const gr = document.querySelector<HTMLElement>(`[data-move="${m}"]`);
    gr?.focus({ preventScroll: true }); gr?.closest('.tile')?.scrollIntoView({ block: 'nearest' });
    const after = tilesNow();
    useDesk.getState().say(ui('tile.moved', { name, dir: ui(`dir.${dir}`), n: after.indexOf(m) + 1, of: after.length }));
  });
}

export function MoveMenu({ m }: { m: string }) {
  const say = useSay(), d = defOf(m), t = targets(m);
  if (!d) return null;
  return (
    <>
      <h4>{say(ui('move.title', { name: defName(d, useDesk.getState().desk) }))}</h4>
      <div className="move-menu" role="menu">
        {DIRS.map(dir => <button key={dir} type="button" role="menuitem" className="small-btn" data-mto={dir} data-mfor={m} disabled={!t[dir]}
          onClick={() => { closePop(); moveTo(m, dir); }}>{say(ui(`move.${dir}`))}</button>)}
      </div>
    </>
  );
}

/* a press on a grip (the arrow keys on it are the slice's, desk/move.ts) */
document.addEventListener('click', e => {
  const b = (e.target as Element).closest?.<HTMLElement>('#bento [data-move]'); if (!b) return;
  const m = b.dataset.move!;
  openPop(b, () => <MoveMenu m={m} />);
});
