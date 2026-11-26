// Quick's fold (the concept's renderDesk, 30-desk.js; A-28, Q-05): past six modules, Quick shows six and folds the rest
// behind one button that names them, but only when the desk holds nine or more and three or more would be folded
// (folding one or two would hide less than the button costs). Never folded: the board's module (it processes every
// song), a source module (it works on the song you chose), and any module holding a value that is not its default, so
// the fold hides nothing you have set. The button says what it hides; open, it says it can fold them again. Whether it
// stands open is the page's (the session's) and is given here. Nothing here has a branch for any one module.

import type { ModuleDef } from '../definitions/types.ts';
import { type Desk, type Index, changed } from './core.ts';
import { EN, list, msg, type Slot } from './messages.ts';
import { tileName } from './desk.ts';

/** The rule's three numbers: Quick shows this many modules that may fold, folds only a desk of this many tiles or
 *  more, and only when at least this many would be folded. (Nine is six and three, so the second follows from the
 *  other two; it is the concept's first test, kept so that the rule reads as the concept states it.) */
export const FOLD = { shows: 6, from: 9, least: 3 } as const;

/** The core's words for the fold's button, added to its table (messages.ts EN) as this file loads. */
const WORDS: { [id: string]: string } = {
  'desk.fold.more': 'Show {n} more: {names}',
  'desk.fold.fewer': 'Show fewer modules',
};
for (const [id, text] of Object.entries(WORDS)) {
  if (id in EN) throw new Error(`the core's word ${id} is defined twice`);
  EN[id] = text;
}

/** A module the fold never takes: the board's, a source, or one holding a changed value. */
export const neverFolded = (ix: Index, d: ModuleDef, desk: Desk): boolean => !!d.board || !!d.source || d.params.some(p => changed(ix, p.key, desk));

/** What the fold does to the desk's tiles (`live`: the tiles the tier shows on this engine, in the user's order): the
 *  modules it folds, in that order (none outside Quick, or when the rule does not fold). */
export function foldedOf(ix: Index, desk: Desk, live: readonly string[]): string[] {
  if (desk.tier !== 'quick' || live.length < FOLD.from) return [];
  const defs = live.map(id => ix.defs.find((d): d is ModuleDef => d.id === id && d.kind === 'module')).filter((d): d is ModuleDef => !!d);
  const folded = defs.filter(d => !neverFolded(ix, d, desk)).slice(FOLD.shows).map(d => d.id);
  return folded.length >= FOLD.least ? folded : [];
}

/** The fold as the desk draws it: whether it stands open, the modules it holds (drawn only while it is open), and its
 *  button's words, which name them while they are folded away. */
export interface FoldModel { open: boolean; ids: string[]; says: Slot }

/** The fold on this desk, open or not, and the tiles drawn with it: null when nothing folds. */
export function foldModel(ix: Index, desk: Desk, live: readonly string[], open: boolean): { tiles: string[]; fold: FoldModel | null } {
  const ids = foldedOf(ix, desk, live);
  if (!ids.length) return { tiles: [...live], fold: null };
  const names = ids.map(id => tileName(ix.defs.find(d => d.id === id)!, desk));
  return {
    tiles: open ? [...live] : live.filter(id => !ids.includes(id)),
    fold: { open, ids, says: open ? msg('desk.fold.fewer') : msg('desk.fold.more', { n: ids.length, names: list(names) }) },
  };
}
