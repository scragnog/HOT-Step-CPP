// The core, indexed once for the page. Everything the view knows about a module, a pedal, a side unit or a value
// comes from here and from src/derive.ts; nothing in src/app restates it.

import { DEFINITIONS } from '../../definitions/index.ts';
import type { Definition, ModuleDef } from '../../definitions/types.ts';
import { indexOf } from '../../derive.ts';

export const ix = indexOf(DEFINITIONS);

const BY_ID = new Map<string, Definition>(DEFINITIONS.map(d => [d.id, d]));
/** A definition by its id (or a former id, `was`). */
export const defOf = (id: string): Definition | undefined => BY_ID.get(id) ?? BY_ID.get(ix.formerIds.get(id) ?? '');

/** The modules built into the page (the prompt, the lyric sheet, the models): always on the desk, never tiles. */
export const BUILT_IN: readonly string[] = DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id);
/** The module that holds the board (the Pedalboard). */
export const BOARD = DEFINITIONS.find((d): d is ModuleDef => d.kind === 'module' && !!d.board)!;
/** Pedals, then side units, each in their place on the board (`at`). */
export const PEDALS = DEFINITIONS.filter(d => d.kind === 'pedal').sort((a, b) => a.at - b.at);
export const UNITS = DEFINITIONS.filter(d => d.kind === 'unit').sort((a, b) => a.at - b.at);
/** Everything the Modules panel offers: the modules (not built in, not a source, not the board), then the pedals,
 *  then the side units. */
export const OFFERED: readonly Definition[] = [
  ...DEFINITIONS.filter((d): d is ModuleDef => d.kind === 'module' && !d.builtIn && !d.source && !d.board).sort((a, b) => a.at - b.at),
  ...PEDALS, ...UNITS,
];
/** Board order: pedals in chain order, then side units. */
export const boardOrder = (ids: Iterable<string>): string[] => {
  const set = new Set(ids);
  return [...PEDALS, ...UNITS].filter(d => set.has(d.id)).map(d => d.id);
};
