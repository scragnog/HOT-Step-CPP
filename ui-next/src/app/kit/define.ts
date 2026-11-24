// The parts the page can draw, by their name in the inventory (src/definitions/parts.ts PARTS). A part's file
// registers its component here when it is loaded (src/app/parts/*.tsx, all loaded by load.ts); the renderer looks a
// node's part up and knows nothing else about it. A part the inventory lists and no package has built yet is drawn as
// a marked seam, so a definition that names it still renders.

//
// A container says, when it registers, the surface it gives its children (a rack, a well, an LCD and a tile's foot are
// surfaces of their own; a terminal's lines, a rack group, a deck and an LCD's row too), so the renderer has no list of
// containers; and a part that draws its own standing back (a terminal fading its lines, a group hatched, a recorder's
// is-off) says so, and the renderer does not wrap it.

import type { ComponentType } from 'react';
import type { PartName } from '../../definitions/parts.ts';
import type { PartView, Surface } from './types.ts';

/** How a part stands among the others: the surface it gives its children (by its node's options, when its look
 *  decides it), and whether it draws its own standing back. */
export interface PartHow { surface?: Surface | ((opts: Record<string, unknown>) => Surface | undefined); dims?: boolean; kidsDim?: boolean }

const VIEWS = new Map<string, ComponentType<PartView>>(), HOW = new Map<string, PartHow>();

/** Registers a part's component under its name in the inventory (one file may register several). */
export function definePart(name: PartName, view: ComponentType<PartView>, how?: PartHow): void {
  if (VIEWS.has(name)) throw new Error(`the part "${name}" is defined twice`);
  VIEWS.set(name, view);
  if (how) HOW.set(name, how);
}
/** The component of a part, or undefined while no package has built it. */
export const partView = (name: string): ComponentType<PartView> | undefined => VIEWS.get(name);
/** The surface a part gives its children, or undefined: they stand on its own. */
export const surfaceOf = (name: string, opts: Record<string, unknown>): Surface | undefined => { const s = HOW.get(name)?.surface; return typeof s === 'function' ? s(opts) : s; };
/** The part draws its own standing back: the renderer leaves it unwrapped. */
export const drawsDim = (name: string): boolean => !!HOW.get(name)?.dims;
/** The container draws its children's standing back (a deck's replay that is not used, said on its own line). */
export const drawsKidsDim = (name: string): boolean => !!HOW.get(name)?.kidsDim;
/** Every part built so far, for the checks. */
export const partsBuilt = (): string[] => [...VIEWS.keys()].sort();
