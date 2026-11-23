// The parts the page can draw, by their name in the inventory (src/definitions/parts.ts PARTS). A part's file
// registers its component here when it is loaded (src/app/parts/*.tsx, all loaded by load.ts); the renderer looks a
// node's part up and knows nothing else about it. A part the inventory lists and no package has built yet is drawn as
// a marked seam, so a definition that names it still renders.

import type { ComponentType } from 'react';
import type { PartName } from '../../definitions/parts.ts';
import type { PartView } from './types.ts';

const VIEWS = new Map<string, ComponentType<PartView>>();

/** Registers a part's component under its name in the inventory (one file may register several). */
export function definePart(name: PartName, view: ComponentType<PartView>): void {
  if (VIEWS.has(name)) throw new Error(`the part "${name}" is defined twice`);
  VIEWS.set(name, view);
}
/** The component of a part, or undefined while no package has built it. */
export const partView = (name: string): ComponentType<PartView> | undefined => VIEWS.get(name);
/** Every part built so far, for the checks. */
export const partsBuilt = (): string[] => [...VIEWS.keys()].sort();
