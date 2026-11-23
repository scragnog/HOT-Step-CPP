// A node bound on the desk: the core's bindPart (setters for exactly what the node binds, each change named by the
// core), its changes committed to the store as one step each. Its children are the renderer's to draw.

import type { Definition, Node } from '../../definitions/types.ts';
import type { PartProps } from '../../definitions/parts.ts';
import { bindPart, type Change, type Desk } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';

const commit = (c: Change) => useDesk.getState().commit(c);

export function bind(def: Definition, node: Node, desk: Desk): Omit<PartProps, 'children'> {
  const props: Partial<PartProps> = bindPart(ix, def, node, desk, commit);
  delete props.children;
  return props as Omit<PartProps, 'children'>;
}
