// The generic renderer: walks a definition's face and draws each node with its part. It knows no module and no part.
// For each node it binds the node (bind.ts: the core's model of it, and actions for exactly what it binds), looks its
// part up (kit/define.ts) and draws it with its children, each drawn the same way on the surface the part gives
// them; a node that stands back is wrapped as the concept wraps it. Each node redraws on its own when its model
// changes, and only then.

import { createElement, memo, useMemo, type ReactNode } from 'react';
import type { Definition, Node } from '../../definitions/types.ts';
import { nodeKeys, type CellWords, type Surface } from '../../derive.ts';
import { ui, useSay } from '../i18n/index.ts';
import { partView } from '../kit/define.ts';
import { usePartView } from './bind.ts';

/** The surface a container gives its children: a rack, a tile's foot, a well and an LCD are surfaces of their own;
 *  every other container passes its own surface on. */
const SURFACE: Record<string, Surface> = { rack: 'rack', foot: 'foot', well: 'well', lcd: 'lcd' };

interface NodeProps { def: Definition; node: Node; nodeKey: string; keys: Map<Node, string>; surface: Surface; cell?: CellWords }

/** One node: its part, bound, with its children. */
export const NodeView = memo(function NodeView({ def, node, nodeKey, keys, surface, cell }: NodeProps): ReactNode {
  const say = useSay(), view = usePartView(def, node, nodeKey, surface, cell);
  if (!view) return null;
  const View = partView(node.part), inner = SURFACE[node.part] ?? surface;
  const kids = (node.children ?? []).map((c, i) => <NodeView key={i} def={def} node={c} nodeKey={keys.get(c) ?? ''} keys={keys} surface={inner} />);
  const body = View ? createElement(View, view, kids) : <span className="lbl" data-seam={node.part}>{say(ui('later'))}</span>;
  return view.dim ? <span className="dimmed" title={say(view.dim)}>{body}</span> : body;
});

/** A definition's face: every top-level node in order, on a surface. */
export function Face({ def, surface, only }: { def: Definition; surface: Surface; only?: (n: Node) => boolean }) {
  const keys = useMemo(() => nodeKeys(def), [def]);
  const face = 'face' in def ? def.face ?? [] : [];
  return <>{face.filter(n => !only || only(n)).map((n, i) => <NodeView key={i} def={def} node={n} nodeKey={keys.get(n) ?? ''} keys={keys} surface={surface} />)}</>;
}
