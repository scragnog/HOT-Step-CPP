// The generic renderer: walks a definition's face and draws each node with its part. It knows no module and no part.
// For each node it binds the node (bind.ts: the core's model of it, and actions for exactly what it binds), looks its
// part up (kit/define.ts) and draws it with its children, each drawn the same way on the surface the part gives
// them; a node that stands back is wrapped as the concept wraps it. Each node redraws on its own when its model
// changes, and only then.

import { createElement, memo, useMemo, type ReactNode } from 'react';
import type { Definition, Node } from '../../definitions/types.ts';
import { nodeKeys, type CellWords, type Surface } from '../../derive.ts';
import { useSay } from '../i18n/index.ts';
import { drawsDim, drawsKidsDim, partView, surfaceOf } from '../kit/define.ts';
import { usePartView } from './bind.ts';

interface NodeProps { def: Definition; node: Node; nodeKey: string; keys: Map<Node, string>; surface: Surface; cell?: CellWords; bare?: boolean }

/** One node: its part, bound, with its children. */
export const NodeView = memo(function NodeView({ def, node, nodeKey, keys, surface, cell, bare }: NodeProps): ReactNode {
  const say = useSay(), view = usePartView(def, node, nodeKey, surface, cell);
  if (!view) return null;
  /* the surface a container gives its children is its own to say (definePart); every other passes its surface on */
  const View = partView(node.part), inner = surfaceOf(node.part, view.opts) ?? surface;
  const kids = (node.children ?? []).map((c, i) => <NodeView key={i} def={def} node={c} nodeKey={keys.get(c) ?? ''} keys={keys} surface={inner} bare={drawsKidsDim(node.part)} />);
  /* every part a definition names has a component (tools/test-code.ts fails until it does) */
  if (!View) return null;
  const body = createElement(View, view, kids);
  return view.dim && !drawsDim(node.part) && !bare ? <span className="dimmed" title={say(view.dim)}>{body}</span> : body;
});

/** A definition's face: every top-level node in order, on a surface. */
export function Face({ def, surface, only }: { def: Definition; surface: Surface; only?: (n: Node) => boolean }) {
  const keys = useMemo(() => nodeKeys(def), [def]);
  const face = 'face' in def ? def.face ?? [] : [];
  return <>{face.filter(n => !only || only(n)).map((n, i) => <NodeView key={i} def={def} node={n} nodeKey={keys.get(n) ?? ''} keys={keys} surface={surface} />)}</>;
}
