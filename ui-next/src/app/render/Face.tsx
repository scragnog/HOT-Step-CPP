// The generic renderer: walks a definition's face for the desk's tier and engine and draws each node with its part.
// It knows no module. For each node it asks the core whether it is drawn (nodeShown), binds it (bindPart: setters for
// exactly what the node binds, each change named by the core and committed to the store), and hands the part its
// props and its children, drawn the same way. A node that stands back (dimOf) is wrapped as the concept wraps it.

import { useMemo, type ReactNode } from 'react';
import type { Definition, Node } from '../../definitions/types.ts';
import { dimOf, nodeKeys, nodeShown, type Desk } from '../../derive.ts';
import { bind } from './bind.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { useSay } from '../i18n/index.ts';
import { PART_VIEWS } from '../parts/registry.ts';
import type { Cell } from '../parts/types.ts';

/** One node: its part, bound, with its children. */
export function NodeView({ def, node, keys, desk, cell }: { def: Definition; node: Node; keys: Map<Node, string>; desk: Desk; cell?: Cell }): ReactNode {
  const say = useSay();
  if (!nodeShown(ix, node, desk)) return null;
  const View = PART_VIEWS[node.part];
  const key = keys.get(node) ?? '';
  const props = bind(def, node, desk);
  const kids = (node.children ?? []).map((c, i) => <NodeView key={i} def={def} node={c} keys={keys} desk={desk} />);
  const body = View
    ? <View {...props} def={def} nodeKey={key} {...(cell ? { cell } : {})}>{kids}</View>
    : <span className="lbl" data-seam={node.part}>{say({ id: 'ui.later' })}</span>;
  const dim = key ? dimOf(ix, node, desk, key) : null;
  return dim ? <span className="dimmed" title={say(dim)}>{body}</span> : body;
}

/** A definition's face at the desk's tier: every top-level node in order. */
export function Face({ def, only }: { def: Definition; only?: (n: Node) => boolean }) {
  const desk = useDesk(s => s.desk);
  const keys = useMemo(() => nodeKeys(def), [def]);
  const face = 'face' in def ? def.face ?? [] : [];
  return <>{face.filter(n => !only || only(n)).map((n, i) => <NodeView key={i} def={def} node={n} keys={keys} desk={desk} />)}</>;
}
