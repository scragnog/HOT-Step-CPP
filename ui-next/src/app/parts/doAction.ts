// Running an action of a node, a strip or an idle reason. What it does is the core's (runAction): a change, committed
// as every step is; going to a tier and ringing controls; adding a module; or a named action the shell runs (the
// handlers for parts.ts ACTIONS, below).

import type { Action, Definition, Node } from '../../definitions/types.ts';
import { runAction } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';
import { setTierFromView } from '../shell/tier.ts';

/** The named actions the shell runs (parts.ts ACTIONS). The slice builds none of them yet: each says what it will do. */
function runNamed(name: string): void {
  useDesk.getState().showToast({ text: ui(`action.${name}`) });
}

/** Runs an action of a node, a strip or an idle reason on the desk. */
export function doAction(d: Definition, a: Action, node?: Node): void {
  const s = useDesk.getState(), did = runAction(ix, d, a, s.desk, node);
  if ('change' in did) s.commit(did.change);
  else if ('go' in did) setTierFromView(did.go, { ring: did.ring });
  else if ('add' in did) s.add(did.add);
  else runNamed(did.call);
}

