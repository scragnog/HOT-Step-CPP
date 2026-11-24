// Two stand-in modules (test build only), for the scenarios and visual states of Quick's fold: no engine has enough
// modules Quick shows for the fold to take three, so the concept's suite registers two of its own (defineModule('tx1',
// { name: 'Stand-in 1', w: 1, keys: [] }) and its face, `<p>stand-in</p>`). Here they are definitions like any other,
// indexed with the real ones in the test build only (model/index.ts) and offered nowhere; the test layer's
// defineModule (testing/compat/fold.ts) lets the concept's scenarios name them.

import type { ModuleDef } from '../../definitions/types.ts';
import { ENGINES } from '../../definitions/engines.ts';

export const STAND_IN_IDS = ['tx1', 'tx2'] as const;

/** The stand-ins: a function, so that no other build keeps any of it. */
export const standIns = (): ModuleDef[] => STAND_IN_IDS.map((id, i) => ({
  kind: 'module', id, name: `Stand-in ${i + 1}`, desc: 'A stand-in module (test build).', at: 300 + i, category: 'Engine',
  engines: ENGINES.map(e => e.id), owner: 'song', stored: 'settings', size: { w: 1 }, params: [],
  face: [{ part: 'note', text: 'stand-in', opts: { look: 'plain' } }],
}));
