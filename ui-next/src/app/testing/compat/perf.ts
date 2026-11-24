// The performance guards' counters in the test layer (tools/behaviour/scenarios/39-perf.mjs): `__tally()` reads what
// the page counted since the last `__tally(true)`, which resets them (tally.ts); `notify` counts the desk store's
// notifications.

import { TALLY, tally } from '../../tally.ts';
import { useDesk } from '../../store/desk.ts';
import { expose } from '../compat.ts';

export function install(): void {
  useDesk.subscribe(() => tally('notify'));
  expose({ __tally: (reset?: boolean) => { const o = { ...TALLY }; if (reset) for (const k of Object.keys(TALLY)) delete TALLY[k]; return o; } });
}
