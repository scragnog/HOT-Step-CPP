// The named actions a definition's node may run (parts.ts ACTIONS: open the VST3 host, the adapter list, compose a
// caption...), each run by the feature that builds it, registered from its own file (src/app/features/). An action no
// package has built yet says what it will do.

import type { ActionName } from '../../definitions/parts.ts';
import { useDesk } from '../store/desk.ts';
import { ui } from '../i18n/index.ts';

const RUN = new Map<string, () => void>();

export function defineAction(name: ActionName, run: () => void): void {
  if (RUN.has(name)) throw new Error(`the action "${name}" is defined twice`);
  RUN.set(name, run);
}
export function runNamed(name: string): void {
  const f = RUN.get(name);
  if (f) f(); else useDesk.getState().showToast({ text: ui(`action.${name}`) });
}
