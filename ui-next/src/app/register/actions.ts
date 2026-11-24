// The named actions a definition's node may run (parts.ts ACTIONS: open the VST3 host, the adapter list, compose a
// caption...), each run by the feature that builds it, registered from its own file (src/app/features/). Every action
// a definition runs has a runner (tools/test-code.ts fails until it does).

import type { ActionName } from '../../definitions/parts.ts';

const RUN = new Map<string, (arg?: string, from?: string) => void>();

/** `run` is given what the user chose when the part passes it (a tag, a file's name), and the part it came from (its
 *  changes then belong to that part's gesture, as its own setters' do). */
export function defineAction(name: ActionName, run: (arg?: string, from?: string) => void): void {
  if (RUN.has(name)) throw new Error(`the action "${name}" is defined twice`);
  RUN.set(name, run);
}
export function runNamed(name: string, arg?: string, from?: string): void {
  const f = RUN.get(name);
  if (!f) throw new Error(`the action "${name}" has no runner: register it with defineAction (tools/test-code.ts)`);
  f(arg, from);
}
