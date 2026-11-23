// The value contract's gestures, for a number part (the concept's 20-controls.js): a key the contract knows moves the
// value and is one step while it is held; a run of wheel steps is one step. Where a key or the wheel takes the value
// is the core's (ParamView.press: keyStep); this only says when a run of changes begins and ends.

import type { KeyboardEvent } from 'react';
import type { Gesture, ParamView } from './types.ts';

/** A key on a number control: true when it was one of the contract's (and the value moved). A held key is one step.
 *  Delete or Backspace returns it to its default, as a double-click does (S-18). */
export function onValueKey(e: KeyboardEvent, p: ParamView, g: Gesture): boolean {
  if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); g.end('key'); p.reset(); return true; }
  if (!/^(Arrow(Up|Down|Left|Right)|Page(Up|Down)|Home|End)$/.test(e.key)) return false;
  e.preventDefault();
  if (!e.repeat) g.end('key');
  g.begin('key');
  return p.press(e.key, e.shiftKey);
}
export const onValueKeyUp = (g: Gesture): void => g.end('key');

/** A wheel run on one control: steps within this long of each other are one step. */
const WHEEL_RUN = 600;
const runs = new WeakMap<Gesture, number>();
export function onWheelStep(up: boolean, shift: boolean, p: ParamView, g: Gesture): void {
  g.begin('wheel');
  p.press(up ? 'ArrowUp' : 'ArrowDown', shift);
  clearTimeout(runs.get(g)); runs.set(g, window.setTimeout(() => g.end('wheel'), WHEEL_RUN));
}
