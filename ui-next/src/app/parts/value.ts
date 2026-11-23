// The value contract every number control keeps (the concept's 20-controls.js): arrows step, Shift steps ten, Page Up
// and Page Down step ten, Home and End go to the ends, double-click resets; a drag, a held key or a run of wheel steps
// on one control is one undo step. A control only says where the hand or the key moved it: the parameter's setter
// cleans the value on the engine's range and grid and the core names the step.

import type { KeyboardEvent } from 'react';
import type { BoundParam } from '../../definitions/parts.ts';
import { useDesk } from '../store/desk.ts';

/** Where a key takes a number, or null when the key is not one of the contract's. */
export function keyTarget(e: Pick<KeyboardEvent, 'key' | 'shiftKey'>, p: BoundParam): number | null {
  const [min, max, step] = p.face.range ?? [0, 1, 0.01];
  const v = typeof p.value === 'number' ? p.value : Number(p.value);
  if (e.key === 'Home') return min;
  if (e.key === 'End') return max;
  const d = ({ ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 10, PageDown: -10 } as Record<string, number>)[e.key];
  if (!d) return null;
  return +(v + d * (e.shiftKey && Math.abs(d) === 1 ? 10 : 1) * step).toFixed(10);
}

/** A run of changes that is one undo step: begun by a press, a held key or a drag, ended when the hand lets go. */
export const gesture = {
  begin: (tag: string) => useDesk.getState().beginGesture(tag),
  end: (tag: string) => useDesk.getState().endGesture(tag),
};

/** A wheel run on one control: steps within this long of each other are one step. */
const WHEEL_RUN = 600;
const wheelEnds = new Map<string, number>();
export function wheelStep(tag: string, fn: () => void): void {
  gesture.begin(tag); fn();
  clearTimeout(wheelEnds.get(tag)); wheelEnds.set(tag, window.setTimeout(() => gesture.end(tag), WHEEL_RUN));
}

/** The keyboard side of the contract, for a control's keydown: steps (a held key is one step), and true when the key
 *  was the contract's. */
export function onValueKey(e: KeyboardEvent, p: BoundParam): boolean {
  const to = keyTarget(e, p);
  if (to === null) return false;
  e.preventDefault();
  const tag = `key:${p.key}`;
  if (!e.repeat) gesture.end(tag);
  gesture.begin(tag);
  p.set(to);
  return true;
}
export const onValueKeyUp = (p: BoundParam) => gesture.end(`key:${p.key}`);
