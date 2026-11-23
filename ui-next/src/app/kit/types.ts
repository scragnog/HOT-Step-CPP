// What a part's component is given, and all it is given (docs/09, "How a part is written"): its node's model, worked
// out by the core (src/derive/models.ts partModel: values, words, options, scale, all decided), and the actions that
// change exactly what its node binds. A part has no route to the store: it cannot read another value, write a key its
// node does not bind, or name an undo step (the core names every step from the definitions). Every setter acts on the
// desk as it is when it is called, so a part that waits (a file that loads, a list that arrives) is right by default:
// its change is cleaned, named and checked against the desk in force, and dropped if what it binds has gone.

import type { ReactNode } from 'react';
import type { MeterScale, ParamModel, PartModel, StateModel, Surface } from '../../derive/models.ts';
import type { Slot } from '../../derive/messages.ts';

export type { MeterScale, Slot, Surface };

/** A parameter a part edits: what the core says about it, and how to change it. */
export interface ParamView extends ParamModel {
  /** Sets it to a value the part chose (a choice's option, a switch's other side): one undo step, named by the core. */
  set(value: string | number | boolean): void;
  reset(): void;
  setAuto(on: boolean): void;
  /** The value contract's keys (arrows, Shift, Page, Home, End): true when the key was one of them. */
  press(key: string, shift: boolean): boolean;
  /** A number typed in, as the user typed it: the core reads it, clamps it and says what it did. */
  typed(text: string): void;
  /** Where a hand is along the range, 0 to 1. */
  toFraction(f: number): void;
  /** A hand dragging a scrub token sideways, `px` pixels on from the value `from` it started at. */
  scrub(from: number, px: number, shift: boolean): void;
  /** A hand on a ruler: its place `f` along a scale spanning `span` from 0, on a track `px` pixels wide. */
  along(f: number, span: number, px: number): void;
  /** Text being typed into a checked field: the check's words while it fails, else null. */
  check(text: string): Slot | null;
}

/** A run of changes that is one undo step (a drag, a held key, typing in one field): begun when the hand starts, ended
 *  when it lets go. `sub` tells two runs of one part apart (the XY pad's drag and its keys). */
export interface Gesture { begin(sub?: string): void; end(sub?: string): void }

/** A state its node names (a drop zone's file): what the core says of it, and its named steps (a file's `load` and
 *  `clear`, the steps its definition lists): one undo step each, named by the definition. A state another definition
 *  lends (`lent`) is shown and never applied. */
export interface StateView extends StateModel {
  apply(step: string, next: string | number | boolean | null | object, slots?: { [slot: string]: string | number }): void;
  /** A file offered to it (chosen or dropped): the core takes it as the `load` step, or refuses it and says why. */
  load(name: string, type: string): void;
}

export interface PartView extends Omit<PartModel, 'params' | 'state'> {
  params: ParamView[];
  state: StateView[];
  /** A Quick choice row: choose choice i (every value it sets, one step named by the row). */
  choose(i: number): void;
  /** Runs the node's action i (actions[i]). */
  run(i: number): void;
  /** Runs fn; every set inside it is one undo step. */
  batch(fn: () => void): void;
  /** A bespoke part's hand, worked out by the core (src/derive/instruments.ts partAct): `name` is what the hand did (a
   *  key on the key wheel, a seed typed, a chip pressed), `arg` what it gives. The change is made through the part's
   *  own setters, cleaned and named as every other, or refused and said. */
  act(name: string, arg?: unknown): void;
  gesture: Gesture;
  children?: ReactNode;
}
