// The overlays' own state (package 8): the picker (open, closing, its filter, its category, what its info line says,
// the snapshot the keys stand on) and the popover (what it holds and what opened it). View state only: nothing here is
// about a value, nothing is kept between visits, and Undo covers none of it.

import { create } from 'zustand';
import type { ReactNode } from 'react';
import type { PickerCat } from '../../derive.ts';
import type { Slot } from '../../derive.ts';

/** What the picker's info line says: the hint (what the keys do now), a snapshot's name and description (pointed at
 *  or focused, or the first that a typed filter shows), what was just added. */
export type PickerInfo = { kind: 'hint' } | { kind: 'first' } | { kind: 'item'; id: string } | { kind: 'added'; id: string; lab?: boolean };

export interface PickerState {
  open: boolean;
  /** Its exit is playing: F9 now opens it again (F-29). */
  closing: boolean;
  q: string;
  cat: PickerCat;
  info: PickerInfo;
  /** The snapshot the keys stand on (the wall's one Tab stop). */
  rove: string | null;
  /** A drag from the wall is on its way to the desk: the picker fades to a peek. */
  peek: boolean;
}
export const usePicker = create<PickerState>()(() => ({ open: false, closing: false, q: '', cat: 'All', info: { kind: 'hint' }, rove: null, peek: false }));

/** Where a popover stands against its anchor (placeFloat): above first, or above a whole element (`over`). */
export interface PlaceOpts { prefer?: 'up'; over?: HTMLElement; alignEnd?: boolean }
/** How a popover opens (pop.ts openPop): its accessible name, where it stands, whether it is the fixed panel that stays
 *  in place as the page scrolls (the drafts above the go bar: `.fixedpop`), and what takes focus (a selector inside it;
 *  else its pressed or primary button, else its first control). */
export interface PopOpts { label?: Slot; place?: PlaceOpts; fixed?: boolean; focus?: string }
export interface PopState {
  /** What opened it: it toggles from there, and focus goes back there. */
  anchor: HTMLElement | null;
  /** What it holds: drawn into #pop. */
  body: (() => ReactNode) | null;
  /** Its accessible name. */
  label: Slot | null;
  place: PlaceOpts;
  fixed: boolean;
  focus: string | null;
  /** Counts each opening, so the same body opened again is placed and focused again. */
  n: number;
}
export const usePop = create<PopState>()(() => ({ anchor: null, body: null, label: null, place: {}, fixed: false, focus: null, n: 0 }));
