// What a part's component is given: the core's PartProps (src/definitions/parts.ts, built by bindPart() for its node
// on the desk) and nothing about values beyond them, plus what the view needs to draw it: its definition, its node's
// translation key, its children already drawn, and, for a part the shell lays out (a pedal's knob), how the shell
// places it. docs/09-architecture.md, "How a part component is written".

import type { ReactNode } from 'react';
import type { Definition } from '../../definitions/types.ts';
import type { PartProps } from '../../definitions/parts.ts';
import type { Slot } from '../../derive/messages.ts';

export interface Cell {
  /** The small knob (rack rows and pedals). */
  small?: boolean;
  /** The read-out shows the number alone, its unit on its own line under it (a pedal's dB, Hz, LUFS). */
  bare?: boolean;
  /** Its printed label, when the shell gives it (a pedal knob's short name). */
  label?: Slot;
  /** Its accessible name, when it is not the printed label (a pedal knob's full name). */
  aria?: Slot;
  /** Its tip, when the shell gives it. */
  title?: Slot;
}

export interface ViewProps extends Omit<PartProps, 'children'> {
  def: Definition;
  /** The node's translation key (src/derive/strings.ts nodeKeys); '' for a part the shell lays out. */
  nodeKey: string;
  children?: ReactNode;
  cell?: Cell;
}
