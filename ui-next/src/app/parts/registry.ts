// Every part the page can draw, by its name in the inventory (src/definitions/parts.ts PARTS). The generic renderer
// looks a node's part up here and knows nothing else about it. Adding a part: its entry in PARTS, its component, its
// line here (docs/06-how-to.md, "Add a new part"). A part the inventory lists and no package has built yet is drawn as
// a marked seam (Missing), so a definition that names it still renders, and says what is missing.

import type { ComponentType } from 'react';
import type { PartName } from '../../definitions/parts.ts';
import type { ViewProps } from './types.ts';
import { Knob } from './Knob.tsx';
import { LedButton } from './LedButton.tsx';
import { PowerButton } from './PowerButton.tsx';
import { TextArea } from './TextArea.tsx';
import { XyPad } from './XyPad.tsx';
import { Pick } from './Pick.tsx';
import { TextField } from './TextField.tsx';
import { Segmented } from './Segmented.tsx';
import { ChoiceRow } from './ChoiceRow.tsx';
import { Meter } from './Meter.tsx';
import { Labelled, Rack, Row } from './containers.tsx';
import { Button, LcdValue, Note, UnitLcd } from './readouts.tsx';

export const PART_VIEWS: Partial<Record<PartName, ComponentType<ViewProps>>> = {
  knob: Knob,
  ledButton: LedButton,
  powerButton: PowerButton,
  textArea: TextArea,
  xyPad: XyPad,
  pick: Pick,
  textField: TextField,
  segmented: Segmented,
  choiceRow: ChoiceRow,
  meter: Meter,
  row: Row,
  rack: Rack,
  labelled: Labelled,
  note: Note,
  lcdValue: LcdValue,
  unitLcd: UnitLcd,
  button: Button,
};
