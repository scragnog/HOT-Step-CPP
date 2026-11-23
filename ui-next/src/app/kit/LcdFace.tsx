// What an LCD prints: its words in capitals, and a unit after a number in small type after a thin space ("-14 LUFS",
// the concept's lcdFace, 22-pick.js). Drawing only: the words are the core's, and where the unit stands is the core's
// splitUnit.

import { splitUnit } from '../../derive/strings.ts';
import type { Slot } from '../../derive/messages.ts';
import { useSay } from '../i18n/index.ts';

/* U+2009 THIN SPACE */
const THIN = String.fromCharCode(0x2009);

export function LcdFace({ text }: { text: Slot }) {
  const say = useSay(), { main, unit } = splitUnit(say(text, { upper: true }));
  return unit ? <><b>{main}</b><small>{THIN + unit}</small></> : <b>{main}</b>;
}
