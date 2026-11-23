// What an LCD prints for a value: its words in capitals, and a unit after a number in small type, after a thin space
// ("-14 LUFS", the concept's lcdFace, 22-pick.js). Drawing only: the words are the core's (lcdMsg, optionsOf).

import { UNITS } from '../../definitions/parts.ts';

/* U+2009 THIN SPACE */
const THIN = String.fromCharCode(0x2009);

export function LcdFace({ text }: { text: string }) {
  const up = text.toLocaleUpperCase();
  const unit = Object.keys(UNITS).find(u => u.length > 1 && up.endsWith(' ' + u.toLocaleUpperCase()) && /^[−-]?[\d.]+$/.test(up.slice(0, -u.length - 1)));
  return unit ? <><b>{up.slice(0, -unit.length - 1)}</b><small>{THIN + unit}</small></> : <b>{up}</b>;
}
