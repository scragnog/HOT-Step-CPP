// A named place in the shell, drawing what the work packages fill it with (register/slots.ts).

import { createElement } from 'react';
import { fillsOf, type SlotName } from './slots.ts';

export function Slot({ name }: { name: SlotName }) {
  return <>{fillsOf(name).map((V, i) => createElement(V, { key: i }))}</>;
}
