// A container's drawn children, by their place among its node's children: the core's kids (PartView.kids) are in the
// same order, so a container reads what each child is from its kid and draws the child itself from here.

import { Children, createContext, type ReactNode } from 'react';

export const itemsOf = (children: ReactNode): ReactNode[] => Children.toArray(children);

/** What an LCD's top row lends the big number it holds (Tempo's): the beat lamps, drawn in the number's unit under its
 *  word, and the buttons that work the number (TAP), drawn with its AUTO, as the concept draws them
 *  (25-module-tempo.js). The row gives them; the number draws them where they belong. */
export const LcdSlots = createContext<{ unit?: ReactNode; btns?: ReactNode }>({});

/** What a terminal says when it stands back (a guard with the planner off: "planner off"), lent to its lines, which say
 *  it after their first control, hidden until it holds (25-module-repetition.js term-plan-off). */
export const TermWords = createContext<ReactNode>(null);
