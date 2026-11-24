// A song value an engine cannot take is held for the engine it came from (the engines' book, derive/engine.ts): its
// control says so where it is ("Your 8 takes return on ACE-Step 1.5", the concept's heldNote, 13-engine-state.js).
// The desk is told what the book holds whenever the book changes, so the core's models can say it (RD-03).

import { useDesk } from '../store/desk.ts';

useDesk.subscribe((st, prev) => { if (st.book.hold !== prev.book.hold) st.told({ hold: st.book.hold }); });
useDesk.getState().told({ hold: useDesk.getState().book.hold });
