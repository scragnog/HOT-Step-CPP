// Whether Quick's fold stands open (the concept's S.qAll): view state kept with the session (kit/session.ts), never a
// step. Which modules fold, and what the button says, are the core's (derive/fold.ts); the desk asks with this.

import { keptFlag } from '../kit/session.ts';

export const deskFold = keptFlag('desk.fold');
