// Quick's fold opens by itself where a person is sent to something it holds (the concept's flash, 30-desk.js, Q-05): a
// ring aimed at a folded tile (adding a module that is already on the desk) opens the fold first, and so does a module
// that arrives on the desk on its own (added, or brought back by Undo or Redo) only to be folded away, so nothing you
// have just put there is hidden. Which modules are folded is the core's (deskModel's fold).

import { deskModel } from '../../derive.ts';
import { ix } from '../model/index.ts';
import { useDesk } from '../store/desk.ts';
import { deskFold } from '../store/fold.ts';

useDesk.subscribe((s, prev) => {
  if (deskFold.get() || s.desk.tier !== 'quick') return;
  const rung = s.flash !== prev.flash && s.flash?.ring === 'flash' ? s.flash.id : null;
  const came = s.mods !== prev.mods ? s.mods.filter(id => !prev.mods.includes(id)) : [];
  const to = rung ?? (came.length === 1 ? came[0]! : null);
  if (to && deskModel(ix, s.desk, s.mods).fold?.ids.includes(to)) deskFold.set(true);
});
