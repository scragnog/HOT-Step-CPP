// Generation (docs/09, package 6): the go bar and Latest in their places in the shell, the generation service's songs
// and queue loaded, Ctrl+Enter, Space for the player, the document's title while songs render, and a warning before
// leaving while they do (F-37).

import { fill } from '../register/slots.ts';
import { enterIsOwn, onKey, isTextField } from '../register/keys.ts';
import { topLayer } from './layers.ts';
import { GoBarFill } from '../generate/Live.tsx';
import { Latest } from '../generate/Latest.tsx';
import { generate } from '../generate/actions.ts';
import { bootGenerate, pausePlayer, useGen } from '../store/generate.ts';

fill('centre.gobar', GoBarFill);
fill('centre.latest', Latest);
void bootGenerate();

/** A layer open over Create (features/layers.ts: the engine panel is one), under which no page shortcut acts. */
const layerOpen = () => !!topLayer() || !!document.getElementById('app')?.hidden;

/* Ctrl+Enter is Generate from every focus position on Create with no layer open, a workflow row, a revision row and a
   Modules card included: it is heard before any row's or card's own Enter (F-07); a held key queues once. A field
   whose Enter is its own (register/keys.ts ownEnter: a name being saved or renamed) keeps it, as the concept's stops
   it there (RC-09) */
addEventListener('keydown', e => {
  if (e.key !== 'Enter' || !(e.ctrlKey || e.metaKey) || e.altKey || e.defaultPrevented || layerOpen() || enterIsOwn(e.target as Element | null)) return;
  e.preventDefault(); e.stopPropagation();
  if (!e.repeat) generate();
}, true);

/* Space plays and pauses from anywhere on the page that does not use Space itself: not a text field, a button, a
   control that takes keys, and not under a layer (C-27) */
onKey({
  id: 'player-space',
  when: e => {
    if (e.key !== ' ' || e.repeat || !useGen.getState().playing || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey || layerOpen()) return false;
    const t = e.target as Element | null;
    if (t && t !== document.body && !t.closest?.('#player') && (isTextField(t) || t.closest?.('button, a, [role="button"], [role="slider"], [role="radio"], [role="menuitem"], [role="option"], [tabindex], select, label'))) return false;
    return !t?.closest?.('#player button');
  },
  run: e => { e.preventDefault(); pausePlayer(); },
});
