# The React layer: architecture, the slice, and the plan for the rest

This page is for the people who build the rest of the Create page and for the maintainer. It says how the React layer
sits on the core ([05](05-what-the-definition-drives.md)), what the first slice built and how it was proved against
the concept, what the page does differently from the concept's markup and why, and how the remaining work splits into
packages that can be built in parallel.

The one rule, from [05](05-what-the-definition-drives.md) "What the view may decide": **the view decides nothing about
a value.** A component draws what the core returns and hands back what the user did. If a component needs to know a
default, a range, a label, an undo name, whether something is changed or in force, or what is sent, it asks the core;
if the core cannot answer, the answer is added to the core with a test (tools/test-view.ts), never to the component.

## What is built

The slice is a working vertical cut of the page:

- the page shell: the top bar (brand, navigation stubs, the engine button and a panel with the five engines, the VRAM
  meter's frame, the theme switch, the notes button), the three panes with the Workflows panel (a shell) and the
  Modules panel (every module, pedal and side unit as a card with its drawing; a click, Enter or Space adds it), the
  desk bar (the drawers' buttons, the tier switch, Saved, Undo and Redo), the go bar's frame and Latest's frame;
- the desk: tiles in the user's order, laid out by the concept's rule (a pure function, `src/app/desk/pack.ts`,
  checked against 90 layouts the concept page chose), Lab modules hidden outside Lab with the note that names them,
  modules and pedals the engine does not use parked, the empty desk, removing (with Undo in the toast, focus to the
  neighbour) and moving a tile from the keyboard (arrows on the grip, one visible step, refused moves said);
- every module the coverage page classes as built from shared parts only (Avoid, Guidance plugin, Throughput, the
  Pedalboard) rendered by the generic renderer at every tier on every engine that has it, and the Pedalboard with every
  such pedal (Tiled decoder, MM3 latent refiner, StableStep, PP-VAE, Spectral lifter, Vocal naturalizer, Pre-VST gain,
  VST3 chain, Final normalizer) and the side unit Whisper lyrics, with the cables, balanced rows and snapped heights;
- the three tiers, structurally (what each tier shows); undo and redo for every change; the autosave and its restore;
  light and dark; the five desktop widths, with the panels as drawers at 1180px and below;
- every part the slice has drawn right on every surface a definition can put it (a tile's body, a row, a rack, a well,
  an LCD, a tile's foot, a labelled pairing, a pedal, a side unit), and the registration points the packages add to.

Not built, and left as seams: the prompt card and the lyric sheet, the bespoke parts and the modules made of them, the
engine panel's models and the VRAM figures, generation, the go bar's contents, Latest and the player,
workflows and revisions, drafts, the remaining overlays (score modal, song menu), and the
choreographed tier transition (a View Transition in the concept; here the
tier changes at once). "The plan for the rest" below divides them into packages.

## The folder

```
ui-next/
  index.html               the concept's document skeleton, with a doctype (standards mode, as the real application's
                           page), the overlays' containers and the one #dd
  vite.config.ts           three builds: dev (dist/), --mode file (dist-file/index.html, one self-contained file),
                           --mode test (dist-test/index.html, the same file with the test layer and the parts gallery);
                           HS_SWITCH=<name> builds the test page with one feature switched off or one defect planted
                           (dist-switch/)
  eslint.config.js         the existing frontend's lint rules and the React layer's two (below, "The rules the lint keeps")
  src/
    definitions/, derive/  the core; derive/models.ts is what the view draws (below, "How a part is written")
    app/
      main.tsx             entry: styles, translation, load.ts, the engine, the test layer in the test build only
      load.ts              loads every file of features/ and parts/: what they register is the page's
      App.tsx              rooted in #app; portals into the top bar, the skip link, the toast and the live region
      kit/                 the part kit: everything a part may use (definePart, PartView, useSay, Icon, LcdFace, the
                           value contract's gestures)
      parts/               one file per part (or a few small ones together); each registers itself with definePart
      render/              Face.tsx (the generic renderer), Tile.tsx, bind.ts (a node's model and its actions),
                           useModel.ts (a view's subscription, redrawn only when its model changes)
      register/            the registration points: keys.ts, slots.ts and Slot.tsx, actions.ts, undoable.ts, services.ts
      features/            what the page does that is not a part: each file registers keys, slot fills, named actions,
                           undoable state (undo-keys, tier-keys, drawers, motion-offscreen)
      model/               the core indexed once (ix; the test build adds the parts gallery), definitions by id
      store/desk.ts        the desk store: desk, engines' book, modules and board order, undo/redo, autosave
      store/panels.ts      which side panels stand open, which drawer is open
      services/            engine.ts (the boundary), mockEngine.ts (the concept's simulation)
      i18n/                i18next on an instance of its own; words/*.ts: the page's own words, one file per package
      styles/              concept/ (the concept's stylesheets, unchanged), index.ts (their order), rebuild/*.css (the
                           rebuild's own, loaded after: parts on every surface, the shared part states, standards mode)
      board/, desk/, shell/, thumbs/   the shells: the Pedalboard, the desk and its layout, the bars and panels
      testing/             compat.ts (the concept's globals for the two tools) and compat/*.ts (one per package);
                           gallery.ts (every part on every surface)
  tools/
    behaviour/             the concept's behaviour suite; acceptance/*.json (one per package), scenarios/30-slice.mjs
                           (the slice's own claims, which pass on the concept too), switches.mjs + switches/*.json
    visual-baseline/       the concept's visual baseline; slice-states/*.mjs (one per package), slice-accepted.json,
                           gallery.mjs (every part on every surface), plants.mjs + plants/*.json (planted defects)
    test-view.ts           the layout, the core's selectors and models, the parts' contract at run time
    test-parts-contract.mjs  the parts' contract in the lint: the review's planted part refused
```

`npm run check` runs both type checks, the validator, the request comparison, the core's tests, tools/test-view.ts and
tools/test-parts-contract.mjs. `npm run dev` serves the page; `npm run build:file` writes the file the owner publishes;
`npm run build:test` the file the tools drive; `npm run lint` lints the React layer.

## Styles

The concept's stylesheets are used as they are: `src/app/styles/concept/` is a byte-for-byte copy of the concept's
`src/css`, imported in its file-name order by `styles/index.ts`, and never edited here. The rebuild's own stylesheets are
`styles/rebuild/*.css`, loaded after the concept's in name order (the number is the layer, as in the concept), each a
file of its own; a package adds one and edits no list:

- `15-parts.css`: parts on the surfaces the concept never drew them on (the meter off the board, the LED switch on paper,
  an XY pad in a row, a small button on paper and on an LCD, a wide control in a well, a segmented choice on a pedal);
- `35-part-states.css`: what every part inherits from its markup alone (root `data-ctl` and `.changed`, a direct `.kl`
  label): the page's mono caps label, its changed mark, focus ring and disabled state, in the hardware's ink on a pedal
  or an LCD. Every rule has no weight (`:where`), so wherever the concept styles a part its own rule wins;
- `10-standards.css`: the two boxes standards mode changes in the slice's part of the page (the engine caret, the LCD
  picker's arrows);
- `15-controls-1.css` (package 1): the shared controls on the surfaces the concept never drew them on (a token, a flag
  or terminal chips off a terminal, an LCD's lit word, bar chips and big number off an LCD, which bring a small LCD of
  their own, a chip selector or the record button on paper, a ruler or a look-back strip in a row or on a pedal);
- `15-containers.css` (package 2): the containers' few rules of their own (a bar-less terminal's first line, the
  planner off, a guard with the planner off, a caption above a field, the speed gauge beside its knob) and every part
  on the surfaces they give (a terminal's line and bar, an LCD's row, a rack group, a strip, a plugin's grid, a deck):
  an XY pad's plane, an LED switch or a chip selector on a group's paper, a wide part across a group, the plan text and
  the beat lamps bringing their own housing, a paper-faced part keeping the page's ink inside an LCD.
- `30-overlays.css` (package 8): the ring on the picker's drawing Enter will take while a filter is typed (F-29).

The components emit the concept's element structure, classes and `data-*` attributes, so its rules apply unchanged;
never style a component inline beyond what the concept sets inline (`--p` on a knob, `--h` on a pedal, the layout's grid
placement). A part's look on every surface is checked by the parts gallery (below, "The test layer").

## The page skeleton

`index.html` is the concept's skeleton with a doctype: the skip link, `header.top`, `#app`, the other pages' stand-in
and every overlay's container, in the concept's order, and the concept's one `#dd` (the dropdown every pick shares), so
that the visual baseline's element keys agree. React is rooted in `#app`; it draws the top bar into `header.top`, the
toast into `#toast`, the live region into `#say` and a pick's list into `#dd` through portals.

## The store

`store/desk.ts` holds the core's `Desk` (engine, tier, values, auto switches, state, what is on it), the engines'
`EngineBook`, the modules in the user's order and the board in chain order, and the undo and redo history. It decides
nothing about a value:

- **A value** changes only through a `Change` a part's setter built (`bindPart`): cleaned, set on the engine's grid and
  named by the core. `commit(change, from)` applies it and records one step; `from` names the part, so a gesture belongs
  to the part that began it and a change from another part meanwhile (an answer from the server) is a step of its own.
- **A gesture** (a drag, a held key, typing in one field, a run of wheel steps, a run of moves on one grip) is one step:
  a part calls `v.gesture.begin()` and `end()`; changes inside merge, named again by the core from the first desk to the
  last (`stepWords`).
- **The make-up**: where something added goes and what leaves with something removed are the core's (`addTo`,
  `removeFrom`), the steps' words too (`deskStep`, `engineStep`, with what Undo puts back: "the modules", "the pedals",
  "the engine"); what a pedal arrives as is the core's (`arrivalChange`); `switchEngine` does the engine switch.
- **Undo** puts back only the paths a step changed and nothing has changed since (a value, an auto switch, a state, the
  modules, the board, the engine, the book, and every package's undoable state), as the concept's desk entries do; redo
  is the same the other way. Its toast and the live region say what was undone or redone; with nothing to undo the keys
  say so.
- **A package's own state** lives in a store of its own (`store/<package>.ts`). If Undo should cover it (a workflow
  opened, the drafts), the package registers it (`register/undoable.ts defineUndoable`) and changes it inside
  `deskStep(words, fn)`: one step, put back path by path like the rest. **State the server owns** (a take that arrives, a
  job's progress, a model list) is not registered: Undo never takes back what the server has done, and an arrival is no
  step; a value an arrival writes into the desk goes through a part's setter, which acts on the desk in force.
- **The autosave** writes `savedDeskOf()` whenever the desk or the book changes (a toast or a focus request writes
  nothing), as `hotstep.create.desk`; the next visit reads it with `readDesk()`, which cleans every value on the way in.

## The service boundary

`services/engine.ts` is the one interface between the page and the server, grouped as the existing frontend's API layer
groups it (`ui/src/services/api.ts`: songs, generate, models, plus the option lists and the VST3 chain). `generate`
takes the Create intent the core builds (`createIntent()`), never anything the view assembles. `mockEngine.ts` is the
concept's simulation. A package that needs more of the server (the build session, the planner's runs) defines a service
of its own in its own file and registers it (`register/services.ts defineService`), its mock beside it; the real
adapters are package 10's.

## How a part is written

A part is given its node's model and the actions that change exactly what its node binds, and nothing else
(`kit/types.ts PartView`). The model is the core's (`derive/models.ts partModel`): every parameter the node binds with
its value, its words as the page writes them, its range and its place on it, its options with their words and which
is chosen, whether it is changed, auto or applies; the node's label (a shell's, else the node's, else the parameter's),
its accessible name, its tooltip, its own words (a note's text, an XY pad's axis words), what it reads, its actions'
words, a choice row's choices and what each sets, a meter's scale, an LCD's lines with their case; the tier, the engine
and the surface it stands on. Its actions are `p.set`, `p.reset`, `p.setAuto`, `p.press` (the value contract's keys),
`p.typed` (a typed number), `p.toFraction` (a hand's place), `p.check` (a field's check while typing), `v.choose`,
`v.run`, `v.batch` and `v.gesture`. Every action acts on the desk as it is when it is called, so a part that waits (a
file that loads, a list that arrives, a run that takes seconds) names its step from the value it really changes, is
cleaned against the engine in force, and changes nothing if what it binds has gone. A part cannot name a step, write
another key, format a value or decide a default; it has no route to the store (the lint, below).

The LED switch, whole:

```tsx
import { definePart, useSay, type PartView } from '../kit/index.ts';

export function LedButton(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p) return null;
  const on = p.value === true;
  return (
    <button type="button" className={`led-btn${on ? ' on' : ''}${p.changed ? ' changed' : ''}`} data-ledtog={p.key} data-ctl={p.key} aria-pressed={on} title={v.title ? say(v.title) : undefined} onClick={() => p.set(!on)}>
      <i />{say(v.label)}
    </button>
  );
}
definePart('ledButton', LedButton);
```

A part's file registers itself (`definePart`) and is loaded by being in `src/app/parts/` (`load.ts`); [06](06-how-to.md)
section 4 has the steps to add one. A part draws the housing the concept gives it on its surface: `labelled` is the
foot pairing on paper (its label first) and an LCD selector on hardware (the control first), a text field is an LCD input
on hardware and a field on paper.

## How the generic renderer works

`render/Face.tsx` walks a definition's face: for each node it asks for its model (`render/bind.ts usePartView`: the
core's `partModel`, null when the core does not draw the node), looks its part up (`kit/define.ts`), and draws it with its
children on the surface the part gives them (a rack, a well, an LCD and a tile's foot are surfaces of their own); a node
that stands back is wrapped as the concept wraps it. It has no branch for any module or part.

The shells are drawn from the core's models too: `render/Tile.tsx` from `tileModel` (its width at the tier and engine,
its group and name, the engine's note, the strip it becomes, the "Also set" line); `board/Board.tsx` from `boardModel`
(the chain, the count, the pedals and units the engine uses) and, for each pedal and unit, `pedalModel` (its state, on,
bypassed, held, idle or always, its classes, its name at the tier, its tip, its foot switch, its body laid out at the
tier); `desk/Desk.tsx` from `deskModel` (the tiles shown, the empty desk's offers and words, what waits parked, the note
naming the hidden Lab modules); the Modules panel from `cardModel` (each card's name, tip, tags and how it matches the
filter, in the page's language).

## Rendering

Each view subscribes to its own model (`render/useModel.ts`): on every change of the store it asks the core for its
model, compares it with the last as JSON, and redraws only when it differs. A knob that turns redraws that knob, the
words that show its value and the Undo button; the tiles, the board and the desk do not redraw. The desk is laid out
again only when its make-up changes (tier, engine, the modules in order, the window's width), a tile's shape changes
(its width class, a strip, the engine's note, the "Also set" line) or a tile's body changes size (a ResizeObserver);
the board's pedals are measured again when what stands on it changes or it resizes.

Measured with the review's method (`perf.js`: every ACE-Step module, all pedals and units at Lab, StableStep's Refine
dragged through 60 pointer moves; CDP metrics, two runs each):

| | Script | Style | Layout | Layouts | DOM mutations outside the pedal |
| --- | --- | --- | --- | --- | --- |
| Concept | 89 to 117 ms | 56 to 60 ms | 23 to 28 ms | 68 to 69 | about 1,780 |
| Before (the slice as reviewed) | 273 to 295 ms | 755 to 846 ms | 81 to 89 ms | 290 to 291 | 4,343 |
| Now | 97 to 144 ms | 41 to 70 ms | 12 to 21 ms | 33 to 35 | 91 |

## The rules the lint keeps

`npm run lint` keeps the two rules the design rests on (eslint.config.js):

1. **A part has no route to the store.** A file in `src/app/parts/` imports react, react-dom, the part kit, its sibling
   parts and `src/app/services`, and nothing else, and names no page global (`window`). The review's planted part (one
   added line that wrote two other modules' values under a made-up undo name) is refused on its imports; written through
   `window` it is refused too (tools/test-parts-contract.mjs, in `npm run check`).
2. **The view decides nothing about a value.** Outside the files whose job is values (the store, the binding, the test
   layer, the model index, translation, the services), the React layer may not name an engine, read a definition field
   that decides a value or what shows (`engines`, `labUnder`, `builtIn`, `source`, `gated`, `when`, `range`,
   `default`, `on`, `per`), import the core's value functions (`valueOf`, `has`, `applies`, `changed`, `resolve`...),
   format, parse or re-case a number or a word, or write English in its markup. It cannot catch a decision written with
   none of these (two values the core gave, compared), English in a template literal or a constant, a definition field
   read through another variable name, or the same rule stated twice inside the core: those remain the review's.

## Adding without colliding

A work package adds files and registers them; it edits none of the shared ones. The registration points:

| What a package adds | Where | How it is found |
| --- | --- | --- |
| A part | `src/app/parts/<Part>.tsx` | `definePart(name, Component, how?)`; loaded by `load.ts`. `how` (package 2): `surface`, the surface a container gives its children (a value, or a function of its options); `dims`, it draws its own standing back (the renderer does not wrap it); `kidsDim`, it draws its children's |
| A part's look, a package's stylesheet | `src/app/styles/rebuild/<NN>-<name>.css` | loaded after the concept's, in name order |
| A keyboard shortcut | a file in `src/app/features/` | `onKey({ id, when, run, order })` (register/keys.ts) |
| Something drawn in the shell (the prompt card, the lyric sheet, the go bar, Latest, the Workflows list, the engine panel's models, an overlay) | a file in `src/app/features/` | `fill(slot, Component)` (register/slots.ts): `desk.lead`, `centre.prompt`, `centre.quickLyrics`, `centre.gobar`, `centre.latest`, `panel.workflows`, `engine.panel`, `overlays` |
| An overlay that takes focus and gives it back (a dialog, a menu) | its own file | `layerOpen(el, { trigger, modal, close })` and `layerClosed(el)` (features/layers.ts): Escape closes the top one, Tab stays in a modal one, Undo waits under one, F6 and F9 wait |
| Something in the popover (a list, a menu, an offer) | its own file | `openPop(button, () => <Body />)`, `closePop()`, `placeFloat` (features/pop.ts) |
| An overlay's exit (a copy of it played out while it is hidden at once, the concept's ovOut) | its own file | `ovOut(el, kind)` just before hiding `el` (features/overlay-exit.ts, package 11) |
| What the queue holds, for the engine log's queue line | the generation package's file | `tellQueue({ rendering, waiting, engine?, loaded? })` (store/notes-logs.ts, package 11); until it is told, the log says idle |
| A named action (parts.ts ACTIONS) | a file in `src/app/features/` | `defineAction(name, run)` (register/actions.ts) |
| State of its own | `src/app/store/<package>.ts` | its own store; `defineUndoable(name, { read, write })` when Undo covers it |
| A server call | `src/app/services/<package>.ts` | `defineService(name, impl)` (register/services.ts) |
| Words | `src/app/i18n/words/<package>.ts` | gathered by i18n/index.ts; an id defined twice fails at load |
| Concept globals its scenarios call | `src/app/testing/compat/<package>.ts` | `install()`: `expose`, `onMirror`, `onScene` |
| Its acceptance | `tools/behaviour/acceptance/<package>.json` | `run.mjs --slice` runs every file |
| Its features switched off | `tools/behaviour/switches/<package>.json` | `switches.mjs` checks each makes a scenario fail, and writes what caught each to `tools/behaviour/caught/<package>.json` |
| Its visual states | `tools/visual-baseline/slice-states/<NN>-<package>.mjs`, ending with its `VISUAL_SET` | `slice-states.mjs` gathers them; `visual-all.mjs` runs every set |
| Its accepted visual differences | entries of its own, passed with `--accept` | counted exactly |

The core and the definitions are shared on purpose and do not collide in practice: the inventory already lists every
part the packages build (`PARTS`, 73 parts, 73 used), every definition exists, and a package changes a definition only
to give a node its part's options.

## Translation

Every word is a message (an id and its slots, `derive/messages.ts`); `useSay()` fills it from the locale's table held by
i18next, on an instance of the page's own (`createInstance`, so the application's default instance and its language
detector are never shared). English is generated: the core's sentences (`coreStrings`, and `EN` in messages.ts, which
holds the core's words for tiles, pedals, cards and the empty desk), each definition's words under the keys `stringsOf()`
gives them, and the page's own chrome (`i18n/words/*.ts`, keyed `ui.*`). A locale adds a table of the same ids.
Components hold no English (the lint); the brand is the one literal. `say(slot, { upper: true })` prints a word in
capitals in the page's language (an LCD).

## Additions to the core made for the view

Each is in `src/derive/` with a test in tools/test-view.ts, and none has a branch for a module:

- `derive/models.ts`: what each view draws (`partModel`, `paramModel`, `tileModel`, `pedalModel`, `boardModel`,
  `deskModel`, `cardModel`), `pedalState`, `chainOf`, `meterScale`, `keyStep` (the value contract's one statement),
  `fractionValue`, `parseTyped` (the inverse of how the page writes a number), the make-up rules `addTo`, `removeFrom`,
  `boardOrder`, `emptyDesk`, `startersOn`.
- `derive/parts.ts bindPart`: setters on the desk in force (a getter), dropping a change to what no longer binds;
  `press`, `typed`, `toFraction`; a typed value's clamp said even when it changes nothing (`tell`).
- `derive/desk.ts`: `nodeShown`, `usedHere`, `tileAtTier`, `tileName`, `deskStep` (with what Undo puts back),
  `engineStep`, `removedWords`, `parkedOn`, `boardCount`, `arrivalChange`; `derive/engine.ts newOn` (the first switch to
  an engine says what it brings); `derive/strings.ts pickMsg`, `splitUnit`, and the rest the view asks.
- The definitions: an option's `pick` words (what an LCD picker prints where it differs from a read-out: closed
  deviations 3 and 4), a module's `starter` (the empty desk's offers and the words it names the first with),
  `CATEGORIES` (the Modules panel's groups, in one place), an engine's `tag`, the power button's `name`.
- `derive/controls.ts` (package 1): what the shared controls draw beyond their parameters (`controlModel`, given to a
  part as `PartModel.control`): a number box's text and its words while empty, the seconds a step count takes
  (`stepSeconds`, an engine's `stepSeconds`), a ruler's scale, fill, caption and markers (`rulerModel`, `rulerScale`,
  `songTempo`: one function for the song's tempo, S-07), a fader's travel and unity line, a look-back strip, a power
  chip, where a Quick row's value stands between its choices (`choicePos`), each option's face in a picker
  (`optionFaces`, `quantOf`); what auto stands for (`autoEstimate`: turning auto off starts there, S-17); where a hand
  moves a ruler or a token (`rulerSnap`, `scrubValue`). `derive/strings.ts keptMsg` (what a field that keeps invalid
  text says is in force) and `optionsOf` keeping a source's `only`; a typed empty box puts a parameter with auto back
  on auto, and text that fails its check is refused and said (`derive/parts.ts`).
- Package 2: `derive/containers.ts kidsOf` (a container's children: each one's part, whether it is drawn, its label,
  why it stands back, the parts it holds, whether a higher tier adds it, the keys it edits: `PartModel.kids`) and
  `fileRefused`, `dropOutside` (what a drop zone takes); `derive/readouts.ts readoutModel` (`PartModel.readout`: the beat
  lamps, the dice, the flow windows and their count, the speed gauge, the plan strength curve, the plan text in its four
  looks, the lyric timestamps, the cover's hue) with `sectionNames` and `spellKey`; `PartModel.dimWords` (what a node
  says when it stands back, held or not) and `PartModel.state` (the state a node names, its owner, its noun, lent or
  not; `PartView.state` applies its steps and `load`s a file); `derive/steps.ts fileStep` (a file's `load` and `clear`
  without saying: "Clear the timbre reference", a replacement from one file to the other); `derive/desk.ts nodeShown`: a
  control none of whose parameters applies is not drawn, as the concept's `ok()` leaves it out (package 1 flagged it);
  `derive/view.ts boardNodes` gives a pedal's or a unit's own nodes as the definition writes them (the renderer found no
  words for a node an engine's or a state's line had resolved: the Mastering pedal's Studio face printed nothing).
- `derive/vram.ts` (package 5): the engine panel's cards (`enginePanelModel`: an engine's tagline, output rate, songs per
  Generate, testing badge, the models heading at the tier), a modelPicker's rows (`modelRows`, given to a part as
  `PartModel.models`: each model's role, list heading, accessible name and option faces), and the VRAM figures
  (`vramNeed`: what is loaded and what the next Generate needs, in how many passes, refused or tight, for the Takes
  tile, the go bar and the log too; `meterModel`, `meterLeds`, `meterEase`). `derive/core.ts`: `Desk.vram` (the
  figures the models service tells), `listKey` (a model bucket's list is the engine's own, `models.<engine>.<bucket>`),
  `selectionKey`, `choiceLists` (one statement of a choice's options for words, faces and formatting),
  `modelDefault` and `shownValue` (a loaded model left at "" is the file the server has selected: shown, written and
  stored so); `formatValue` takes the engine. The definitions: `OptionsFrom.loaded`, the Models parameters' `short`
  (their roles: Sound, Planner, Decoder), the Models face split so Quick shows the sound model only, an engine's
  `khz`, `tagline` and `testing`. The engine switch's toast names a model whole (F-34).
- Package 3a: `derive/instruments.ts instrumentModel` (`PartModel.instrument`: what a bespoke instrument draws: the key
  wheel's segments spelled as a musician does, the prompt chips and which are in, the split-flap's cells and the LM seed's
  link, the saved seeds' rows and the save form's defaults, the reels and the VRAM a count costs, the scope's schedule,
  band, cutoff, read-out and handles, the latent curves, the repaint region in the song's times, the blend's percent) and
  `partAct` (`PartView.act(name, arg)`: what a hand on a bespoke part does, made through the part's own setters, so it is
  cleaned, bound and named as every other change, or refused and said, with Undo in the toast where a step's words call
  for it): `wheelWalk`, `capToggle`, `seedTyped`, `cleanSeeds`, `takesCost`, `sigmasOf`, `distOf`, `regionMove`.
  `ParamModel.canAuto` and `run`, a number on auto standing at what auto stands for (`at`), a list of option words
  (`corners.0`...) in `PartModel.words`, `ReadModel.changed`. `derive/steps.ts`: the seed's named step is named as every
  change of the seed is (random or the number kept), a span is said in the song's times; a change carries the named step
  it took (`Change.named`), so a gesture of several (a region dragged) is named by it. `derive/desk.ts nodeShown`: a
  bespoke part that houses children stands, on an engine without what it edits, while one of them does (the key wheel
  houses Tempo's LCD).
- Package 3b: `derive/cartridges.ts` (`InstrumentModel.carts`: the Adapters' cartridges, each with its housing, label
  art, name in pieces, faders, share, phase window and section masks; the empty slot, the planner adapter, the column
  beside one cartridge, the adapters loaded but not used, what another engine keeps, Quick's line, the adapter lists;
  the hands `load`, `remove`, `planner`, `phase`, `mask` through the slots state's steps; `directivesOf`,
  `setDirective`, `dropDirectiveSlot`, `phaseMove`, `adapterLists`, `adaptersAway`; and the XY pad's off rail:
  `InstrumentModel.rail`, `xyRail`, `xyRailValue`, `xyRailKey`, the hands `y` and `yKey`), registered through
  `derive/instruments.ts defineInstrument(part, draw, acts)`, so a core file of its own adds instruments without editing
  instruments.ts' tables. The definitions: `ParamFace.forced` (a choice the engine runs at another value while a
  condition holds: shown pressed, the others disabled, what returns after said; `strings.ts forcedOf`,
  `ParamModel.forced`, an option's `off`, `KidModel.forced`), `Size.narrowWhen` (one column at Quick and Studio while it
  holds), `Desk.lists` entries' `kind`; a fader's `name` and `tape` (`ControlModel.fader`: "Self-Attn strength", its
  short tape carrying the full name, V-01); a bespoke part that holds children it places itself is given their `kids`;
  a state's named step may say its own `{from}` and `{to}` (a phase window's span).
- `derive/steps.ts stepName`: a parameter is named with its owner when another parameter **on the desk** prints the same
  words, so adding a module no one has put on the desk renames nothing (SL-13).
- `derive/workflows.ts` (package 7): the user's saved work. One rule for names (`normName`, `nameKey`, `sameName`: NFC,
  trimmed, a run of white space as one, any capitals; F-02, F-10) and the clashes it decides (`wfNameClash`,
  `revNameClash`, `saveCheck`); the lists as pure operations by name (`putWorkflow`, `dropWorkflow`, `putRevision`,
  `dropRevision`, `renameWorkflow`, `renameRevision`); reading a stored list record by record, the page's shape and the
  concept's (`readSaved`, `cleanWorkflow`, `cleanRevision`, `readDrafts`, `cleanDraft`: C-08); the merge with what another
  tab saved (`merge3`, `mergeSaved`, `mergeDrafts`: C-09); a workflow on the desk (`layoutOf`, `workflowModified`) and a
  revision against it (`revisionDiff`, `revisionModified`, `revisionDetail`: F-04, C-36); the list as the panel draws it
  (`listModel`) and the save row's hint (`saveHint`); the steps' words (`openStep`, `loadStep`, `recordStep`,
  `draftStep`, `putWords`, `loadedWords`, `unloadedFiles`); a revision as a file and back (`exportFile`,
  `importRevision`: C-24, F-20); drafts (`draftSub`, `draftTitle`, `draftWhen`, `sameDraft`, `applyDraft`,
  `draftsAbout`); storage's words (`saveState`, `storeFailed`, `storeSkipped`, `storeUnreadable`, `storeTold`: F-15,
  F-16). Its words are added to the core's table (messages.ts `EN`) as the file loads. `derive/records.ts loadRevision`:
  content a revision holds as empty text replaces nothing. Tested by `tools/test-workflows.ts` (in `npm run check`).
- `derive/picker.ts` (package 8): the picker's model (`pickerModel`: each drawing's group, whether the category bar
  hides or dims it, how it matches the filter, what it says, the groups' counts, the count's words by kind, the order
  the keys walk, what Enter takes and what the hint names), `pickerGroupOf`, the category bar (`PICKER_CATS`) and the
  Lab offer's words (`labOffer`). `derive/tips.ts`: a tip's sentences with every digit kept (`tipSentences`), the
  sentences that teach how to work a control and are said once a session (`tipLesson`, `isHowTo`), a finger's words
  (`touchTip`) and the rule that a tip repeating a printed line does not open (`sameAsLabel`); the last two languages'
  rules are the English table's, keyed by language. Tested in `tools/test-overlays.ts` (in `npm run check`).
- `derive/logs.ts` (package 11): what the engine log says (`engineLogModel`: its head, the engine and the models it
  renders with, named whole, F-34; its lines: what the server says of itself, `EngineStatus`, the models by role, the
  plugins the server lists, the VRAM as the meter's, `vramNeed`, and the queue, `QueueSummary`, with what another engine
  rendering holds, B-34). Its words are added to the core's table as the file loads. Tested in `tools/test-notes-logs.ts`
  (in `npm run check`).
- `derive/generate.ts` (package 6): what a press of Generate makes of the desk (`refusalOf`: an empty box per engine,
  a score to preview, two modules that cannot run together, a run certain not to fit, F-21; `runOf`: the run as it is
  started, its engine and desk kept with it, one song per take with its title, card line, length and colour, the
  pedals that run, play while rendering, the re-planning tries; `pageIntent`: the request, the desk in real names), what
  a run says while it renders (`stageOf`, `takeProgress`, `readyOf`, `attemptsAt`, `queueStage`), the server's refusal
  (`refusedWords`), the go bar's line (`goLine`, `changedKeys`, `changesUnmarked`), Generate and Cancel (`goModel`,
  `docTitle`), cancelling (`cancelWords`, `cancelBack`), Latest (`latestShown`: whole rows, the fold, the count;
  `songCard`, `songArt`, `songWave`, `songName`), the song menu (`songMenu`, `repaintFrom`, `repaintDesk`, `reuseWords`,
  `sentTo`) and the player (`playerWords`, `playerSeek`); the `gen.*` messages. `derive/vram.ts meterModel(ix, desk,
  run)`: while another engine renders, what that engine holds (B-34); `meterTarget`: the needle's swing while a run
  renders. `Desk.songs`: what the generation service tells (the newest song's colour, the tries the run has used),
  which the cover swatch and the dice read (`readoutModel`).
- Package 3c: `derive/build.ts` (Build by section: `buildSections` (a section known by its name and its ordinal, B-05),
  `sectionColour`, `cleanBuild`, `cleanSessions`, `buildAt`, `buildModel` (`InstrumentModel.build`: the sections, the
  drift banner, the section being built and what Generate asks for, its takes, the keeps in order, the locked settings,
  the foot, what Render asks for) and the board's hands `planned`, `keep`, `discard`, `relyric`, `rendered`, `reset`, each
  the record state's step of that name); `derive/recording.ts` (`spanAt`, `spanModel`, `spanMove`, `spanHit`: the
  recording's part, its keys and drags); `derive/score.ts` (the ABC foot's model, `scoreAsk`, `abcInfo`, `abcFile`,
  `readsAsAbc`, `scoreLayout`, `scoreModel`: the preview's words, every mark of its staff and where the playhead stands).
  The three register through package 3b's `defineInstrument` (one mechanism; each declares its field of `InstrumentModel`
  where it is drawn) and `Tell`'s `'say'` (said in the live region only). `Desk.sessions` (what a server session tells the page: never a step,
  never saved). `controls.ts rulerStep`. `models.ts`: a strip carries its Also set line (the concept's every tile does);
  `PartModel.actions[].auto`, `StateModel.detail` (a file's length). `tiers.ts alsoSet`: the tier a setting is in is
  where the face edits it, strip or not (E-25; `core.ts editsAt(..., strips)`). `desk.ts nodeShown`: a control with a
  `dim` stays and stands back, as the rule's comment always said. `core.ts WORD_FIELDS` and `WordsOverride`: a word line
  may change a placeholder. `containers.ts KidModel.look`. Tested by `tools/test-instruments-3c.ts` (in `npm run check`).

## The test layer, and running the tools against the rebuild

`src/app/testing/compat.ts` gives the two tools what they reach into the concept page for, on top of the store, in the
test build only: `S` (set it, then `render()` applies it as a scene, not a step), `render`, `setVal`, `get`, `toast`, the
catalogue, `UNDO` and `REDO`, and `vsStray`, which detects a change to the desk that no undo or redo recorded (outside a
scene). A package's additions are a file of its own in `testing/compat/` (the slice's: `addItem`, `switchEngine`,
`enterEngine`, `setTier`, `shown`, `removeMod`, `paintUndo`, `S.eng`).

```
npm run build:test
set PLAYWRIGHT_MODULE=<node_modules/playwright>
node tools/behaviour/run.mjs --slice                 # the acceptance: every file in tools/behaviour/acceptance/
npm run switches:apply                               # each switch's text still applies exactly once (no builds)
npm run switches                                     # each feature switched off makes a scenario fail
npm run visual:all                                   # every package's visual set against the concept
node tools/visual-baseline/vb.mjs capture concept-slice --states slice --scope slice --page <concept>/hotstep-tiers.html
node tools/visual-baseline/vb.mjs capture rebuild-slice --states slice --scope slice --plan-from concept-slice
node tools/visual-baseline/vb.mjs compare concept-slice rebuild-slice --scope slice --accept tools/visual-baseline/slice-accepted.json
node tools/visual-baseline/gallery.mjs               # every part on every surface, both themes, every tier
node tools/visual-baseline/plants.mjs                # each planted defect is caught
```

- **Behaviour.** `acceptance/slice.json` lists 38 scenarios, every one whose subject is something the slice builds: 31
  of the concept's suite and 7 of the slice's own (`scenarios/30-slice.mjs`: knob keys, undo and redo by key, button and
  toast with what they say, the autosave and its restore, the theme, an engine switch, the tier keys, adding and
  removing; each passes on the concept page too). 10 that look like the slice's are excluded, each with its reason (4
  vacuous without a picker, 1 without a prompt card, 2 that contradict F-25, 3 that need later packages). All 38 pass.
  `switches.mjs` builds the page with each of 11 slice features switched off in turn (undo, redo, autosave, restore,
  knob keys, keyboard moves, add, remove, tier switch, engine switch, theme) and runs the acceptance: each makes 1 to 6
  scenarios fail.
- **Looks, against the concept.** 161 states (the concept's matrix cut to the slice, the dropdown open, the engine
  toast, and three that record parts the slice's modules do not draw where the concept does: Flow's rack and foot, the
  Mastering VU meter, a Quick choice row), in standards mode, compared on every element's computed style (with forced
  hover, focus and active), its role, accessible name, own words and attributes, 11 screenshots of the desk, and the
  parts on their own; the scope is the desk, the bars' slice, the Modules panel, the toast and the dropdown. Result:
  **identical**, with the differences accepted on purpose (`slice-accepted.json`, each with its reason and its exact
  count, so a new one of an accepted kind fails).
- **Every part on every surface.** `gallery.mjs` draws the parts gallery (`testing/gallery.ts`: every control and
  read-out of the inventory, generated from it, in a tile's body, a row, a rack, a well, an LCD, a tile's foot, a
  labelled pairing, on a pedal and on a side unit, plus a bare new part's markup on each), both themes, every tier,
  values changed: drawn inside its container, words at 3:1, the page's label, a changed mark, a choice that says which
  is chosen, the concept's type and colours where the concept has the same part on the same kind of surface (20
  references), and every look as it was recorded (`tools/fixtures/gallery.json`).
- **Planted defects.** `plants.mjs` plants 15 defects one at a time (the review's 0 to 11, and three of a part's look off
  its concept surface) and runs the comparison and the gallery: 15 caught. The review's 12 (knob keys dead) is
  behaviour, caught by the switches.

**Package 1 (shared controls)** adds `acceptance/1-controls.json` (24 scenarios: 10 of `scenarios/31-controls.mjs`, which
hold on the concept page too, 8 of `scenarios/32-controls-gallery.mjs`, which drive the controls no module draws yet in
the parts gallery, and 6 of the concept's suite, three of them pending there: S-18 twice and S-31), `switches/
1-controls.json` (25 features), `slice-states/10-controls.mjs` (18 states of the controls on their own where the concept
draws them, compared with `controls-accepted.json`: `vb.mjs compare concept-controls rebuild-controls --accept
tools/visual-baseline/controls-accepted.json`) and `tools/test-controls.ts` (in `npm run check`). The gallery draws a
part with variants once in each (`testing/gallery.ts`).
**Package 8 (the picker, the popover, tooltips, touch, the focus ring, F6, drag and drop)** adds
`acceptance/8-overlays.json` (43 scenarios: 10 of `scenarios/33-overlays.mjs`, which hold on the concept page too, each a
concept scenario's checks with the steps that need another package's part replaced by parts both pages draw; 33 of the
concept's suite, six of them pending there, F-18, F-19, F-29 three times and S-36; nine of those wait on packages 3, 6
and 7 for a part they draw), `switches/8-overlays.json` (32 features), `slice-states/80-overlays.mjs` (22 states: the picker in
both themes, at four sizes, filtered, by category, a Lab module's offer and an arrival; the Lab offer and a tile's moves
in the popover; the tip by keyboard and by pointer; four drags in progress; compared with `overlays-accepted.json`:
`vb.mjs compare concept-overlays rebuild-overlays --scope overlays --accept tools/visual-baseline/overlays-accepted.json`)
and `tools/test-overlays.ts` (in `npm run check`). The `overlays` slot is drawn at the page's root (`App.tsx`); an
overlay of another package registers with `features/layers.ts` and puts what it holds in the popover with
`features/pop.ts` (above, "Adding without colliding").
An acceptance entry may wait on a later package (`"waiting": { "package", "because" }`, tools/behaviour/README.md):
`run.mjs --slice` reports it apart and fails if it passes, `switches.mjs` leaves it out. The slice's
`01-desk/add-by-keyboard` and `10-desk-modules/undo-covers-structure` wait on package 3 (Tempo's LCD rows and key wheel
decide the tile heights their moves need).

**Package 2 (containers and read-outs)** adds `acceptance/2-containers.json` (17 scenarios: 11 of `scenarios/33-containers.mjs`,
which hold on the concept page too, and 6 of the concept's suite, one of them pending there: S-28), `switches/2-containers.json`
(39 features), `slice-states/20-containers.mjs` (35 states of the containers and read-outs on their own, compared with
`containers-accepted.json`), the parts gallery's seven new surfaces (a terminal's line and its bar, an LCD's row, a rack
group, a strip, a plugin's controls, a deck: `testing/gallery.ts`, `gallery.mjs`) and `tools/test-containers.ts` (in `npm
run check`). `testing/compat/containers.ts` gives the scenarios the drop zones' files (`S.timbreFile`, `S.contFile`,
`S.masterRef`), the adapters loaded (`S.slots`) and `isAuto`.
**Package 5 (engine panel and models)** adds `acceptance/5-models.json` (12 scenarios: 9 of `scenarios/35-models.mjs`,
which hold on the concept page too, and 3 of the concept's suite), `switches/5-models.json` (17 features) and
`slice-states/50-models.mjs` (the panel, a model's list and the meter, compared in the scope `models` with
`models-accepted.json`: `vb.mjs compare concept-models rebuild-models --scope models --accept
tools/visual-baseline/models-accepted.json`). The models service (`services/models.ts`) is the concept's catalogue and
figures behind `ModelsService` (`catalogue(engine)`, `vram()`); the simulated Engine's `models.list` reads the same
catalogue.
**Package 6 (generation, the go bar, Latest, the player, the song menu, the toasts)** adds `acceptance/6-generate.json`
(45 scenarios: 25 of `scenarios/36-generate.mjs`, 10 of the concept's suite, and 10 of the concept's suite waiting on
package 4 because they type into the prompt card), `switches/6-generate.json` (63 features), `slice-states/
60-generate.mjs` (23 states, scope `generate`: the go bar, Latest, the song menu and the toasts, compared with
`generate-accepted.json`: identical but for 6 accepted kinds, 499 differences, each counted). The generation service (`services/generate.ts`) is the concept's simulation behind
`GenerateService` (`songs`, `submit`, `cancel`, `live`, `watch`): one run renders at a time, its songs are kept in
this browser as a server keeps them. `store/generate.ts` holds Latest's cards, the queue as the page started it and the
player; none of it is undoable desk state, but cancelling is a step that undoes itself (`store/desk.ts recordStep`,
package 7's record steps): Undo queues the cancelled songs again with the settings they started with. The views are in
`src/app/generate/` (the go bar, Latest, the song menu, the player), filled into `centre.gobar` and `centre.latest`
(`features/generate.tsx`); the toast stack is `shell/Toast.tsx`. It tells the engine log the queue (`tellQueue`), the
song menu is one of the page's layers (`features/layers.ts`), and Latest's "in sight" and the toast stack stand on the
docked log (`placeToast.ts dockH`). `switches.mjs --scenarios <regex>` runs only some of
the acceptance's scenarios for each switch.

**Package 3a (the instruments of Tempo, Seed, Takes, the Sampler, Repaint, Timbre reference, Latent & correction and
Performance trades)** adds `acceptance/3a-instruments.json` (22 scenarios: 13 of `scenarios/34-instruments.mjs`, which hold
on the concept page too, and 9 of the concept's suite; 9 more waiting on packages 4, 6 and 8), `switches/3a-instruments.json`
(38 features), `slice-states/30-instruments.mjs` (24 states of the instruments on their own, compared with
`instruments-accepted.json`) and `tools/test-instruments.ts` (in `npm run check`). Its parts are `parts/Tempo.tsx`
(keyWheel, tapTempo, capChips), `Seed.tsx` (splitFlap, savedSeeds), `Takes.tsx`, `Scope.tsx` (samplerScope, latentCurve),
`Repaint.tsx`, `Timbre.tsx` (recentRefs, timbreBlend), with `parts/motion.ts` (the page's motion tokens read for their
animations); `services/references.ts` (the recent reference tracks, server-owned) and `features/references.ts` (a file a
drop zone takes is remembered); `testing/compat/instruments.ts` (`S.seed`, `S.stars`, `S.refs`, `S.source`, `S.region`).
The slice's `01-desk/add-by-keyboard` and `10-desk-modules/undo-covers-structure` no longer wait. What the instruments'
modules show of others' parts is drawn as the concept draws it there: a knob's auto pill and its symbol labels, an XY
pad's corner words, the scope's power switch (`look: 'scope'`), a row that is a pad with a column beside it
(`.padrow`, `.padside`) or a row of plugin pickers (`.picks`), the Sampler's well holding its pickers beside its numbers
(`.with-picks`), a note's `cap` and `hint` looks, an LCD word's changed mark, and a word written into a number box put
back.

**Package 3b (the Adapters' cartridges; the planner's XY pad: its top-p off rail and its fade with the planner off)** adds
`acceptance/3b-cartridges.json` (18 scenarios: 10 of `scenarios/36-cartridges.mjs`, which hold on the concept page too,
and 8 of the concept's suite), `switches/3b-cartridges.json` (46 features), `slice-states/31-cartridges.mjs` (17
states, compared with `cartridges-accepted.json`) and `tools/test-cartridges.ts` (in `npm run check`). Its parts are
`parts/Cartridges.tsx` (cartridges, placing its node's children: Quick's strength row, the stack bar, the fader banks,
the foot) and `parts/CartArt.tsx` (the label art); `features/adapters.tsx` runs the named actions `loadAdapter` and
`loadPlannerAdapter` (the adapter list, in the page's popover: package 8's `features/pop.ts openPop`) and
tells the page the installed adapters (`services/adapters.ts`) and what another engine keeps (`adaptersAway`, from the
engines' book; `store/desk.ts told` takes it). The Adapters' face is restructured: the cartridges node holds what stands
around the cartridges. The Repetition guard, Mastering and the side units were drawn whole by packages 1 and 2 and match
the concept. Parts of others given what these modules show: the fader bank's heading (its small words and the rebase
row on its line), a labelled pairing's `rebase` look, a symbol label and the `forced` mark, a segmented choice's disabled
options and forced note, the fader's tape tip and name, the LED switch's tip, the XY pad's rail and standing back,
Quick's How adventurous standing back with its note.

**Package 7 (workflows, revisions, drafts, storage)** adds `acceptance/7-workflows.json` (40 scenarios: 21 of the
concept's suite, three of them pending there: F-17, F-36, S-17; 14 of `scenarios/33-workflows.mjs`, nine of which hold
on the concept page too; and 5 waiting on packages 4 and 8, each with what it needs), `switches/7-workflows.json`
(38 features), `slice-states/70-workflows.mjs` (17 states of the Workflows panel, the row menu and the Drafts list,
compared with `workflows-accepted.json`: `vb.mjs compare concept-workflows rebuild-workflows --accept
tools/visual-baseline/workflows-accepted.json`) and `tools/test-workflows.ts` (68 checks, in `npm run check`). Its store is `store/workflows.ts` (the lists, the
open workflow and revision, registered as the undoable `workflow`) and `store/stored.ts` (every store's reads and
writes: what cannot be read is copied aside before anything writes over it, and a refused write is reported, so the
read-out says Not saved). Opening a workflow, loading a revision or a draft is a desk step (`deskStep`); replacing,
updating, renaming, deleting and importing a record change no desk, so they are **record steps**
(`store/desk.ts recordStep(words, run, back)`: the entry carries its own undo and redo, which act by name on the lists as
they are then and say what they left alone; C-02, C-07). The shell gained four slots for it: `panel.workflows.head`
(Save and Import after the panel's heading), `panel.workflows.save` (the save row, between the head and the list),
`centre.drafts` (the go bar's Drafts) and `centre.saveState` (the read-out before Undo).

**Package 11 (the Concept notes and the engine log)** adds `acceptance/11-notes-logs.json` (14 scenarios: 10 of
`scenarios/36-notes-logs.mjs`, nine of which hold on the concept page too, the tenth being the concept's pending F-18 for
the notes; the concept's pending F-08, `21-chrome-round5/tier-keys-wait-under-a-modal`; and 3 waiting on packages 4 and
6, each for the prompt card or generation), `switches/11-notes-logs.json` (30 features), `slice-states/110-notes-logs.mjs`
(15 states of the drawer and the log, compared in the scope `notes` with `notes-accepted.json`: `vb.mjs compare
concept-notes rebuild-notes --scope notes --accept tools/visual-baseline/notes-accepted.json`) and
`tools/test-notes-logs.ts` (17 checks, in `npm run check`). The notes are drawn into the skeleton's `#notes`
(`features/notes.tsx`, opened and closed by `notes-open.ts` through `layers.ts`), the log into `#logs`
(`features/logs.tsx`, `logs-dock.ts`), both always drawn, as the concept paints them; what the log says is the core's
(`derive/logs.ts`) from the logs service (`services/logs.ts`) and the queue the generation package tells
(`store/notes-logs.ts tellQueue`).
**No id is on the page twice** (found by package 11's full visual run after packages 7 and 8 merged): a drawing that
stands in several places at once carries ids of its own in each. An engine's mark takes the place it stands in
(`shell/engines.ts logoOf(engine, size, place)`: the engine button, the engine panel's card, a revision's or a draft's
row), which numbers its gradient's id; a thumbnail takes its place (`thumbs/index.ts thumbOf(..., place)` and
`thumbBare`: `card`, the picker's `pk`, a drag's `drag`), and every id inside it and every reference to it carry the
definition and the place (`ownIds`). Before, HeartMuLa's gradient (`hmRainbow`) stood in the engine panel and in every
HeartMuLa revision's row, and a module drawing's fixed `thg13` beside a pedal's counted one: a gradient inside a hidden
element leaves every other drawing that names it unpainted. Every state of every package's visual set was checked for
a repeated id: none. A new drawing with ids of its own goes through `ownIds` or a place in the same way.

**Package 3c (the instruments of YuE2 and MiniMax-Music3: Build by section, Continue from recording, the Lead sheet and
its score preview, and the Flow renderer's row)** adds `acceptance/3c-instruments.json` (25 entries: 12 of
`scenarios/36-instruments-3c.mjs`, ten of which hold on the concept page too, the other two being its pending E-46 and
E-47 stated in full; 11 of the concept's suite, five of them pending there: E-09, E-23, E-24, E-46, E-47; and 2 waiting on
packages 4 and 6), `switches/3c-instruments.json` (47 features), `slice-states/32-instruments-3c.mjs` (24 states,
compared with `instruments-3c-accepted.json`) and `tools/test-instruments-3c.ts` (in `npm run check`). Its parts are
`parts/Build.tsx` (buildBoard), `Recording.tsx` (recordingSpan), `Score.tsx` (abcFoot); the build session is a service
(`services/build.ts`, server-owned: the takes as they land and the section looked at, told to the desk by
`store/build.ts` as `desk.sessions.build`), as is a recording's length (`services/recording.ts`) and the planner's score
for the preview (`services/score.ts`, with `scoreRuns`, which generation registers to render a score with its seed).
What the user decides about a build is the module's record state `build` (its session, seed, plan, keeps, discards and
the songs it rendered), changed only by the board's hands through the core, each a step; a section looked at and a take
arriving are none. The score preview is `features/score-preview.tsx` with its state in `score/state.ts`, drawn into
`#scoreModal` through the `overlays` slot, which `App.tsx` now draws (`<Slot name="overlays" />`). The page's
`kit/glyphs.tsx` holds three more icons (play, stop, the padlock) in a file of its own.

## Deviations from the concept's markup

1. **The skeleton is static, React draws into it.** `#app` is React's root; the top bar, the toast, the live region and
   the dropdown are portals into the concept's elements; the overlays' containers are empty until their packages fill
   them (the `overlays` slot, drawn at the root by `App.tsx`; the picker portals into `#pickerWrap`, the popover into
   `#pop`). Package 10 undoes this when the page moves into the real application's root.
2. **Adding a Lab module outside Lab offers Lab in the popover by its card**, as the concept does (package 8): closed.
3. **The engine panel holds the five engine cards only**: closed (package 5 draws the taglines, specs and models).
4. **An XY pad changed from elsewhere is drawn where it is**, as the concept draws a pad it has just rebuilt; it springs
   only while it is being worked.
5. **The page keeps its records under its own keys** (`hotstep.create.desk`, `hotstep.create.engines`,
   `hotstep.theme`); the core reads the concept's saved shape (`migrate`, version 0) for when importing is built.
6. **A knob says its value with its unit to a screen reader** (`aria-valuetext`), and a segmented choice is a named
   group; every button and input says its type. The concept's give the bare number and no names.
7. **Adding from the Modules panel keeps focus on the card** (F-25, found by the review loop); the concept moves it to
   the new tile (two of its scenarios, excluded from the acceptance with that reason).
8. **An XY pad's key run is a step of its own**, and a double-click reset is a "Reset" step; the concept merges keys
   pressed soon after a drag into the drag's step and lets a reset back to the start of a run cancel it. Each gesture
   one step is what every other control does here.
9. **The toasts are a stack of three** (F-12) that Tab reaches (C-13, F-11): closed by package 6, with the concept's
   pending F-18 (a plain message never pushes out an Undo; Escape on a toast's button dismisses it) and S-17 (a toast
   claims Tab only while focus is where it was when it appeared and no layer has opened since).
10. **The concept's delegation hooks are absent** (`data-sig`, `data-toast`, `data-luaon`, `data-setq`, `data-base`):
    the rebuild's buttons run their own actions. A dropdown option's `data-opt` is its value, not its label.
11. **A Quick row's marker for a value between its choices**: built (package 1); closed.
12. **A tile moved up or down by the keyboard moves its top** that way; the concept compares the middles of the places
    it could take, which a shorter place in the same row can pass as "up". The same on every layout the concept's
    suite checks.
13. **Delete or Backspace resets a value control** (a knob, a token, a fader, a ruler, the look-back strip), as a
    double-click does: the concept's pending S-18, in the value contract (`kit/value.ts`). The controls' tips keep the
    concept's words; S-18's sentence about the tip is not done.
14. **A hand moves some values by a rule of its own**: dragging a length ruler snaps to 5 s (15 s above 2:00) and to a
    labelled tick within 6px (pending S-31); a token over a range of more than 500 steps moves with the value, a
    hundredth of it a pixel (one at 100, ten at 1000; pending S-28). Both are the core's (`derive/controls.ts`
    `rulerSnap`, `scrubValue`, reached through the setters `along` and `scrub`).
15. **The switches inside a module's look name their parameter** to a screen reader (a flag, an LCD's lit word, the
    record button, a power chip: aria-label "Chain of thought" where the concept's says "--cot"), and every one
    carries the changed mark (the concept's power chip and its rulers without auto have none; a look-back strip off a
    terminal carries it after its box). The record button off the plans deck has a paper face.
16. **The LCD number says "type it, or drag up or down"**: the concept's says "or tap" because Tempo's TAP stands
    beside it; here TAP is another part (package 3).
17. **The language stamp remembers the languages used last** under the page's own key (`hotstep.create.langRecent`,
    deviation 5), so the concept's 04-controls/language-stamp, which reads `hs3-lang-recent`, cannot pass as written.
18. **A plugin's family, symbol, file and curve** are the concept's tables (`src/definitions/option-faces.ts`), keyed
    by the option's words as the server lists them; the simulated engine lists the concept's plugins and languages.
19. **A side unit's chip selector carries its own caps label** (the concept's `chipsw`): the board lays it out without
    the LCD selector it gives a pick (`board/Board.tsx`, one line).
39. **A drag's preview is an inert copy** (package 8): the tile's own face, as the concept's, but without ids or drag
    handles of its own, `inert` and hidden from screen readers, so no id is on the page twice while it shows. A pedal
    or side unit dragged while the Pedalboard is not on the desk previews the Pedalboard holding its drawing (the
    picker's), where the concept draws the whole pedal. The tile that lands does not play the concept's landing (it
    settles from the preview's box with a lift and a shadow): that is motion, package 9's.
40. **The picker's buttons, the Lab offer's and a grip's moves say their type** (deviation 6).
41. **A move from a grip's menu merges with a key move on the same grip made in the last two seconds**: one rule of
    the store's (`move`) for a run of moves on one grip; the concept's menu moves are steps of their own.
42. **Escape in the Modules panel opened over the desk at 1280 closes the panel too while a popover is open from one
    of its cards**: the panel's own Escape (`shell/panelState.ts`) runs before the layers' (`features/layers.ts`).

20. **A terminal line hangs under its first flag wherever it stands** (the planner's rule, L-24): a repetition guard's
    line is the same part, so it carries `--hi` too; its stylesheet takes the hang off a bar-less terminal's first line,
    so it is drawn as the concept's plain line.
21. **A note in a terminal's bar is in the bar's own ink** (#a39cae, the flags' and chips'), which reads at 3:1; the
    concept's quiet ink reads at 2.88:1.
22. **With the planner off, the repetition guard says "planner off" through its terminal's standing back**, its line
    through its own: two `dim`s in the definition (the terminal's and its line's) where the concept wrote two spans by
    hand; MiniMax-Music3's guard, whose terminal never stands back, has no "planner off" span.
23. **The planner's bar says what Plans & endings set** through the note's own words (`{value:mm.reuse}`, a line per
    replay), not a paint hook; the plan text's MiniMax-Music3 line is the concept's ("plan --model 8B").
24. **A control none of whose parameters applies is not drawn** (`nodeShown`), as the concept leaves out what `ok()`
    refuses; a node that should stay and stand back says so with `dim`. Package 1 flagged the gap.
25. **(Closed by package 6: the cover swatch shows the newest song's colour.)** It showed the concept's colour for no song (300) until the generation package listed songs; the
    concept's page has its demo songs (215).
26. **A drop zone names its file field and its clear button by its state's noun** ("Choose the timbre reference"); the
    concept wrote each by hand ("Choose a reference track").
27. **ACE-Step's card says ×8**, the most takes the definitions give it (the server takes 8 at most); the concept
    says ×9.
28. **A model left at "" (the engine's default) is the file the server has selected**: the row shows that file, its
    list has no "Engine default" row while the server lists it, a step names it, and choosing it stores "" (question
    5). The concept's default is the first file of the list; with the simulated server, which selects the first, the
    page looks and behaves the same.
29. **(Closed by package 6.)** The VRAM meter swings while a run renders and shows another engine rendering (B-34).
    Its click opens the engine log, and its `aria-expanded` says whether the log is open (package 11).
30. **Ctrl+Z with focus in the engine panel undoes** (the page's undo keys); the concept's does nothing there.
31. **LM seed follows stands on the Seed tile while the planner is on the desk**: it is the planner's switch, and with the
    planner off the desk it would change nothing that is sent (the value in force). The concept shows it at Lab always.
32. **The saved seeds are the desk's state, and every change of them is a step** (save, update, star, unstar, delete),
    as every change of the desk is; the concept's save and star are no steps. The test page starts with the concept's two
    demo seeds (`testing/compat/instruments.ts`); the product starts with none.
33. **The recent references are the server's reference tracks** (`services/references.ts`, as the existing frontend's
    `useReferenceTracks`), remembered when a drop zone takes a file and never taken back by Undo; the definition's
    `recentRefs` state stays declared and holds nothing.
34. **The Takes tile's VRAM line counts the engines' default models** until the engine panel's package gives the loaded
    models' figures; ACE-Step has eight reels (docs/03, Takes), three a row, the footprint of the concept's nine (its
    rule would put eight in two rows of four, a shorter tile, and the desk would pack differently from the concept's).
35. **A seed is typed up to 2^64-1**, the application's range (the concept's 4294967295), and the field takes 20 digits.
36. **A repaint's song is a record of its own** (`{ song, title, style, dur, hue }`, written by the song menu's "Repaint a
    region"), so the region needs no song list; the region is kept in seconds, as the real application keeps it.
37. **At Studio Repaint's mode and crossfade stand in a row (`.knobs`) beside the wave**, where the concept draws a
    `.rp-ctl`; at Lab they stand on the wave's caption line, as the concept's.
38. **A seed's refusal, the saved seeds' update and delete, and the plain loads say what they did in a toast through
    the core** (`partAct`'s words), the update and delete with Undo; the concept's Update toast offered Undo too.
Package 7's (numbered P7- so that the packages built beside it number theirs without a clash):

- P7-1. **The row menu and the Drafts list are a popover of the package's own** (`workflows/pop.tsx`, drawn into the
  skeleton's `#pop` with the concept's placement, Escape and outside press); package 8's popover takes them over.
- P7-2. **Saved records are kept under the page's own keys and in its own shapes** (deviation 5): `hotstep.create.
  workflows` (`{ v: 1, workflows, revisions }`, a revision naming its workflow), `hotstep.create.drafts`,
  `hotstep.create.workflow` (the open workflow and revision, the workflows whose revisions show). The concept's stored
  list (version 0) and its revision files are read. The test layer reads the concept's keys as the page's.
- P7-3. **A revision holds every setting of its engine, defaults included** (`revisionOf`), so its details list those
  not at their default; the concept's holds the values that were set, and lists those.
- P7-4. **"Modified" is worked out for a workflow too**: its layout differs from the desk's (what Revert would put back,
  F-04). The concept's flag is set by any change to the layout and cleared only by Revert or Update, so a module added
  and taken off again reads modified there and not here.
- P7-5. **Content held as empty text replaces nothing on load**: the concept's rule for the lyrics and the caption, for
  every content text.
- P7-6. **The export file is the page's record**: `{ format: 'hotstep-revision', v, workflow, revision }`, the revision
  naming its engine (`engine`), the file named with the engine's short id (`.mm3.json`); the concept's file (`backend`,
  `slots`, `content`) imports.
- P7-7. **A workflow renamed in another tab is followed** (pending F-17) when a workflow of the same tier, modules and
  board appears there as the open one goes.
- P7-8. **The Workflows list's keys** (pending F-36): F2 renames a row of yours, Delete opens its menu with Delete armed,
  a letter moves to the next row whose name starts with it.

Package 11's (the Concept notes and the engine log):

- P11-1. **The notes say what the rebuild does where it differs on purpose.** Four sentences changed: Alt+1, 2 and 3
  choose a tier "from anywhere while no dialog is open" (pending F-08, built); Ctrl+Z changes nothing under a dialog
  "(the engine panel aside: there it is the page's Undo)" (deviation 30); the held takes read "Your 8 takes return on
  ACE-Step 1.5" (deviation 27: 8 at most); a knob's "double-click or Delete still resets it" (deviation 13). Every other
  sentence holds of the rebuild as written.
- P11-2. **The log tags the engine's models line with the engine's short id** (`engines.ts short`: `[yue2]`), where the
  concept writes its own (`[yue]`); ACE-Step, MiniMax-Music3, Stable Audio 3 and HeartMuLa read the same.
- P11-3. **An overlay's exit copy carries no ids** (`ovOut`), so no id is on the page twice while it plays (deviation
  39); the concept's keeps them.
- P11-4. **A toast that shows while any modal layer is open joins its Tab cycle** (pending F-18, for the picker and the
  notes; `layers.ts` does it for every modal layer, the picker's keys for the picker).
- P11-5. **The tier keys wait under a modal layer or a side panel open as a drawer** (pending F-08): `modal-keys.ts`.
- P11-6. **(Closed by package 6, which tells the queue as runs come and go: `store/generate.ts`.)** The log's queue line said idle until the generation package told the queue (`tellQueue`); the concept's
  counts its songs rendering and waiting.
- P11-7. **The notes' × and the log's Close say their type** (deviation 6). The notes button and the VRAM meter open
  their overlays through a delegated click, the meter's `aria-expanded` set by the log as it opens and closes, so the top
  bar (package 5's) is not edited; the same markup and behaviour as the concept's.

43. **The empty Adapters tile is its definition's strip**, drawn by the tile: its buttons run the named actions and carry
    none of the concept's hooks (`data-cartadd`, `data-plannerpick` on the strip; deviation 10). The cartridges' own
    buttons keep them.
44. **A phase window's and a section mask's steps** are named as the concept names them (L-01, L-09), and their Undo and
    Redo toasts say that name; the concept's said "… back to 0–100%". Several masks painted in one drag join their
    files in the core's English ("Section masks of a and b").
45. **Apply as while runtime is forced** is the definition's `forced` (ParamFace): the segmented choice draws it,
    wherever a choice has one; what is sent does not change (the server forces runtime itself).
46. **The LED switch carries its parameter's tip** wherever it has one (E-17: the plans' Endings); the concept wrote
    those three by hand.
47. **The planner adapter's list holds the server's planner LM adapters** (`adapters.lm`, told by
    features/adapters.tsx from services/adapters.ts), as `ad.planner`'s options; their value is the file, their words
    the concept's ("folk-voice-planner (LoRA)").

Package 6's (numbered P6 so that packages built at the same time do not collide):

- P6-1. **Generate does not start a run certain not to fit** (F-21, pending in the concept): the line's "Use N takes"
  pulses and the line is said. A "Didn't run" notice comes only from a refusal the server makes (the simulated server
  checks its own card; `mockCard` makes it smaller in the test build). So 06-prompt-generate/generate-engines' first
  check and the pending F-13 and E-03 scenarios, which get their notice from pressing Generate, cannot pass as written.
- P6-2. **The document's title while songs render is "(3) " before the page's own title** ("HOT-Step"), not
  "(3) HOT-Step Tiers".
- P6-3. **A double-click on Generate is one press, and separate presses each queue** (F-22): the click's own count,
  where the concept ignores any press within 350ms.
- P6-4. **The song menu opens above its button and never over the go bar** (F-28): with no room on either side and no
  page to scroll it takes the roomier side and scrolls inside. The concept opens it below and moves the page.
- P6-5. **The queue is the server's**: after a reload the page says the songs the server no longer has (F-27, pending
  in the concept); the simulated server loses its queue on a reload, as the concept page does, so that is every song
  that was rendering. Leaving while songs render asks first (F-37).
- P6-6. **Songs are the server's**: the simulated server keeps them under `hotstep.mock.songs`; the page's "not played
  yet" marks are not kept over a reload (the concept keeps them in its autosave).
- P6-7. **The cover swatch follows the newest song as it arrives**; the concept repaints it only when the board is drawn
  again.
- P6-8. **"N changes" counts the built-in boxes' switches and choices, not their text** (`changedKeys`): the concept's
  list differs in the trigger word (text, not counted here) and MiniMax-Music3's and YuE2's caption source and dataset
  (choices, counted here).
- P6-9. **Ctrl+Enter is heard on the window in the capture phase**, before any row's or card's own Enter (F-07, pending
  in the concept), not through `onKey` (which the page dispatches after its controls').
- P6-10. **The request is `pageIntent`** (the desk in the real application's names, request.ts realState), not
  `createIntent` with the application's resolver, which needs the real application (package 10).
- P6-11. **"Open the Library" in Latest goes nowhere**: no package of this branch builds the navigation to the stub
  pages.

Package 3c's:

- P3c-1. **A build is the user's record and the server's session.** The record (state `build`) holds what Undo covers:
  the session, the score's seed, the sections planned, the keeps, the takes discarded and the songs rendered; the session
  holds the takes and the section looked at. The concept keeps both in `S.build`. The first plan is a step ("Plan the
  score"); the concept's makes none.
- P3c-2. **Discard and Re-plan leave the server's takes alone**: a discarded take is marked so in the record, and Re-plan
  opens a new session; Undo of either brings the takes back from the session that still holds them.
- P3c-3. **Undoing a render withdraws its song from the Library** through the build session (the record's songs follow
  Undo and Redo; `store/build.ts` keeps them among Latest's cards, package 6's, while their render stands).
- P3c-4. **Follow score and Plan strength are locked while any section is kept**, worked out from the keeps, so
  discarding the last kept take frees them (pending E-24).
- P3c-5. **Play kept** beside Render (E-46) and **Continue's play button** (E-47), added as the review asked; the
  simulated server plays no sound, the playhead runs; the player's audio is package 6's.
- P3c-6. **A recording's length is what the page is told of it** (`services/recording.ts`; the simulated one is two to four
  minutes by its name, the concept's a random length each load), said on the sleeve ("3:51 long"). The part chosen is kept
  while the recording holds it, so replacing the recording with a long enough one keeps it (the concept goes back to the
  first 20 seconds).
- P3c-7. **With Chain of Thought off the lead sheet's paper stands back and says why** (its text is not sent then), with
  the concept's placeholder; the concept leaves it at full strength.
- P3c-8. **The score preview is a modal layer** in `#scoreModal` (package 8's layers: Tab goes round it, Escape and the
  scrim close it, focus goes back to what opened it; package 11's exit). Continue with this score renders it with its
  seed through `scoreRuns` (registered with package 6's `generate({ seed })`, which then does not preview it again).
- P3c-9. **The planner's composer CFG token is not painted as clashing** by the lead sheet (the concept's cross-module
  paint, B-09); the lyric-timing line warns, in the warning's ink, with Set CFG to auto.
- P3c-10. **A tile that is a strip says what is set at another tier** (its Also set line), as every tile of the concept's
  does; the slice left it out on strips.

The doctype (the review's deviation 1) is gone: the page is in standards mode; `styles/rebuild/10-standards.css` keeps
the two boxes that would move (the engine caret and the LCD picker's arrows) where the concept draws them. Deviations
3, 4 (an LCD picker's words) and 9 (Avoid's ellipsis) are closed.

## Adding from the how-to, on a copy

On a scratch copy of `ui-next`, following [06](06-how-to.md) as it is (the trials are in the revision's notes), each
with `npm run check` and `npm run lint` passing and looked at in both themes at every tier:

| Job | Files written by hand | Lines | Besides |
|---|---|---|---|
| Add a parameter (APG eta on two engines) | 1 (`guidance.ts`) | 11 | `npm run docs` (2 generated docs) |
| Add a pedal (a stereo widener) | 2 (its definition, the index) | 55 + 2 | its drawing (13 lines), written by `node tools/thumb.ts widener`; docs |
| Add a module of existing parts, a face per tier (Dynamics) | 4 (definition, a unit in `parts.ts`, index, drawing) | 37 + 1 + 2 + 17 | docs |
| Add a part (a stepped slider) used by two definitions | 5 (`PARTS`, component, stylesheet, the two definitions) | 4 + 24 + 17 + 1 + 1 | `gallery.mjs --record` once it looks right; docs |
| Remove a module (Avoid) | 4 (its file deleted, index, `retired.ts`, the factory workflow naming it) | 31 deleted; 2 + 2 + 2 | docs. The slice's own browser checks add Avoid (`30-slice.mjs`, `00-slice.mjs`): removing a slice module means giving them another, as 06 says |

A new part is checked on every surface by the gallery with no line written there; until its looks are recorded the
gallery fails on it alone.

## How the rebuild will sit inside the real application beside Tailwind

The existing frontend loads Tailwind (with its preflight reset) and its own `index.css`; the concept's stylesheets carry
their own reset (`01-base.css`: `body`, `button`, `input`...) and tokens on `:root` (`--bg`, `--fg`, `--line`; none
collide with the existing names, which are `--bg-primary-*`, `--color-*` and the like). Loaded together on one document,
each reset reaches the other's pages. The choices, for the maintainer:

1. **Scope the concept's stylesheets under the Create page's root** at build time (a PostCSS step that prefixes every
   selector with `.hs-create` and moves `:root` tokens and element resets onto it). One document, one React tree; the
   selectors' specificity rises evenly, which the visual baseline can prove changes nothing.
2. **Render the Create page in a shadow root.** Exact isolation without rewriting a rule, but portals, the fonts,
   document-level keys and the tools' element keys all need care.
3. **Keep the Create page its own entry** (its own HTML document under the same server) until the other screens move
   to the same design language, then merge.

The page is already in standards mode, as the real application's is, and proved so against the concept. Translation
runs on an instance of its own, so it cannot take over the application's i18next. Recommendation: option 1.

## The plan for the rest

Each package adds files and registers them (above, "Adding without colliding"); it owns the slice files named for it,
and no other package edits them. Each is accepted when its acceptance file passes, its switches each make a scenario
fail, its visual states match the concept with its accepted differences counted, the gallery passes with its parts in
it, and `npm run check` and `npm run lint` pass. Order: 1 and 2 first, then 3 to 8 in parallel, then 9 and 10.

**How a package is accepted, and how a merge is verified.** A package's three test files register themselves: its
acceptance file is run by `run.mjs --slice`, its switches file by `switches.mjs`, and its states file, which ends with
its visual set (`export const VISUAL_SET = { name, only, scope, accept }`: the states by id, their scope, its accepted
differences; tools/visual-baseline/README.md, "Every set at once"), by `visual-all.mjs`. Nothing else is added to any
runner. Then, on the package's branch and again on the tree after each merge:

```
npm run check && npm run lint
npm run build:test && npm run behaviour -- --slice --workers 6   # the whole acceptance (under 3 minutes)
npm run switches:apply                               # every switch still applies exactly once (a second; merges break this most)
npm run visual:all                                   # every package's visual set, one line each (about 13 minutes)
npm run visual:all -- --reuse-concept                # the same with the kept concept recordings (about 7 minutes)
npm run switches                                     # every switch built and tried (about an hour once caught/ is filled)
node tools/visual-baseline/gallery.mjs && node tools/visual-baseline/plants.mjs
```

The times are at the tools' default of 6 workers (half of a 12-thread machine; `--workers` lowers it). A switch
costs about 9 seconds once `caught/` names what catches it, about 16 seconds (more for a large package) when it starts
from its own package's acceptance, and a whole acceptance run (about 2 minutes 40) when only another package's scenarios
catch it; the old way, the whole acceptance for every switch, would take about 18 hours for the 387 switches.

`visual:all` prints each set's states and `identical`, or how many differences go beyond its accepted ones, and exits 1
if a set differs. `switches` tries each switch first against the scenarios that caught it the last time
(`tools/behaviour/caught/<package>.json`, written by the run and committed with the package), then the rest of its own
package's acceptance, then the rest of the acceptance, and counts a scenario as catching it only if it passes with the
feature on; the verdict for each switch is the one the whole acceptance would give (tools/behaviour/README.md,
"Switches"). A package's own switches alone: `npm run switches -- --package "^<package>$"`.

| Package | Adds (files of its own) | Registers | Slice files it owns and edits |
| --- | --- | --- | --- |
| 1. Shared controls | `parts/` fader, ruler, numberBox, lcdNumber, chips, termChips, lcdChips, pill, flag, lcdButton, recButton, lockButton, powerChip, token, lookback; `styles/rebuild/15-controls-1.css`; `acceptance/1-controls.json`, `switches/1-controls.json`, `slice-states/10-controls.mjs`, `compat/controls.ts` | `definePart` | `parts/Pick.tsx` and `parts/Dropdown.tsx` (the plugin, chip, model and stamp variants), `parts/ChoiceRow.tsx` (its marker), `parts/TextField.tsx`, `parts/TextArea.tsx` (their looks) |
| 2. Containers and read-outs | `parts/` lcdRow, terminal, termBar, termLine, group, deck, pair, bank, link, dice, beatPulse, flowWindows, speedGauge, adhereCurve, planText, lrcPreview, coverSwatch, unitStatus, unitBadge, dropZone; `styles/rebuild/15-containers.css`; its acceptance, switches, states | `definePart` | `parts/containers.tsx` (if a container's housing changes) |
| 3. Bespoke instruments, one each | `parts/<Part>.tsx`, `styles/rebuild/25-<module>.css`, `services/<module>.ts` (Build's session, the planner's runs), `store/<module>.ts` for server-owned state, `compat/<module>.ts`, its acceptance, switches, states | `definePart`, `defineService`, `defineUndoable` where Undo covers it | none |
| 4. Prompt card and lyric sheet | `parts/PromptBox.tsx`, `parts/LyricSheet.tsx`, `features/prompt.tsx`, `features/field-undo.ts`, `store/fields.ts`, `styles/rebuild/25-prompt.css`, its words, acceptance, switches, states | `fill('centre.prompt' / 'centre.quickLyrics' / 'desk.lead')`, `onKey({ id: 'field-undo', order: 10 })` (Ctrl+Z in a field first) | `tools/visual-baseline/slice-states/00-slice.mjs`: the lyric sheet's withdrawal goes |
| 5. Engine panel and models | `parts/ModelPicker.tsx`, `features/engine-panel.tsx`, `services/models.ts`, its words, acceptance, states | `fill('engine.panel')`, `defineService` | `shell/TopBar.tsx` (the VRAM meter's figures) |
| 6. Generation, go bar, Latest, player, song menu | `features/generate.tsx`, `generate/`, `player/`, `store/generate.ts` (the queue: server-owned, not undoable), `services/generate.ts` | `fill('centre.gobar' / 'centre.latest' / 'overlays')`, `onKey({ id: 'generate' })` (Ctrl+Enter) | `shell/Toast.tsx` (the stack of three, Tab into it) |
| 7. Workflows, revisions, drafts, storage | `features/workflows.tsx`, `workflows/`, `store/workflows.ts` | `fill('panel.workflows')`, `defineUndoable('workflow')`, `deskStep` for opening one | `src/derive/records.ts` (unreadable records kept aside) |
| 8. Picker, overlays, tooltips, touch, drag and drop | `features/picker.tsx`, `features/popover.tsx`, `features/tooltips.ts`, `features/touch.ts`, `features/focus-ring.ts`, `features/dnd.ts` (delegated listeners on the desk and the board, `add(id, before)` and `move`) | `fill('overlays')`, `onKey` (F9, Escape, F6) | `shell/Panels.tsx` (the Lab offer becomes the popover) |
| 9. Motion | `features/motion.ts`, `styles/rebuild/35-motion.css` | `onKey` none | `shell/tier.ts` (the View Transition around the tier change), `desk/layout.ts` (FLIP around a pack), `board/cables.ts` (cables drawing out) |
| 10. The real application | `services/real/*.ts`, the Tailwind scoping | `defineService` real adapters | `main.tsx`, `index.html`, `App.tsx` (its root inside the real app) |
| 11. Concept notes and the engine log | `features/notes.tsx`, `features/notes-open.ts`, `features/logs.tsx`, `features/logs-dock.ts`, `features/overlay-exit.ts`, `features/modal-keys.ts`, `services/logs.ts`, `store/notes-logs.ts`, `src/derive/logs.ts`, `i18n/words/notes.ts`, `tools/test-notes-logs.ts`, its acceptance, switches, states | `fill('overlays')`, `defineService('logs')`, `layerOpen` (the notes), `onKey({ id: 'logs-escape', order: 90 })`, `onKey({ id: 'modal-waits', order: 6 })` (F-08) | none; two lines in others' files: `features/layers.ts` (a toast joins a modal layer's Tab cycle, F-18), `shell/placeToast.ts` (the toast stands on the docked log); at the coordinator's request, unique ids in drawings: `shell/engines.ts` and `thumbs/index.ts` (a place per drawing), their callers `shell/TopBar.tsx`, `workflows/List.tsx`, `workflows/Drafts.tsx`, `features/picker.tsx`, `features/dnd.tsx` (one argument each) |

No file is edited by two packages, with three planned exceptions: `src/definitions/parts.ts`, where a package that
needs a new option on a part appends it (the type check refuses a clash, and the rows are independent); the generated
pages of `docs/` (`npm run docs` rewrites them; a merge regenerates them); and the slice's acceptance and states files,
which only package 4 touches (one line) and which every other package leaves alone by adding its own.
