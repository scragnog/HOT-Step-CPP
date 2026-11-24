# The React layer: architecture, the slice, and the plan for the rest

This page is for the people who build the rest of the Create page and for the maintainer. It says how the React layer
sits on the core ([05](05-what-the-definition-drives.md)), what the first slice built, how the page is checked, the
design decisions that explain why it behaves as it does, and how the remaining work splits into packages that can be
built in parallel.

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

Not built, and left as seams: the bespoke parts and the modules made of them, the
engine panel's models and the VRAM figures, generation, the go bar's contents, Latest and the player,
workflows and revisions, drafts, the remaining overlays (score modal, song menu), and the
choreographed tier transition. "The plan for the rest" below divides them into packages; package 9 ("Motion", below)
built the transition and every other motion.

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
                           undoable state (undo-keys, tier-keys, drawers, motion-offscreen); motion.ts,
                           motion-blocks.ts, motion-kicks.ts (package 9: the desk's own transition, what pushes what
                           follows, the small motions)
      model/               the core indexed once (ix; the test build adds the parts gallery), definitions by id
      store/desk.ts        the desk store: desk, engines' book, modules and board order, undo/redo, autosave
      store/panels.ts      which side panels stand open, which drawer is open
      services/            index.ts (the boundary: every service by name), one interface a file; mock/ (the simulated
                         server, registered by main.tsx in one line)
      i18n/                i18next on an instance of its own; words/*.ts: the page's own words, one file per package
      styles/              concept/ (the concept's stylesheets, unchanged), index.ts (their order), rebuild/*.css (the
                           rebuild's own, loaded after: parts on every surface, the shared part states, standards mode)
      board/, desk/, shell/, thumbs/   the shells: the Pedalboard, the desk and its layout, the bars and panels
      testing/             compat.ts (the concept's globals for the two tools) and compat/*.ts (one per package);
                           gallery.ts (every part on every surface)
  tools/
    motion/                frames.mjs (each motion against its approved samples, step by step), approved/*.json,
                           cost.mjs (what each costs the main thread), motions.mjs (package 9)
    behaviour/             the behaviour suite; acceptance/*.json (one per area), scenarios/*.mjs, switches.mjs +
                           switches/*.json, caught/*.json
    visual-baseline/       the visual baseline; slice-states/*.mjs (one per area, each with its visual set),
                           approved/<set>/ (the approved recordings), visual-all.mjs,
                           gallery.mjs (every part on every surface), plants.mjs + plants/*.json (planted defects)
    test-view.ts           the layout, the core's selectors and models, the parts' contract at run time
    test-parts-contract.mjs  the parts' contract in the lint: the review's planted part refused
```

`npm run check` runs both type checks, the validator, the request comparison, every `tools/test-*.ts` file (the
core's, the view's and each package's), the parts' contract (`tools/test-parts-contract.mjs`), the code's contract with
the definitions (`tools/test-code.ts`: no key or id named in the code, every part and action built) and that every
switch and planted defect still applies (`switches.mjs --applies`); [07](07-checking.md) lists them. `npm run dev` serves the page; `npm run build:file` writes the file the owner publishes;
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

Motion (package 9) needs no stylesheet of its own: the concept's `35-motion.css` (the tier change's choreography, the
overlays' entries, reduced motion as a designed alternative) and each component's transitions apply as they are.

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

Every call the page makes to the server is a method of a named service, and the services are listed in one place,
`services/index.ts` (`Services`: `lists`, `models`, `generate`, `adapters`, `references`, `logs`, `recording`, `score`,
`build`), each interface in a file of its own beside it, grouped as the existing frontend's API layer groups the routes
(`ui/src/services/api.ts`). A consumer asks for a service by name, `service('models')` (`register/services.ts`, typed
by `Services`), and never imports an implementation: the lint refuses an import of `services/mock` anywhere but
`main.tsx` and the test layer. `main.tsx` registers one implementation of every service before anything asks for one;
today that is the simulated server, `services/mock/index.ts` (the concept's catalogue, queue, sessions and lists), and
package 10's real adapters take its place there: one import, nothing else changes. What is sent is built by the core,
never here: `generate.submit` takes the run the core worked out (`derive/generate.ts runOf`) and its request is
`createIntent(ix, spec.desk, real)` (`derive/request.ts`) with the application's own resolver and content builder, the
one request the comparison verifies over the value space ([07](07-checking.md)); the simulated server sends nothing.
(The round found that a second request function, `pageIntent`, the desk in real names without the application's
resolver, differed from the verified one in 650 fields once the server resolved it, RA-20; it is gone.) What the server decides about a run's
songs (a title from the words, a length where the engine ends the song, the art's colour) is the service's `draft`, not
the core's. A bespoke part responsible for its own conversation with the server (Build by section's session, `PARTS`
`responsible`) reaches the boundary through the kit's `service`.

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

## Motion

Motion is drawing only, and it never fights React's rendering: React draws each state as it always does, and motion reads
where things stood before a change and plays the difference afterwards with the Web Animations API, on `transform`,
`opacity` and `clip-path` (nothing that lays the page out on every frame, and no component drawn again per frame). Its
vocabulary is the concept's (`00-tokens.css`: `--t-*`, `--e-*`, read through `kit/motion.ts`), and under reduced motion
each motion plays the concept's designed alternative: nothing moves, every change of state still shows (a 120ms fade,
rings as still outlines).

- **The tier change** (`shell/tier.ts`) is one View Transition in which every part present in both tiers is one object:
  `PARTS` names each part with its `view-transition-class` (`frame`, `keep`, `head`, `slide`, `swap`, `band`, `still`,
  `only`; the choreography is the concept's `35-motion.css`, `html.tier-morph`), React draws the new tier inside the
  transition's update (`flushSync`) and the desk is laid out there (`desk/layout.ts flushPack`), so what is captured is
  the page as it will stay; the tile being read keeps its place (focus, else the pointer, else the bars' edge, Q-14).
  A new part of the page that should take part is one line in `PARTS`. `instant` changes the tier without it, for a
  caller that acts in the new tier at once (Lab's offer).
- **The desk's own transition** (`features/motion.ts`): the store tells it, before the page is drawn again, that the
  modules, the board, the engine or the tier change; where every tile and pedal stands is taken then, and once the desk
  has been laid out (`desk/layout.ts onPacked`, after the commit) arrivals scale in, exits fade where they stood (the
  node React let go of, without ids), moves slide, a drop lands from its preview's box, a pedal drops onto the board, the
  new cables draw out and every cable follows the pedals (`board/cables.ts drawOut`, `followCables`); what was added is
  brought into view. Quick's fold opening or folding again (`store/fold.ts`, kept with the session) plays the same
  transition. A change to the desk's make-up made some other way than the store is wrapped in `transition(fn)`.
  A content re-pack (a tile that grew) slides what it moves (`desk/layout.ts slidePack`).
- **What pushes what follows** (`features/motion-blocks.ts`): the prompt box growing or shrinking by more than a line and
  a half, a suggestion, Song info and the save row opening and closing; watched with a ResizeObserver (after layout,
  before paint), what follows slides from where it stood and the card is revealed down to its new edge.
- **The small motions** (`features/motion-kicks.ts`): an LED switch's kick, a stamp's thunk, a side panel's fade,
  Generate's label and the saved state's word cross-fading, the decorative loops resting (`body.calm`). Each is read
  from the markup React draws (a MutationObserver on one attribute, or on the two bars), so no part carries it.
- **Parts and shells that move on their own** keep their motion where it is (the split-flap, the reels, the key wheel's
  nudge, the XY pad, knob and fader settle in the concept's CSS, the go bar, Latest, the toasts and overlays of packages
  6, 8 and 11); a playhead or the VRAM meter, which move on every frame, are set on the page from the animation loop
  (`parts/Build.tsx`, `parts/Recording.tsx`, `shell/TopBar.tsx`), never by drawing the component again.

Checked step by step (`tools/motion/frames.mjs`: 20 motions, every running animation moved to 0, 16, 32 ... ms and the
moving boxes compared with their approved samples within 0.5px; until `rebuild-parity`, with the concept's within 2px)
and in the acceptance (`scenarios/38-motion.mjs`, which reads the page at moments of each motion). What each costs the
main thread on a full desk is `tools/motion/cost.mjs` (package 9's notes give the numbers before and after).

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
| A keyboard shortcut | a file in `src/app/features/` | `onKey({ id, when, run, order, create })` (register/keys.ts); `create: true`, a key of the Create page, is not asked while another page shows (F9, Alt+1/2/3, F6, Space for the player, the drawers' keys) |
| Something drawn in the shell (the prompt card, the lyric sheet, the go bar, Latest, the Workflows list, the engine panel's models, an overlay) | a file in `src/app/features/` | `fill(slot, Component)` (register/slots.ts): `desk.lead`, `centre.prompt`, `centre.quickLyrics`, `centre.gobar`, `centre.latest`, `panel.workflows`, `engine.panel`, `overlays` |
| An overlay that takes focus and gives it back (a dialog, a menu, a panel) | its own file | `layerOpen(el, { trigger, modal, undo, close })` and `layerClosed(el)` (features/layers.ts): Escape closes the top one, Tab stays in a modal one (a toast raised since it opened joins it), Undo waits under one (`undo`: it acts there, the engine panel's deviation 30), Ctrl+Enter, F6 and F9 wait |
| Something in the popover (a list, a menu, an offer) | its own file | `openPop(button, () => <Body />, { label, place, fixed, focus })`, `closePop()`, `placeFloat` (features/pop.ts): the page's one popover (RA-04) |
| An overlay's exit (a copy of it played out while it is hidden at once, the concept's ovOut) | its own file | `ovOut(el, kind)` just before hiding `el` (features/overlay-exit.ts, package 11) |
| What the queue holds, for the engine log's queue line | the generation package's file | `tellQueue({ rendering, waiting, engine?, loaded? })` (store/notes-logs.ts, package 11); until it is told, the log says idle |
| A named action (parts.ts ACTIONS) | a file in `src/app/features/` | `defineAction(name, run)` (register/actions.ts); `run(arg, from)` is given what the part chose and the part, whose gesture its change joins: a bespoke part reaches it by `v.act(name, arg)` when its node lists the action |
| State of its own | `src/app/store/<package>.ts` | its own store; `defineUndoable(name, { read, write })` when Undo covers it |
| View state a reload brings back (a fold or a section standing open) | its own file, or a part's | `keptState(name, fallback, take)` or `keptFlag(name)` (kit/session.ts): kept with the session under one key, never a step and never in the desk's record; `useKept(k)` in a component. The test layer gives them the concept's names (`testing/compat/fold.ts`) |
| A server call | `src/app/services/<package>.ts` | `defineService(name, impl)` (register/services.ts) |
| Words | `src/app/i18n/words/<package>.ts` | gathered by i18n/index.ts; an id defined twice fails at load |
| Handles its scenarios and states drive the page by | `src/app/testing/compat/<package>.ts` | `install()`: `expose`, `onMirror`, `onScene` |
| A part of the page that takes part in the tier change | `shell/tier.ts` | one line in `PARTS`: its selector, its name, its class (package 9) |
| A change to the desk's make-up made outside the store | its own file | `transition(fn)` (features/motion.ts): what fn adds, removes or moves animates as the desk's own changes do |
| Its acceptance | `tools/behaviour/acceptance/<package>.json` | `run.mjs` runs every file; every scenario is in one |
| Its features switched off | `tools/behaviour/switches/<package>.json` | `switches.mjs` checks each makes a scenario fail, and writes what caught each to `tools/behaviour/caught/<package>.json` |
| Its visual states | `tools/visual-baseline/slice-states/<NN>-<package>.mjs`, ending with its `VISUAL_SET` | `slice-states.mjs` gathers them; `visual-all.mjs` runs every set against `approved/<set>/` |
| Its approved recording | `tools/visual-baseline/approved/<set>/` | `npm run visual:approve -- --only "^<set>$"`, committed with the change |

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
- The definitions: an option's `pick` words (what an LCD picker prints where it differs from a read-out), a module's `starter` (the empty desk's offers and the words it names the first with),
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
- `derive/steps.ts stepName`: a module's parameter is named with its owner when another parameter **on the desk** prints
  the same words, so adding a module no one has put on the desk renames nothing (SL-13). A pedal's or side unit's is
  counted as the concept counts it, among every definition on every engine, so a generic knob on the board ("Mix") always
  says whose it is (the fix round's RD-11); an owner whose name already ends with the label is said once (RD-18).
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
  pedals that run, play while rendering, the re-planning tries; its request is `createIntent`, built by the service), what
  a run says while it renders (`stageOf`, `takeProgress`, `readyOf`, `attemptsAt`, `queueStage`), the server's refusal
  (`refusedWords`), the go bar's line (`goLine`, `changedKeys`, `changesUnmarked`), Generate and Cancel (`goModel`,
  `docTitle`), cancelling (`cancelWords`, `cancelBack`), Latest (`latestShown`: whole rows, the fold, the count;
  `songCard`, `songArt`, `songWave`, `songName`), the song menu (`songMenu`, `repaintFrom`, `repaintDesk`, `reuseWords`,
  `sentTo`) and the player (`playerWords`, `playerSeek`); the `gen.*` messages. `derive/vram.ts meterModel(ix, desk,
  run)`: while another engine renders, what that engine holds (B-34); `meterTarget`: the needle's swing while a run
  renders. `Desk.songs`: what the generation service tells (the newest song's colour, the tries the run has used),
  which the cover swatch and the dice read (`readoutModel`).
- `derive/fields.ts` (package 4): the prompt card and the lyric sheet. What the sheet draws (`sheetModel`, given to the
  part as `PartModel.sheet`: its sections with their colours, where each starts and the short labels a narrow segment
  tries, the line count and its estimate by the song's tempo (`estimateTempo`: the BPM field, else a BPM written in the
  box, S-07), the tabs, whether the stamp shows (Q-07), Instrumental standing everything back); where it goes
  (`lyricsPlace`: a sheet, or the note that the engine has no lyrics, S-11); what the card draws (`promptModel`, as
  `PartModel.prompt`: the box's name and examples by engine (S-05), the suggestions its words call for, the caption a
  dataset supplies (`trackNow`, Automatic by the tempo field, E-06) and the tags, which tools show); and what each tool
  writes, as one of the prompt's named steps with what it says after (`writeForMe`, `composeCaption`, `expandStep`,
  `moveToLyrics`, `tempoStep` and `latentStep`, which bring the module their value lives on, `plainStep`, `copyStep`,
  `tagStep`, `datasetStep`, `latentClear`, `insertTag`). `derive/models.ts`: a node's named actions say their name
  (`PartModel.actions[].name`; a part runs one by `v.act(name, arg)`); a text parameter's `maxLength`. `derive/core.ts`: a server list's entry may carry
  `data` (a caption dataset's tracks and tags).

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

- The consolidation round (review r3; one rule in one place, after eleven packages met):
  - **Roles** (`src/definitions/roles.ts`, a `role` on a parameter or a state; `core.ts roleKey`, `roleKeys`,
    `roleOwner`, `roleOn`; `ParamModel.role`): what the core and the view need in particular is found by its role,
    never its key, and a module by the role its values play (`roleOn(ix, 'repaintSource', desk)`: the repaint module
    is on the desk) or by what it holds (`model/index.ts LEAD`, `BOARD`, `MODELS`). The validator checks every role;
    `tools/test-code.ts` refuses a key, a state or an id named in the code.
  - **Engine facts** (`engines.ts ENGINE_FACTS`): the box's word, lyrics alone, the tempo field reaching the song, a
    structured caption, how a length is taken; the core reads them, and branches on no engine id.
  - **One VRAM figure** (`vramNeed`, with `first` and `fits`): the Takes tile, the go bar, the meter, the refusal and
    "Use N takes" read it.
  - **One undo-step identity**: `commit`, `deskStep`, `setValue` and `add` return the step they took (`Entry`), and a
    toast's Undo is that step, never "the newest" in the history.
  - **One way in**: what is on the desk is `core.ts onOf` (the store's `applyDesk` sets it from the modules and the
    board); a value set for the user from outside a part is the store's `setValue` (the core's `paramSetter`); a text
    field's Undo amends its step through `amendStep`; a server session is told (`told({ sessions })`).
  - **One instrument registry**: every bespoke part, package 3a's too, registers through `defineInstrument`.
  - **A run's songs are the server's**: `runOf` gives what the page knows of each (`SongAsk`), the generation service's
    `draft` says the rest (`services/mock/songs.ts` for the simulated server).
  - **One motion helper, one icon set, one convenience store**: `kit/motion.ts` (`reduced`, `mtime`, `motionVar`,
    `easing`), `kit/icons.tsx`, `kit/storage.ts` (`readPref`, `writePref`; the records stay `store/stored.ts`'s).

## The test layer

The rebuild is its own reference. Its browser checks compare it with itself: the behaviour suite with what each
scenario says is right, the visual baseline and the motion samples with recordings of the page approved after they were
looked at. Until the tag `rebuild-parity` they compared it with the concept page it was ported from; that tag marks the
last commit proved against the concept (every visual set identical with its accepted differences, every motion within
2px of the concept's, the concept's own scenarios run or excluded with their reasons), and nothing after it is.

`src/app/testing/compat.ts` is the test build's compatibility layer, the handles the tools drive the page by, on top of
the store, in the test build only: `S` (set it, then `render()` applies it as a scene, not a step), `render`, `setVal`,
`get`, `toast`, the catalogue, `UNDO` and `REDO`, and `vsStray`, which detects a change to the desk that no undo or redo
recorded (outside a scene). Its names are the concept's, which the scenarios and states were first written in (an
engine by its short id, a choice by its label); it translates both ways. An area's additions are a file of its own in
`testing/compat/` exporting `install()` (`addItem`, `switchEngine`, `enterEngine`, `setTier`, `S.slots`, `S.build`,
...).

```
npm run build:test
set PLAYWRIGHT_MODULE=<node_modules/playwright>
node tools/behaviour/run.mjs --workers 6              # the acceptance: every scenario of every acceptance file
npm run switches:apply                               # each switch's text still applies exactly once (no builds)
npm run switches                                     # each feature switched off makes a scenario fail
npm run visual:all                                   # every visual set against its approved recording
npm run visual:approve -- --only "^<set>$"           # after a deliberate change: approve that set again
node tools/motion/frames.mjs                         # every motion against its approved samples
node tools/visual-baseline/gallery.mjs               # every part on every surface, both themes, every tier
node tools/visual-baseline/plants.mjs                # each planted visual defect is caught
```

- **Behaviour.** `tools/behaviour/acceptance/*.json`, one file per area of the page, list every scenario of
  `scenarios/`, each with what it exercises (`run.mjs` refuses a scenario in no list). The files numbered below 30 are
  the page's oldest scenarios, written against the concept during the design rounds and kept because each checks
  something the rebuild does; those that only drove the concept's own script, or held the concept's behaviour where the
  rebuild decided otherwise, were rewritten to the rebuild's behaviour or removed at the cutover (commit 1f974479 lists
  which). Every scenario ends with three checks: every change to the desk was an undo step, every step is named in the
  page's words, no console error. `switches/*.json` switch each feature off in turn (`HS_SWITCH`, `vite.config.ts`):
  each must make a scenario fail that passes with it on.
- **Looks.** Each file of `tools/visual-baseline/slice-states/` adds states and ends with its visual set
  (`export const VISUAL_SET = { name, only, scope }`); `visual-all.mjs` captures every set (every element's computed
  style with forced hover, focus and active, its role, accessible name, own words and attributes, the parts on their
  own, and screenshots) and compares it with the set's approved recording in `tools/visual-baseline/approved/<set>/`,
  committed. Computed styles and semantics must be identical; screenshots have one tolerance for anti-aliasing
  (tools/visual-baseline/README.md, "The tolerance").
- **Motion.** `tools/motion/frames.mjs` sets off each motion of `motions.mjs`, holds every animation and moves it to 0,
  16, 32 ... ms, and compares the moving boxes and opacities at each step with the motion's approved samples
  (`tools/motion/approved/<motion>.json`, committed) within 0.5px and 0.01 of opacity. `cost.mjs` measures what each
  costs the main thread (a measure, not a check).
- **Every part on every surface.** `gallery.mjs` draws the parts gallery (`testing/gallery.ts`: every control and
  read-out of the inventory in a tile's body, a row, a rack, a well, an LCD, a tile's foot, a labelled pairing, on a
  pedal and on a side unit, plus a bare new part's markup on each), both themes, every tier, values changed: drawn
  inside its container, words at 3:1, the page's label, a changed mark, a choice that says which is chosen, and every
  look as it was recorded (`tools/fixtures/gallery.json`).
- **Planted defects.** `plants.mjs` plants each defect of `plants/*.json` one at a time and runs the slice's visual set
  and the gallery on it: each must be caught.

**After a deliberate change that shows** (a module added, a part restyled, a motion retimed), the checks that see it
fail, naming what changed. Read the differences, look at the screenshots (`out/now-<set>/shots/`, `frames.mjs
--shots`), then approve the sets and motions that were meant to change, and only those, and commit the approved files
with the change ([06](06-how-to.md), "Approve a change the page shows").

The core's tests (`tools/test-core.ts`, `tools/test-view.ts`, in `npm run check`) still read what the concept page's
own functions returned on recorded desks (`tools/fixtures/concept.json`, `concept-tiers.json`, `pack.json`), recorded
before the cutover; they are the core's expected outputs now, with `concept-differences.json` saying where the core
returns something else on purpose. The scripts that recorded them from the concept are in the tag `rebuild-parity`.

## Design decisions

What the rebuild does on purpose where a reader might expect otherwise, and why. Each keeps the number it had in the
list of deviations from the concept that stood here until the cutover (code comments, acceptance entries and
[08](08-open-questions.md) cite them by it); the deviations that only recorded a difference from the concept, or that
later work closed, went with the cutover and are in the tag `rebuild-parity`.

The page and its markup:

1. **The skeleton is static, React draws into it.** `#app` is React's root; the top bar, the toast, the live region and
   the dropdown are portals into the skeleton's elements; the overlays' containers are filled through the `overlays`
   slot (the picker portals into `#pickerWrap`, the popover into `#pop`). Package 10 undoes this when the page moves
   into the real application's root.
5. **The page keeps its records under its own keys** (`hotstep.create.desk`, `hotstep.create.engines`,
   `hotstep.create.session`, `hotstep.create.langRecent`, `hotstep.theme`, and P7-2's); the core reads the concept's
   saved shape (`migrate`, version 0) for importing an older file.
6. **Every control says what it is to a screen reader**: a knob its value with its unit (`aria-valuetext`), a segmented
   choice is a named group, every button and input says its type (the picker's, the Lab offer's, a grip's moves, the
   notes' × and the log's Close too).
10. **No delegation hooks.** The buttons run their own actions; a dropdown option's `data-opt` is its value, not its
    label. Scenarios find an option by its value or by `aria-selected`.
15. **The switches inside a module's look name their parameter** (aria-label "Chain of thought", not "--cot"), and every
    one carries the changed mark.
18. **A plugin's family, symbol, file and curve** are tables keyed by the option's words as the server lists them
    (`src/definitions/option-faces.ts`); the simulated engine lists the same plugins and languages.
24. **A control none of whose parameters applies is not drawn** (`nodeShown`); a node that should stay and stand back
    says so with `dim`.
22. **A terminal that stands back says why through the definition's `dim`s** (the repetition guard's "planner off"),
    not spans written by hand; 23, **a note's words come from its own `{value:...}`** (the planner's bar saying what
    Plans & endings set), not from a paint hook.
21. **A note in a terminal's bar is in the bar's own ink** (#a39cae), which reads at 3:1.
26. **A drop zone names its file field and its clear button by its state's noun** ("Choose the timbre reference").
39. **A drag's preview is an inert copy**: the tile's own face without ids or drag handles of its own, `inert` and
    hidden from screen readers, so no id is on the page twice while it shows. A pedal or side unit dragged while the
    Pedalboard is not on the desk previews the Pedalboard holding its drawing.
43. **The empty Adapters tile is its definition's strip**, drawn by the tile; its buttons run the named actions.
45. **Apply as while runtime is forced** is the definition's `forced` (ParamFace), drawn by the segmented choice
    wherever a choice has one; what is sent does not change (the server forces runtime itself).
P11-3. **An overlay's exit copy carries no ids** (`ovOut`), so no id is on the page twice while it plays.
The page is in standards mode; `styles/rebuild/10-standards.css` keeps the two boxes that would move (the engine caret
and the LCD picker's arrows) where the design draws them.

Controls and steps:

4. **An XY pad changed from elsewhere is drawn where it is**; it springs only while it is being worked.
8. **An XY pad's key run is a step of its own**, and a double-click reset is a "Reset" step: each gesture one step, as
   every other control.
12. **A tile moved up or down by the keyboard moves its top** that way (comparing middles lets a shorter place in the
    same row pass as "up").
13. **Delete or Backspace resets a value control** (a knob, a token, a fader, a ruler, the look-back strip), as a
    double-click does (S-18), in the value contract (`kit/value.ts`).
14. **A hand moves some values by a rule of its own**: dragging a length ruler snaps to 5 s (15 s above 2:00) and to a
    labelled tick within 6px (S-31); a token over a range of more than 500 steps moves with the value, a hundredth of it
    a pixel (S-28). Both are the core's (`derive/controls.ts` `rulerSnap`, `scrubValue`, through the setters `along`
    and `scrub`).
41. **A move from a grip's menu merges with a key move on the same grip made in the last two seconds**: one rule of
    the store's (`move`) for a run of moves on one grip.
48. **A text field's own Ctrl+Z** (F-22) is kept by the page for every text field (`store/fields.ts`,
    `features/field-undo.ts`); a page step that a field's Ctrl+Z took back to where it started goes. A field bound to
    nothing (a number box's typing, a filter) starts its history afresh when the page draws it again.
FD-1. **A setting that leaves the value in force is no step** (RD-14): a key pressed at the end of a range.
FD-3. **An owner whose name ends with the label is said once** ("Pre-VST gain 0.0 dB → 0.5 dB", RD-18).
FD-5. **A cleared file's toast names the file** ("Cleared take-2.flac"), with Undo.
FD-6. **The reference blend is a percent everywhere** (Format `percent`), its Also set line too.
38. **A seed's refusal, the saved seeds' update and delete, and the plain loads say what they did in a toast through
    the core** (`partAct`'s words), the update and delete with Undo.

Keys, focus and layers:

7. **Adding from the Modules panel keeps focus on the card** (F-25), so several can be added in a row; a starter of the
   empty desk moves it to the new tile.
9. **The toasts are a stack of three** (F-12) that Tab reaches (C-13, F-11); a plain message never pushes out an Undo,
   Escape on a toast's button dismisses it (F-18), and a toast claims Tab only while focus is where it was when it
   appeared and no layer has opened since (S-17).
30. **Ctrl+Z with focus in the engine panel undoes** (the page's undo keys).
42. **Escape in the Modules panel opened over the desk at 1280 closes the panel too while a popover is open from one
    of its cards**: the panel's own Escape (`shell/panelState.ts`) runs before the layers' (`features/layers.ts`).
P11-1. **The Concept notes say what the page does**: Alt+1, 2 and 3 choose a tier "from anywhere while no dialog is
    open"; Ctrl+Z changes nothing under a dialog, the engine panel aside (30); the held takes read "Your 8 takes return
    on ACE-Step 1.5" (27); a knob's "double-click or Delete still resets it" (13).
P11-4. **A toast that shows while any modal layer is open joins its Tab cycle** (F-18; `layers.ts` for every modal
    layer, the picker's keys for the picker).
P11-5. **The tier keys wait under a modal layer or a side panel open as a drawer** (F-08): `modal-keys.ts`.
P6-9. **Ctrl+Enter is heard on the window in the capture phase**, before any row's or card's own Enter (F-07), not
    through `onKey` (which the page dispatches after its controls'); FC-3, **in a field whose Enter is its own** (the save
    row's name, a rename, the draft's name: `register/keys.ts ownEnter`) it is that Enter (RC-09).

Modules:

27. **ACE-Step's card says ×8**, the most takes the definitions give it (the server takes 8 at most); 34, **ACE-Step has
    eight reels**, three a row (docs/03, Takes), whose tile has the footprint the desk's packing was designed around.
28. **A model left at "" (the engine's default) is the file the server has selected**: the row shows that file, its
    list has no "Engine default" row while the server lists it, a step names it, and choosing it stores "" (question 5).
31. **LM seed follows stands on the Seed tile while the planner is on the desk**: it is the planner's switch, and with
    the planner off the desk it would change nothing that is sent (the value in force).
32. **The saved seeds are the desk's state, and every change of them is a step** (save, update, star, unstar, delete),
    as every change of the desk is. The test build starts with two demo seeds (`testing/compat/instruments.ts`); the
    product starts with none.
33. **The recent references are the server's reference tracks** (`services/references.ts`, as the existing frontend's
    `useReferenceTracks`), remembered when a drop zone takes a file and never taken back by Undo; the definition's
    `recentRefs` state stays declared and holds nothing.
35. **A seed is typed up to 2^64-1**, the application's range, and the field takes 20 digits.
36. **A repaint's song is a record of its own** (`{ song, title, style, dur, hue }`, written by the song menu's "Repaint
    a region"), so the region needs no song list; the region is kept in seconds, as the real application keeps it.
47. **The planner adapter's list holds the server's planner LM adapters** (`adapters.lm`, told by
    `features/adapters.tsx` from `services/adapters.ts`), as `ad.planner`'s options: their value is the file.
50. **The caption dataset's default is the definition's**, the adapter's own dataset ("The adapter's dataset", value
    "", as the real application sends it); the supplied caption reads the first dataset the server lists for "".
54. **Compose caption writes the key as the page writes it** ("F# major": keys are stored with sharps).
FD-4. **An empty box is not drawn**: HeartMuLa's planner terminal at Lab ends after its sample line, as HeartMuLa has
    no Lab tokens or flags.
P3c-1. **A build is the user's record and the server's session.** The record (state `build`) holds what Undo covers:
    the session, the score's seed, the sections planned, the keeps, the takes discarded and the songs rendered; the
    session holds the takes and the section looked at. The first plan is a step ("Plan the score").
P3c-2. **Discard and Re-plan leave the server's takes alone**: a discarded take is marked so in the record, and Re-plan
    opens a new session; Undo of either brings the takes back from the session that still holds them.
P3c-3. **Undoing a render withdraws its song from the Library** through the build session (`store/build.ts` keeps the
    record's songs among Latest's cards while their render stands).
P3c-4. **Follow score and Plan strength are locked while any section is kept**, worked out from the keeps, so
    discarding the last kept take frees them (E-24).
P3c-5. **Play kept** beside Render (E-46) and **Continue's play button** (E-47); the simulated server plays no sound,
    the playhead runs.
P3c-6. **A recording's length is what the page is told of it** (`services/recording.ts`; the simulated one is two to
    four minutes by its name), said on the sleeve ("3:51 long"). The part chosen is kept while the recording holds it.
P3c-7. **With Chain of Thought off the lead sheet's paper stands back and says why** (its text is not sent then).
P3c-8. **The score preview is a modal layer** in `#scoreModal` (Tab goes round it, Escape and the scrim close it, focus
    goes back to what opened it). Continue with this score renders it with its seed through `scoreRuns`.
P3c-9. **The planner's composer CFG token is not painted as clashing** by the lead sheet (no cross-module paint); the
    lyric-timing line warns, in the warning's ink, with Set CFG to auto.
P3c-10. **A tile that is a strip says what is set at another tier** (its Also set line), as every tile does.
FD-2. **A side unit's lip buttons draw their keyboard ring inside the button** (RD-15, the owner's rule 1).

Generation:

P6-1. **Generate does not start a run certain not to fit** (F-21): the line's "Use N takes" pulses and the line is said.
    A "Didn't run" notice comes only from a refusal the server makes (the simulated server checks its own card;
    `mockCard` makes it smaller in the test build).
P6-2. **The document's title while songs render is "(3) " before the page's own title** ("HOT-Step").
P6-3. **A double-click on Generate is one press, and separate presses each queue** (F-22): the click's own count.
P6-4. **The song menu opens above its button and never over the go bar** (F-28): with no room on either side the page
    moves first; with no page to move it stands beside its card, level with its button, between the go bar and the
    window's foot (F-34), and scrolls inside only for what that room lacks (RC-04).
P6-5. **The queue is the server's**: after a reload the page says the songs the server no longer has (F-27); the
    simulated server loses its queue on a reload. Leaving while songs render asks first (F-37).
P6-6. **Songs are the server's**: the simulated server keeps them under `hotstep.mock.songs`; the page's "not played
    yet" marks are not kept over a reload.
P6-7. **The cover swatch follows the newest song as it arrives.**
P6-8. **"N changes" counts the built-in boxes' switches and choices, not their text** (`changedKeys`).

Workflows and storage:

P7-2. **Saved records are kept under the page's own keys and in its own shapes** (5): `hotstep.create.workflows`
    (`{ v: 1, workflows, revisions }`, a revision naming its workflow), `hotstep.create.drafts`,
    `hotstep.create.workflow` (the open workflow and revision, the workflows whose revisions show). The concept's
    stored list (version 0) and its revision files are read.
P7-3. **A revision holds every setting of its engine, defaults included** (`revisionOf`), so its details list those not
    at their default.
P7-4. **"Modified" is worked out, for a workflow too**: its layout differs from the desk's (what Revert would put back,
    F-04), so a module added and taken off again does not read modified.
P7-5. **Content held as empty text replaces nothing on load**, for every content text.
P7-6. **The export file is the page's record**: `{ format: 'hotstep-revision', v, workflow, revision }`, the revision
    naming its engine (`engine`), the file named with the engine's short id (`.mm3.json`); the concept's file
    (`backend`, `slots`, `content`) imports.
P7-7. **A workflow renamed in another tab is followed** (F-17) when a workflow of the same tier, modules and board
    appears there as the open one goes.
P7-8. **The Workflows list's keys** (F-36): F2 renames a row of yours, Delete opens its menu with Delete armed, a
    letter moves to the next row whose name starts with it.
FC-1. **A tab reloads with its own desk** (RC-08): each tab keeps the desk it last showed in `sessionStorage` beside
    the shared one, so a reload shows that desk when another tab has written since, and offers the newer one. A new tab
    opens with the desk changed last.
FC-2. **A desk step puts back the tier it changed** (Use that desk, Open in its tier): the tier is one of the undo's
    paths (`store/desk.ts flatOf`).

Motion:

P9-1. **What pushes what follows moves on the compositor** (the owner's rule for motion: no property that lays the
    page out is animated): the prompt box's height, a suggestion, Song info and the save row change at once, and what
    follows slides from where it stood (`transform`) while the card is revealed down to its new edge (`clip-path`); a
    block that opens is at its full height from the first frame and fades in, one that closes leaves a copy fading
    where it stood.
P9-2. **The prompt box eases when it is typed into**, by more than a line and a half (F-09), as when Undo or a tool
    sets the text.
P9-3. **The player joins the go bar on the compositor**: the line beside it slides over (`transform`) and the player is
    revealed from its left edge (`clip-path`).
P9-4. **Under reduced motion every tile that arrives is rung** with the still outline (an add, a drop, an Undo or Redo
    that brings one back), so every change of state still shows (S-27).
P9-5. **A tile moved by an arrow key starts where it stood**, also when it moves up a row.
P9-6. **The playheads and the VRAM meter are set on the page from the animation loop**, not drawn again by React on
    each frame (Build by section's take, Continue's recording, the meter's LEDs): the same markup at every moment.
P9-7. **A tile or pedal that leaves fades exactly where it stood** at every width.
FO-1. **A module that arrives on the desk alone, only to be folded away, opens the fold**, whether it was added or
    brought back by Undo or Redo: nothing just put on the desk is hidden. A make-up that brings several at once (a
    workflow, a scene) leaves the fold as it is.
FO-2. **The fold's open state is view state kept with the session** (`hotstep.create.session`), not part of the
    desk's record: a workflow, a revision, a draft or another tab's desk never opens or closes it.

## Adding from the how-to, on a copy

Re-measured in the consolidation round (review r3, RA-03) on a scratch copy of the merged branch, following
[06](06-how-to.md) as it now is, with `npm run check` and `npm run lint` after each and the browser acceptance after
the additions and after the removal:

| Job | Files written by hand | Lines | Besides (what 06 says to run) |
|---|---|---|---|
| Add a parameter (YuE2's plan token limit, 06 section 1) | 1 (`planner.ts`) | 10 | `npm run docs`; the request comparison lists its field as new (question 19) |
| Add a pedal with a knob, a choice and a switch (a stereo widener) | 2 (its definition, the index) | 24 + 2 | its drawing, written by `node tools/thumb.ts widener`; docs. The picker's counts follow (the tests count the inventory) |
| Add a module of existing parts, a face per tier, a Quick row (Dynamics) | 4 (definition, a unit in `parts.ts`, index, drawing) | 28 + 1 + 2 + a drawing | docs; and its id in `factory.ts ORDER`, which the layout scenarios asked for ("Everything" left a hole) |
| Add a bespoke instrument (a loudness gauge in Dynamics, 06 section 4b) | 6 (`PARTS` row, core file, the line in `src/derive.ts`, component, stylesheet, the node) | 5 + 19 + 1 + 16 + 2 + 1 | docs |
| Remove a module (Timbre reference) | 4 (its file deleted, index, `retired.ts`, the two Mastering notes the validator named) | 84 deleted; 2 + 3 + 19 deleted | docs; `npm run core -- --recount` (took one entry out); 12 acceptance scenarios that use Timbre fail naming it, to be excluded as 06 says; one test section skipped and said |

All four additions together left `npm run check`, `npm run lint` and the whole acceptance passing, but for the layout
scenarios' hole, which the `ORDER` line closed. Not re-measured in this round: a new part's gallery run (06 section 4
gives its numbers from the first measurement).

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
fail, its visual sets match their approved recordings (approved once its screenshots look right), the gallery passes
with its parts in it, and `npm run check` and `npm run lint` pass. Order: 1 and 2 first, then 3 to 8 in parallel, then 9 and 10.

**How a package is accepted, and how a merge is verified.** A package's three test files register themselves: its
acceptance file is run by `run.mjs`, its switches file by `switches.mjs`, and its states file, which ends with
its visual set (`export const VISUAL_SET = { name, only, scope }`: the states by id and their scope;
tools/visual-baseline/README.md), by `visual-all.mjs`. Nothing else is added to any
runner. Then, on the package's branch and again on the tree after each merge:

```
npm run check && npm run lint
npm run build:test && npm run behaviour -- --workers 6   # the whole acceptance (about 4 minutes 20)
npm run switches:apply                               # every switch still applies exactly once (a second; merges break this most)
npm run visual:all                                   # every visual set against its approved recording (about 7 minutes)
npm run motion                                       # every motion against its approved samples (about 15 seconds)
npm run switches                                     # every switch built and tried (about an hour once caught/ is filled)
node tools/visual-baseline/gallery.mjs && node tools/visual-baseline/plants.mjs
```

The times are at the tools' default of 6 workers (half of a 12-thread machine; `--workers` lowers it). A switch
costs about 9 seconds once `caught/` names what catches it, about 16 seconds (more for a large package) when it starts
from its own package's acceptance, and a whole acceptance run (about 4 minutes 20) when only another package's scenarios
catch it; the old way, the whole acceptance for every switch, would take about 18 hours for the 387 switches.

`visual:all` prints each set's states and `identical`, or how many differences it has from its approved recording, and
exits 1 if a set differs; a merge where both sides re-approved a set approves it again on the merged tree. `switches` tries each switch first against the scenarios that caught it the last time
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
| Quick's fold (after the consolidation round) | `derive/fold.ts`, `desk/Fold.tsx`, `store/fold.ts`, `features/fold.ts`, `kit/session.ts` (the kept-state registration point), `testing/standins.ts`, `testing/compat/fold.ts`, `tools/test-fold.ts`, its acceptance, switches, states | `keptFlag('desk.fold')`, `keptFlag('prompt.songInfo')` | `derive/models.ts deskModel` (`open`, `fold`), `desk/Desk.tsx` (draws the button, passes `open`), `features/dnd.tsx` and `model/index.ts` (one line each), `parts/PromptBox.tsx` (Song info kept), `switches/4-fields.json` (its Song info switch's text) |
| 11. Concept notes and the engine log | `features/notes.tsx`, `features/notes-open.ts`, `features/logs.tsx`, `features/logs-dock.ts`, `features/overlay-exit.ts`, `features/modal-keys.ts`, `services/logs.ts`, `store/notes-logs.ts`, `src/derive/logs.ts`, `i18n/words/notes.ts`, `tools/test-notes-logs.ts`, its acceptance, switches, states | `fill('overlays')`, `defineService('logs')`, `layerOpen` (the notes), `onKey({ id: 'logs-escape', order: 90 })`, `onKey({ id: 'modal-waits', order: 6 })` (F-08) | none; two lines in others' files: `features/layers.ts` (a toast joins a modal layer's Tab cycle, F-18), `shell/placeToast.ts` (the toast stands on the docked log); at the coordinator's request, unique ids in drawings: `shell/engines.ts` and `thumbs/index.ts` (a place per drawing), their callers `shell/TopBar.tsx`, `workflows/List.tsx`, `workflows/Drafts.tsx`, `features/picker.tsx`, `features/dnd.tsx` (one argument each) |

No file is edited by two packages, with three planned exceptions: `src/definitions/parts.ts`, where a package that
needs a new option on a part appends it (the type check refuses a clash, and the rows are independent); the generated
pages of `docs/` (`npm run docs` rewrites them; a merge regenerates them); and the slice's acceptance and states files,
which only package 4 touches (one line) and which every other package leaves alone by adding its own.
