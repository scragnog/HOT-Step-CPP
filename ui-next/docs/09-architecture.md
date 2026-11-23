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
  light and dark; the five desktop widths.

Not built, and left as seams: the prompt card and the lyric sheet, the bespoke parts and the modules made of them, the
picker, the engine panel's models and the VRAM figures, generation, the go bar's contents, Latest and the player,
workflows and revisions, drafts, drag and drop, the remaining overlays (popover, notes, score modal, song menu, engine
log), tooltips and touch gestures, and the choreographed tier transition (a View Transition in the concept; here the
tier changes at once). "The plan for the rest" below divides them into packages.

## The folder

```
ui-next/
  index.html               the concept's document skeleton (no doctype: see "Deviations"), the overlays' containers
  vite.config.ts           three builds: dev (dist/), --mode file (dist-file/index.html, one self-contained file),
                           --mode test (dist-test/index.html, the same file with the compatibility layer)
  eslint.config.js         the existing frontend's lint rules, for src/app
  tsconfig.json            the core and the tools (Node); tsconfig.app.json: the React layer (DOM, JSX), same strictness
  src/
    definitions/, derive/  the core (unchanged in kind; additions below)
    app/
      main.tsx             entry: styles, translation, the engine, the test layer in the test build only
      App.tsx              rooted in #app; portals into the top bar, the skip link, the toast and the live region
      model/               the core indexed once (ix), definitions by id, board order
      store/desk.ts        the desk store: desk, engines' book, modules and board order, undo/redo, autosave
      store/panels.ts      which side panels stand open (view state)
      services/            engine.ts (the boundary), mockEngine.ts (the concept's simulation)
      i18n/                i18next with the generated English table; ui.ts: the page's own words
      styles/              concept/ (the concept's 62 stylesheets, unchanged) and index.ts (their load order)
      parts/               one component per part of the inventory, registry.ts (name to component), value.ts (the
                           number contract and gestures), xyMotion.ts, LcdFace.tsx, Dropdown.tsx, doAction.ts
      render/              Face.tsx (the generic renderer), Tile.tsx (the tile shell), bind.ts
      board/               Board.tsx (the Pedalboard's body, pedals and side units laid out), cables.ts
      desk/                Desk.tsx, pack.ts (the pure layout), layout.ts (measuring and placing), move.ts, focus.ts
      shell/               TopBar, Theme, Panels (+ panelState.ts), Centre, Toast, tier.ts, engines.ts (marks, colours)
      thumbs/              the thumbnail frames around the definitions' drawings
      testing/compat.ts    the test-only compatibility layer (never in dist/ or dist-file/)
  tools/
    behaviour/             the concept's behaviour suite, brought here (+ slice.json, --slice)
    visual-baseline/       the concept's visual baseline, brought here (+ slice-states.mjs, --states, --scope)
    concept/collect-pack.mjs  records the concept's desk layouts into tools/fixtures/pack.json
    test-view.ts           the layout against those layouts, and the core's selectors the view uses
```

`npm run check` runs both type checks, the validator, the request comparison, the core's tests and tools/test-view.ts.
`npm run dev` serves the page; `npm run build:file` writes the file the owner publishes; `npm run build:test` the file
the two tools drive; `npm run lint` lints the React layer.

## Styles

The concept's stylesheets are used as they are: `src/app/styles/concept/` is a byte-for-byte copy of the concept's
`src/css`, imported in its file-name order by `styles/index.ts`, and the build neither minifies nor rewrites them
(`cssMinify: false`; the built stylesheet equals the concatenation, comments and spacing aside). The components emit the
concept's element structure, classes and `data-*` attributes, so these rules apply unchanged and the concept's visual
baseline compares the two pages directly. To change the look, change a stylesheet here (and only here, from now on);
never style a component inline beyond what the concept sets inline (`--p` on a knob, `--h` on a pedal, the layout's
grid placement).

## The page skeleton

`index.html` is the concept's skeleton: the skip link, `header.top`, `#app`, the other pages' stand-in and every
overlay's container, in the concept's order, so that the visual baseline's element keys (an element is keyed by the
nearest unique id above it, or by its position among its siblings) agree. React is rooted in `#app`; it draws the top
bar into `header.top`, the toast into `#toast` and the live region into `#say` through portals, and the engine panel and
the dropdown into their containers.

## The store

`store/desk.ts` holds the core's `Desk` (engine, tier, values, auto switches, state, what is on it), the engines'
`EngineBook`, the modules in the user's order and the board in chain order (the desk's `on` is built from them, in that
order, which is the order the records keep), and the undo and redo history. It decides nothing about a value:

- **A value** changes only through a `Change` a part's setter built (`bindPart`): cleaned, set on the engine's grid and
  named by the core. `commit(change)` applies it and records one step with the core's words.
- **A gesture** (a drag, a held key, typing in one field, a run of wheel steps, a run of moves on one grip) is one step:
  a part calls `gesture.begin(tag)` and `gesture.end(tag)` (parts/value.ts); changes inside merge, and the merged step is
  named again by the core from the first desk to the last (`stepWords`).
- **The make-up** (add, remove, move) and an engine switch are steps named by the core (`deskStep`, `engineStep`);
  what a pedal arrives as is the core's (`arrivalChange`); `switchEngine` does the engine switch.
- **Undo** puts back only the paths a step changed and nothing has changed since (a value, an auto switch, a state, the
  modules, the board, the engine, the book), as the concept's desk entries do; redo is the same the other way.
- **The autosave** writes `savedDeskOf()` on every change (`hotstep.create.desk`); the next visit reads it with
  `readDesk()`, which cleans every value on the way in. The panels' choice is kept beside it (`hotstep.create.panels`),
  the theme as `hotstep.theme`.

## The service boundary

`services/engine.ts` is the one interface between the page and the server, grouped as the existing frontend's API layer
groups it (`ui/src/services/api.ts`: songs, generate, models, plus the option lists and the VST3 chain). `generate`
takes the Create intent the core builds (`createIntent()`), never anything the view assembles. `mockEngine.ts` is the
concept's simulation (its songs, its model catalogue, a queue that renders on a clock with the concept's stage words),
and is what the single-file build runs. The real implementation is a thin adapter over `generateApi`, `songApi`,
`modelApi` and `vstApi`, chosen in `main.tsx`; the page needs no other change.

## How a part component is written

A part's component receives `ViewProps`: the core's `PartProps` for its node on the desk (`bindPart`: its node
resolved for the engine and tier, `params` with a setter for exactly what the node binds, `state`, `reads`, `batch`),
plus its definition, its node's translation key and its children already drawn. It emits the concept's markup for the
part, draws from what it is given, and changes the desk only through the setters. A worked example, the LED switch:

```tsx
export function LedButton(props: ViewProps) {
  const say = useSay(), desk = useDesk(s => s.desk);
  const p = Object.values(props.params)[0];          // the one parameter its node binds
  if (!p) return null;
  const on = p.value === true;                        // the value in force, as the core gives it
  const label = props.cell?.label                     // a shell's label (a pedal), else the node's words, else the
    ?? (props.nodeKey ? nodeWord(ix, props.node, desk, props.nodeKey, 'label') : undefined)
    ?? paramWords(ix, p.key, desk).label;             // parameter's: every word is a message with its key
  return (
    <button type="button" className={`led-btn${on ? ' on' : ''}${p.changed ? ' changed' : ''}`}
      data-ledtog={p.key} data-ctl={p.key} aria-pressed={on} onClick={() => p.set(!on)}>
      <i />{say(label)}
    </button>
  );
}
```

It never computes the changed marker (`p.changed`), the words (`nodeWord`, `paramWords`) or the step (`p.set` returns
nothing; the store records the step the core named). A number control keeps the value contract by calling
`onValueKey(e, p)` (arrows, Shift, Page, Home, End) and the gesture helpers; it may compute where a hand or a key moves
the value (that is how it is operated) and the setter cleans it. To add a part: its entry in `PARTS`
(src/definitions/parts.ts), its component in `src/app/parts/`, its line in `parts/registry.ts`.

## How the generic renderer works

`render/Face.tsx` walks a definition's `face`: for each node it asks the core whether it is drawn (`nodeShown`: shown
at this tier, on this engine and under its condition, bound to something the engine has, and for a container at least
one child drawn), binds it (`bind`), looks its part up in the registry and renders it with its children, and wraps it
when the core says it stands back (`dimOf`). Node keys come from the core (`nodeKeys`), so a node's words are
translated under the same key `stringsOf()` gives them. It has no branch for any module.

The shells are drawn from the definition too. `render/Tile.tsx` is the tile (its width class from `size`, its group,
its name at the tier from `tileName`, the strip it becomes from `stripOf`/`stripWords`, the "Also set" line from
`alsoSet`). `board/Board.tsx` is the Pedalboard's body: each pedal and side unit is laid out from its parameters by the
core (`boardNodes`), each cell bound like a node; the board's count is `boardCount`, a pedal's idle reason `idleOf`.

## A bespoke part, or a mostly bespoke module

A bespoke part is a part like any other: an entry in `PARTS` with `responsible` saying exactly what it does, a component
under its name, its line in the registry. Its definition binds the parameters and state it edits and lists what it
reads. It is given `ViewProps` and nothing else. It may draw anything, keep view state (a caret, a hover, an
animation), talk to the engine through `services/` for what its row says (the build board's session, the model list),
and ask for one of its definition's named steps (`p.set(v, 'step', slots)`, `state.apply('step', next)`). It may not
keep its own copy of a value, decide a default, a range, a label or an undo name, write a value its node does not bind,
or reach into the store. A module that is mostly one bespoke part (Seed, Takes, Adapters, Build) is still a definition
rendered by the generic renderer: its face is that part's node beside the data nodes around it. Until a package builds
a part, its node renders as a marked seam (`data-seam`), so its module still appears with everything else data.

## Translation

Every word is a message (an id and its slots, `derive/messages.ts`); `useSay()` fills it from the locale's table held by
i18next, the existing frontend's mechanism. English is generated: the core's sentences (`coreStrings`), each
definition's words under the keys `stringsOf()` gives them, and the page's own chrome (`i18n/ui.ts`, keyed `ui.*`). A
locale adds a table of the same ids. Components hold no English; the brand is the one literal.

## Additions to the core made for the view

Each is in `src/derive/` with a test in tools/test-view.ts, and none has a branch for a module:

- `derive/desk.ts`: `nodeShown`, `usedHere`, `tileAtTier`, `tileName` (a module's name at Quick is its quickName),
  `deskStep`, `engineStep`, `removedWords`, `parkedOn`, `boardCount`, `arrivalChange`.
- `derive/strings.ts`: `nodeKeys`, `nodeWord`, `nodeOptWord`, `actionKey`, `stripKey`, `stripWords`, `actionWords`,
  `bareText` (a pedal knob's number without its unit), `lcdMsg`, `optionsOf`, `checkText`.
- Messages for the make-up steps, the removal toasts, the parked tray and the board's count.

Defects found while building the view, fixed in the core or the definitions:

- `boardNodes` laid out a text that stands `beside` a switch twice (StableStep's fixed seed);
- `idleOf` keyed a pedal's idle words under a key `stringsOf()` never makes (no translation could reach them), and
  `rowOf` keyed a Quick choice given by an engine line without its line;
- `workflowOf` and `savedDeskOf` wrote the modules in definition order, losing the user's order;
- "Also set" named a side unit's setting without the unit ("Beam 8" for the concept's "Whisper lyrics beam 8");
- the Tiled decoder lacked the concept's Lab note on Stable Audio 3 while Lua plugins are off (pedals/tile.ts).

## The test layer, and running both tools against the rebuild

`src/app/testing/compat.ts` gives the two tools what they reach into the concept page for, on top of the store, in the
test build only: `S` (engine, tier, modules, board, values, panels; set it, then `render()` applies it as a scene, not a
step), `render`, `setVal` (a value set as a control would set it), `get`, `toast`, the catalogue (`MODS`, `PEDALS`,
`EXTRAS`, `PED`, `BK`), `UNDO` and `REDO`, `vsStray` (always empty: every change is a step), and each step's words for
the suite's naming check. It translates the concept's engine ids (`mm3`) and choice values (`Crossover`, by label).

```
npm run build:test
set PLAYWRIGHT_MODULE=<node_modules/playwright>
node tools/behaviour/run.mjs --slice                 # the scenarios in the slice (tools/behaviour/slice.json)
node tools/behaviour/run.mjs --slice --pending       # the pending ones in the slice
node tools/visual-baseline/vb.mjs capture concept-slice --states slice --scope slice --page <concept>/hotstep-tiers.html
node tools/visual-baseline/vb.mjs capture rebuild-slice --states slice --scope slice --plan-from concept-slice
node tools/visual-baseline/vb.mjs compare concept-slice rebuild-slice --scope slice
```

Both tools default to `dist-test/index.html`. The visual baseline's slice matrix (`slice-states.mjs`, 155 states) is
the concept's own matrix cut to what the slice draws: every engine at every tier with every slice module, pedal and side
unit, light and dark, the five desktop widths, reduced motion, parked, the empty desk, the hidden-modules note, changed
values, the chain bypassed, a removal with Undo showing; on the concept page the lyric sheet is withdrawn first (it
heads the desk and the lyrics package has not been built). `--scope slice` compares the desk, the note above it, the
tier switch, Undo and Redo, and the top bar's brand, navigation, engine button and theme switch.

**Results.** Visual baseline: 155 states compared, **99 identical** element for element (with forced hover, focus and
active states), **56 differing**, every one of them only in the two elements listed as deviations 3 and 4 below (and
the boxes that hold them). Behaviour: the 6 normal scenarios in the slice pass (50 checks), and both pending scenarios in
the slice now pass (F-20, F-25). Over the whole suite, which mostly exercises what later packages build, 27 of 250
normal scenarios pass today, and 3 of 36 pending.

## Deviations from the concept's markup

1. **No doctype.** The concept's page has none and renders in quirks mode; so does the rebuild's, so the stylesheets
   lay out identically. The real application's page has a doctype; see the questions below.
2. **The skeleton is static, React draws into it.** `#app` is React's root; the top bar, the toast and the live region
   are portals into the concept's elements; the overlays' containers are empty until their packages fill them.
3. **The Tiled decoder's LCD picker prints the option's `lcd` words, "V2.1"; the concept prints "TILED V2.1".** The
   definition has one `lcd` for both the picker and the Studio read-out, which in the concept differ.
4. **Whisper's Language picker prints "ANY LANGUAGE" (the option's `lcd`, which the unit's LCD needs); the concept
   prints its label, "AUTO-DETECT".** Same cause: one option field serves two LCDs that print different words. Both
   want a definition-level fix (an option's picker words beside its `lcd`), a question for the maintainer.
5. **Adding a Lab module outside Lab offers Lab in a toast with an action**, not in the popover (#pop is the overlays
   package's).
6. **The engine panel holds the five engine cards only** (their marks and names); the taglines, specs and models are
   the engine panel package's.
7. **An XY pad changed from elsewhere (an undo, a scene, another tier) is drawn where it is**, as the concept draws a
   pad it has just rebuilt; it springs only while it is being worked.
8. **The page keeps its records under its own keys** (`hotstep.create.desk`, `hotstep.theme`), not the concept's
   `hs3-*`; the core reads the concept's saved shape (`migrate`, version 0) for when importing is built.
9. **Words the definitions hold differ in a few places from the concept's** (Avoid's placeholder "jazz, piano, slow..."
   for "…"): text, not structure; the computed styles agree.

A design gap found by the how-to below: the `meter` part's look exists only on the Pedalboard (`.board .vu` in
27-pedalboard.css), so a meter in a module's tile (the how-to's Dynamics example) is drawn unstyled. The inventory says
`meter` is usable anywhere; its stylesheet should say so too.

## Adding a module and a pedal from the how-to, on a copy

On a scratch copy of `ui-next`, following [06](06-how-to.md) sections 2 and 3 and nothing else: `pedals/widener.ts` as
written (36 lines), `modules/dynamics.ts` as written with its options and wire notes filled in (24 lines), 4 lines in
`index.ts`, 1 unit in `parts.ts`, `node tools/thumb.ts widener` (9 lines), the Dynamics drawing copied from a similar
module (22 lines), `npm run docs`: 9 files, 125 lines, no component. `npm run check` passed. In the single-file build both
appear in the Modules panel with their drawings and on the desk at Quick (the Loudness row), Studio and Lab (the meter,
the two knobs, the segmented choice; the pedal's knob and switch), with no seam; "Loud" is one step named "Loudness:
Standard → Loud", a knob is one step ("Dynamics ceiling −0.3 dBTP → −0.4 dBTP", "Stereo widener width 100% → 105%"),
the changed marker shows, Quick's "Also set" names the widener, Undo and Redo work, and a reload restores both. It took
three parts the slice had not needed (segmented, choiceRow, meter), now built, and found two defects, now fixed: the
Modules panel had no group for a new category ("Chain"), and redrawing a card's drawing on every render swallowed a
click that began on it.

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

And one question that comes first: the real application's page is in standards mode (it has a doctype) and the concept
in quirks mode. The concept's stylesheets were tuned in quirks mode; before the rebuild moves into the real page, the
concept and the rebuild should both be put in standards mode and the visual baseline captured again, so that the move
itself is proved. Recommendation: option 1, after that switch.

## The plan for the rest

Each package owns the files named, depends on what is listed, and is accepted when its behaviour scenarios pass (with
`--only`) and its visual-baseline states match the concept with the scope widened to its region (states.mjs families;
add a slice-states entry when the concept's state needs a withdrawal). Order: 1 and 2 first (every later module needs
them), then 3 to 8 in parallel, then 9 and 10.

1. **Shared controls.** `parts/`: fader, ruler, numberBox, lcdNumber, chips, termChips, lcdChips, pill, flag,
   lcdButton, recButton, lockButton, powerChip, token, lookback, the textField and textArea looks, choiceRow's marker
   for a value between choices, the pick variants plugin, chip, model and stamp with the dropdown's variants. Depends on:
   the slice. Acceptance: `04-controls/*`, `10-desk-modules/keyboard-contracts`, `13-quick-studio/choice-rows-are-radio-groups`,
   `14-lab-desk/knob-value-can-be-typed`, `18-tips-touch/short-plugin-menu-is-named-cards`; states `ov-dd-*`.
2. **Containers and read-outs.** `parts/`: well, lcd, lcdRow, terminal, termBar, termLine, group, foot, deck, pair,
   bank, link, dice, beatPulse, flowWindows, speedGauge, adhereCurve, planText, lrcPreview, coverSwatch, unitStatus,
   unitBadge, dropZone, and the meter's look outside the board. With 1, every "data plus parts of its own" module whose
   own part is a read-out renders (Repetition guard, Flow, Plans, Re-planning, Plan adherence, the side units). Acceptance:
   `20-lab-round5/side-units-say-their-actions`, `22-engines-round5/side-units-fit-a-narrow-board-*`, `08-persistence-cables/cables`;
   states `desk-*`, `w*-*` with those modules.
3. **Bespoke instruments**, one package each, in parallel: Tempo (keyWheel, tapTempo, capChips), Sampler
   (samplerScope), Seed (splitFlap, savedSeeds), Takes (takesReels), Adapters (cartridges), Timbre (timbreBlend,
   recentRefs), Latent (latentCurve), Lead sheet (abcFoot and the score modal), Build by section (buildBoard, its
   session calls in `services/`), Continue and Repaint (recordingSpan, repaintRegion). Each owns its part files and its
   `services/` calls. Depends on 1 and 2. Acceptance: its `05-modules/*` scenario, its `22-` and `23-` scenarios, and the
   states that show it (`pop-seeds`, `pop-adapters*`, `pop-build-*`, `repaint-source*`, `sched-*`, `latent-on`, `tap-armed`).
4. **The prompt card and the lyric sheet.** `parts/` promptBox and lyricSheet; `shell/Centre.tsx`'s quickcol and
   `desk/Desk.tsx`'s `LyricsSeam` are theirs; remove the withdrawal from `slice-states.mjs`. The layout already handles a
   lyric sheet at the head of the desk (`lead`, `fill`). Acceptance: `06-prompt-generate/prompt-*`, `05-modules/lyrics`,
   `01-desk/tiers`, `09-undo-storage/undo-keeps-a-later-edit-of-the-same-thing`; states `pop-lyrics*`, `pop-wildcards`,
   `pop-mm3-caption`, `pop-yue-tags`, and every `desk-*` state without the withdrawal.
5. **The engine panel and models.** `shell/TopBar.tsx`'s panel and meter, modelPicker, the engine log; `services`
   models. Acceptance: `15-other-engines/model-list-as-wide-as-its-row`, `21-chrome-round5/model-steps-named-by-role`;
   states `ov-engine*`, `ov-log*`, `pop-vram-full*`.
6. **Generation, the go bar, Latest, the player and the song menu.** `shell/Centre.tsx`'s gobar and latest, new
   `generate/` and `player/` folders; the real `services/` adapter (see 10). Acceptance: `06-prompt-generate/generate`,
   `songs-player`, `00-smoke/smoke`, `21-chrome-round5/quick-player-names-its-song-*`; states `pop-running*`,
   `pop-finished`, `pop-error*`, `pop-player*`, `ov-song-menu*`.
7. **Workflows, revisions, drafts and storage.** `shell/Panels.tsx`'s Workflows body, a `workflows/` folder, the
   unreadable-record handling over `records.ts`. Acceptance: `03-workflows/*`, `07-overlays-keys/drafts`,
   `09-undo-storage/*`, `08-persistence-cables/old-shape-ignored`, `21-chrome-round5/unreadable-desk-kept-aside`; states
   `pop-modified*`, `pop-rev-detail*`, `pop-rename*`, `ov-save*`, `ov-drafts*`.
8. **The picker, the overlays and drag and drop.** The picker, the popover (and the Lab offer, deviation 5), notes,
   tooltips and touch gestures, the focus ring, drag and drop onto the desk and the board. Acceptance: `07-overlays-keys/*`,
   `02-dnd-panels/drag-*`, `18-tips-touch/*`, `01-desk/lab-modules`; states `ov-picker*`, `ov-notes*`, `ov-ideas*`, `drag-*`.
9. **Motion.** The tier change as a View Transition (`shell/tier.ts`), the desk's FLIP slides and entrances, the
   pedals' landing and the cables drawing out (`board/cables.ts`). Acceptance: `13-quick-studio/tier-change-keeps-the-place-*`,
   `19-desk-round5/rapid-tier-changes-raise-no-error`, `23-sa3-hm-round5/reduced-motion-arrivals-are-ringed`.
10. **The real application.** The real `Engine` over `ui/src/services`, the Tailwind coexistence chosen above, the
    route in the real app, and the doctype question. Acceptance: the whole suite and the whole matrix, then the concept's
    own states with no withdrawal.

What the slice will need changed as the packages land: `Desk.tsx`'s `LyricsSeam` and the quickcol (package 4); the
Lab offer's toast becomes the popover (8); `TopBar.tsx`'s engine panel grows its models (5); `main.tsx` chooses the real
engine (10); `slice-states.mjs` and `slice.json` grow with each package, and the deviations 3 and 4 close with a
definition change.
