# How to do the common jobs

Each job below was done on a copy of `ui-next` and checked; the line counts are what it took, from `git diff --stat`
of that copy, in the files' own style (one field per line when a line would pass 120 characters). Each job ends with
**What else to run**: what it takes beyond the definition, measured on that copy; nothing outside it needs touching.
After any job, from `ui-next`:

```
npm run check
npm run docs        when the check says the generated pages are out of date
```

It must pass. Every message names the file, the parameter or node, the engine, and what to change. The tests count
what the definitions hold (the picker's "35 in all" is the inventory's count, not a number in a test) and skip, saying
so, the sections that name a definition that is gone, so adding or removing one fails nothing that is not about it.

Where the page needs one parameter or state in particular (the main box, the takes count, the song a repaint works on),
the definition says so with its `role` ([02](02-definition-format.md), "Roles"), and the code asks for the role, never
the key: renaming such a key the documented way keeps it working, and removing it is named by the validator.

## 1. Add a parameter to an existing module

Example: YuE2's plan stage takes a token limit, `yue2PlanMaxTokens` (256 to 4096 in steps of 64, default 4096), which
the server declares and no definition carries yet. It belongs on the Planner's plan line, after top-P, at Lab.

1. If the server has only just gained the field, refresh the fixture so the validator knows it: `npm run fixtures`. (A
   parameter the server does not have yet is written with `wire: { none: '...' }` saying what is missing.)
2. Add the parameter to the module's `params` in `modules/planner.ts`, next to its neighbours:

   ```ts
   {
     key: 'yp.maxtok',
     type: 'number',
     label: 'Plan max tokens',
     range: [256, 4096, 64],
     default: 4096,
     engines: ['yue2'],
     wire: 'ext:yue2PlanMaxTokens',
   },
   ```

   `engines` is needed because the Planner is on four engines; without it the parameter would be on all four, and the
   validator would say the other three backends do not have the field and that no control edits it there. The range
   and default are the server's; if they were not, the validator would say so, and the wire would name the check it
   waives in `differs` and say why in its `note`. Add `tip`, `unit` or `format` if the number needs them.
3. Show it: add a token after top-P in the YuE2 `plan` line of the terminal, at Lab:

   ```ts
   { part: 'token', param: 'yp.topp', label: '--top-p' },
   { part: 'token', param: 'yp.maxtok', label: '--max-tokens', tiers: ['lab'] },
   ```

**One file, 10 lines.** Its default, cleaning (it is a whole number), changed marker, undo name ("Plan max tokens 4096 →
2048"), tip, revision storage, what an engine switch does with it, the request (it is sent once changed, as
`yue2PlanMaxTokens`, and the request comparison tries it at 256, 4096 and 4032) and the "Also set" line at Studio all
follow. If you forget step 3 the validator says "no control edits it on yue2 at any tier". Run `npm run docs`: the
generated tables, the counts and the list of shipped keys change, and the check fails until you do.

A new parameter whose field the existing page never sends needs nothing in `tools/existing-ui.ts`: the request
comparison lists it as having no control on the existing page and its field as new ([08](08-open-questions.md)
question 19), and does not fail. **When the existing page already sends the field** (it sends `storkSubsteps: 10`
with the STORK solvers whether or not it has a control for it), the comparison compares the two: give the parameter the
default the existing page sends, and it passes; with another default, it fails naming the field, the engine and both
values ("new: ace | storkSubsteps ... existing 10, definitions 1"), and the difference is either corrected or accepted
on purpose with an entry in `tools/fixtures/request-differences.json` (its format is in [07](07-checking.md), "An
accepted difference names exactly what it covers", and it names the question of [08](08-open-questions.md) that decides
it: add a question there if none does). A parameter for something the existing page has a control for gets one line in
`tools/existing-ui.ts` naming that control by its place and words ("Generation > Shift").

**What else to run**: `npm run docs`.

On a pedal there is no step 3: a pedal's controls are laid out from its parameters, in the order of `params`.

## 2. Add a new pedal

Example: a stereo widener after the Spectral lifter (the real app has no field for it yet).

1. Create `pedals/widener.ts`:

   ```ts
   // Pedal "Stereo widener".

   import type { PedalDef } from '../types.ts';

   export const widener = {
     kind: 'pedal',
     id: 'widener',
     name: 'Stereo widener',
     desc: 'Widens or narrows the stereo image.',
     at: 55,
     engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
     hue: 100,
     on: 'pp.widen.on',
     main: 'pp.widen.amt',
     owner: 'song',
     stored: 'settings',
     params: [
       {
         key: 'pp.widen.on',
         type: 'boolean',
         label: 'On',
         default: false,
         wire: { none: 'No stereo widener in the real app yet.' },
       },
       {
         key: 'pp.widen.amt',
         type: 'number',
         label: 'Width',
         range: [0, 200, 5],
         unit: '%',
         default: 100,
         wire: { none: 'No stereo widener in the real app yet.' },
       },
     ],
     thumb: { file: 'pd-widener' },
   } satisfies PedalDef;
   ```

   `at` places it in the chain: between the Spectral lifter (50) and the Vocal naturalizer (60). `hue` is its face
   colour: pick one at least 15 degrees from its neighbours' (the lifter's 140, the naturalizer's 38).
2. Add two lines to `index.ts`: its import, and its name in `DEFINITIONS`.
3. Draw its thumbnail: `node tools/thumb.ts widener` writes `thumbs/pd-widener.svg`, a knob at its default; the frame is
   drawn for you.

**Three files: 36 lines of definition, 2 in the index, and a 9-line drawing the tool writes.** **What else to run**:
`npm run docs`. How its controls lay out at Lab: a number is a knob, a choice of up to four options a segmented choice
(more, an LCD picker), a switch an LED button, in the order of `params`; `beside` puts a switch on its knob's row. At Lab it is a 1U pedal with one knob and its
foot switch; at Studio its knob; at Quick its name and switch; it is in the Modules panel and the picker under Pedals, in
the chain at its place, counted on the board, and its settings travel with the song and are saved in revisions. When the
server gains the field, its wire becomes the field's name.

## 3. Add a new module out of existing parts

Example: "Dynamics" on ACE-Step and Stable Audio 3, with a target loudness, a true-peak ceiling and a character; Quick
shows one plain choice of three presets that set all three; Studio adds the three knobs and a meter that reads the target.

1. If a unit is new, add it to `UNITS` in `parts.ts` (one line: `dBTP: 'decibels true peak; ...'`).
2. Create `modules/dynamics.ts`:

   ```ts
   export const dynamics = {
     kind: 'module', id: 'dynamics', name: 'Dynamics', desc: 'How loud the song is pushed, and how hard.',
     at: 215, category: 'Chain', engines: ['ace', 'sa3'], owner: 'song', stored: 'settings', size: { w: 1 },
     tierSummary: ['Three presets', '+ target, ceiling, character and a meter', 'Same as Studio'],
     params: [
       { key: 'dyn.target', type: 'number', label: 'Target', range: [-24, -6, 0.5], unit: 'LUFS', default: -14, wire: { none: '...' } },
       { key: 'dyn.ceiling', type: 'number', label: 'Ceiling', range: [-3, 0, 0.1], unit: 'dBTP', default: -1, wire: { none: '...' } },
       { key: 'dyn.char', type: 'choice', label: 'Character', default: 'clean', options: [...], wire: { none: '...' } },
     ],
     face: [
       { part: 'choiceRow', label: 'Loudness', tiers: ['quick'], choices: [
         ['Gentle', { 'dyn.target': -16, 'dyn.ceiling': -1.5, 'dyn.char': 'clean' }],
         ['Standard', { 'dyn.target': -14, 'dyn.ceiling': -1, 'dyn.char': 'clean' }],
         ['Loud', { 'dyn.target': -9, 'dyn.ceiling': -0.3, 'dyn.char': 'punchy' }],
       ] },
       { part: 'meter', reads: ['dyn.target'], opts: { red: -9 }, tiers: ['studio', 'lab'] },
       { part: 'row', tiers: ['studio', 'lab'], children: [
         { part: 'knob', param: 'dyn.target' }, { part: 'knob', param: 'dyn.ceiling' }, { part: 'segmented', param: 'dyn.char' },
       ] },
     ],
     thumb: { file: 'dynamics' },
   } satisfies ModuleDef;
   ```

   A choice that sets several values names them by key; one of the choices must be the defaults. A part whose drawing
   should show a value reads it (`reads`); a `meter` that reads a number is on that number's range (a `min` or `max`
   beside it is refused: the range is stated once), and `red` says where it turns red.
3. Add two lines to `index.ts`.
4. Add `thumbs/dynamics.svg`: a drawing in the thumbnail kit's classes on the 176 x 112 canvas, kept inside the 15-unit
   inset (start from a module with similar controls).

**Four files: 26 lines of definition written as above (about 68 with one field a line), 1 in `parts.ts`, 2 in the
index, a drawing.** **What else to run**: `npm run docs`; then the browser acceptance (`npm run build:test`,
`node tools/behaviour/run.mjs --page dist-test/index.html --slice`): a module joins the factory workflow "Everything" by
itself, and the layout scenarios (`12-desk-layout/factory-desks-compose-*`) check that every factory desk composes.
Dynamics at the end of it left a hole at every width; one word, its id in `ORDER` in `factory.ts` where the desk
composes (after `avoid`), put it right. Which parts give which look: [01](01-parts-inventory.md)'s
table; a face at Lab is paper like Studio's (a `rack` or a `well` gives a row its hardware ground). To see it: `npm run dev`, add it from the Modules panel, and switch tiers and themes; every part
it names already looks right on every surface (the parts gallery, section 4, step 5). Choosing parts:
the table in [01](01-parts-inventory.md) says what each part is for, what it edits and reads, and which modules use it.
For a desk module (not Lab-only), tier the nodes that Quick or Studio should not show, give Quick a `choiceRow` when a
plain choice makes sense, and set `size`.

## 4. Add a new part to the inventory

When no part does what a module needs. Example: a small stepped slider with labelled stops, used by Dynamics for its
character and by YuE2 Batch for its noise variations.

1. Its entry in `PARTS` in `src/definitions/parts.ts` (the inventory; `npm run docs` puts it in the table of
   [01](01-parts-inventory.md)):

   ```ts
   stepSlider: {
     kind: 'control', edits: ['choice', 'number'], drive: 'choose', css: ['15-step-slider.css'],
     doc: 'A short slider that snaps to labelled stops: a choice\'s options, or a small number range\'s steps.',
   },
   ```

2. Its component, a file of its own, `src/app/parts/StepSlider.tsx`. It is loaded and registered by being there
   (`src/app/load.ts`); nothing else is edited. It imports only the part kit (`../kit/index.ts`), react and its sibling
   parts; the lint refuses anything else (eslint.config.js rule 1), so it cannot reach the store or decide a value. What
   it is given (`PartView`, `src/app/kit/types.ts`) is everything it draws, already decided by the core: each parameter
   with its value, its words, its options with theirs, whether it is changed; and setters that write exactly what its
   node binds, on the desk as it is when they are called:

   ```tsx
   // stepSlider: a short slider that snaps to labelled stops.
   import type { KeyboardEvent } from 'react';
   import { definePart, useSay, type PartView } from '../kit/index.ts';

   export function StepSlider(v: PartView) {
     const say = useSay(), p = v.params[0];
     if (!p?.options) return null;
     const at = Math.max(0, p.options.findIndex(o => o.on));
     const key = (e: KeyboardEvent<HTMLDivElement>) => {
       const d = ({ ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 } as Record<string, number>)[e.key];
       if (!d) return; e.preventDefault();
       const o = p.options![Math.min(p.options!.length - 1, Math.max(0, at + d))]; if (o) p.set(o.value);
     };
     return (
       <div className={`stepsl${p.changed ? ' changed' : ''}`} data-ctl={p.key}>
         <span className="kl">{say(v.label)}</span>
         <div className="stepsl-track" role="radiogroup" aria-label={say(v.label)} tabIndex={0} onKeyDown={key}>
           {p.options.map(o => <button key={String(o.value)} type="button" role="radio" aria-checked={o.on} tabIndex={-1}
             className={o.on ? 'on' : ''} onClick={() => p.set(o.value)}><i /><span>{say(o.label)}</span></button>)}
         </div>
       </div>
     );
   }
   definePart('stepSlider', StepSlider);
   ```

   A number of twelve steps or fewer has its steps as `options` too, each written as the page writes it (YuE2 Batch's
   1 to 9 variations draw as nine stops with no line more); a longer range has no `options`, and a part that needs
   stops draws them from `p.range`; `p.press(key, shift)`, `p.typed(text)` and `p.toFraction(f)` keep the value contract
   for a number. `v.gesture.begin()` and `end()` make a drag or a held key one undo step; `v.batch(fn)` makes several sets
   one. It never names a step, never formats a value, never computes a default or a range: those are in `v` already.

3. Its look, a stylesheet of its own: `src/app/styles/rebuild/15-step-slider.css` (loaded after the concept's by being
   there; the number is its layer). Use the tokens of `00-tokens.css`. What every part gets from its markup alone needs
   no line: its root has `data-ctl` and `.changed`, its label is a direct `.kl` child, and then the page's mono caps label,
   its changed mark, its focus ring and its disabled state come from `styles/rebuild/35-part-states.css`. Where the part
   would look wrong on a surface (on hardware: a pedal, a side unit, an LCD; on paper: a tile, a row, a well, a foot),
   say so in its own stylesheet, by the surface's container (`.pedal .stepsl`, `.lcd .stepsl`).

4. Use it in the definitions that need it: `{ part: 'stepSlider', param: 'dyn.char' }`.

5. Look at it everywhere, and keep it so: `npm run build:test`, then `node tools/visual-baseline/gallery.mjs`. The parts
   gallery (the test build only) draws every part of the inventory on every surface, both themes, every tier, with its
   value changed, and checks it is drawn inside its container, its words read at 3:1, its label is the page's, it is
   marked when changed, and that every part looks as it did when last recorded. A new part is in it with no line
   written; its screenshots are in `tools/visual-baseline/out/gallery/`. Until it is recorded the run fails with one
   "new since the last record" per place it stands (50 for a control: 9 surfaces at 3 tiers and 2 themes, less a
   pedal's at Quick and Studio, more where a definition places it too); the run says how many of its failures are new
   since the last record and, past 80 lines, how many it did not print (`GALLERY_MAX=<n>` prints them all). The gallery's
   bare new part (a markup-only control drawn beside every part on the deck) is recorded by its place among those, so
   it may move by one with a part added before it. Once the screenshots look right, `gallery.mjs --record` keeps them
   (`tools/fixtures/gallery.json`).

**4 files: the `PARTS` entry (4 lines), the component (one file), its stylesheet (one file), and a line in each
definition that uses it.** **What else to run**: `npm run docs`; `npm run build:test`, `gallery.mjs`, then
`gallery.mjs --record`. `npm run check` fails while a definition names a part that has no component
(`tools/test-code.ts`: "names the part ..., which has no component"). A bespoke part also says in its `PARTS` row what it is `responsible` for, the named `steps`
it takes and, when its drawing needs particular inputs, its `reads` signature; the validator checks every node against
both. A part that talks to the server does it through a service of its own (`src/app/register/services.ts`), and any
answer it writes goes through its setters, which act on the desk in force when the answer arrives (a module removed
meanwhile takes nothing).

## 4b. Add a bespoke instrument

When a part's drawing or what a hand on it does cannot be said as a value (a wheel of keys, a gauge, a board of takes),
it is a **bespoke** part: its component draws what the core works out for it (`v.instrument`), and every hand on it is
one of its core's acts (`v.act(name, arg)`), made through its own setters, so it is cleaned, bound and named as every
other change. Example: a loudness gauge for Dynamics that a click sets and the arrows step.

1. Its `PARTS` row, `kind: 'bespoke'`, with what it edits, what it is `responsible` for and, when it takes named steps,
   their names (`steps`).
2. Its core, a file of its own in `src/derive/` (say `gauge.ts`): what it draws and what its hands do, registered once,
   and its model's field declared on `InstrumentModel` where it is drawn:

   ```ts
   import { defineInstrument, type ActIn } from './instruments.ts';
   import { EN, msg, type Slot } from './messages.ts';
   Object.assign(EN, { 'gauge.aria': 'Target loudness {v}' });          // the core's words, added to its table
   declare module './instruments.ts' { interface InstrumentModel { gauge?: { at: number; aria: Slot } } }
   defineInstrument('loudGauge',
     (ix, node, desk) => ({ gauge: { at: ..., aria: msg('gauge.aria', { v: ... }) } }),   // what it draws
     { step: (x: ActIn) => { const p = x.v.params[x.node.param!]; if (p) p.set(...); } }); // what a hand does
   ```

   and one line in `src/derive.ts`, `export * from './derive/gauge.ts';` (the core in one import: the registry is
   filled when the file is loaded, in the page and in every tool).
3. Its component, `src/app/parts/LoudGauge.tsx`, reading `v.instrument?.gauge` and calling `v.act('step', ...)`,
   registered with `definePart('loudGauge', LoudGauge)`; its stylesheet as in section 4.
4. Its node in the definition: `{ part: 'loudGauge', param: 'dyn.target' }`.

**5 files and a line**: the `PARTS` row, the core file, the line in `src/derive.ts`, the component, its stylesheet,
and the node. **What else to run**: as section 4 (`npm run docs`, the gallery and `--record`). A hand that needs the
shell (a step that brings a module, the live region, focus elsewhere) is a named action instead (`ACTIONS` in
`parts.ts`, `defineAction` in a file of `src/app/features/`); a part responsible for a conversation with the server
asks for a service by name through the kit's `service` (`src/app/services/index.ts` lists them).

## 5. Change a range or default on one engine

Give the parameter a line for that engine, and its `engines` beside the lines:

```ts
engines: ['ace', 'minimax-m3', 'yue2', 'heartmula'],
per: [{ engines: ['heartmula'], range: [1, 32, 1] }],
```

A song value whose range differs per engine also needs `held`, the words for when an engine cannot take the value
([02](02-definition-format.md), "Held"):

```ts
held: { note: 'Capped at {shown} steps here · your {value} return on {engine}', toast: 'StableStep steps capped at {shown}' },
```

**3 to 4 lines.** A default outside the new range, or a Quick choice outside it, is refused.

## 6. Make a control depend on two others

```ts
when: { all: [{ param: 'pp.ss.blend', is: 'crossover' }, { param: 'pp.ss.dyn', is: true }] },
```

**1 line.** A condition that can never hold, a value outside a range, or two parameters that wait for each other are
refused. On a pedal, a knob that `when` can hide keeps its room behind a plate whose words are worked out from the
condition.

## 7. The server renamed a field

Change the wire, then refresh the fixture:

```ts
wire: { to: 'param:stableStepRefine', divide: 100 },
```

`npm run fixtures`. **1 line**, and `was: ['param:stableStepStrength']` on the wire if saved profiles or songs hold the
old name, so reading them back still finds it. Saved revisions keep loading: they are keyed by the definition's key, not
the server's name. If the definition's key changes too, list the old one: `was: ['pp.ss.str']`; if an option's value changes, list
the old value on the option: `{ value: 'xover', label: 'Crossover', was: ['crossover'] }`, and change the conditions
that compare with the old value, which the validator names ("it is now "xover", which lists "crossover" in was"). For
the blend's value: **1 line, and 2 conditions.**

## 8. Remove a parameter

Delete it from `params`, and the nodes that edit it. The validator names every other place that refers to it (a
condition, an action, a read, a `beside`, a factory revision), and says saved data may hold the key: add it to `keys` in
`src/definitions/retired.ts`, so a record that has it loads without it and without a word. For StableStep's fixed seed:
**10 lines deleted, 1 added.** A field of the format, a part or a transform no definition uses any more is only a note.

## 8b. Move, rename or remove other things

| Job | What to write | What checks it |
| --- | --- | --- |
| Rename a parameter key | The new key, and `was: ['old.key']`. `npm run check` then names the tools that name the old key: `tools/existing-ui.ts` (its line for the parameter), the patterns of `tools/fixtures/concept-differences.json` and `request-differences.json` that match it, and a test's desk that sets it; rename it there too | Saved values read as the new key; its role (if it has one) goes with it, so the page keeps working |
| Rename a definition's id | The new id, its file name, `was: ['old']` on the definition | Saved workflows and desks read it |
| Rename a state key | The new key, `was: ['old']` on the state | Saved revisions read it |
| Move a parameter to another module | Cut and paste; if its owner or what saves it changes, `npm run docs` records the change once you mean it | `tools/fixtures/shipped.json`: the validator says where it was kept and where it is now |
| Remove a module, pedal or unit | Delete the file and its two index lines; add its id, its keys and its state keys to `retired.ts`; take it out of what the validator names (a factory workflow that lists it, another definition's condition or action that names it: Timbre's removal names two notes of the Mastering pedal; a role it played: a role nothing plays is a note saying what the page does without it, or an error for a role the page needs); `npm run docs`; `npm run core -- --recount` (the concept comparison's accepted differences that now cover fewer cases are recounted, and one that covers none is taken out). The core's tests skip, and say so, the sections that name it. Then the browser acceptance: the scenarios that use it fail naming it ("no such element .tile[data-m=\"timbre\"]": for Timbre, 12 scenarios, 27 checks): move each to its acceptance file's `excluded` with the reason "uses <module>, removed"; a visual state that draws it is taken out of its states file | The validator names what refers to it, and what saved data may hold; `npm run check` passes |

## 9. Add an engine

Example: a sixth engine, "X6", with half the modules.

1. When the server has its backend, add it to `ENGINES` in `engines.ts` (one line: id, name, backend folder, its output
   rate `khz` and the engine panel's two-line `tagline`, `testing: true` while it is in testing) and its row of
   `ENGINE_FACTS` beside it (what its main box is called, whether it renders from lyrics alone, whether its tempo field
   reaches the song, whether it structures its caption, how it takes a song's length), add its model buckets in
   `tools/make-fixtures.ts`, and run `npm run fixtures`, so the validator knows what it declares and reads. The type
   check then names the rest, each typed by engine so none can be forgotten: its mark and colour in
   `src/app/shell/engines.ts` (`LOGO`, `ENGINE_COLOUR`, and the concept's short id the test layer speaks,
   `CONCEPT_ID`), and the simulated server's catalogue in `src/app/services/mock/models.ts` (`MODEL_FILES`; its VRAM
   figures, what it always loads and what a take adds, beside them in `FIGURES`, and its adapters in
   `services/mock/adapters.ts` if it takes any). The real server says these itself once package 10 is in.
2. Add `'x6'` to the `engines` of each module, pedal and unit it should have.
3. Run the validator. It names every parameter of those definitions that differs per engine (it has `per` lines and
   lists its own engines, so the new engine does not get another engine's names, ranges or defaults by accident), every
   parameter that is on every other engine of its definition but not the new one (a Prompt with no prompt text: it
   fails until you add the engine or, if it is left off on purpose, name it in `ABSENT_ON` in the validator), every
   real name the new backend does not read, and every Quick row with no choices for it. For each, add `'x6'` to the
   parameter's `engines` and a line for it, or leave it off the engine.

**1 line in `engines.ts`, 1 in the fixture tool, 1 per definition, and for each parameter that differs per engine its
`engines` and a line**: 47 lines for the example's 12 definitions, 33 of them in the 4 that have parameters differing
per engine (the Sampler, Tempo, Takes and Models).

## 10. Add a side unit

Copy `units/wh.ts` (a power switch, a server-listed choice, a read-out of its values): a unit is `kind: 'unit'`, an
`on` switch, its parameters (laid out at Lab: numbers as knobs, a choice of up to four options as chips, longer choices
as LCD pickers, switches as LED buttons), and a `face` with the object it shows at every tier (an `unitLcd`, a status, a
badge), with `opts: { notAtLab: true }` on a read-out that would repeat the controls at Lab. Add its two index lines and
a drawing (its middle; the strip is drawn for you). It is in "Everything" by itself. A unit not on the board is off: its
switch sends its off value.

## 11. Add a state entry

A file, a seed, a span, slots, a record or a list a module keeps: one entry in its `state` with `key`, `kind`, `noun`,
`owner`, `stored`, `wire`, and the named `steps` its part takes. Name it in the node of the part that holds it. Its
kind says what it may hold, and cleaning keeps nothing else.

## 12. Add a transform

When a wire's fields cannot say a packing (several parameters into one field) or a change of type: an entry in
`transforms.ts` with `send` and `read`, named by `via` on every parameter it packs. Its line in `tools/existing-ui.ts`
says how the existing page holds the same values; the request comparison then compares it over the value space and
fails until every branch of its `send` is reached.

## Smaller jobs

| Job | Where |
| --- | --- |
| Rename a control | Its `label` (and `short`) in the parameter; per engine in a line |
| Make a control apply only sometimes | `when` on the parameter |
| Make a control stand back with a reason | `dim` on its node |
| Change the plain choices at Quick | `choices` on the `choiceRow` node, per engine in its lines |
| Move a control to another tier | `tiers` on its node |
| Different words at one tier or on one engine | a line in the node's `per` |
| Take a module off an engine | its `engines` |
| A new unit | one line in `UNITS` in `parts.ts` |
| A new text check or desk fact | one line in its list in `parts.ts`, with the rule it is worked out by |
| A new option source or named action | one line in its list in `parts.ts`, and the code that fetches the list or runs the action in the shell |
| A value a wire cannot say (a new packing) | an entry in `transforms.ts`, named by `via` |
