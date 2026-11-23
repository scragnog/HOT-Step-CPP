# How to do the common jobs

Each job below was done on a copy of `ui-next` and checked; the line counts are what it took, from `git diff --stat`
of that copy, in the files' own style (one field per line when a line would pass 120 characters). After any job, from
`ui-next`:

```
npm run check
```

It must pass. Every message names the file, the parameter or node, the engine, and what to change.

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

A new parameter needs nothing in `tools/existing-ui.ts`: the request comparison lists it as having no control on the
existing page and its field as new ([08](08-open-questions.md) question 19), and does not fail. A parameter for
something the existing page already has a control for gets one line there naming that control by its place and words
("Generation > Shift"), and the comparison asks for it when the field differs.

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

**Three files: 36 lines of definition, 2 in the index, and a 9-line drawing the tool writes.** At Lab it is a 1U pedal with one knob and its
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
       { part: 'meter', reads: ['dyn.target'], opts: { min: -24, max: -6, red: -9 }, tiers: ['studio', 'lab'] },
       { part: 'row', tiers: ['studio', 'lab'], children: [
         { part: 'knob', param: 'dyn.target' }, { part: 'knob', param: 'dyn.ceiling' }, { part: 'segmented', param: 'dyn.char' },
       ] },
     ],
     thumb: { file: 'dynamics' },
   } satisfies ModuleDef;
   ```

   A choice that sets several values names them by key; one of the choices must be the defaults. A part whose drawing
   should show a value reads it (`reads`); `meter` takes its scale as options.
3. Add two lines to `index.ts`.
4. Add `thumbs/dynamics.svg`: a drawing in the thumbnail kit's classes on the 176 x 112 canvas, kept inside the 15-unit
   inset (start from a module with similar controls).

**Four files: 26 lines of definition written as above (about 68 with one field a line), 1 in `parts.ts`, 2 in the
index, a drawing.** Choosing parts:
the table in [01](01-parts-inventory.md) says what each part is for, what it edits and reads, and which modules use it.
For a desk module (not Lab-only), tier the nodes that Quick or Studio should not show, give Quick a `choiceRow` when a
plain choice makes sense, and set `size`.

## 4. Add a new part to the inventory

When no part does what a module needs:

1. Add an entry to `PARTS` in `parts.ts`: its `kind`, what it `edits` (and `reads`, `state`), its `drive`, a `doc`
   sentence, its `opts` with each one's type and what it does, its `states`, and its `css`. A bespoke part also says what
   it is `responsible` for.
2. Write its component under the same name. It receives `PartProps` ([01](01-parts-inventory.md), "What a bespoke part is
   given"), and its stylesheet follows the design language: tokens from `00-tokens.css`, the shared states of layer 35,
   and a file of its own.
3. Use it in the definition that needed it, in place of what it replaces; a part no definition uses any more is only a
   note.

Example: a two-handled range slider for CADS's two thresholds in Plan adherence:

```ts
rangeSlider: {
  kind: 'control', edits: ['pair'], drive: 'keys', css: ['15-range.css'],
  doc: 'One track with two handles for the two ends of a range, the first of its params below the second.',
},
```

and in `modules/adhere.ts`, in place of the two labelled tokens (and, in the same way, of the Plan CFG range's `pair`):

```ts
{ part: 'rangeSlider', params: ['yue.cads1', 'yue.cads2'], label: 'Clean below · full above' },
```

**4 lines in `parts.ts`; in the definition, 7 lines become 2**, plus the component and its stylesheet, which are the real work
and outside this format. The two ends never cross because the parameters say so (`staysBelow` on the lower end): the
core's `setValue` stops an end a step from the other, and cleaning puts crossed ends in order, whichever part draws
them. A bespoke part also says in its row the named `steps` it takes and, when its drawing needs particular inputs,
its `reads` signature; the validator checks every node against both. The inventory table in
[01](01-parts-inventory.md) is regenerated by `npm run docs`.

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
| Rename a parameter key | The new key, and `was: ['old.key']` | Saved values read as the new key |
| Rename a definition's id | The new id, its file name, `was: ['old']` on the definition | Saved workflows and desks read it |
| Rename a state key | The new key, `was: ['old']` on the state | Saved revisions read it |
| Move a parameter to another module | Cut and paste; if its owner or what saves it changes, `npm run docs` records the change once you mean it | `tools/fixtures/shipped.json`: the validator says where it was kept and where it is now |
| Remove a module, pedal or unit | Delete the file and its two index lines; add its id and its keys to `retired.ts` | The validator names what refers to it, and what saved data may hold |

## 9. Add an engine

Example: a sixth engine, "X6", with half the modules.

1. When the server has its backend, add it to `ENGINES` in `engines.ts` (one line: id, name, backend folder), add its
   model buckets in `tools/make-fixtures.ts`, and run `npm run fixtures`, so the validator knows what it declares and
   reads.
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
