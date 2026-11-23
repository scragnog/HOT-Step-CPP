# How to do the common jobs

Each job below was done on a copy of `ui-next` and checked; the line counts are what it took, from `git diff --stat`
of that copy, in the files' own style (one field per line when a line would pass 120 characters). After any job, from
`ui-next`:

```
npm run check
```

It must pass. Every message names the file, the parameter or node, the engine, and what to change.

## 1. Add a parameter to an existing module

Example: YuE2 gained a plan min-P, `yue2PlanMinP` (0 to 1, default 0.05). It belongs on the Planner's plan line, beside
top-P, at Lab, and it stays with the engine.

1. Refresh the fixture so the validator knows the new name: `npm run fixtures`. (Until the server declares it, the
   validator says the name is neither declared nor read by the backend; a parameter the server does not have yet is
   written with `wire: { none: '...' }` saying what is missing.)
2. Add the parameter to the module's `params` in `modules/planner.ts`, next to its neighbours:

   ```ts
   {
     key: 'yp.minp',
     type: 'number',
     label: 'Plan min-P',
     range: [0, 1, 0.01],
     default: 0.05,
     engines: ['yue2'],
     owner: 'engine',
     wire: 'ext:yue2PlanMinP',
   },
   ```

   `engines` is needed because the Planner is on four engines; without it the parameter would be on all four, and the
   validator would say that the other three backends do not have `yue2PlanMinP` and that no control edits it there. `owner: 'engine'` because the Planner's settings travel
   with the song unless they say otherwise. The range and default are the server's; if they were not, the validator
   would say so. Add `tip`, `unit` or `format` if the number needs them.
3. Show it: add a token after top-P in the YuE2 `plan` line of the terminal, at Lab:

   ```ts
   { part: 'token', param: 'yp.topp', label: '--top-p' },
   { part: 'token', param: 'yp.minp', label: '--min-p', tiers: ['lab'] },
   ```

**One file, 11 lines.** Its default, clamping, changed marker, undo name ("Plan min-P 0.05 → 0.10"), tip, revision
storage, engine ownership, the request (it is sent once changed, as `yue2PlanMinP`) and the "Also set" line at Studio all
follow; there is nothing to write for "summarised at Studio". If you forget step 3 the validator says "no control edits
it on yue2 at any tier". The generated tables in [04](04-parameter-names.md) and the counts change: `npm run docs`
rewrites them, and the check fails until you do.

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

**Four files: 68 lines of definition in the files' style, 1 in `parts.ts`, 2 in the index, a drawing.** Choosing parts:
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
and outside this format. The two ends never cross because the parameters say so (`staysBelow`), whichever part draws
them. The inventory table in [01](01-parts-inventory.md) is regenerated by `npm run docs`.

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

`npm run fixtures`. **1 line.** Saved revisions keep loading: they are keyed by the definition's key, not the server's
name. If the definition's key changes too, list the old one: `was: ['pp.ss.str']`; if an option's value changes, list
the old value on the option: `{ value: 'xover', label: 'Crossover', was: ['crossover'] }`, and change the conditions
that compare with the old value, which the validator names ("it is now "xover", which lists "crossover" in was"). For
the blend's value: **1 line, and 2 conditions.**

## 8. Remove a parameter

Delete it from `params`, and the nodes that edit it. The validator names every other place that refers to it (a
condition, an action, a read, a `beside`, a factory revision). For StableStep's fixed seed: **10 lines deleted, nothing
else.** A field of the format, a part or a transform no definition uses any more is only a note.

## 9. Add an engine

Example: a sixth engine, "X6", with half the modules.

1. When the server has its backend, add it to `ENGINES` in `engines.ts` (one line: id, name, backend folder), add its
   model buckets in `tools/make-fixtures.ts`, and run `npm run fixtures`, so the validator knows what it declares and
   reads.
2. Add `'x6'` to the `engines` of each module, pedal and unit it should have.
3. Run the validator. It names every parameter of those definitions that differs per engine (it has `per` lines and
   lists its own engines, so the new engine does not get another engine's names, ranges or defaults by accident), every
   real name the new backend does not read, and every Quick row with no choices for it. For each, add `'x6'` to the
   parameter's `engines` and a line for it, or leave it off the engine.

**1 line in `engines.ts`, 1 in the fixture tool, 1 per definition, and for each parameter that differs per engine its
`engines` and a line**: 47 lines for the example's 12 definitions, 33 of them in the 4 that have parameters differing
per engine (the Sampler, Tempo, Takes and Models).

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
| A new unit, text check, option source, desk fact or named action | one line in its list in `parts.ts` |
| A value a wire cannot say (a new packing) | an entry in `transforms.ts`, named by `via` |
