# How to do the common jobs

Each job below was done on a copy of `src/definitions` and checked with the compiler and the validator; the line counts
are from `git diff --stat` of that copy. After any of them run, from `ui-next`:

```
npx tsc -p tsconfig.json
node tools/validate-definitions.ts
```

Both must pass. The validator names the file and the field when something is wrong.

## 1. Add a parameter to an existing module

Example: YuE2 gained a plan token budget, `yue2PlanMaxTokens` (it is a real extension today with no control on the
concept). It belongs with the planner's other plan settings in Lab.

1. Find the real name and its range: `GET /api/capabilities?backend=yue2` lists it under `extensions` with its type,
   range and default (or read `server/src/services/backends/yue2/index.ts`).
2. Add the parameter to the module's `params` in `modules/planner.ts`, next to its neighbours:

   ```ts
   {
     key: 'yp.maxtok', type: 'number', label: 'Plan max tokens', engines: ['yue2'],
     range: [256, 4096, 64], default: 2048, wire: 'ext:yue2PlanMaxTokens',
   },
   ```

   Add `tip`, `unit` or `format` if the number needs them. Leave out `owner` and `stored`: the module's apply.
3. Show it: add the key where it belongs in the face. Here, the YuE2 `set` line of the terminal at Lab:

   ```ts
   params: ['lm.plaus', 'lm.resc', 'yp.maxtok'],
   opts: { command: 'set', flags: ['--cfg-plausibility', '--cfg-rescale', '--max-tokens'] },
   ```

**One file, 6 lines added, 2 changed.** Its default, clamping, changed marker, undo name ("Plan max tokens 2048 → 2112"),
tip, revision storage, engine ownership and "Also set" line at Studio all follow. If you forget step 3 the validator says
"no face shows this parameter at any tier"; a default outside the range, or a `when` naming a key that does not exist,
fails the same way.

On a pedal there is no step 3: a pedal's controls are laid out from its parameters, in the order of `params`.

## 2. Add a new pedal

Example: a stereo width pedal (made up for the trial).

1. Create `pedals/width.ts`:

   ```ts
   // Pedal "Stereo width".

   import type { PedalDef } from '../types.ts';

   export const width = {
     kind: 'pedal', id: 'width', name: 'Stereo width', desc: 'Widens or narrows the stereo image.',
     at: 75, engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'], hue: 250,
     on: 'pp.width.on', main: 'pp.width.amt', thumbWidth: 1, owner: 'song', stored: 'settings',
     params: [
       { key: 'pp.width.on', type: 'boolean', label: 'On', default: false, wire: 'param:stereoWidthEnabled' },
       { key: 'pp.width.amt', type: 'number', label: 'Width', range: [0, 200, 5], unit: '%', default: 100, wire: 'param:stereoWidth' },
     ],
     thumb: { file: 'pd-width' },
   } satisfies PedalDef;
   ```

   `at` places it in the chain: between Pre-VST gain (70) and the VST3 chain (80). `hue` is its face colour.
2. Add two lines to `index.ts`: its import, and its name in `DEFINITIONS`.
3. Add its thumbnail middle, `thumbs/pd-width.svg`. Start from the closest existing pedal's (`pd-gain.svg` is one knob)
   and move its knob values; the frame is drawn for you.

**Three files: 14 lines of definition, 2 in the index, a 9-line drawing.** At Lab it is a 1U pedal with one knob and its
foot switch; at Studio its knob; at Quick its name and switch; it is in the Modules panel and the picker under Pedals, in
the chain at its place, counted on the board, and its settings travel with the song and are saved in revisions. If the
real app has no field for it yet, write `wire: { none: '...' }` and say what is missing; that is a finding for the server.

## 3. Add a new module out of existing parts

Example: YuE2's song-ending controls (`yue2EndThreshold`, `yue2EndBias`, `yue2EndBiasFrom`, `yue2EndBiasRamp`), real
extensions with no control on the concept, as a Lab module under the Planner.

1. Create `modules/endings.ts`:

   ```ts
   // Module "Song endings" (YuE2, Lab).

   import type { ModuleDef } from '../types.ts';

   export const endings = {
     kind: 'module', id: 'endings', name: 'Song endings', desc: 'When the composer may end the song, and how hard it is pushed to.',
     at: 175, category: 'Engine', labUnder: 'planner', engines: ['yue2'], owner: 'engine', stored: 'settings', size: { w: 1 },
     params: [
       { key: 'yue.endth', type: 'number', label: 'End threshold', range: [0, 1, 0.01], default: 0, wire: 'ext:yue2EndThreshold' },
       { key: 'yue.endbias', type: 'number', label: 'End bias', range: [-20, 20, 0.5], default: 0, wire: 'ext:yue2EndBias' },
       { key: 'yue.endfrom', type: 'number', label: 'Bias from', unit: 's', range: [0, 900, 5], default: 0, wire: 'ext:yue2EndBiasFrom', when: { param: 'yue.endbias', not: 0 } },
       { key: 'yue.endramp', type: 'number', label: 'Bias ramp', unit: 's', range: [0, 120, 5], default: 0, wire: 'ext:yue2EndBiasRamp', when: { param: 'yue.endbias', not: 0 } },
     ],
     face: [
       { part: 'row', children: [{ part: 'knob', param: 'yue.endth', label: 'Threshold' }, { part: 'knob', param: 'yue.endbias', label: 'Bias' }] },
       { part: 'row', children: [{ part: 'knob', param: 'yue.endfrom', label: 'From' }, { part: 'knob', param: 'yue.endramp', label: 'Ramp' }],
         dim: { when: { param: 'yue.endbias', is: 0 }, says: 'No end bias' } },
     ],
     thumb: { file: 'endings' },
   } satisfies ModuleDef;
   ```

2. Add two lines to `index.ts`.
3. Add `thumbs/endings.svg`: a drawing in the thumbnail kit's classes on the 176 x 112 canvas, kept inside the 15-unit
   inset (start from a module with similar controls; `replan.svg` has knobs).

**Three files: 20 lines of definition, 2 in the index, a drawing.** Choosing parts: look in
[01-parts-inventory.md](01-parts-inventory.md) for the part whose "What it is for" matches, and copy a node from a
module that uses it (the "Used by" column). For a desk module (not Lab-only), tier the nodes that Quick or Studio should
not show (`tiers: ['studio', 'lab']`), give Quick a `choiceRow` when a plain choice makes sense, and set `size`.

## 4. Add a new part to the inventory

When no part does what a module needs:

1. Add an entry to `PARTS` in `parts.ts`: its `kind`, what it `binds`, a `doc` sentence, its `opts` with what each does,
   its `states`, and its `css`. A bespoke part also says what it is `responsible` for, and so what it leaves to the
   generic machinery: it must not keep its own copy of a value, decide a default, write an undo name or draw the changed
   marker.
2. Write its component under the same name, and its stylesheet in the design language: tokens from `00-tokens.css`, the
   shared states of layer 35, and a file of its own, as the concept's rules for adding a component say.
3. Use it in the definition that needed it. The validator refuses a part no definition uses, and an option no part lists.
4. Print the inventory table again (`node tools/validate-definitions.ts --inventory`) into
   [01-parts-inventory.md](01-parts-inventory.md).

**About 6 to 10 lines in `parts.ts`**, plus the component and its stylesheet, which are the real work and are outside this
format. From then on every definition can name it.

## Smaller jobs

| Job | Where |
| --- | --- |
| Rename a control | Its `label` (and `short`) in the parameter; per engine in `per` |
| Change a range or default on one engine | `per.<engine>.range` / `default` |
| Make a control apply only sometimes | `when` on the parameter |
| Make a control stand back with a reason | `dim` on its node |
| Change the plain choices at Quick | `choices` on the `choiceRow` node |
| Move a control to another tier | `tiers` on its node |
| Take a module off an engine | its `engines`; parameters follow |
| The real name changed on the server | the parameter's `wire`, then print the names table into [04](04-parameter-names.md) |
