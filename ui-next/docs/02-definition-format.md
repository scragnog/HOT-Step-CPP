# The definition format

Every module, pedal and side unit of the Create desk is one definition: a plain data object that says what it is, which
engines have it, every parameter it owns and what changes per engine, where each value goes in the real application,
and what it shows at each tier, out of the parts in [01-parts-inventory.md](01-parts-inventory.md). The types are in
`src/definitions/types.ts`; this page explains them, each with a piece of a real definition, and records the choices
made and why.

## Why one definition

In the concept a module is spread over several registries: `defineControls` (its parameters), `defineModule` (its place
and size), `defineFace` (its tile, as hand-written HTML per tier), `defineThumb`, and for the board `definePedal` and
`defineUnit`. Facts about one parameter were stated in more than one of them, and five rounds of review kept finding the
same kind of defect where two statements drifted apart.

A definition states each fact once. Defaults, clamping, undo names, the changed marker, what a revision stores, engine
ownership, the "Also set" line and the request are all derived from it by code that knows nothing about any one module
(`src/derive/`, described in [05](05-what-the-definition-drives.md)), and tested against the concept page.

## Choices

**Data in `.ts` files.** A definition is a `.ts` file that exports one object literal followed by `satisfies ModuleDef`
(or `PedalDef`, `UnitDef`). [00](00-index.md) says why, and how the same definitions are checked as JSON.

**One file per module, not one per engine.** Most of a module is the same on every engine; per-engine files would repeat
it five times. Engine differences are lines inside the one file (`per`).

**Overrides are shallow lines.** A parameter's base fields apply on every engine it is on; a line in `per` names some
engines and replaces whole fields for them. A replaced field is replaced entirely: a line's `range` is the whole
`[min, max, step]`, its `options` the whole list, its `wire` the whole mapping. There is no deep merge, so what applies on
an engine is always "the base, then that engine's line", readable without running anything. Nodes have lines too, for an
engine, a tier or a state.

**No engine inherits another's by accident.** A parameter with engine lines lists its own `engines` beside them, so adding
an engine to a module does not hand it, say, ACE-Step's range and real name for the steps; and the validator checks every
real name against what that engine's backend declares or reads ([07](07-checking.md)). The engine list itself is stated
once, in `engines.ts`.

**Presence is a list.** `engines` on a definition says which engines have it; `engines` on a parameter narrows that. A
parameter an engine does not have is absent there: it is not drawn, not sent, not saved in that engine's revision, and a
condition on it is false.

**One face list for all three tiers.** A module's `face` is a single list of nodes; a node with `tiers` appears only at
those tiers, and a node's lines change it at some tiers or on some engines. The concept wrote each tier's face
separately, and the same LCD row written three times is three chances to drift.

**The concept's keys stay; the real names are recorded beside them.** A definition keeps the concept's key
(`samp.steps`, `pp.ss.str`), because one control can be five different fields in the real app: Steps is `inferenceSteps`
on ACE-Step, `mm3Steps`, `yue2OdeSteps`, `sa3Steps` and `heartmulaCodecSteps` elsewhere. `wire` says where the value goes
on each engine. See [04-parameter-names.md](04-parameter-names.md), and [08](08-open-questions.md) question 2.

**Stored values are the real values.** An option's `value` is what the server takes (`'double'`, `'presence'`,
`'crossover'`, `'ggml-medium.bin'`); its `label` is what the desk prints.

**Ownership and saving are stated, not inferred.** `owner` (engine or song) and `stored` (settings, content, none) are
set once per definition, and per parameter where one differs, as the concept keeps them.

**Strings live in the definition, in English.** Translation keys are derived from where a string sits, so a definition
never names a key ([05](05-what-the-definition-drives.md), "Strings and translation").

**Bespoke is a part, not an escape hatch.** When something cannot be said as data, the definition names a bespoke part
from the inventory and binds the parameters and state that part edits. Nothing in a definition is code; the few things a
wire cannot say are named transforms in `transforms.ts`.

**Thumbnails are drawings, kept as files.** A thumbnail is an SVG fragment in the thumbnail kit's classes, in
`src/definitions/thumbs/`, captured from the concept. A pedal's or side unit's file holds only its middle; the frame is
drawn from the definition. `node tools/thumb.ts <pedal>` draws a new pedal's middle from its parameters.

## The fields

### Definition (all three kinds)

| Field | Says | Example |
| --- | --- | --- |
| `kind` | `'module'`, `'pedal'` or `'unit'` | |
| `id` | Unique across all definitions; also its file name | `'tempo'` |
| `was` | Ids it was saved under before, so a saved workflow or desk naming one has it | `was: ['ss']` |
| `name`, `desc` | Its name and one-line description, in English | `'Tempo, key & length'` |
| `at` | Its place: Modules panel and picker order for modules, chain order for pedals, board order for units. Neighbours are 10 apart; pick a number between | `at: 20` |
| `engines` | Engine ids (`engines.ts`) that have it | `['ace', 'minimax-m3', 'yue2']` |
| `owner`, `stored` | Defaults for its parameters (below) | `owner: 'song', stored: 'settings'` |
| `params` | Its parameters | |
| `state` | What it keeps that is not a setting: a file, the seed, a selection, its adapters, a list | |
| `thumb` | Its thumbnail file, and other files on some engines | `{ file: 'tempo', per: [{ engines: ['sa3', 'heartmula'], file: 'tempo.sa3' }] }` |

### Module

| Field | Says |
| --- | --- |
| `category` | `Compose`, `Engine`, `Chain` or `Words`: its group in the panel and the picker |
| `board` | Holds the pedalboard: the pedals and side units on the board are drawn in its tile and answer to its "Also set" line (the Pedalboard) |
| `labUnder` | A Lab module: shown only at Lab, under this desk module. Its settings apply at every tier, and a note above the desk names them when they are hidden |
| `builtIn` | Part of the page, not added from the panel (the prompt, the lyric sheet, the models) |
| `source` | Brought by an action on a song (Repaint); its values are not part of a Create request |
| `tierSummary` | What each tier adds, for the picker: `[Quick, Studio, Lab]` |
| `size` | Layout hints: `w` (1, 2 or 4 columns), `quickW`, `grow`, `fold`, `foldAt3`, `stretch`, `stripKeepsWidth` |
| `quickName` | Its name at Quick when Quick shows less of it |
| `per` | Lines for some engines: `name`, `quickName`, `desc`, `tierSummary`, `size` |
| `face` | What the tile shows (below) |
| `strip` | When the tile becomes a one-line strip: what it says and the actions that fix it |

From `modules/tempo.ts`: on Stable Audio 3 and HeartMuLa the tile is a length ruler alone, so its name, words and what
each tier adds differ there.

```ts
per: [
  { engines: ['sa3'], name: 'Length', quickName: 'Length', desc: 'How long the song runs, ...', size: { grow: true },
    tierSummary: ['Length on a timeline', '+ Beat I/O and tempo and key chips for the prompt', 'Same as Studio'] },
  ...
],
```

### Pedal

| Field | Says |
| --- | --- |
| `short` | Its name outside Lab ("MM3 refiner") |
| `hue` | Its face colour in degrees. The board reads as a sequence of colours; pick a hue at least 15 degrees from its neighbours in the chain (`at`) |
| `on` | Its foot switch, one of its switches; `null` for always on, drawn with the neutral face |
| `main` | The knob it shows at Studio, one of its numbers; `null` when it shows its Studio face nodes instead |
| `face` | Only what is special, every node tiered `studio` or `lab`; the rest is laid out from its parameters ([01](01-parts-inventory.md)) |
| `idle` | Engaged with nothing to run: the amber LED and the reason, per tier ("Needs a reference track. Set it in Lab") |

Its width (1U or 2U) follows from the number of its numbers; there is no field for it.

### Side unit

| Field | Says |
| --- | --- |
| `on` | Its power switch |
| `wide` | Takes the board's full width |
| `gated` | Runs after the chain, so it stops while the chain is bypassed |
| `face` | The object it shows at every tier; at Lab its parameters follow, laid out |

### Parameter

| Field | Says |
| --- | --- |
| `key` | Unique across every definition; letters and digits in dot-separated parts |
| `type` | `number`, `boolean`, `choice` or `text`. A number whose step is whole on every engine is a whole number: cleaned, set and sent whole |
| `label`, `short`, `undoName`, `tip` | Its words: printed label; where room is short; its name in undo when the label is ambiguous; its tooltip |
| `range` | `[min, max, step]`, for numbers |
| `unit`, `format` | How the number is written (below); `unit` is one of `UNITS` in `parts.ts` |
| `default` | Its default, inside its range, on its step grid, and among its options |
| `auto` | A number that can be left to the engine; `says` is what auto stands for. `false` in a line turns it off on those engines. A choice says "Auto" with an option instead |
| `options` | A list of `{ value, label, short?, tip?, lcd?, detail?, was? }`, or `{ from, bucket?, include?, only? }` when the server owns the list |
| `placeholder`, `example`, `check`, `maxLength` | For text: the empty hint, what Insert example puts in, a check from `CHECKS` with its message, and the longest text it takes |
| `when` | It applies only while this holds; otherwise hidden (a pedal keeps its room), unchanged, not sent |
| `engines` | Narrows the definition's engines; required beside engine lines |
| `owner`, `stored` | Override the definition's |
| `held` | A song number whose limit differs per engine: the words of its note and toast (below) |
| `steps` | The named undo steps its part takes on it besides setting it (below, "Named steps") |
| `staysBelow` | Stays a step below that parameter, which stays a step above this one: a range's two ends never cross. The two share a range |
| `beside` | On a pedal, stands beside this switch, shown while the switch is off |
| `run` | Presses within this many milliseconds are one undo step (taps on TAP) |
| `was` | Keys it was saved under before (below, "Saved data") |
| `wire` | Where the value goes in the real application (below) |
| `per` | Engine lines (`{ engines, ...fields }`, each engine in one line at most), then word lines (`{ when, label?, short?, tip?, auto? }`). A word line changes words only; the type check refuses any other field in one |

From `modules/sampler.ts`, one control that differs on every engine:

```ts
{
  key: 'samp.steps', type: 'number', label: 'Steps', range: [1, 300, 1], default: 12,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  wire: 'param:inferenceSteps',
  per: [
    { engines: ['minimax-m3'], label: 'Flow steps', range: [2, 60, 1], default: 20, wire: 'ext:mm3Steps' },
    { engines: ['yue2'], label: 'ODE steps', range: [8, 64, 1], default: 32, wire: 'ext:yue2OdeSteps' },
    { engines: ['sa3'], range: [1, 64, 1], default: 8, wire: 'ext:sa3Steps' },
    { engines: ['heartmula'], label: 'Codec steps', range: [1, 100, 1], default: 10, wire: 'ext:heartmulaCodecSteps' },
  ],
},
```

and a word line, from `modules/planner.ts`: YuE2's composer CFG says what auto means, which depends on Chain of Thought:

```ts
auto: { says: 'auto (1.00)' },
per: [{ when: { param: 'lm.cotmode', is: 'off' }, auto: { says: 'auto (1.01)' } }],
```

#### Owner and stored

- `owner: 'engine'`: the value belongs to the engine. Leaving an engine puts its values aside; coming back brings them
  back; an engine reached for the first time starts on its own defaults. The sampler and the Lab modules under it,
  adapters and models.
- `owner: 'song'`: the value travels to whichever engine is chosen. Tempo and key, the lyrics, the planner, the pedals.
- `stored: 'settings'`: a revision saves it. `'content'`: drafts save it, and a revision saved "with content".
  `'none'`: only the desk keeps it (the seed, a loaded file, a repaint region's settings).

A value kept per engine is never song content (the validator refuses `owner: 'engine'` with `stored: 'content'`).

#### Held

```ts
held: { note: 'Capped at {shown} here · your {value} returns on {engine}', toast: 'Length capped at {shown}' },
```

The note is said by the control while the value is held, the toast in the engine switch's message. `{value}` is the held
value and `{shown}` the limit in force, both written as the parameter writes numbers; `{engine}` is the engine it
returns on ([05](05-what-the-definition-drives.md), "Engine switches"). `value` says how the toast names the held value
when it needs more than the number (`value: '{value} takes'`: "your 9 takes wait on ACE-Step 1.5").

#### Named steps

A bespoke part that changes a value in a way the user would name (Compose caption, Add 120 BPM to the prompt) takes one
of the parameter's named steps; the words are the definition's, so the part never names a step itself:

```ts
steps: {
  compose: 'Compose caption',
  addChip: 'Add {chip} to {name}',
  tempo: { says: 'Set the tempo to {bpm} BPM', changes: ['tempo.bpm'] },
},
```

`{name}` is the parameter's name in undo steps; the part fills any other slot. `changes` lists the parameters the step
may set besides this one, and nothing else may be set in it. A state's `steps` work the same way.

#### Format

[05](05-what-the-definition-drives.md), "How a number is written", gives the rule for a number without `format`.
`format` covers the rest, each field taken from a real case:

| Field | Case |
| --- | --- |
| `decimals` | YuE2 penalties step by 0.005 and read to three places |
| `template`, `perSecond` | `'{v} codes ({s})'` with 5 codes a second: "64 codes (13 s)" |
| `words` | Exact values said as words: window 0 is "whole history", Top-P 0 is "off", the chain switch off is "bypassed" |

The same format is used for the read-out, the tip, the undo step, the held note and the "Also set" line.

#### Wire

Where the value goes, by channel:

| Channel | Where |
| --- | --- |
| `param:` | A field of `GENERATION_DEFAULTS` (`server/src/contracts/generationControls.ts`): the input of the real app's `resolveGenerationParams()` |
| `ext:` | A backend's extension knob (`capabilities().extensions`), sent in `backendParams`, or a field its backend reads without declaring it |
| `content:` | A field of `CreateContent` (`ui/src/components/create/createContent.ts`): the input of `createContentParams()` |
| `body:` | A top-level request field set after both (HeartMuLa's frozen model choices) |
| `intent:` | A part of the Create intent beside its params: `compose.autoExpand`, `compose.loraTrigger`, `compose.beatIntro`, `compose.introBars` (the caption steps the server runs) and `captionSource` |
| `model:` | Engine state chosen through `POST /api/backends/models`; never part of the request |
| `ui:` | Form state the browser keeps; never sent |

A name may be a path into its field: `param:adapterGroupScales.self_attn`, `param:adapterStack[0].scale`. A wire is a
string, an object when the value needs changing on the way, a list when one control sets several fields, or `none`:

```ts
wire: { to: 'param:skipLm', invert: true }                                  // the real field means the opposite
wire: { to: 'param:autoTrimFadeMs', scale: 1000 }                           // seconds to milliseconds
wire: { to: 'ext:yue2MaxMinutes', divide: 60 }                              // seconds to minutes
wire: { to: 'param:advancedAdapters', map: { simple: false, advanced: true } } // an option's value to the real one
wire: { to: 'content:duration', auto: -1 }                                  // what is sent while on auto (null: nothing)
wire: { to: 'param:denoiseStrength', when: { param: 'den.on', is: true }, else: 0 } // sent while this holds, else 0
wire: { to: 'param:scheduler', via: 'scheduler' }                           // a named transform (transforms.ts)
wire: ['param:adapterScale', 'param:adapterStack[0].scale']                 // several fields
wire: { none: 'No chain switch in the real app: ...' }                      // no counterpart; say what the app does
```

Several controls may write one field when their conditions keep them apart: the normalizer's preset maps to
`lufsTarget` for every preset but Custom, and its Custom knob applies only on Custom. Several controls packed into one
field name one transform, which lists them in its `packs`.

A wire that deliberately differs from the real application names the check it waives in `differs` and says why in its
`note`: `'range'` (its range or step), `'options'`, `'default'`, or `'unread'` (a field the engine does not read). A note
alone waives nothing; it is for the reader (a field read but not declared). The request comparison runs over the whole
value space whatever a wire waives. `was` lists the real field's former names, for reading back a profile saved under
one:

```ts
wire: { to: 'param:stableStepStrength', divide: 100, differs: ['range'], note: 'Real is 0..1; the existing UI steps by 5%, the concept by 1%.' }
```

[05](05-what-the-definition-drives.md), "What is sent", says how the request is built from wires.

### Conditions

```ts
{ param: 'pp.ss.blend', is: 'crossover' }        // equal to one value (or one of a list)
{ param: 'yue.cfg', auto: false }                  // its auto switch
{ param: 'samp.retake', above: 0 }                 // numbers: above, below
{ state: 'adapters', atLeast: 1 }                  // a state key, or a desk fact (parts.ts DESK_FACTS)
{ state: 'timbreFile', set: false }
{ onDesk: 'timbre' }                               // a module, pedal or unit is on the desk
{ engine: ['ace', 'yue2'] }
{ all: [...] }, { any: [...] }, { not: ... }
```

A condition on a parameter the engine does not have is false, so `{ param: 'lm.on', is: false }` needs no engine guard
on a node shown on several engines.

### Words that name values

Any text may name values to fill in. The validator checks every name.

| Slot | Is |
| --- | --- |
| `{engine}` | The engine's name: "MiniMax-Music3 uses its own guidance" |
| `{value:<key>}` | A parameter's value, written as everywhere else: "CFG++ has no settings of its own" |
| `{count:<state or fact>}` | How many: "0 plugins" |
| `{file:<state>.<n>\|<words>}` | A loaded file's name, or the words when there is none: "tape-lofi-v2 strength", "Adapter 1 strength" |
| `{say:<fact>}` | A desk fact that is words: the adapters another engine keeps |
| `{min}`, `{max}`, `{default}` | In a parameter's own words: "up to {max}", "{default}, exact" |

Templates in a part's words (`{v}`, `{s}` in a format, `{value}` and `{shown}` in a held note, the slots of a state's
steps) are their own.

### Faces

A face is a list of nodes. A node names a part and binds it ([01](01-parts-inventory.md), "What a node binds"):

```ts
{ part: 'knob', param: 'yue.recompose', label: 'Recompose' }
{ part: 'xyPad', params: ['perf.cfgcut', 'perf.cache'], opts: { look: 'heat', xl: 'CFG cutoff', yl: 'step cache' } }
{ part: 'lcdValue', reads: ['pp.tile.plugin'], label: 'decoder', tiers: ['studio'] }
{ part: 'dropZone', state: ['timbreFile'], opts: { look: 'sleeve', empty: 'Drop a reference track' } }
{ part: 'row', children: [ ... ] }
```

and may also carry:

| Field | Says |
| --- | --- |
| `label`, `tip`, `text` | Its own words where they are not the parameter's (a short knob label, a flag `--temp` in a terminal line, a note) |
| `opts` | The part's options, typed in the inventory, so a wrong one fails the type check |
| `tiers` | Shown only at these tiers |
| `engines` | Shown only on these engines |
| `when` | Shown only while this holds |
| `dim` | Stands back at 45% while this holds, says why, stays operable |
| `choices`, `between` | A Quick choice row (below) |
| `actions` | Buttons or links: set a value, turn on auto, go to a tier and ring controls (or a state's drop zone), insert an example, add a module, reset a group's parameters, add a section tag, or a named action from `ACTIONS` |
| `per` | Lines `{ engines?, tiers?, when?, ...fields }` that change the node's fields where they hold; later lines win |

One node with lines replaces the copies a face would otherwise need. The Sampler's scope reads the CFG interval at Studio
and edits it with two handles at Lab:

```ts
{
  part: 'samplerScope', reads: ['samp.steps', 'samp.shift', 'samp.sched', 'samp.cs', 'samp.ce'],
  children: [ ... the Lua plugins switch ... ],
  per: [{ tiers: ['lab'], params: ['samp.cs', 'samp.ce'], reads: ['samp.steps', 'samp.shift', 'samp.sched'], opts: { interval: true } }],
  tiers: ['studio', 'lab'],
},
```

and the Planner's "How adventurous" row is one node whose parameter and choices change per engine.

#### Quick choices

Quick shows plain words that each set a detailed value. The row is a `choiceRow` bound to the detailed parameter; its
choices differ per engine through lines; `between` says how the row names a value set at another tier that matches no
choice:

```ts
{
  part: 'choiceRow', param: 'samp.steps', label: 'Quality', between: '{v} steps', tiers: ['quick'],
  per: [
    { engines: ['ace'], choices: [['Draft', 8], ['Standard', 12], ['High', 30]] },
    { engines: ['minimax-m3'], choices: [['Draft', 10], ['Standard', 20], ['High', 40]] },
    ...
  ],
},
```

A choice may set several values at once, by key: `['Gentle', { 'dyn.target': -16, 'dyn.ceiling': -1.5 }]`. The row then
checks the choice whose values are all in force. The validator checks that every choice is on its parameter's range and
step grid or among its options, that no two share a word or a value, that the row has choices on every engine it is shown
on, and that one choice is the default.

#### Strips

A strip is what a tile becomes when it has nothing to show; for a pedal, the same shape is its idle reason. The first
that holds at the tier wins:

```ts
strip: [
  { when: { param: 'lm.cotmode', is: 'off' }, tiers: ['quick'],
    says: 'The Planner is set to write nothing down, and this needs a score',
    actions: [{ do: 'set', param: 'lm.cotmode', value: 'full', label: 'Write a full score' }] },
  { when: { not: { state: 'sectionTags', atLeast: 1 } }, says: 'Tag the lyrics by section to build one at a time',
    actions: [{ do: 'tag', tag: '[Verse]', label: '+ [Verse]' }, { do: 'tag', tag: '[Chorus]', label: '+ [Chorus]' }] },
],
```

#### How a tier acknowledges what it cannot show

Nothing to write. Each tier's edited parameters are the ones edited by nodes shown at that tier, so a changed parameter
outside that set is listed on the tile's foot ("Also set: Key D major") with a link to the lowest tier that edits it,
which rings it on arrival ([05](05-what-the-definition-drives.md)).

### State

```ts
state: [
  { key: 'seed', kind: 'seed', noun: 'Seed', empty: 'random', steps: { set: '{noun} {from} → {to}' },
    owner: 'song', stored: 'none', wire: 'param:seed' },
],
```

`kind` is `file`, `seed`, `span`, `slots`, `record` or `list`, and says what it may hold ([05](05-what-the-definition-drives.md),
"Values"). `noun` names it; `empty` is how an empty value reads in a step ("Seed random → 1834920213"). `steps` are its
named steps, as a parameter's: the part asks for a step by name and the words come from here; a file has `load` and
`clear` without saying. Loading an adapter is `{ says: 'Load {file}', changes: ['ad.adv'] }`, because loading a second
one while Stack is Single stacks them. `was` lists keys it was saved under before. `wire` is a wire, or a list of
`{ engines, wire }` when it differs per engine.

## Saved data

Every record the page saves carries `v`, the version of its shape, and its types are in `types.ts` beside the
definitions:

| Record | Holds |
| --- | --- |
| `Workflow` | The desk's layout: its modules, its board (pedals and units in order) and its tier. No values |
| `Revision` | One engine's settings, saved under a workflow, and with `withContent` the song's content too; the adapter slots in `state` |
| `Draft` | One song's content for one engine |
| `SavedDesk` | Everything, for a reload: values, auto switches, state, the engines' put-aside values and the held values |

Values are keyed by parameter key, in the definition's own units; auto switches are a map of their own. A renamed key
keeps old records loading through `was` on the parameter, a renamed option value through `was` on the option, a renamed
definition or state through `was` on it, and a renamed real field through `was` on the wire. No definition uses them
yet, because nothing has been saved in this format (the core's tests exercise them on a renamed copy); the first rename
after the page ships would look like this:

```ts
{ key: 'pp.ss.refine', was: ['pp.ss.str'], ... }                    // the key renamed
{ value: 'ggml-medium.bin', label: 'Medium', was: ['medium'] }       // an option's value renamed
```

Something removed on purpose is listed in `src/definitions/retired.ts`, and saved data naming it loads without it and
without a word. `tools/fixtures/shipped.json` lists every key, state and id shipped, and the validator fails until each
one that has gone is named in a `was` or retired. A record of an older shape is read by `migrate()`
([05](05-what-the-definition-drives.md), "Saved data").

The factory workflows and revisions (`factory.ts`, the concept's seven desks and eleven revisions) are records of these
types, and the validator checks them as it checks a loaded file: every module and pedal exists, and every value is one
its parameter can hold on the revision's engine.

## What is data and what is not

| Data | Not data |
| --- | --- |
| Every parameter: type, range, step, default, options, units, words, per engine | How a part draws and moves (its component) |
| Which engines have a module or a parameter, and every per-engine difference | The packer's placement of tiles (it reads `size` hints) |
| What each tier shows, Quick's plain choices and the values they set | Bespoke instruments: the key wheel, the scope, the split-flap, the cartridges, the build board |
| Conditions: when a control applies, when it stands back and what it says | Computed read-outs whose formula is the module's own: the speed estimate, the flow windows, the plan text |
| Owner, storage, held values, undo names, tips | Animations and motion |
| Where every value goes in the request, and how it changes on the way | The few packings a wire cannot say (transforms.ts), and server calls a bespoke part makes |
| Thumbnail file names | The thumbnail drawings themselves (asset files) |

The line is drawn where a statement could drift from another statement. Everything that the generic code reads lives in
the definition; everything that is one module's own behaviour lives in its bespoke part, named in the inventory with what
it is responsible for.
