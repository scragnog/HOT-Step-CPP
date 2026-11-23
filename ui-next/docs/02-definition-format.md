# The definition format

Every module, pedal and side unit of the Create desk is one definition: a plain data object that says what it is, which
engines have it, every parameter it owns and what changes per engine, and what it shows at each tier, out of the parts
in [01-parts-inventory.md](01-parts-inventory.md). The types are in `src/definitions/types.ts`; this page explains them,
each with a piece of a real definition, and records the choices made and why.

## Why one definition

In the concept a module is spread over several registries: `defineControls` (its parameters), `defineModule` (its place
and size), `defineFace` (its tile, as hand-written HTML per tier), `defineThumb`, and for the board `definePedal` and
`defineUnit`. Facts about one parameter were stated in more than one of them, and five rounds of review kept finding the
same kind of defect where two statements drifted apart: a revision that did not store a module's settings, an undo step
named after an internal key, a changed marker missing on one control type, a Quick summary that disagreed with Lab.

A definition states each fact once. Defaults, clamping, undo names, the changed marker, what a revision stores, engine
ownership and the "Also set" line are all derived from it by code that knows nothing about any one module
([05-what-the-definition-drives.md](05-what-the-definition-drives.md)). The concept's per-module code that worked these
things out (`isEngOwn`, `CONTENT_KEYS`, `PROFILE_SKIP`, `HELD_KEYS`, `hiddenChanged`, each face's own `changed(k)` calls)
has no counterpart in the rebuild.

## Choices

**Typed `.ts` data files, not `.json`.** A definition is a `.ts` file that exports one object literal followed by
`satisfies ModuleDef` (or `PedalDef`, `UnitDef`). The object is plain data, and the validator checks that it survives a
JSON round trip unchanged, so it could be moved to JSON at any time. The reason for `.ts`: the compiler checks part names,
engine ids, value types and field names as you type, which an imported JSON file does not get (its strings widen to
`string`). A JSON Schema could give the editor the same help, but it would be a second statement of the format.

**One file per module, not one per engine.** Most of a module is the same on every engine; per-engine files would repeat
it five times. Engine differences are overrides inside the one file (`per`).

**Overrides are shallow and per field.** A parameter's base fields apply on every engine it is on; `per[engine]` replaces
whole fields for that engine. A replaced field is replaced entirely: an override of `range` gives the whole
`[min, max, step]`, an override of `options` the whole list, an override of `wire` the whole mapping. There is no deep
merge, so what applies on an engine is always "the base, then that engine's line", readable without running anything.

**Presence is a list, not an override.** `engines` on a definition says which engines have it; `engines` on a parameter
narrows that. A parameter an engine does not have is absent there: it is not drawn, not sent, not saved in that engine's
revision, and a condition on it is false.

**One face list for all three tiers.** A module's `face` is a single list of nodes; a node with `tiers` appears only at
those tiers. The concept wrote each tier's face separately, and the same LCD row written three times is three chances to
drift. Here the Tempo tile's BPM row is written once and appears at every tier, while its key row says
`tiers: ['studio', 'lab']`.

**The concept's keys stay; the real names are recorded beside them.** A definition keeps the concept's key
(`samp.steps`, `pp.ss.str`), because one control on the desk can be five different fields in the real app: Steps is
`inferenceSteps` on ACE-Step, `mm3Steps`, `yue2OdeSteps`, `sa3Steps` and `heartmulaCodecSteps` elsewhere. `wire` records
the real name per engine. See [04-parameter-names.md](04-parameter-names.md).

**Stored values are the real values.** An option's `value` is what the server takes (`'double'`, `'presence'`,
`'crossover'`); its `label` is what the desk prints. The concept stored the label and translated it in places. Storing
the real value means a revision, a draft and a request all carry the same thing.

**Ownership and saving are stated, not inferred.** `owner` (engine or song) and `stored` (settings, content, none) are
set once per definition, and per parameter where one differs. The concept inferred them from a value's shape and from
lists in other files; that inference is the source of two of the review findings.

**Strings live in the definition, in English.** Translation keys are derived from where a string sits, so a definition
never names a key. See "Strings and translation" in [05](05-what-the-definition-drives.md).

**Bespoke is a part, not an escape hatch.** When something cannot be said as data, the definition names a bespoke part
from the inventory and still declares the parameters that part edits. Nothing in a definition is code.

**Thumbnails are drawings, kept as files.** A thumbnail is an SVG fragment in the thumbnail kit's classes, in
`src/definitions/thumbs/`, captured from the concept. A pedal's or side unit's file holds only its middle: the frame
(body in its hue, screws, name, foot switch or plate) is drawn from the definition. Drawing a thumbnail as data
(primitives with coordinates) was tried on paper and rejected: it is the same drawing, harder to read and to edit.

## The fields

### Definition (all three kinds)

| Field | Says | Example |
| --- | --- | --- |
| `kind` | `'module'`, `'pedal'` or `'unit'` | |
| `id` | Unique among its kind; also its file name | `'tempo'` |
| `name`, `desc` | Its name and one-line description, in English | `'Tempo, key & length'` |
| `at` | Its place: Modules panel and picker order for modules, chain order for pedals, board order for units. Neighbours are 10 apart; pick a number between | `at: 20` |
| `engines` | Engine ids that have it | `['ace', 'minimax-m3', 'yue2']` |
| `owner`, `stored` | Defaults for its parameters (below) | `owner: 'engine', stored: 'settings'` |
| `params` | Its parameters | |
| `state` | What it keeps that is not a setting: a file, the seed, a selection, its adapters | |
| `thumb` | Its thumbnail file, and other files per engine | `{ file: 'tempo', per: { sa3: 'tempo.sa3' } }` |

### Module

| Field | Says |
| --- | --- |
| `category` | `Compose`, `Engine`, `Chain` or `Words`: its group in the panel and the picker |
| `labUnder` | A Lab module: shown only at Lab, under this module. Its settings apply at every tier, and a note above the desk names them when they are hidden |
| `builtIn` | Part of the page, not added from the panel (the prompt, the lyric sheet, the models) |
| `source` | Brought by an action on a song (Repaint) |
| `tierSummary` | What each tier adds, for the picker: `[Quick, Studio, Lab]` |
| `size` | Layout hints: `w` (1, 2 or 4 columns), `quickW`, `grow`, `fold`, `foldAt3`, `stretch`, `stripKeepsWidth` |
| `quickName` | Its name at Quick when Quick shows less of it |
| `per` | Per engine: `name`, `quickName`, `desc`, `size` |
| `face` | What the tile shows (below) |
| `strip` | When the tile becomes a one-line strip, what it says and the one action that fixes it |

From `modules/tempo.ts`: the name and size differ per engine, because on Stable Audio 3 and HeartMuLa the tile is a
length ruler alone.

```ts
per: {
  'minimax-m3': { name: 'Tempo & key', quickName: 'Tempo', desc: 'BPM with tap tempo, key on a circle of fifths, and Beat I/O.' },
  sa3: { name: 'Length', quickName: 'Length', desc: 'How long the song runs, ...', size: { grow: true } },
},
```

### Pedal

| Field | Says |
| --- | --- |
| `short` | Its name outside Lab ("MM3 refiner") |
| `hue` | Its face colour in degrees |
| `on` | Its foot switch, one of its boolean parameters; `null` for always on |
| `main` | The knob it shows at Studio, one of its numbers; `null` when it shows its Studio face nodes instead |
| `thumbWidth` | Its thumbnail's width in pedal units, 1 or 2 |
| `face` | Only what is special, every node tiered `studio` or `lab`; the rest is laid out from its parameters ([01](01-parts-inventory.md)) |
| `idle` | Engaged with nothing to run: the amber LED and the reason ("Needs a reference track") |
| `blank` | What the blanking plate over hidden knob cells says ("with Target: Custom") |

`pedals/ppvae.ts` is a whole pedal: identity, its switch and main knob, two parameters, a thumbnail. Its Lab body, Studio
knob, foot switch, LED, cables, undo names, "Also set" wording and revision storage all follow from that.

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
| `key` | Unique across every definition |
| `type` | `number`, `boolean`, `choice` or `text` |
| `label`, `short`, `undoName`, `tip` | Its words: printed label; where room is short; its name in undo when the label is ambiguous ("Plan strength start" for a knob labelled "Start"); its tooltip |
| `range` | `[min, max, step]`, for numbers |
| `unit`, `format` | How the number is written (below) |
| `default` | Its default, inside its range and among its options |
| `auto` | It can be left to the engine; `says` is what auto stands for. `false` in an override turns it off for that engine |
| `options` | A list of `{ value, label, short?, tip?, lcd?, detail? }`, or `{ from, bucket?, include? }` when the server owns the list |
| `placeholder`, `example`, `check` | For text: the empty hint, what Insert example puts in, and a named check with its message |
| `when` | It applies only while this holds; otherwise hidden (a pedal keeps its room), unchanged, not sent |
| `labelWhen` | A label that depends on another setting |
| `engines` | Narrows the definition's engines |
| `owner`, `stored` | Override the definition's |
| `held` | A song value whose limit differs per engine (length, takes) |
| `below`, `above` | Stays a step below or above another parameter: a range's two ends never cross |
| `beside` | On a pedal, stands beside this switch, shown while the switch is off |
| `run` | Presses within this many milliseconds are one undo step (taps on TAP) |
| `wire` | Where the real application keeps it (below) |
| `per` | Per engine overrides of any of the fields above from `label` to `wire` |

From `modules/sampler.ts`, one control that differs on every engine:

```ts
{
  key: 'samp.steps', type: 'number', label: 'Steps', range: [1, 300, 1], default: 12,
  wire: 'param:inferenceSteps',
  per: {
    'minimax-m3': { label: 'Flow steps', range: [2, 60, 1], default: 20, wire: 'ext:mm3Steps' },
    yue2: { label: 'ODE steps', range: [8, 64, 1], default: 32, wire: 'ext:yue2OdeSteps' },
    sa3: { range: [1, 64, 1], default: 8, wire: 'ext:sa3Steps' },
    heartmula: { label: 'Codec steps', range: [1, 100, 1], default: 10, wire: 'ext:heartmulaCodecSteps' },
  },
},
```

#### Owner and stored

- `owner: 'engine'`: the value belongs to the engine. Leaving an engine puts its values aside; coming back brings them
  back; an engine reached for the first time starts on its own defaults. The sampler, the planner, adapters and models.
- `owner: 'song'`: the value travels to whichever engine is chosen. Tempo and key, the lyrics, the pedals, the seed.
- `stored: 'settings'`: a revision saves it. `'content'`: drafts save it, and a revision saved "with content". `'none'`:
  only the desk keeps it (a repaint region's settings belong to the song being repainted).

#### Format

Without `format`, a number shows as many decimals as its step needs, then its unit ("9.0", "4.5 Hz", "30%"); `unit: 's'`
reads as m:ss from a minute. `format` covers the rest, each field taken from a real case:

| Field | Case |
| --- | --- |
| `decimals` | YuE2 penalties step by 0.005 and read to three places |
| `template`, `perSecond` | `'{v} codes ({s})'` with 5 codes a second: "64 codes (12.8 s)" |
| `words` | Exact values said as words: window 0 is "whole history", Top-P 0 is "off", the chain switch off is "bypassed" |
| `percent` | 0..1 written as a percentage: the reference blend |

The same format is used for the read-out, the tip, the undo step and the "Also set" line, so they never disagree.

#### Wire

A string `'<channel>:<name>'`, or an object when the mapping needs more:

```ts
wire: { to: 'param:skipLm', invert: true, note: 'skipLm is the opposite of Plan the song first.' }
wire: { to: 'param:autoTrimFadeMs', scale: 1000, note: 'Real is milliseconds.' }
wire: { to: 'content:duration', auto: -1, note: 'Auto is -1. SA3 renders 30 s for -1.' }
wire: { none: 'No switch in the real app: the denoiser runs while denoiseStrength is above 0.' }
```

The channels are listed in `types.ts` (`Channel`). `none` is a finding, not a gap in the file: the text says what the
real app does instead.

### Conditions

```ts
{ param: 'pp.ss.blend', is: 'crossover' }        // equal to one value (or one of a list)
{ param: 'yue.cfg', auto: false }                  // its auto switch
{ param: 'samp.retake', above: 0 }                 // numbers: above, below
{ state: 'adapters', atLeast: 1 }                  // desk state: a module's state key or a DESK_STATES fact
{ state: 'timbreFile', set: false }
{ onDesk: 'timbre' }                               // a module is on the desk
{ engine: ['ace', 'yue2'] }
{ all: [...] }, { any: [...] }, { not: ... }
```

A condition on a parameter the engine does not have is false.

### Faces

A face is a list of nodes. A node names a part and binds it:

```ts
{ part: 'knob', param: 'yue.recompose', label: 'Recompose' }
{ part: 'xyPad', params: ['perf.cfgcut', 'perf.cache'], opts: { look: 'heat', xl: 'CFG cutoff', yl: 'step cache' } }
{ part: 'row', children: [ ... ] }
```

and may also carry:

| Field | Says |
| --- | --- |
| `label`, `tip`, `text` | Its own words where they are not the parameter's (a short knob label; a note's words) |
| `opts` | The part's options, as listed for it in the inventory |
| `tiers` | Shown only at these tiers |
| `engines` | Shown only on these engines |
| `when` | Shown only while this holds |
| `dim` | Stands back at 45% while this holds, says why, stays operable |
| `choices`, `between` | A Quick choice row (below) |
| `action` | A button: set a value, turn on auto, go to a tier and ring controls, insert an example, add a module, or a named bespoke action |

#### Quick choices

Quick shows plain words that each set a detailed value. The row is a `choiceRow` bound to the detailed parameter, with its
choices per engine when they differ, and `between` says how the row names a value set at another tier that matches no
choice:

```ts
{
  part: 'choiceRow', param: 'samp.steps', label: 'Quality', between: '{v} steps', tiers: ['quick'],
  choices: { ace: [['Draft', 8], ['Standard', 12], ['High', 30]], 'minimax-m3': [['Draft', 10], ['Standard', 20], ['High', 40]], ... },
}
```

A value between the choices checks none of them; a marker stands at its place on the row and the label says
"Quality · 20 steps". The validator checks every choice is inside the parameter's range on its engine.

#### How a tier acknowledges what it cannot show

Nothing to write. Each tier's shown parameters are the ones bound by nodes visible at that tier, so a changed parameter
outside that set is listed on the tile's foot ("Also set: Key D major") with a link to the lowest tier that shows it,
which rings it on arrival ([05](05-what-the-definition-drives.md)).

### State

```ts
state: [{ key: 'timbreFile', kind: 'file', noun: 'the timbre reference', load: 'Load {file} as the timbre reference',
  owner: 'song', stored: 'content', wire: 'param:timbreAudioPath' }],
```

`kind` is `file`, `seed`, `span`, `slots` or `record`. `noun` and `load` name its undo steps. A part that edits state
names it in `opts.state`.

## What is data and what is not

| Data | Not data |
| --- | --- |
| Every parameter: type, range, step, default, options, units, words, per engine | How a part draws and moves (its component) |
| Which engines have a module or a parameter, and every per-engine difference | The packer's placement of tiles (it reads `size` hints) |
| What each tier shows, Quick's plain choices and the values they set | Bespoke instruments: the key wheel, the scope, the split-flap, the cartridges, the build board |
| Conditions: when a control applies, when it stands back and what it says | Computed read-outs whose formula is the module's own: the speed estimate, the flow windows, the plan text |
| Owner, storage, held values, undo names, tips | Animations and motion |
| The real application's name for every value | Server calls a bespoke part makes (score preview, build sessions, VST host) |
| Thumbnail file names | The thumbnail drawings themselves (asset files) |

The line is drawn where a statement could drift from another statement. Everything that the generic code reads lives in
the definition; everything that is one module's own behaviour lives in its bespoke part, named in the inventory with what
it is responsible for.

## Rules the validator adds

`tools/validate-definitions.ts` checks what the types cannot ([07-checking.md](07-checking.md)): keys unique, defaults in
range and among their options, option lists non-empty, every part named exists and fits what it is bound to, options
passed to a part are ones it takes, overrides and conditions name real engines, parameters and state, every module
parameter is shown at some tier, every thumbnail exists and is used, every field of the format is used by some definition,
and every part of the inventory by some definition.
