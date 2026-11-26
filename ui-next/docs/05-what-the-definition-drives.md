# What the definition drives

Everything on this page is derived from the definitions by code that has no branch for any one module: pure functions
in `src/derive/`, which take definitions and a desk and return values. The validator and the tools call them, and the
page will be a view over them. If a module needs one of these to behave differently, the difference belongs in its
definition (a field) or in its bespoke part, never in this code. The behaviour is the concept's
(`C:\Users\sneth\misc\HOT-Step-UI\src\js`, cited by file), and `tools/test-core.ts` checks it against what the concept
page does ([07](07-checking.md)).

| File | What it derives |
| --- | --- |
| `derive/core.ts` | A parameter on an engine, conditions, the value in force, the desk facts, the changed marker, how a number is written, which nodes show and what they edit, a tile's strip |
| `derive/values.ts` | Cleaning a value, a record's values and its state; whole numbers; a range's two ends; setting a value from a control |
| `derive/steps.ts` | Undo step names |
| `derive/engine.ts` | An engine switch: what is put aside, what comes back, what is held, and the toast |
| `derive/tiers.ts` | Each tile's "Also set" line and where its link goes; a control's tip; a hidden knob's plate |
| `derive/request.ts` | The desk in real names, the Create intent, and reading real fields back |
| `derive/records.ts` | What a revision, a draft, a workflow and the desk autosave hold; loading them; older shapes |
| `derive/parts.ts` | What a part is given at run time ([01](01-parts-inventory.md)) |
| `derive/view.ts` | What the view asks: a Quick row, a pedal laid out, an action, a node standing back, loading a workflow or draft |
| `derive/strings.ts` | The translation key of every string, and the messages that name values |
| `derive/messages.ts` | Messages: an id and its slots, the core's English table, and `say()`, which fills one |

## The desk

What everything reads (`Desk` in `derive/core.ts`): the engine in use, the tier, the stored values (parameter key to
value), the auto switches (key to on or off; absent is on), the module state, what is on the desk, the two facts the
page is told (`told`: the plugin host's count and the words for another engine's adapters), the models the server has
selected (`selection`) and the server's lists with their words (`lists`). Every other desk fact in `parts.ts`
`DESK_FACTS` is worked out from its rule when it is read (`factOf`), so none is stored and none can be stale.

## The value in force

One answer to "what does this value say now", used by conditions, desk facts, words, the changed marker, the "Also
set" line and the request alike:

```
onDesk(d)        = d is a built-in module, or d is on the desk (a module the tier hides is still on the desk)
valueOf(key)     = onDesk(its definition) ? the stored value as the engine takes it (storedOn) : its default
stateOf(key)     = onDesk(its definition) ? the state it holds : nothing
factOf(name)     = the fact's rule over values in force, or what the page was told
```

A definition off the desk keeps its values for when it comes back, but nothing reads them, so the page and the request
never describe two different songs. A pedal's switch defaults to off (the validator checks it), so a pedal off the
board is off. What is about stored values whatever the desk shows (what an engine switch puts aside and holds, what a
revision saves) reads `everythingOn(desk)`. The existing page has no desk; question 11 in [08](08-open-questions.md)
asks whether the request may differ from it here.

## A parameter on an engine

```
resolve(p, E)  = the base fields, then the engine line that names E (shallow, per field)
wordsOf(p, E)  = resolve(p, E), then the label, short, tip and auto words of every word line whose condition holds
has(p, E)      = E is in the parameter's engines and in its definition's
applies(p, E)  = has(p, E) and its `when` holds
```

`applies` is the concept's `ok(k)`. A condition on a parameter the engine does not have is false. A word line changes
words only; the type check refuses any other field in one.

## Values: one rule for every path

A stored value is in the parameter's own units, kept in the parameter's domain: the union over all its engines. Which
engine is in use matters only when the value is read or sent.

**Cleaning** (`cleanValue`, `cleanValues`, `cleanStates`) runs on every path that brings values in from outside the
desk's memory: a reload, another tab's desk, an import, a restore, an engine's stash, a factory record (the validator
runs it on them).

| Type | Kept as | Not kept |
| --- | --- | --- |
| number | A finite number, clamped to the union of its engines' ranges. A whole number (its step is whole on every engine) is rounded; any other number is never rounded | Anything else |
| boolean | `true` or `false` | Anything else |
| choice | One of the union of its engines' fixed options; for a server-listed choice, any value of its default's type, until a list has loaded that lacks it | Anything else |
| text | A string, cut to its `maxLength` (else 20,000 characters), that passes its `check` on every engine that checks it | Anything else |
| auto switch | `true` or `false`, for a parameter with `auto` on some engine | Anything else |

A range's two ends (`staysBelow`) that arrive crossed are put in order. A key that is no parameter's is read as the one
that lists it in `was`, an option value likewise, a state key and a definition id too. A key in `retired.ts` is dropped
without a word; anything else not kept is counted, and the toast says so ("2 settings could not be used").

State is kept by its kind: a file is a string; a seed a whole number from 0 to 18446744073709551615 (the server's
64-bit seed), as a number while it is exact and as its decimal digits above that, or null; a span `{ start, end }` with
start below end; the adapter slots up to three `{ file, type, ph? }`, `ph` being the active phase `[from %, to %]`; a
list an array; a record an object.

**Setting a value** from a control (`setValue`) cleans it against the engine in use: a number is clamped to that
engine's range and put on its step grid, a range's end stops a step from the other end, and a typed number out of range
is said once ("Steps: 1 to 64. Set to 64."). A value that is not a number at all (NaN) is refused with an error, as is
state a part gives that its kind does not keep (`BoundState.apply` cleans it on the way in): a part's mistake fails
where it is made.

**Reading a value** on engine `E` (`storedOn`) gives the stored value when `E` can take it, else the nearest `E` can
take for a number (its limit), else the default. Reading never writes. `valueOf` is that while the definition is on
the desk ("The value in force").

## The changed marker

```
changed(p, E) = applies(p, E) and (auto ? not on auto : stored and valueOf(p, E) differs from the default)
```

A value that does not apply is never changed, so a value kept for a condition that does not hold is not marked, counted
or listed. The marker is drawn after a control's words by the wrapper every control has.

## How a number is written

`formatValue`, the concept's `fmt` (`03-controls.js`):

1. A `format.words` entry for the exact value wins; a switch is "on" or "off"; a choice is its option's label, a
   server-listed one its name from the server's list.
2. A `format.template` fills `{v}` with the number and `{s}` with the number over `perSecond`, as seconds.
3. Seconds (`unit: 's'`) read as m:ss from a minute ("5:00"); below a minute with one decimal under 10 s when the step is
   below 1, else none ("9.5 s", "13 s").
4. Otherwise two decimals when the step is below 0.1, one below 1, none from 1 (`format.decimals` overrides), a true
   minus sign, then the unit after a space, or right after the number for `%`.

The same function writes the read-out, the tip, the undo step, the held note and the "Also set" line.

## Undo step names

Every change is one undo step, named by `stepWords(before, after)` from the desk before and after it (the concept's
`vsWords`, `14-value-steps.js`). Each name has the label the Undo button shows and what its toasts and preview say.

| Change | Label | Undo says, Redo says, Undo will put back |
| --- | --- | --- |
| A named step a part takes | The step's words, its slots filled | Puts back what it changed |
| The seed, or only its lock | `Seed {from} → {to}`, the number while kept, else "random" | `Seed back to {from}`, `Seed {to}`, `Puts the seed back to {from}` |
| One text | `Edit {the name}` | `Puts back {the name} as it was` (as they were, for a plural undo name) |
| A choice of a Quick row, one value or several | `{row}: {from} → {to}`, in the row's words (a value between the choices is said as its values) | `{row} back to {from}`, `{row}: {to}` |
| One value | `{name} {from} → {to}`, "auto" for auto | `{name} back to {from}`, `{name} {to}` |
| Two numbers together (an XY pad) | `{a} {from} → {to}, {b} {from} → {to}`, the owner said once; `Reset {a} and {b}` when both go back | `{a} and {b} put back` |
| Anything else | `Change {what}` | `Puts back {what}` |

`{name}` is the parameter's `undoName` if it has one; a pedal's or unit's switch labelled On is the pedal or unit; any
other is its label, with its definition's name on the engine in use first when another parameter has the same label
("Repetition guard mode"). A drag, a held key or a run of scroll steps on one control is one step, as is a run of
presses within the parameter's `run`, and everything a part sets inside one `batch`. A part never names a step: it
asks for one of its definition's named steps ([02](02-definition-format.md), "Named steps"), or sets values, and the
core names what changed.

## Engine switches

`switchEngine(desk, book, to)` (the concept's `enterEngine` and `engineNews`, `13-engine-state.js`), one undo step:

1. Every value and state whose owner is `engine` is put aside for the engine left (`book.stash`); what was put aside for
   the new one comes back. An engine reached for the first time starts on its defaults.
2. Song values stay, whether the new engine has them or not.
3. A `held` parameter, not on auto, whose value is outside the new engine's range reads at its limit and is held, with
   the engine it was last in use on; it comes back on an engine that can take it, unless it was changed meanwhile.
4. A source module (Repaint) the new engine does not have leaves the desk.
5. The toast says, in this order: what is capped and where the values wait, what the engine left keeps (its models by
   name, its adapters, a count of its changed settings), and that the new engine's own settings are back.

`heldNote()` gives the held parameter's note while the value is held.

## What is sent

The page submits the Create intent to `/api/resolve/preview`, as the existing Create page does; the server's
`resolveCreateIntent()` composes the caption and finishes the request. `createIntent()` builds it in three steps, two
of which are the real application's own code, handed in (`RealApp`):

1. **The desk in real names** (`realState`): every definition on the desk on the engine in use, every parameter that
   applies, through its wires, into five bags: `params` (`GENERATION_DEFAULTS` fields), `backendParams` (extension
   knobs), `content` (`CreateContent` fields), `body` (fields set after the rest) and `intent` (the intent's `compose`
   and `captionSource`).
2. **The real application's code**: `resolveGenerationParams()` turns `params` and `backendParams` into the engine
   fields; `createContentParams()` turns `content` into the content fields.
3. **The intent**: `{ kind: 'create', params, compose, captionSource }`, the params being those, then `body`, the raw
   caption and lyrics, `backendOverride` on an engine marked `jobOverride`, and the app's own settings that every
   Create request carries.

The rules of step 1:

- A definition not on the desk sends nothing, so the engine's defaults apply, which are its values in force; its values
  are kept. Built-in modules are always on the desk; a source module is never part of a Create request. The existing
  page sends what the user last set whatever is shown; the request comparison accepts this as one rule
  (`off-the-desk`), checked exactly by sending with everything on the desk, and question 11 asks the maintainer.
- A parameter that does not apply sends nothing.
- A value is sent unless it is untouched and equals the field's real default (`GENERATION_DEFAULTS` for `param:`, the
  backend's declared default for `ext:`). A field whose real default is unknown is sent once touched; a `body:` and an
  `intent:` field always.
- A model choice left at "" (the engine's default) is sent as the model the server has selected for its bucket
  (`desk.selection`), so a queued job keeps the models it was queued with.
- A wire changes the value in this order: `when` (else `else`, or nothing), auto (`auto`, null for nothing), then either
  `via` (a named transform, which does the whole change) or `map` (an option the map leaves out sends nothing through
  it), `invert`, `scale`, `divide`. A path writes into its field, starting from the field's real default; an entry of a
  list is written only when something sent the list.
- A packing transform runs once for its field, with the values of everything it `packs`.
- State is sent through its wire when it holds something, and through a transform even when empty.

The named transforms (`transforms.ts`): `int`, `text`, `scheduler`, `firstPath`, `stackPaths` (the adapters as the
stack, each with its phase window), `continuation`, `mm3CaptionSource` and `yue2CaptionSource`. Each also reads the
real value back. `fromReal()` reads a set of real fields back into desk values, through each wire backwards and under a
field's former names (`was` on the wire), for loading a saved profile or a song's settings.

`tools/compare-request.ts` compares this over the value space with what the existing Create page sends when its own
controls are used ([07](07-checking.md)); each difference it accepts is written down exactly and asked about in
[08](08-open-questions.md).

## Saved data

| Record | Holds | From the definition |
| --- | --- | --- |
| Revision (`revisionOf`) | One engine's settings: every parameter on that engine with `stored: 'settings'`, its stored value as set, or its default when none is (whether or not its module is on the desk), and its auto switch, and the state with `stored: 'settings'` | `stored`, `engines` |
| Revision "with content" | The above and every `stored: 'content'` parameter on that engine | `stored` |
| Draft (`draftOf`) | The `stored: 'content'` values set | `stored` |
| Workflow (`workflowOf`) | The desk: modules (never a source module), pedals and units, its tier. No values | `id`s |
| Desk autosave (`savedDeskOf`) | Everything, the engines' stashes and the held values included | |

Loading a revision (`loadRevision`) first resets every parameter a revision keeps for its engine, then applies its
values cleaned as above, so what a revision holds is decided by `stored` alone. A factory revision lists only what it
sets.

Every record carries `v`, the version of its shape. `migrate()` reads a record of an older shape by running each step
from its version up to the current one; version 0 is the concept's own saved shape, so its exported workflows and
revisions load. A record of a newer or unknown shape is kept aside unread. A rename is not a migration step: `was` reads
the old name in every version. `tools/fixtures/shipped.json` lists every key, state and id the definitions have shipped
with, and where each value is kept; the validator fails when one of them is neither in a definition (under its name or
in a `was`) nor retired in `src/definitions/retired.ts`, or when where it is kept changed without `npm run docs`
recording it. So removing, renaming or moving anything saved data holds is a decision written down.

## What each tier says about what is set

**The tile's "Also set" line** (`alsoSet`, the concept's `paintAlso`, `30-tile.js`). For a module at a tier below Lab:

```
editsAt(def, tier, E) = the parameters edited by nodes of def's face shown at tier on E, that apply; none while the tile is a strip
hidden = def's changed parameters not in editsAt
```

For the module that holds the board (`board`) the parameters are its own and those of every pedal and unit on the board, where a
pedal edits its switch at Quick, its switch and main knob at Studio, everything at Lab. If `hidden` is not empty the
tile ends with "Also set: {items}." Each item is the parameter's `short` when it has more than two letters, else its
label without a parenthesis, after the pedal's name for a pedal's; a switch is its name, or "{name} off"; anything else
"{name} {value}". From Quick the link goes to Studio when Studio edits every one of them, else to Lab; from Studio to
Lab.

**Quick's plain words.** A choice row checks the choice whose values are all in force, or none, and its label then says
the value (`between`).

**Tips.** `tipOf(node)` is the node's or its parameter's tip, then its part's `howTo`.

## Strings and translation

`stringsOf(definition)` gives every string with its key. A key is made from where the string sits and what it is bound
to, never from its place in a list, so reordering lines, options, choices, nodes or strips keeps every translation, and
no two strings share a key (`tools/test-core.ts` checks both).

| String | Key |
| --- | --- |
| A definition's words | `create.<id>.name`, `.desc`, `.quickName`, `.short`, `.tier.quick` ...; an engine line's `@<engines>` |
| A parameter's words | `create.<id>.p.<key>.label` ... (dots in the key as `-`), its held words, its named steps (`.step.<name>`) |
| An option's words | `create.<id>.p.<key>.o.<value>.label`: letters and digits kept, any other character as its hex code between two `_` (`F_23__20_major` for F# major), "" as `_` |
| A parameter line's words | the same key with `@<engines>` for an engine line, `@when-<hash of its condition>` for a word line |
| A node's words | `create.<id>.n.<part>.<what it binds>`, or its words when it binds nothing; siblings of one kind and binding add a hash of themselves; a node line adds `@<hash of its engines, tiers and condition>` |
| A strip or idle reason | `create.<id>.strip.<hash of its condition>` |

A key that follows words (an unbound note) changes when the English does; the extraction that writes `en.json` lists
the keys no definition makes any more. Slots stay in the translated text (`{engine}`, `{value:samp.guide}`).

**Messages.** Nothing the core returns is a finished English sentence. An undo name, a toast, an "Also set" line, a
plate, a load note, a part's text and a value's words are a message: an id and its slots (`derive/messages.ts`). A
definition's own words are messages keyed as above, with their English as the template; the core's own sentences are
the `EN` table, keyed `step.*`, `switch.*`, `also.*`, `load.*` and so on (`coreStrings()` lists them). A slot is text,
a number, another message, or a list joined as the locale joins one. Plurals, selects and the case of a word inside a
sentence are written in the template (`{n, plural, one {# setting} other {# settings}}`, `{label|lower}`), so a
locale's table decides its own grammar. `say(message, table, lang)` fills one; the view calls it with the locale's
table.

## What the view may decide

The page is a view over the core. A component draws what the core returns and hands back what the user did.

The view may decide: layout, styling and animation; which locale's table `say()` uses; how a part is drawn and
operated (a knob's drag, a field's caret), as long as it gives the core only what its part's binding takes; when to
fetch a server list and what to show while it loads; focus, scrolling and which panel is open.

The view may not decide: whether a value applies, is changed, or is in force (`applies`, `changed`, `valueOf`); which
nodes show at a tier and what they edit (`editsAt`, `boardNodes`); which choice of a Quick row is checked and what
pressing one sets (`choiceOf`, `applyChoice`); what an action does (`runAction`); whether a node stands back and why
(`dimOf`), or a pedal's idle reason (`idleOf`); the name of an undo step (`stepWords`); what an engine switch does and
says (`switchEngine`); what "Also set" lists (`alsoSet`) and which Lab modules a tier hides (`hiddenLabModules`); how a
number is written (`formatMsg`); what a load keeps and says (`loadRevision`, `loadWorkflow`, `loadDraft`); and what is
sent (`createIntent`). If a component needs a different answer, the difference belongs in a definition.

## The thumbnail

The Modules panel and the picker draw `thumb.file` (or its engine line's) from `src/definitions/thumbs/`. A module's file
is the whole drawing on the 176 x 112 canvas. A pedal's frame is drawn from the definition (1U or 2U from the number of
its numbers, the face from `hue`, the name bar, the LED and switch, or "Always on" when `on` is null), with the file's
middle inside. A side unit is the rack strip with the file as its object.

## Pedals and side units

Laid out from their parameters by `layOut()` in `parts.ts` ([01](01-parts-inventory.md)); `boardNodes()` in
`view.ts` gives the cells at the desk's tier, a hidden knob as a plate with its words.

## Automated tests

`tools/test-core.ts` runs the derivations on the states the concept page was put in and compares what they return with
what the page did ([07](07-checking.md)). The generic walker of the rebuilt page will, for each engine and definition
on it, put it on the desk at each tier, operate each parameter in `editsAt` the way its part's `drive` says, and check
from the definition alone: the stored value is valid, the marker is drawn, Undo's label is `stepWords`'s, a reload and a
saved revision restore it as `stored` says, an engine switch does what `switchEngine` says, the tier below lists it as
`alsoSet` says, and `createIntent` carries it.
