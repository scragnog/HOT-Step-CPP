# What the definition drives

Everything on this page is derived from the definitions by code that has no branch for any module. Most of it is
written once in `src/derive.ts`, which the validator and the tools already call and the page will call; the rest is
described here for the page to implement the same way. If a module needs one of these to behave differently, the
difference belongs in its definition (a field) or in its bespoke part, never in this code. The behaviour is the
concept's, cited by file (`C:\Users\sneth\misc\HOT-Step-UI\src\js`).

## The desk

What everything below reads (`Desk` in `derive.ts`): the engine in use `E`, the tier, the stored values (parameter key to
value), the auto switches (parameter key to on or off; absent is on), the module state, the desk facts (`DESK_FACTS`) and
what is on the desk.

## A parameter on an engine

```
resolve(p, E)  = the base fields, then the engine line that names E (shallow, per field)      derive.ts resolve
wordsOf(p, E)  = resolve(p, E), then every word line whose condition holds                     derive.ts wordsOf
has(p, E)      = E is in the parameter's engines and in its definition's
applies(p, E)  = has(p, E) and its `when` holds                                                derive.ts applies
```

`applies` is the concept's `ok(k)`. A condition on a parameter the engine does not have is false.

## Values: one rule for every path

A stored value is in the parameter's own units, and it is kept in the parameter's own domain: the union over all its
engines. Which engine is in use matters only when the value is read or sent. The concept works this way
(`cleanVals`, `10-working-state.js`, and `enterEngine`, `13-engine-state.js`), and the walks listed at the end of this
section were run on the concept's page to confirm it.

**Cleaning a stored value** happens on every path that brings values in from outside the desk's own memory: a reload,
another tab's desk, undo of a structural step (a module added or removed, a revision loaded), importing a revision,
workflow or draft file, and loading a factory record. It is the same function on every path:

| Type | Kept as | Not kept |
| --- | --- | --- |
| number | A finite number, clamped to the union of its engines' ranges (lowest minimum to highest maximum). Never rounded to a step | Anything else |
| boolean | `true` or `false` | Anything else |
| choice | One of the union of its engines' fixed options, or of `include`; for a server-listed choice, any value, until a list has loaded that lacks it | A value no list has |
| text | A string, cut to 20,000 characters, that passes its `check` if it has one | Anything else |
| auto switch | `true` or `false`, for a parameter with `auto` on some engine | Anything else |

A key that is no parameter's is read as the parameter that lists it in `was`; an option value likewise. Anything not kept
is counted, and an import says so in its toast ("2 settings could not be used").

**Setting a value** from a control cleans it against the engine in use: a typed number out of range is clamped and said
once ("Steps: 1 to 64. Set to 64."). Rounding to the step happens only here, where the control's own keys and drags make
steps. Nothing else rounds what the user set.

**Reading a value** on engine `E` (`valueOf` in `derive.ts`) gives the stored value when `E` can take it, else the
nearest `E` can take for a number (its limit), else the default. Reading never writes. So a value only one engine can take
waits, unchanged, for an engine that can.

**An engine switch** from `A` to `B` is one undo step (`enterEngine`):

1. Every stored value whose parameter is `owner: 'engine'` is put aside for `A`, with `A`'s engine-owned state; what was
   put aside for `B` comes back. An engine reached for the first time starts on its own defaults.
2. Song values stay where they are, whether `B` has them or not.
3. A `held` parameter, not on auto, whose stored value is outside `B`'s range reads at `B`'s limit, and the desk records
   where the value was last in use: `A` if `A` has the parameter, else where it was recorded before (300 s set on
   ACE-Step, through MiniMax-Music3, which has no length, to HeartMuLa: "returns on ACE-Step 1.5"). Its control says the
   held note, and the switch's toast the held toast. Changing it on `B` stores the new value and ends the hold; an engine
   that can take it reads it again, and the hold ends.
4. The toast says what `A` keeps, what came back, and what is held.

The desk autosave keeps the records of where held values came from, and a reload checks them (a number, a known engine)
as it checks values.

**Undo** of a value step puts the value back; undo of a structural step restores the desk as it was, cleaned as above.

The walks on the concept page, with what it did and what these rules give:

| Walk | Concept | Here |
| --- | --- | --- |
| BPM 82 set on ACE-Step, switch to Stable Audio 3 (no BPM), reload, back | 82 kept throughout | Same: kept in its domain, absent on Stable Audio 3 |
| 33 s set on Stable Audio 3 (step 1), switch to ACE-Step (step 5), reload, back | 33 throughout | Same: never rounded |
| 300 s on ACE-Step, to HeartMuLa (up to 240), to MiniMax-Music3, back to HeartMuLa, reload | Held, "returns on ACE-Step 1.5" throughout | Same: the hold's origin is where the value was last in use |
| 300 s on auto, to HeartMuLa | Not held | Same: auto is never held |
| 9 takes on ACE-Step, to Stable Audio 3 | "Your 9 takes return on ACE-Step 1.5" | Same |
| A saved desk with length 9999, steps 12.5, StableStep 99 %, key "H major" | 600, 12.5, 60, key dropped | Same |

## The changed marker

```
changed(p, E) = applies(p, E) and (auto ? not on auto : stored and valueOf(p, E) differs from the default)   derive.ts changed
```

Numbers compare after cleaning, so 9 and 9.0 are equal. Every control part is wrapped by one component that adds the
marker after the control's words, or after its box when it has none; a part never draws the marker itself. For an XY pad
each axis carries its own parameter's marker. The marker is drawn at every tier.

## How a number is written

`formatValue` in `derive.ts`, the concept's `fmt` (`03-controls.js`):

1. A `format.words` entry for the exact value wins ("off", "whole history"); a switch is "on" or "off" unless words say
   otherwise; a choice is its option's label.
2. A `format.template` fills `{v}` with the number and `{s}` with the number divided by `perSecond`, as seconds.
3. Seconds (`unit: 's'`, and `{s}`) read as m:ss from a minute ("5:00", "1:20"); below a minute with one decimal under 10
   s when the step is below 1, else none ("9.5 s", "13 s", "10 s").
4. Otherwise two decimals when the step is below 0.1, one when below 1, none from 1 (`format.decimals` overrides),
   a true minus sign, then the unit after a space, or right after the number for `%`.

The same function writes the read-out, the tip, the undo step, the held note and the "Also set" line, so they never
disagree.

## Words that name values

`fill` in `derive.ts` fills the slots of any text ([02](02-definition-format.md), "Words that name values"): `{engine}`
with the engine's name, `{value:<key>}` with the parameter's value as above, `{count:<name>}` with a state's or desk
fact's count, `{file:<state>.<n>|<words>}` with a loaded file's name or the words, `{say:<fact>}` with a fact's words,
and, in a parameter's own words, `{min}`, `{max}` and `{default}`. So "MiniMax-Music3 uses its own guidance", "up to
4:00" and "tape-lofi-v2 strength" are one string each, true on every engine and in every state.

## Undo step names

Every change is one undo step, named from the definition (`14-value-steps.js`). Each derived name is one whole sentence
with named slots, so a translation can put the slots where its language wants them:

| Step | Sentence | Slots |
| --- | --- | --- |
| A value | `{name} {from} → {to}` ("Guidance 9.0 → 20.0", "Chain on → bypassed") | `name`: the parameter's `undoName`, else its `label` (word lines applied), filled; `from`, `to`: the values as written everywhere |
| A value whose name is shared | `{owner} {name} {from} → {to}` ("Repetition guard mode presence → DRY") | `owner`: the definition's name. Used when another parameter on the engine in use has the same label |
| A pedal's or unit's switch | `{owner} {from} → {to}` ("StableStep off → on") | Decided by the key: the parameter is the definition's `on`, not by its words |
| Text | `Edit {name}` ("Edit the lyrics") | One step per visit to the field |
| A Quick row | `{row}: {from} → {to}` ("Quality: Standard → High") | The row's label and the choices' words |
| An XY pad | One step naming both values | |
| State | The StateDef's step ("Load {file} as the timbre reference", "Keep take {n} for {section}", "Seed random → 1834920213") | Its slots; a file has `load` and `clear` |

A drag, a held key or a run of scroll steps on one control is one step. Presses on one control within its `run`
milliseconds are one step (taps on TAP). A part never names a step: it asks for a state's step by name, and a value's
step is made by `set`.

## What is sent

The request is derived from the definitions alone, in three steps, two of which are the real application's own code:

1. **The desk in real names** (`realState` in `derive.ts`): for every definition on the desk on the engine in use, every
   parameter that applies, through its wires. The result is four bags: `params` (`GENERATION_DEFAULTS` fields),
   `backendParams` (extension knobs), `content` (`CreateContent` fields) and `body` (fields set after the rest).
2. **The real application's resolver**: `resolveGenerationParams()` (`server/src/services/generation/defaults.ts`) turns
   `params` and `backendParams` into the engine fields, with every conditional emission rule (`cfgCutoffRatio` only below
   1, the StableStep fields only while it is on, the scaler routing of DCW). `createContentParams()`
   (`ui/src/components/create/createContent.ts`) turns `content` into the content fields, with the caption's trigger word
   and Beat I/O words. The rebuild imports both rather than restating their rules, as the existing UI already imports
   `GENERATION_DEFAULTS`.
3. **The request**: those, then `body`, then `backendOverride` on an engine marked `jobOverride` in `engines.ts`, then the
   app's own settings that every Create request carries (`source`, `coResident`, `cacheLmCodes` and the `parallel*`
   switches, from the Settings panel, not the desk).

The rules of step 1:

- A definition not on the desk sends nothing; its values are kept and apply again when it is added back. Built-in modules
  are always on the desk; a Lab module on the desk applies at every tier. A source module (Repaint) is never part of a
  Create request.
- A parameter that does not apply on the engine sends nothing.
- A value is sent unless it is untouched (not in the stored values or auto switches) and equals the field's real default:
  `GENERATION_DEFAULTS` for `param:`, the backend's declared default for `ext:`. A field whose real default is unknown,
  one a backend reads without declaring it, is sent once touched, and its wire's note says so. A `body:` field is always
  sent. So an untouched desk sends what the existing UI sends, and a definition whose default differs from the real one
  still gets its default applied: the validator requires a note on every such wire.
- A wire changes the value on the way in this order: `when` (else `else`, or nothing), auto (`auto`, null for nothing),
  `via` (a named transform), `map` (an option the map leaves out sends nothing through it), `invert`, `scale`, `divide`.
  A path writes into its field, starting from the field's real default (`adapterGroupScales.self_attn`); an entry of a
  list is written only when something sent the list (a slot's strength with no adapter loaded sends nothing).
- A packing transform runs once for its field, with the values of everything it `packs`: the schedule and its shape
  become `'beta:0.60:0.70'`; the recording and its span become `yue2Continuation`.
- State is sent through its wire when it holds something; through a transform even when empty, so the transform says
  what empty is (no adapter is `''`).

The named transforms (`transforms.ts`) are the cases fields cannot say: `int` (a seed typed as text, sent as a number),
`text` (YuE2's composer CFG, a number sent as text), `scheduler` and `continuation` (packing), `firstPath` and
`stackPaths` (the adapter slots as Simple mode's single adapter and Advanced mode's stack). Each also reads the real
value back, for loading a saved profile or a song's settings ([08](08-open-questions.md) question 4).

`tools/compare-request.ts` builds the request this way for every engine at default values and compares it field by field
with what the existing frontend sends at its defaults (`getGlobalParams()` and `createContentParams()`, joined as
`App.tsx` joins them). The one difference is on HeartMuLa: the existing UI freezes the scalar decoder the server has
selected into the request, and no definition carries it ([08](08-open-questions.md) question 9). `npm run check` fails on
any difference that `tools/fixtures/request-differences.json` does not list with its reason.

## Saved data

| Record | Holds | From the definition |
| --- | --- | --- |
| Revision | One engine's settings: every parameter on that engine with `stored: 'settings'`, its auto switch, and state with `stored: 'settings'` (the adapter slots) | `stored`, `engines` |
| Revision "with content" | The above plus every parameter with `stored: 'content'` on that engine (lyrics, tempo, song info, the prompt) | `stored` |
| Draft | `stored: 'content'` only | `stored` |
| Workflow | The desk: modules, pedals and units in their order, and its tier. No values | `id`s |
| Desk autosave | Everything, `stored: 'none'` included: values, auto switches, state, what each engine keeps aside, held values | |

A revision the user saves holds every value it covers, the defaults included, so a default changed in a later definition
does not change what a saved revision sounds like. A factory revision (`factory.ts`) lists only what it sets, because it
ships with the definitions it was written for. Loading a revision first resets every parameter a revision keeps for its
engine, then applies its values cleaned as above, so what a revision holds is decided by `stored` alone, whether or not
its module is on the desk.

Every record carries `v: 1`. A record with another `v` is kept aside unread and said once, as the concept does with a
desk it cannot read (`deskUnreadable`); a later shape adds its migration beside the reader.

## What each tier says about what is set

**The tile's "Also set" line.** For a module at a tier below Lab:

```
editsAt(def, tier, E) = the parameters edited by nodes of def's face shown at tier on E, that apply   derive.ts editsAt
hidden = def's parameters with changed(p, E) and not in editsAt
```

Only `param` and `params` count, never `reads`: the Sampler's scope reads the CFG interval at Studio and edits it only at
Lab, so a changed interval is on the Studio tile's foot. For the Pedalboard, its parameters plus those of every pedal and
unit on the board, where a pedal at Studio edits its `main` knob and switch. If `hidden` is not empty the tile ends with
"Also set: {items}." Each item is `{name} {value}`, the name being the parameter's `short` or its `label`, prefixed with
the pedal's name when it is a pedal's; a switch is its name, or "{name} off". The link goes to the lowest tier at which
the first of them is edited, and rings them there.

**Lab modules on a lower tier.** A note above the desk names the Lab modules on the desk that hold changed values.

**Quick's plain words.** A choice row checks the choice whose values are all in force, or none, with the marker and the
value in its label (`between`). The row's words and the detailed values never disagree because the row has no value of
its own.

**The picker.** `tierSummary` says what each tier adds; `name`, `desc` and `tierSummary` are read per engine.

## Tips

A control's tip is built from the definition: its full label (the `label` even where the face prints `short` or the
node's own `label`), its `tip`, then how to work the part, said once a session (the part's own sentence, such as the
knob's "Drag up or down (hold Shift for fine steps), scroll, or use the arrow keys. Double-click to reset."). A value in a
tip is written as above. An option's `tip` is its tooltip in a dropdown or chip row.

## Strings and translation

A definition holds English. The real app uses i18next with flat dotted keys in `ui/src/i18n/locales/en.json` and
`t('key', 'English default')`. Keys are derived from where a string sits, so a definition never names one:

| String | Key |
| --- | --- |
| A definition's `name`, `quickName`, `desc`, `short` | `create.<id>.name`, `.quickName`, `.desc`, `.short` |
| `tierSummary` | `create.<id>.tier.quick`, `.studio`, `.lab` |
| A parameter's `label`, `short`, `undoName`, `tip`, `placeholder` | `create.<id>.p.<key>.label`, ... (dots in the key written as `-`: `create.sampler.p.samp-steps.label`; a key has no `-` of its own, so two keys never share one) |
| An option's words | `create.<id>.p.<key>.o.<value>.label`, ..., the value written as letters and digits joined by `-`, and `""` as `none` |
| A line's words | the same key with `@<engines>` or `@<n>` (the line's place): `create.sampler.p.samp-steps.label@minimax-m3` |
| Words in a node, a strip, an idle reason, a check message, a held note, format words and templates | `create.<id>.t.<slug>`, the slug being the English lower-cased to letters and digits joined by `-`, cut to 48 characters, with a 4-character hash when two strings share a slug |

Slots stay in the translated text as they are (`{engine}`, `{value:samp.guide}`), and the derived sentences of the undo
steps and the "Also set" line are keyed once each in the shell, with their slots, so no string is ever made by joining
two translated pieces. Node words are keyed by their text, so reordering a face keeps its translations and changing the
English makes a new key; the extraction that writes the keys into `en.json` also lists the keys no definition makes any
more, for removal. The app looks a string up with the English as the default, so a missing translation shows English.

Words that are not in definitions (engine names, "Also set", "Edit in Lab", toasts) belong to the shell and are keyed by
hand, as now.

## The thumbnail

The Modules panel and the picker draw `thumb.file` (or the file its `per` line gives for the engine) from
`src/definitions/thumbs/`. For a module the file is the whole drawing on the 176 x 112 canvas. For a pedal the frame is
drawn from the definition (1U or 2U from the number of its numbers, the face gradient from `hue`, four screws, the name
bar sized from `name`, the LED and foot switch, or the "Always on" plate when `on` is null) and the file's middle drawn
inside it; in the panel the pedal is mounted on a piece of board with a cable each side. A side unit is the rack strip
with the file as its object. Every class in a drawing is prefixed `t-` on the way in (`thSvg`, `24-thumbnails.js`), and
gradient ids are made unique per drawing.

## Pedals and side units

Laid out from their parameters as [01-parts-inventory.md](01-parts-inventory.md) describes, by `layOut()` in `parts.ts`:
the definition supplies the order (its `params`), the switch (`on`), the Studio knob (`main`), what stands beside a
switch (`beside`) and the extra nodes (`face`); the blanking plate's words come from the hidden knobs' conditions. The
count on the board ("10 pedals · 9 on · 1 needs setting up") reads `on` and `idle`.

## Automated tests

The concept's behaviour suite (`tools/behaviour/run.mjs`) drives the page through hand-written scenarios. With
definitions, most of those scenarios can be generated. A generic walker, for each engine, for each definition on it:

1. Puts it on the desk (a pedal or unit on the board), at each tier.
2. For each parameter in `editsAt` at that tier, finds its control by `data-ctl="<key>"` (the wrapper adds it to every
   control part) and changes it the way its part's `drive` says (`parts.ts`): `keys` (focus, press End, or Home at the
   maximum), `click`, `choose` (the next option or choice), `type` (a valid string: the `example`, or "x", then leave the
   field), `file` (drop a test file).
3. Checks, from the definition alone:
   - the stored value changed and is valid (cleaning leaves it unchanged);
   - the control carries the changed marker;
   - Undo's label equals the derived step name, and Undo puts back the default and removes the marker; Redo restores it;
   - a reload restores it;
   - saving a revision and loading it on a cleared desk restores it when `stored` is `settings`, and not otherwise;
   - switching to another engine and back restores it when `owner` is `engine`; when `owner` is `song`, the value is
     still in force on the other engine if that engine has the parameter;
   - one tier down, if it is not in `editsAt` there, the tile's "Also set" line names it with its value;
   - the request (step 1 of "What is sent") carries the new value through the parameter's wire.
4. For each parameter with `when`, sets the condition false and checks the control is hidden (or its room kept on a
   pedal), the value kept, not marked changed and not sent.
5. For each `choiceRow`, presses each choice and checks the detailed values; sets a value between choices at Lab and
   checks Quick's row checks none and its label contains the `between` text.
6. For each `held` parameter, the walks of "Values: one rule for every path".

Bespoke parts are covered by this walk for everything that is the parameter's (marker, undo, saving, request); what they
do beyond that keeps hand-written scenarios, which is where the suite's remaining effort belongs.
