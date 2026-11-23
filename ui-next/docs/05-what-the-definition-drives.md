# What the definition drives

Everything on this page is derived from the definitions by code that has no branch for any module. If a module needs
one of these to behave differently, the difference belongs in its definition (a field) or in its bespoke part, never in
this code. Function names are suggestions; the behaviour is what the concept does, cited by file
(`C:\Users\sneth\misc\HOT-Step-UI\src\js`).

Notation: `E` is the engine in use, `S.vals` the stored values (a map from parameter key to value, with `<key>.auto` for
auto switches), `S.state` the module state.

## Resolving a parameter on an engine

```
resolve(p, E) = { ...p, ...(p.per?.[E] ?? {}) }        // shallow, per field
has(p, E)     = E in (p.engines ?? definition.engines)
applies(p, E) = has(p, E) && (resolve(p, E).when is absent or holds)
```

`applies` is the concept's `ok(k)`. A condition on a parameter the engine does not have is false. A condition on state
reads `S.state[key]` (`set`: not empty; `atLeast`: a count) or one of the computed `DESK_STATES`.

## Defaults

```
default(p, E) = resolve(p, E).default
value(p, E)   = S.vals[p.key] if it is valid for E (below), else default(p, E)
isAuto(p, E)  = resolve(p, E).auto is an object && S.vals[p.key + '.auto'] !== false
```

A parameter with `auto` starts automatic. What an automatic parameter sends is its wire's `auto` value (bpm 0, duration
-1, keyScale ''); what it shows is `auto.says`, or the default as an estimate ("≈ 3.0").

Reset (double-click, a Reset action) deletes `S.vals[key]` and `S.vals[key + '.auto']`. It never writes the default:
an untouched parameter is absent from `S.vals`, so a later change to the definition's default reaches it.

## Validation and clamping

One function, `clean(p, E, raw)`, used on every path. It returns the value to keep, or "absent", and whether it had to
change it.

| Type | Rule |
| --- | --- |
| number | Must be finite (a string is parsed). Rounded to the step from the minimum: `min + round((v - min) / step) * step`, to 4 decimals. Clamped to `[min, max]` of `resolve(p, E).range`. Then `below`/`above`: kept a step below or above the partner's current value. |
| boolean | `true` or `false`; anything else is absent. |
| choice | Must be one of the option values. For a server-listed choice (`OptionsFrom`), the `include` values are always valid; any other value is kept as it is until the server's list has loaded, then dropped to the default if the list does not have it (a model file that was deleted). |
| text | A string, trimmed of a trailing newline where the part is one line. With a `check`, an invalid string is refused: the field keeps what was typed and says why. With `check.keepInvalid`, the field keeps the invalid text visibly, applies nothing, and says "Still using <keepInvalid>" (`12-values.js`, `keptMsg`). |

Where it runs:

- **On input**: every control calls `set(key, raw)`, which cleans against the engine in use. A typed number out of range is
  clamped and said once: "Steps: 1 to 64. Set to 64." (`rangeSaid`, `03-controls.js`).
- **On import** of a revision or workflow file: each value is cleaned against the engine the revision is for. Unknown keys
  are dropped. Anything changed or dropped is counted and said in the import toast.
- **On restore** (reload, another tab's desk, undo of a structural step): the desk's values are cleaned against the
  desk's engine; each stashed engine's values against that engine. A value that cannot be cleaned is dropped, never
  carried as is (the concept's `cleanVals`, `10-working-state.js`).
- **On engine switch**: see "Engine ownership". Song values are not re-cleaned destructively: a value valid on the old
  engine and out of range on the new one is a held value, which only a `held` parameter can have (the validator refuses
  a song number whose range differs per engine without `held`).

## The changed marker

```
changed(p, E) = applies(p, E) && (auto ? !isAuto(p, E) : p.key in S.vals && value(p, E) !== default(p, E))
```

Numbers compare after cleaning, so 9 and 9.0 are equal. Every control part is wrapped by one component that adds the
marker after the control's words, or after its box when it has none; a part never draws the marker itself. That is what
makes "a changed marker missing on some control types" impossible: there is no per-part code to forget. For an XY pad
each axis carries its own parameter's marker. The marker is drawn at every tier.

## Undo step names

Every change to a parameter is one undo step, named from the definition (`14-value-steps.js`):

1. The name: `undoName`, else `label` (with `labelWhen` applied, for the engine in use).
2. If that name is `On`, the step is named after the pedal or unit ("StableStep off → on").
3. If another parameter anywhere has the same label (case-insensitive), the definition's name goes first and the label's
   first word drops its capital unless it is an initialism: "Repetition guard mode".
4. The values, as the parameter is written everywhere (its `format`, its option `label`, booleans as `on` and `off` unless
   `format.words` says otherwise): "Guidance 9.0 → 20.0", "Chain on → bypassed".
5. Text: one step per visit to the field, "Edit the lyrics", "Edit the LM negative prompt".
6. A Quick choice row names the row and the words: "Quality: Standard → High".
7. Two numbers moved together on one XY pad are one step naming both.

A drag, a held key or a run of scroll steps on one control is one step. Presses on one control within its `run`
milliseconds are one step (taps on TAP). State changes are steps named by their StateDef: a file is "Load {file} as the
timbre reference", clearing it "Clear the timbre reference", a span "<noun> 0:12 → 0:20".

## What saving stores

| What | Holds | From the definition |
| --- | --- | --- |
| Revision | One engine's settings: every parameter on that engine with `stored: 'settings'`, its `.auto` switch, and state with `stored: 'settings'` (the seed, the adapter slots) | `stored`, `engines` |
| Revision "with content" | The above plus every parameter and state with `stored: 'content'` (lyrics, tempo, song info, the reference files) | `stored` |
| Draft | `stored: 'content'` only | `stored` |
| Workflow | The desk: modules, pedals and units in their order, and its tier. No values | `id`s |
| Desk autosave | Everything, `stored: 'none'` included | |

A revision stores only values that differ from their defaults (absent means default), so loading one first clears every
`stored: 'settings'` parameter of its engine and then applies its values. This is what makes "a revision that did not
store a module's settings" impossible: what a revision holds is the set of parameters that say `stored: 'settings'`,
whether or not their module is on the desk, with no list kept anywhere else.

## Engine ownership

On switching from engine A to engine B (`13-engine-state.js`, `enterEngine`), in one undo step:

1. For every key in `S.vals` whose parameter has `owner: 'engine'` (resolved), move it to `stash[A]`; move state with
   `owner: 'engine'` there too.
2. Bring back `stash[B]` if there is one. An engine never used starts on its own defaults.
3. For every `held` parameter: if the value (or the held value) fits B's range, use it and drop the hold; if not, show B's
   limit and keep `hold[key] = { value, from: A }`. Changing it on B drops the hold. The control says it while it holds:
   "Your 9 takes return on ACE-Step 1.5".
4. Song values that B does not have simply stay in `S.vals`; they are absent on B and come back on an engine that has them.
5. The toast says what A kept, what came back, and what is held.

Nothing about a module is consulted except `owner` and `held`. That is what makes "a module's settings travelling between
engines when they should stay" impossible.

## What each tier says about what is set

**The tile's "Also set" line.** For a module at a tier below Lab:

```
shownAt(def, tier, E) = keys bound by nodes in def.face visible at tier on E (tiers, engines and when all pass)
hidden = def's parameters with changed(p, E) and not in shownAt
```

For the Pedalboard, its parameters plus those of every pedal and unit on the board. If `hidden` is not empty the tile ends
with: "Also set: Key D major · StableStep refine 45%." Each item is the parameter's `short` (when longer than two
characters) or its `label` without a trailing parenthesis, prefixed with the pedal's name when it is a pedal's, then the
value written as everywhere else; a switch is its name, or "<name> off". The link goes to the lowest tier at which all of
them are shown and rings them there. The concept found hidden values by searching the drawn tile for `data-ctl`
attributes (`30-tile.js`); here the set is known from the definition, so a face that forgets to draw a control does not
silently hide its value.

**Lab modules on a lower tier.** A note above the desk names the Lab modules on the desk that hold changed values.

**Quick's plain words.** A choice row checks the choice equal to the value, or none, with the marker and the real value
in its label (`between`). The row's words and the detailed value never disagree because the row has no value of its own.

**The picker.** `tierSummary` says what each tier adds; `desc` and `name` are read per engine.

## Tips

A control's tip is built from the definition: its full label (the `label` even where the face prints `short` or the
node's own `label`), its `tip`, then how to work the part, said once a session (the part's own sentence, such as the knob's
"Drag up or down (hold Shift for fine steps), scroll, or use the arrow keys. Double-click to reset."). A value in a tip is
written with the parameter's format. An option's `tip` is its tooltip in a dropdown or chip row.

## Strings and translation

A definition holds English. The real app uses i18next with flat dotted keys in `ui/src/i18n/locales/en.json` and
`t('key', 'English default')`. Keys are derived from where a string sits, so a definition never names one:

| String | Key |
| --- | --- |
| A definition's `name`, `quickName`, `desc`, `short` | `create.<id>.name`, `.quickName`, `.desc`, `.short` |
| `tierSummary` | `create.<id>.tier.quick`, `.studio`, `.lab` |
| A parameter's `label`, `short`, `undoName`, `tip`, `placeholder` | `create.<id>.p.<key>.label`, ... (dots in the key written as `-`: `create.sampler.p.samp-steps.label`) |
| An option's `label`, `short`, `tip`, `lcd`, `detail` | `create.<id>.p.<key>.o.<value>.label`, ... |
| A per-engine override | the same key with `@<engine>`: `create.sampler.p.samp-steps.label@minimax-m3` |
| Words in a node (`label`, `tip`, `text`), a strip, an idle reason, a check message, format words and templates | `create.<id>.t.<slug>`, where the slug is the English lower-cased to letters and digits joined by `-`, cut to 48 characters, with a 4-character hash when two strings share a slug |

Node words are keyed by their text, not their position, so reordering a face keeps its translations, and changing the
English makes a new key (so a stale translation is never shown for new words). The app looks a string up with the English
as the default, so a missing translation shows English, as the real app's other locales already do. An extraction script
walks `DEFINITIONS` and writes the keys and English into `en.json`'s `translation` root (the repository's
`ui/add_strings.mjs` takes exactly that shape); the existing `generate_translations.mjs` then fills the other locales.

Words that are not in definitions (engine names, "Also set", "Edit in Lab", toasts) belong to the shell and are keyed by
hand, as now.

## The thumbnail

The Modules panel and the picker draw `thumb.file` (or `thumb.per[E]`) from `src/definitions/thumbs/`. For a module the
file is the whole drawing on the 176 x 112 canvas. For a pedal the frame is drawn from the definition (1U or 2U from
`thumbWidth`, the face gradient from `hue`, four screws, the name bar sized from `name`, the LED and foot switch, or the
"Always on" plate when `on` is null) and the file's middle drawn inside it; in the panel the pedal is mounted on a piece
of board with a cable each side. A side unit is the rack strip with the file as its object. Every class in a drawing is
prefixed `t-` on the way in (`thSvg`, `24-thumbnails.js`), and gradient ids are made unique per drawing, as the concept
does with its counter.

## Pedals and side units

Laid out from their parameters as described in [01-parts-inventory.md](01-parts-inventory.md): the definition supplies the
order (its `params`), the switch (`on`), the Studio knob (`main`), what stands beside a switch (`beside`), what a hidden
knob's plate says (`blank`), and the extra nodes (`face`). The count on the board ("10 pedals · 9 on · 1 needs setting
up") reads `on` and `idle`.

## Automated tests

The concept's behaviour suite (`tools/behaviour/run.mjs`) drives the page through hand-written scenarios. With
definitions, most of those scenarios can be generated. A generic walker, for each engine, for each definition on it:

1. Puts it on the desk (a pedal or unit on the board), at each tier.
2. For each parameter that applies on that engine and is shown at that tier (`shownAt`), finds its control by
   `data-ctl="<key>"` (every control part must carry it; the wrapper adds it) and changes it with the driver for its part:

   | Parts | Driver |
   | --- | --- |
   | knob, fader, ruler, numberBox, lcdNumber, token, lookback, powerChip, xyPad, takesReels, timbreBlend | focus, press End (or Home when the value is at the maximum) |
   | ledButton, pill, flag, lcdButton, recButton, powerButton, lockButton | click |
   | pick, chips, segmented, termChips, lcdChips, choiceRow, keyWheel | choose the next option (or next choice) |
   | textField, textArea, lyricSheet, promptBox | type a valid string (the `example`, or "x") and leave the field |

3. Checks, from the definition alone:
   - the stored value changed and is valid (clean leaves it unchanged);
   - the control carries the changed marker;
   - Undo's label equals the derived step name, and Undo puts back the default and removes the marker; Redo restores it;
   - a reload restores it;
   - saving a revision and loading it on a cleared desk restores it when `stored` is `settings`, and not otherwise;
   - switching to another engine and back restores it when `owner` is `engine`; when `owner` is `song`, the value is
     still in force on the other engine if that engine has the parameter;
   - one tier down, if it is not shown there, the tile's "Also set" line names it with its value.
4. For each parameter with `when`, sets the condition false and checks the control is hidden (or its room kept on a
   pedal) and the value kept, and that it is not marked changed.
5. For each `choiceRow`, presses each choice and checks the detailed value; sets a value between choices at Lab and checks
   Quick's row checks none and its label contains the `between` text.
6. For each `held` parameter, sets a value only one engine can take, switches, and checks the held note and the return.

Bespoke parts are covered by this walk for everything that is the parameter's (marker, undo, saving); what they do
beyond that keeps hand-written scenarios, which is where the suite's remaining effort belongs.
