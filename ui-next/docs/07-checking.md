# Checking that the format is sound

The checks, run from `ui-next`. They need Node 22.18 or later (the tools run the TypeScript sources directly) and
`npm install` once (TypeScript is the only dependency).

```
npm run check      the type check, the validator, the request comparison and the core's tests; all must pass
npm test           the validator's own test: planted mistakes it must catch (about ten minutes)
npm run docs       rewrites the generated parts of these pages and the list of shipped keys
npm run fixtures   reads what the real application declares into tools/fixtures/real-app.json
npm run concept    collects what the concept page does into tools/fixtures/concept.json (needs Playwright)
```

## The type check

`tsconfig.json` checks `src` and `tools` with `strict` and `exactOptionalPropertyTypes`. Every definition file ends in
`satisfies ModuleDef` (or `PedalDef`, `UnitDef`), so a misspelt field, a part not in the inventory, an option a part does
not take, an engine, unit, check, option source, action or transform that is not listed, a value of the wrong type, or
a word line that changes anything but words fails here, at the line in the definition.

## The validator

`tools/validate-definitions.ts` loads every definition, checks its JSON form against `types.ts` with the TypeScript
compiler, then what the types cannot see. Each message names the file, the parameter or node, the engine and tier, and
what to change. `--json <file>` runs every check on a definition that arrives as JSON, as if it were in the index (it
takes the place of the listed definition with its id).

What it checks:

- every definition file is listed in `index.ts`, and every definition is plain JSON data of the format's shape;
- ids, parameter keys and state keys unique; former keys (`was`) not in use; two of a kind not at one `at`;
- every condition: real names, values the parameter can hold, a condition that can hold, no circles;
- every parameter on every engine: a range with a positive step and a default on its grid, options without repeats,
  formats and words that fit, a text default and example that pass the parameter's own check, a range's two ends with
  the lower end's default below the upper's and neither below the other, `held` exactly where a song number's range
  differs per engine, an engine line whose `when` keeps the names the parameter's own condition tests, two options
  never claiming one former value;
- every wire: a channel; a `param:` field that `GENERATION_DEFAULTS` has, of the type of its real default, and that the
  engine's code reads; an `ext:` key the backend declares, or reads with a note; a `content:` or `intent:` field the
  real app has; an option value the real field takes, where the real app spells its values (`values` in the fixture,
  read from `generationControls.ts`, `ui/src/types.ts` and the guards a backend throws from); a range, options or a
  default that differ from the server's or the existing UI's only with `differs` naming the check and a note saying why
  (a note alone waives nothing); a default that is the real one whether or not the parameter applies; `wire.auto` of
  the field's type; `via` alone (a transform does the whole change); a mapped choice whose every option is mapped, or
  written by another parameter while it is chosen;
- every state: its words, its named steps and what they may change, its wire;
- every face node, on every engine and tier it can show at: a part from the inventory, bindings of the types its part
  edits, `reads` that are parameters, state or desk facts and that match the part's `reads` signature, the named steps
  its part takes declared by its state or parameter, tiers inside its parent's, conditions and dims that can hold,
  actions on real things (a tier action rings what that tier shows), Quick choices on every engine the row shows on and
  setting the row's own definition's values, words whose `{value:...}` the engine has, the same control not twice at
  one tier;
- every strip and idle reason; every pedal's switch and main knob, a Studio face for a pedal without a main knob, a
  face for every side unit, each pedal's hue at least 15 degrees from its neighbours' in the chain;
- two parameters of one definition with one name; what Quick or Studio edits, Lab edits;
- every module parameter edited by a control on every engine it is on; every thumbnail file exists and draws;
- on every engine, no two parameters write one real field at once, and at defaults every value sent that differs from
  the real default is waived with `differs: ['default']`;
- the factory records as a loaded file is read: values that cleaning keeps unchanged, parameters that apply, state of
  its kind;
- `tools/fixtures/shipped.json`: every key, state and id ever shipped is still a definition's, or named in a `was`, or
  retired in `src/definitions/retired.ts`, and is kept where it was (owner and stored) unless `npm run docs` recorded
  the change;
- the generated parts of these pages are up to date.

Notes, printed with `--notes`, are worth a look and not wrong: a part, option, action, desk fact or transform no
definition uses, a thumbnail file no definition draws, a default off the step grid that the server declares that way.

## The request comparison

`tools/compare-request.ts` builds the Create request from the definitions and the one the existing Create page sends
for the same intent, over the value space of every engine, and compares them field by field: the resolved request, and
the intent's `compose` and `captionSource`. Both sides end in the server's own `resolveCreateIntent()`, with the tracks a
caption source picks from.

The definitions' side is `createIntent()` ([05](05-what-the-definition-drives.md), "What is sent"). The existing side
runs the existing code (`getGlobalParams()`, `createContentParams()`, `resolveGenerationParams()`) through
`tools/real-app/`, which lets those files load in Node without the frontend's dependencies; `tools/real-app/app.ts`
restates the two things the existing page does in React components (CreatePanel's compose block, App's merge) and fails
when the lines it restates change. The stand-in server reports real HeartMuLa model names. How the existing page holds
each intent is `tools/existing-ui.ts`, written from the existing UI's controls: a parameter whose wire is a plain name
is held as the wire says, and every other one has a line there, so a wire and its holding are two statements of one
fact. The holdings are set through the store's own setters.

The states, per engine: every parameter at its minimum, its maximum and a step off its default, each option, each
switch both ways, auto on and off, text set and empty, each with its condition made to hold; with each condition one of
its wires or holdings sends under, alone and with every switch of its definition on; every parameter with a condition
set while the condition does not hold; every module, pedal and unit taken off the desk with its values changed; adapter
slots in Simple and Advanced with a phase window; a kept seed; every file and span; the caption source with its tracks;
and 300 random combinations from a fixed seed.

A difference is a defect in a definition, fixed there, or a class listed in `tools/fixtures/request-differences.json`
(engines, fields and the kind of difference) with why it is accepted. A difference is classed "existing sends a value
that does not apply" when it goes away once the existing side holds only the values that apply. A listed class that no
longer occurs fails the run. `--print` prints every class, `--show <engine> <words>` both sides of one state, `--quick`
leaves out the random states.

```
> npm run request
request over the value space, 5 engines:
  ace: 794 states, ... fields compared, ... differences
  ...
```

## The core's tests

`tools/test-core.ts` runs the derivations on the states the concept page was put in and compares what they return with
what the page did (`tools/fixtures/concept.json`, written by `npm run concept`, which runs the page's own functions in
Chromium): undo step names, what each tile shows at each tier, the "Also set" line and its link, engine switches and
their toasts, what a revision stores, and what cleaning keeps. A value the definitions cannot say (a server-listed choice
whose names the server gives) is left out of a case. A disagreement is a defect, fixed, or listed in
`tools/fixtures/concept-differences.json` with the reason the page is not followed (its own fault, or a decision of the
rebuild).

It also checks, without the page: the parts' contract (`bindPart` gives setters for exactly what a node binds, names
every step, refuses a value or step the definition does not give it, and makes a pad's drag one step), the translation
keys (none shared, none moved by reordering every list), reading the real fields back from what each value sends, and
saved records of older shapes (the concept's own revision, a desk under former ids and keys, a newer shape kept aside).

## The validator's own test

`tools/test-validator.ts` makes one plausible mistake at a time on a fresh copy of `ui-next` and runs the type check and
the validator on it, and for a mistake marked so the request comparison or the core's tests. It holds the 40 mistakes
of the format's first review, those found since, and the 30 of the second review with the five wrong sends it found.
Every one must be caught, except those this format made impossible or that no check can see without someone reading
them (a label wrong for its engine, a strip action whose words do not fit what it sets); those must pass, and the list
says why. Add a mistake when one gets through.

## The fixtures

`tools/fixtures/real-app.json` is what the real application declares, read from its source by `tools/make-fixtures.ts`
(from the repository `ui-next` sits in, or `HOTSTEP_APP`): `GENERATION_DEFAULTS`, the Create form's content and fields,
each backend's extensions, the fields each backend reads, the plugin ids, the model buckets, the existing UI's sliders,
the values a field takes, each backend's batch and length limits, and the Create intent's compose fields. Run
`npm run fixtures` after the server changes and commit the result. `concept.json` is the concept page's behaviour;
`shipped.json` the keys saved data may hold; `request-differences.json` and `concept-differences.json` the accepted
differences.
