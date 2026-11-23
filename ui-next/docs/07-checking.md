# Checking that the format is sound

Four checks, run from `ui-next`. They need Node 22.18 or later (the tools run the TypeScript sources directly) and
`npm install` once (TypeScript is the only dependency).

```
npm run check      the type check, the validator and the request comparison; all must pass
npm test           the validator's own test (a few minutes)
npm run docs       rewrites the generated parts of these pages
npm run fixtures   reads what the real application declares into tools/fixtures/real-app.json
```

## The type check

`tsconfig.json` checks `src` and `tools` with `strict` and `exactOptionalPropertyTypes`. Every definition file ends in
`satisfies ModuleDef` (or `PedalDef`, `UnitDef`), so a misspelt field, a part not in the inventory, an option a part does
not take or of the wrong type, an engine, unit, check, option source, action or transform that is not listed, or a value
of the wrong type fails here, once, at the line in the definition.

## The validator

`tools/validate-definitions.ts` loads every definition and checks it twice: first its JSON form against `types.ts` with
the TypeScript compiler (so a definition that arrives as JSON, or that skipped its `satisfies`, gets the same check), then
what the types cannot see. Each message names the file, the parameter or node, the engine and tier, and what to change.

What it checks, in the order it runs:

- every definition file is listed in `index.ts`, and every definition is plain JSON data of the format's shape;
- ids unique across all definitions; parameter and state keys unique, in letters and digits; no two of a kind at one
  `at`; former keys (`was`) not in use;
- every condition: real parameters, state, desk facts, modules and engines; values among the parameter's options (the
  server's list, from the fixture, for a server-listed choice) or inside its range; a condition that can never hold, or
  two parameters whose `when` depend on each other in a circle;
- every parameter on every engine it is on: numbers have a range, a positive step and a default inside it and on its
  step grid (off the grid only where the server declares it so); choices a non-empty option list without repeated values
  or labels and a default among them; formats and words that fit the value; auto only on numbers; engine lines that name
  engines the parameter is on, once each, with the parameter's `engines` beside them; `held` only where a song number's
  range differs per engine, and required there; `staysBelow` joining two numbers with one range; kept per engine and
  saved as content refused;
- every wire: a channel; a `param:` field that `GENERATION_DEFAULTS` has and the engine's code reads; an `ext:` key the
  engine's backend declares, or reads with a note saying so; a `content:` field `CreateContent` has; a `model:` bucket
  the engine has; a range or options that differ from the server's or the existing UI's slider only with a note;
  `invert`, `scale`, `divide`, `map`, `via`, `auto` and `when` used where they fit; and a transform that packs a
  parameter named by that parameter's wire;
- every state: its words, its wire;
- every face node, on every engine and tier it can be shown at, with each of its lines: a part from the inventory; edits
  only through `param`/`params` of the types its part edits, reads only through `reads`, state only of the kinds it holds
  and only its own definition's; a control bound to something; bindings some engine it is shown on has; tiers inside its
  parent's; conditions and dims that can hold; actions on real things with values that fit (an example that exists, a set
  value inside the range); Quick choices on every engine the row is shown on, on the range and step grid, distinct, one of
  them the default; words whose slots name real things;
- every strip and idle reason: a condition that can hold, words, actions that fit;
- every pedal's switch and main knob its own and never hidden by `when`; every pedal face node tiered;
- every definition on each of its engines has something there, and a module a tile that is not empty;
- every module parameter is edited by a control of its own module on every engine it is on, at some tier;
- every thumbnail file exists and draws something;
- on every engine, no two parameters write one real field at the same time (their conditions are evaluated over every
  combination of the values they test);
- on every engine, the request at defaults: every value sent that differs from the real app's default has a note saying
  why;
- the factory workflows and revisions: modules and pedals that exist, values their engine can take, only what a revision
  keeps;
- the generated parts of these pages are up to date.

Things that are worth a look but not wrong are notes, printed with `--notes`: a part, an option, an action, a desk fact or
a transform no definition uses, a thumbnail file no definition draws, a default off the step grid that the server
declares that way. Nothing in the inventory has to be used: replacing a part, or removing the last parameter that used a
field of the format, is ordinary maintenance.

Output on the definitions as committed:

```
> node tools/validate-definitions.ts
(the counts, as in 00-index.md, "In numbers")

1 notes (--notes prints them).

OK: no errors.
```

The note is MiniMax-Music3's repetition window: it runs from 25 in steps of 25 with a default of 320, in the server's
manifest and in the concept alike.

## The request comparison

`tools/compare-request.ts` builds the Create request from the definitions on every engine at default values and
compares it field by field with what the existing frontend sends at its defaults ([05](05-what-the-definition-drives.md),
"What is sent"). It runs the existing code itself (`getGlobalParams()`, `createContentParams()`,
`resolveGenerationParams()`) through `tools/real-app/`, which lets those files load in Node without the frontend's
dependencies installed. It fails on a difference `tools/fixtures/request-differences.json` does not list with its reason,
or on a listed one that has gone. `--print` prints every difference.

```
> npm run request
request at defaults, 5 engines: 1 differences from the existing UI, 1 accepted in tools/fixtures/request-differences.json.
```

The one is HeartMuLa's scalar decoder ([08](08-open-questions.md) question 9).

## The validator's own test

`tools/test-validator.ts` makes one plausible mistake at a time on a fresh copy of `ui-next` and runs the type check and
the validator on it. The list holds the 40 mistakes of the format's review, rewritten for this format, and those found
since. Every one must be caught; three are no longer mistakes in this format (an auto switch's key can no longer collide
with a parameter's, a terminal flag is its control's label, and a pedal's width is worked out), and those must pass. Add a
mistake to the list when one gets through.

```
> npm test
...
Every mistake caught.
```

## The fixtures

`tools/fixtures/real-app.json` is what the real application declares, read from its source by `tools/make-fixtures.ts`:
`GENERATION_DEFAULTS`, the Create form's content and its fields, each backend's extensions, the backend-prefixed fields
each backend reads, the `GENERATION_DEFAULTS` fields each engine's code reads (directly, or through a field the resolver
makes from them), the plugin ids (the engine's registries and the Lua plugins), the model buckets, and the existing UI's
slider for each field. Run `npm run fixtures` after the server, the engine's registries or the Lua plugins change, and
commit the result; the validator then shows what the change means for the definitions.
