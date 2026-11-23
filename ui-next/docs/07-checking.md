# Checking that the format is sound

Two checks, both run from `ui-next`. They need Node 22.18 or later (the validator runs the TypeScript sources directly)
and `npm install` once (TypeScript is the only dependency).

```
npx tsc -p tsconfig.json
node tools/validate-definitions.ts
```

or both at once, `npm run check`.

## The type check

`tsconfig.json` checks `src` and `tools` with `strict` and `exactOptionalPropertyTypes`: every definition file ends in
`satisfies ModuleDef` (or `PedalDef`, `UnitDef`), so a misspelt field, a part name not in the inventory, an engine id
that does not exist, or a value of the wrong type fails here.

Result on the definitions as committed:

```
> npx tsc -p tsconfig.json
(no output)
exit 0
```

## The validator

`tools/validate-definitions.ts` loads every definition through `index.ts` and checks what the types cannot:

- every definition file is listed in `index.ts`, and every definition is plain JSON data;
- ids unique per kind; every parameter key and state key unique across all definitions;
- for every parameter on every engine it is on: numbers have a range, a positive step and a default inside it; choices
  have a non-empty option list without repeats and a default among the options (or among `include` for a server-listed
  list); switches and text have defaults of their type; format words match the value type; `{s}` comes with `perSecond`;
- a song number whose range differs per engine is `held`, and only those are;
- every per-engine override names an engine the parameter is on; every parameter's engines are its definition's;
- every condition names a real parameter (with values among its options, of its type), a real state, a real module or a
  real engine; a condition on a parameter an engine lacks is an error unless an `any`, a `not` or an engine test guards it;
- `below`, `above`, `beside`, a pedal's `on` and `main`, a unit's `on` refer to real parameters of the right type;
- every face node names a part in the inventory, binds parameters that exist on its engines and fit the part's value
  types, passes only options the part lists, holds children only if it is a container or bespoke part, and a terminal
  line has as many flags as controls; every Quick choice is inside its parameter's range or among its options;
- every action names a real parameter (with a value of the right type), module or tier;
- every module parameter is shown by some node at some tier (a pedal's and unit's are laid out);
- every thumbnail file exists and every file in `thumbs/` is used;
- every field declared in `types.ts` is used by at least one definition (read from the source with the TypeScript
  compiler API), and every part in the inventory by at least one definition.

Output on the definitions as committed:

```
> node tools/validate-definitions.ts
definitions: 40 (25 modules, 10 pedals, 5 side units)
parameters: 284; parameter-engine pairs: 515 (501 with a real name: content 49, ui 18, model 19, param 276, ext 139; 14 with none)
parameters with no real counterpart on at least one engine: 9 (cap.ds, s0, ad.trigcap, den.on, lss.on, yue.btakes, yue.bmode, takes.n, pp.vst.on)
state entries: 9; parts in the inventory: 73, used: 73; format fields: 142, unused: 0

notes (1):
  repetition › mm.win [minimax-m3]: default 320 is off the step grid (25 + n × 25)

OK: no errors.
exit 0
```

The note is real: MiniMax-Music3's repetition window runs from 25 in steps of 25 with a default of 320, in the server's
manifest and in the concept alike. A keyboard step from 320 lands on 325, not 345. It is reported, not refused.

### That it catches what it should

While writing the how-to ([06-how-to.md](06-how-to.md)) on a copy of the definitions, the validator was run against
deliberate mistakes. Each produced one error naming the place:

```
planner › yp.maxtok: no face shows this parameter at any tier, so nothing can change it or say it is changed
planner › yp.maxtok [yue2]: default 5000 is outside 256..4096
```

and while the definitions were being written it caught, among others: parameters left on engines their module is not
on, a condition testing a parameter on an engine that does not have it, a part given an option it does not take, a
read-out given children, a thumbnail that did not exist, an option list whose default was not one of its values, and two
parts of the inventory no definition used (one was then used, the other removed).

## Other output

```
node tools/validate-definitions.ts --usage       where each part is used
node tools/validate-definitions.ts --inventory   the inventory table in docs/01
node tools/validate-definitions.ts --names       the names table in docs/04
```
