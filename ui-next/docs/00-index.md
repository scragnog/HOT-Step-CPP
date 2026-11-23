# ui-next: the module definition format

`ui-next` is where the Create page's redesign (the "Tiers" concept) is being rebuilt in React, beside the existing
frontend in `ui`. This folder holds its foundation: the format in which every module of the desk is defined, and
every existing module, pedal and side unit written in it; and the first slice of the page in React (`src/app`,
[09](09-architecture.md)). What runs beneath the page is the data, the core that derives everything else from it (`src/derive/`, pure
functions of the definitions and a desk, which the React layer will be a view over), and the tools that check both
against the two references.

Read this page first, then the page for what you are about to do.

## What it is for

Two things, equally:

1. **Adding things without having been part of the redesign.** A new engine parameter, a new pedal or a new module is a
   small piece of data that names existing parts, and it comes out looking and behaving like every other one. A part,
   a unit or a named transform that does not exist yet is added once to the inventory and is then available everywhere.
2. **Making a class of defect impossible.** In the concept, facts about one parameter were stated in several places and
   drifted apart: a revision that did not store a module's settings, a value left out of range after an engine switch,
   undo steps named after internal keys, a missing changed marker, a Quick summary that disagreed with Lab. Here each fact
   is stated once, in the definition, and everything else is derived from it, the request sent to the engine included.

## The definitions are data in `.ts` files, not `.json` files

The owner asked for "a standardized JSON". A definition is exactly that in content: one plain data object, with no
code, that the validator converts to JSON and checks in that form. It is written in a `.ts` file that ends in
`satisfies ModuleDef` because of who edits it and when: the maintainer, in his editor, months from now. In a `.ts` file
the editor refuses a misspelt field, a part that is not in the inventory, an engine that does not exist, an option a part
does not take or a unit that is not listed, at the moment it is typed, and the file can carry a comment saying why.
A `.json` file with a generated JSON Schema would give completion and some of those refusals, but not the per-part
options, and no comments.

For the day the data arrives as JSON: the validator checks every definition in its JSON form against `types.ts` with
the TypeScript compiler, so there is one statement of the format, and `node tools/validate-definitions.ts --json <file>`
runs every check on a definition that comes as a file, as if it were in the index. What does not exist yet is a loader
that adds such a file to the page's definitions at run time; it would be a few lines beside `index.ts`. If the server
one day supplies parameter lists, [08](08-open-questions.md) question 1 says what would change.

## The pages

| Page | What it answers |
| --- | --- |
| [01-parts-inventory.md](01-parts-inventory.md) | Which parts a definition may name, what each edits, reads and takes, what a bespoke part is given, and the other lists a definition draws from |
| [02-definition-format.md](02-definition-format.md) | The format, field by field with real examples, the saved records, and the choices behind it |
| [03-definitions-coverage.md](03-definitions-coverage.md) | Every module, pedal and side unit: how much of each is data, and what is handled differently from the concept |
| [04-parameter-names.md](04-parameter-names.md) | The real application's name for every parameter, and the lists of mismatches |
| [05-what-the-definition-drives.md](05-what-the-definition-drives.md) | The core: how values, cleaning, undo names, the changed marker, the request, saving, engine switches, the "Also set" line, translation keys and the parts' runtime interface are derived |
| [06-how-to.md](06-how-to.md) | Add a parameter, a pedal, a module, a side unit, a part, an engine, a state entry, a transform; change a range, a condition, a real name; move, rename or remove things. Steps and line counts |
| [07-checking.md](07-checking.md) | The type check, the validator, the request comparison over the value space, the core's tests against the concept page, and the planted mistakes |
| [08-open-questions.md](08-open-questions.md) | What needs the maintainer's decision, and what was chosen meanwhile |
| [09-architecture.md](09-architecture.md) | The React layer: its folders, how a part and the generic renderer work, the store, the service boundary, the test layer and its results, the deviations from the concept's markup, and the plan for the rest |

## The folder

```
ui-next/
  docs/                      these pages
  src/
    derive.ts                the core, in one import
    derive/                  core, values, steps, engine, tiers, request, records, parts, view, strings, messages ([05])
    definitions/
      types.ts               the format, and the saved records
      engines.ts             the engines, in one place
      parts.ts               the parts inventory, what a part is given, and the other lists (units, checks, option
                             sources, desk facts, actions)
      transforms.ts          the named transforms a wire may use
      index.ts               every definition, in one list
      factory.ts             the factory workflows and revisions
      retired.ts             keys, state and ids removed on purpose
      modules/               25 module definitions (22 desk modules, plus the prompt, lyrics and models)
      pedals/                10 pedal definitions
      units/                 5 side-unit definitions
      thumbs/                their thumbnail drawings (SVG fragments in the thumbnail kit's classes)
  tools/
    validate-definitions.ts  the checks
    compare-request.ts       the request over the value space against the existing Create page's
    existing-ui.ts           which control of the existing page a user would use for each parameter
    test-core.ts             the core's derivations against the concept page, and its other tests
    test-validator.ts        the validator's own test: planted mistakes it must catch
    make-fixtures.ts         reads what the real application declares into tools/fixtures/real-app.json
    thumb.ts                 draws a pedal's thumbnail from its definition
    real-app/                runs the existing Create page's bar and CreatePanel in Node (page.ts), with a small React
    concept/                 collects what the concept page does (concept.json, concept-tiers.json)
    fixtures/                real-app.json, the existing page's recorded requests, the concept's, shipped.json and the
                             two lists of accepted differences
  package.json, tsconfig.json
```

## In numbers

<!-- generated: counts (npm run docs) -->
- 40 definitions: 25 modules, 10 pedals, 5 side units, on 5 engines.
- 284 parameters; 515 parameter-engine pairs. Of those, 500 say where the value goes (content 41, intent 16, ui 11, model 19, body 4, param 281, ext 138): 438 as a plain name, 41 through wire fields (invert, scale, divide, map, auto, when, several fields), 21 through a named transform. 15 pairs, on 9 parameters, have no counterpart in the real app.
- 13 state entries; 73 parts in the inventory, 73 used; 8 transforms; 7 factory workflows and 11 factory revisions.
<!-- end generated -->

## The two references

- **The concept**: `C:\Users\sneth\misc\HOT-Step-UI`. Open `hotstep-tiers.html` in a browser; read its `README.md` for
  the layout of `src`. When this format and the concept disagree about how something looks or behaves, the concept is
  right and the definition is wrong, unless [03](03-definitions-coverage.md) or `tools/fixtures/concept-differences.json`
  records the difference.
- **The real application**: this repository. The real names, ranges and defaults are read from its source into
  `tools/fixtures/real-app.json` by `npm run fixtures` ([07](07-checking.md)); the request is built with its own
  `resolveGenerationParams()` and `createContentParams()` and finished by the server's `resolveCreateIntent()`
  ([05](05-what-the-definition-drives.md), "What is sent").

## Checking

```
cd ui-next
npm install
npm run check
```

runs the type check, the validator, the request comparison and the core's tests. All four must pass before a definition
change is done; `npm test` runs the planted mistakes.
