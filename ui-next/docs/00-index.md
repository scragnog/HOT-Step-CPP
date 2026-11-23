# ui-next: the module definition format

`ui-next` is where the Create page's redesign (the "Tiers" concept) is being rebuilt in React, beside the existing
frontend in `ui`. This folder holds its foundation only: the format in which every module of the desk is defined, and
every existing module, pedal and side unit written in it. There is no React application here yet: no components, no Vite
setup. The only code is types, data and one validation script.

Read this page first, then the page for what you are about to do.

## What it is for

Two things, equally:

1. **Adding things without having been part of the redesign.** A new engine parameter, a new pedal or a new module is a
   small piece of data that names existing parts, and it comes out looking and behaving like every other one. A part
   that does not exist yet is added once to the inventory and is then available everywhere.
2. **Making a class of defect impossible.** In the concept, facts about one parameter were stated in several places and
   drifted apart: a revision that did not store a module's settings, a value left out of range after an engine switch,
   undo steps named after internal keys, a missing changed marker, a Quick summary that disagreed with Lab. Here each fact
   is stated once, in the definition, and everything else is derived from it.

## The pages

| Page | What it answers |
| --- | --- |
| [01-parts-inventory.md](01-parts-inventory.md) | Which parts a definition may name, what each takes, its states, where its styles are, who uses it |
| [02-definition-format.md](02-definition-format.md) | The format, field by field with real examples, and the design choices behind it |
| [03-definitions-coverage.md](03-definitions-coverage.md) | Every module, pedal and side unit: which are pure data, which need a bespoke part, and why |
| [04-parameter-names.md](04-parameter-names.md) | The real application's name for every parameter, and the two lists of mismatches |
| [05-what-the-definition-drives.md](05-what-the-definition-drives.md) | How defaults, clamping, undo names, the changed marker, saving, engine ownership, tier summaries, tips, translation, thumbnails and tests are derived |
| [06-how-to.md](06-how-to.md) | Add a parameter, a pedal, a module, a part: steps and line counts |
| [07-checking.md](07-checking.md) | The type check and the validator, and their output |
| [08-open-questions.md](08-open-questions.md) | What needs the maintainer's decision, starting with the server supplying parameter lists |

## The folder

```
ui-next/
  docs/                      these pages
  src/definitions/
    types.ts                 the format
    parts.ts                 the parts inventory, the desk states, the text checks
    index.ts                 every definition, in one list
    modules/                 25 module definitions (22 desk modules, plus the prompt, lyrics and models)
    pedals/                  10 pedal definitions
    units/                   5 side-unit definitions
    thumbs/                  their thumbnail drawings (SVG fragments in the thumbnail kit's classes)
  tools/validate-definitions.ts
  package.json, tsconfig.json
```

## In numbers

- 40 definitions, all five engines: 18 pure data, 13 data plus a named part, 9 mostly bespoke.
- 284 parameters, every control of the concept; 515 parameter-engine pairs, 501 of them with the real application's
  name found.
- 73 parts in the inventory, every one used; 142 fields in the format, every one used.

## The two references

- **The concept**: `C:\Users\sneth\misc\HOT-Step-UI` at the tag `vanilla-reference`. Open `hotstep-tiers.html` in a
  browser; read its `README.md` for the layout of `src`. When this format and the concept disagree about how something
  looks or behaves, the concept is right and the definition is wrong.
- **The real application**: this repository. The real names come from `ui/src/stores/globalParamsStore.ts`,
  `server/src/contracts/generationControls.ts`, the backends' `capabilities().extensions` in
  `server/src/services/backends/`, and `ui/src/components/create/createContent.ts`.

## Checking

```
cd ui-next
npm install
npm run check
```

runs the TypeScript compiler and the validator. Both must pass before a definition change is done.
