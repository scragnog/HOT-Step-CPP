# Checking that the format is sound

The checks, run from `ui-next`. They need Node 22.18 or later (the tools run the TypeScript sources directly) and
`npm install` once (TypeScript, ESLint, Vite and React; the browser tools need Playwright with Chromium, below).

```
npm run check      the everyday check (about a minute): both type checks, the validator, the request comparison, every
                   tools/test-*.ts (the core's derivations against the concept, the view's selectors and each package's
                   core), the parts' contract in the lint, the code's contract with the definitions (test-code.ts: no
                   key, state or id named in the code; every part a definition names has a component and every action
                   a runner), and that every switch and planted defect still applies once (switches.mjs --applies)
npm run lint       the React layer's lint (eslint.config.js: the two rules the design rests on, and one boundary)
npm test           before a merge: the planted mistakes, then the request comparison driven live (about 10 min)
npm run docs       rewrites the generated parts of these pages
npm run record     drives the existing Create page over the value space and records what it sends
npm run ship       records the keys, state and ids a release ships with (only when a release goes out)
npm run fixtures   reads what the real application declares into tools/fixtures/real-app.json
```

The browser tools (Playwright with Chromium; `PLAYWRIGHT_MODULE` may point at an installed `node_modules/playwright`),
on the test build (`npm run build:test`):

```
npm run behaviour                  the acceptance: every scenario of tools/behaviour/acceptance (about 4 minutes 20)
npm run visual:all                 every visual set against its approved recording (about 7 minutes)
npm run motion                     every motion against its approved samples, every 16ms of its run (about 15 seconds)
npm run visual:approve -- --only "^<set>$"    after a deliberate change: approve that set's recording again
npm run motion:approve -- --only "^<id>$"     after a deliberate change to a motion: approve its samples again
node tools/visual-baseline/gallery.mjs   every part on every surface, both themes, every tier, against its record
npm run switches                   each feature switched off makes a scenario fail (slow: a build per switch)
npm run switches:apply             each switch's and planted defect's text is still in its file, once (in npm run check)
node tools/visual-baseline/plants.mjs    each planted visual defect is caught
node tools/motion/cost.mjs          what each motion costs the main thread on a full desk (a measure, not a check)
npm run perf                        every action's speed, the load, idle and memory (a measure; tools/perf/README.md)
```

The times are at 6 workers on a 12-thread machine. These checks compare the page with itself: the scenarios with what
each says is right, the visual sets and the motions with recordings approved after they were looked at, committed in
`tools/visual-baseline/approved/` and `tools/motion/approved/`. A deliberate change that shows fails them until the sets
and motions it was meant to change are approved again ([06](06-how-to.md), "Approve a change the page shows"). Until
the tag `rebuild-parity` they compared the page with the concept it was ported from; that tag is where parity was last
proved. [09](09-architecture.md), "The test layer", says what each covers; the READMEs of `tools/behaviour`,
`tools/visual-baseline` and `tools/motion` say how.

The acceptance holds the performance guards too (`scenarios/39-perf.mjs`, `acceptance/perf.json`): what a knob's
change asks again, what an Undo notifies, what a removal, a move and a drag measure, each held to a limit;
`switches/perf.json` brings each cause back and a guard fails.

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
- the definitions that are a server subsystem (StableStep, Whisper lyrics, Lyric timestamps, Forced alignment) are on
  exactly the engines whose `capabilities().features` has it, except where [08](08-open-questions.md) question 17 asks;
- every condition: real names, values the parameter can hold, a condition that can hold, no circles;
- every parameter on every engine: a range with a positive step and a default on its grid, options without repeats,
  formats and words that fit, a text default and example that pass the parameter's own check, a range's two ends with
  the lower end's default below the upper's, `held` exactly where a song number's range differs per engine, an engine
  line whose `when` keeps the names the parameter's own condition tests, two options never claiming one former value;
- every parameter with engine lines is said on every engine its definition is on: a definition put on a new engine
  without the parameter (a Prompt with no prompt text) is refused;
- every wire: a channel; a `param:` field that `GENERATION_DEFAULTS` has, of the type of its real default, and that the
  engine's code reads; an `ext:` key the backend declares, or reads with a note; a `content:` or `intent:` field the
  real app has; an option value the real field takes; a range, options or a default that differ from the server's or
  the existing UI's only with `differs` naming the check and a note saying why; `wire.auto` of the field's type; `via`
  alone; a mapped choice whose every option is mapped, or written by another parameter while it is chosen;
- every state: its words, its named steps and what they may change, its wire;
- every face node, on every engine and tier it can show at: a part from the inventory, bindings of the types its part
  edits, `reads` that match the part's `reads` signature, the named steps its part takes declared, tiers inside its
  parent's, conditions and dims that can hold, a node whose own condition can show it only where something it edits
  applies (the Power group is not shown on Cosine), actions on real things, Quick choices on every engine the row shows
  on, words whose `{value:...}` the engine has, the same control not twice at one tier;
- every strip and idle reason; every pedal's switch (defaulting to off, so a pedal off the board is off) and main knob,
  a Studio face for a pedal without a main knob, a face for every side unit, each pedal's hue at least 15 degrees from
  its neighbours' in the chain on every engine;
- two parameters of one definition with one name; what Quick or Studio edits, Lab edits;
- every module parameter edited by a control on every engine it is on; every thumbnail file exists and draws;
- on every engine, no two parameters write one real field at once, and at defaults every value sent that differs from
  the real default is waived with `differs: ['default']`;
- the factory records as a loaded file is read;
- `tools/fixtures/shipped.json`: every key, state and id a release shipped is still a definition's, or named in a `was`,
  or retired in `src/definitions/retired.ts`, and kept where it was (owner and stored). It is empty until the first
  release, and only `npm run ship` writes it, so moving a value is never recorded as a side effect;
- the generated parts of these pages are up to date.

Notes, printed with `--notes`, are worth a look and not wrong: a part, option, action, desk fact or transform no
definition uses, a thumbnail file no definition draws.

## The request comparison

`tools/compare-request.ts` builds the Create request from the definitions and the one the existing Create page sends
when a user does the same things on it, over the value space of every engine, and compares them field by field: the
resolved request, and the intent's `compose` and `captionSource`. Both end in the server's own `resolveCreateIntent()`.

**The definitions' side** is `createIntent()` ([05](05-what-the-definition-drives.md), "What is sent").

**The existing side is the existing page itself.** `tools/real-app/page.ts` runs the bar of the existing Create page
(`GlobalParamBar` and the panels it opens for each engine) and `CreatePanel` in Node, as functions over the real
`globalParamsStore`, with a React that only calls components and keeps their state (`react.mjs`); what the panels draw
from elsewhere (shared controls, queries, modals) is read as an element with its props. So every control the page
draws is found by what it shows: its place and words ("Generation > Shift", "Post-Processing > StableStep > Refine
strength"). For each state, the comparison uses the control a user would use for each value the desk holds, through
the page's own `onChange` (a switch is turned on first if it hides what is behind it, a button is pressed, a field is
typed into and left), then presses CreatePanel's Generate, whose `handleGenerate` hands App the intent. The one line of
`App.tsx` that merges the bar's params under it is restated in `tools/real-app/app.ts` and checked against its text.

Which control stands for which parameter is `tools/existing-ui.ts`: one line per parameter naming the control by its
place and words on the existing page, with a factor where the existing control works in another unit. It never names
a request field: the field comes from the existing page's code, so a wire that sends to the wrong field, in the wrong
unit or under the wrong condition shows as a difference. What a control in Node cannot set is restated there and
counted apart: the engine panel's model rows and MiniMax-Music3's renderer row (drawn from the server's catalogue), the
planner adapter (picked from the server's list), and files and adapters chosen in a dialog. A parameter with no line is
not driven, and the run lists it as having no control on the existing page.

The run says how many pairs were driven through the existing page's controls, how many restated, and which have no
control. At this writing: 456 of the 479 wired pairs on engines that send them are driven, 10 restated, 13 have no
control on the existing page (the caption source and dataset, which the page fills from Lyric Studio or a dataset, and
controls the existing page does not offer on that engine, each also a question in [08](08-open-questions.md)).

**The states**, per engine: a description and lyrics on every desk (the existing page offers Generate only then);
every parameter at its minimum, its maximum and a step off its default, each option (every id a server list has), each
switch both ways, auto on and off, text set and empty, each with its condition made to hold, from every value a
condition names; with each condition one of its wires sends under, alone and with every switch of its definition on;
every parameter set while its condition does not hold, in the first three ways it can fail; every parameter with its
module's state held (adapters loaded, a file chosen); every module, pedal and unit taken off the desk with its values
changed; adapter slots in Simple and Advanced with a phase window; seeds up to 2^64-1; every file and span; the
caption source with its tracks, with and without an adapter; and 300 random combinations, each parameter drawn from a
generator seeded by its own key, so adding a definition does not move the others' states.

**The transforms' branches.** The run measures, with V8's block coverage, every block of every transform's `send` in
`src/definitions/transforms.ts`, prints how many it reached, and fails when one is never reached. A new transform is
compared over the value space only once every branch of it is reached.

**An accepted difference names exactly what it covers**: in `tools/fixtures/request-differences.json` each entry is one
field, the engines, the state it occurs in (a condition on the desk in the definitions' terms, and parameters set or
left alone), what each side sends there (a value, `absent`, `sent`, `value:<key>`), why, and the question in
[08](08-open-questions.md) that decides it. Any other difference fails, and so does an entry that no longer occurs on
one of its engines. One entry is a rule rather than a field: a difference that goes away when every module, pedal and
unit is put on the desk is the desk's own (question 11), and is accepted only if the definitions with everything on the
desk send exactly what the existing page sends. A field the existing page never sends on that engine, sent by a
parameter it has no control for, has nothing to compare with: the run lists it as new (question 19) and does not fail.

**What the existing page sent is recorded** in `tools/fixtures/existing-requests.json`, with the commit of `ui/` it was
recorded at and a fingerprint of the files it comes from (the bar, CreatePanel, the stores, `App.tsx`, the settings).
The run says which mode it ran in:

- *recorded*: the files are as recorded, so the recording stands for them (the everyday check: about 6 s; states not
  in the recording are driven live);
- *live*: the files changed since the recording, so the page is driven, and a recorded request it no longer sends fails
  the run until `npm run record` records it again on purpose; `--live` forces this;
- *recorded, absent*: the existing frontend is gone, and the recording stands for it (`--as-if-absent` runs this mode
  with the files present).

So the comparison outlives `ui/`, as long as the definitions' side keeps `createContentParams()` and the app settings
([08](08-open-questions.md), question 4).

```
> npm run request
request over the value space, 5 engines; existing side recorded: the existing frontend is unchanged since 25eb7bd3d2 ...
  ace: 982 states, ... fields compared, ... differences
  pairs: 456 driven through the existing page's own controls, 10 restated, 13 with no control on the existing page: ...
  transforms reached (blocks of send reached/all): int 7/7, text 1/1, scheduler 9/9, ...
```

`--print` prints every difference class, `--show <engine> <words>` both sides of one state and what was done on the
existing page, `--quick` leaves out the random states, `--missing` the controls the page did not show when they were to
be used.

## The core's tests

`tools/test-core.ts` runs the derivations on the states the concept page was put in and compares what they return with
what the page did (`tools/fixtures/concept.json` and `concept-tiers.json`, recorded from the page's own functions in
Chromium before the cutover; they are the derivations' expected outputs now, and their collectors are in the tag
`rebuild-parity`): undo step names, what each tile shows at each tier at default values and with each switch and choice of the
module on its other values, the "Also set" line and its link, engine switches and their toasts, what a revision
stores, and what cleaning keeps. A value the definitions cannot say (a server-listed choice whose names the server
gives) is left out of a case; an option of a fixed list the page shows that is none of the definition's labels fails.

A disagreement is a defect, fixed, or listed in `tools/fixtures/concept-differences.json` with the reason the page is
not followed. An entry names its derivation and a pattern the case matches, and where the disagreement can be said
exactly, `rewrite` (replacements that turn the page's output into the core's, which must then be equal) or `core` (a
pattern the core's output must match), so a misspelt undo name or a count that is off does not pass as accepted.

It also checks, without the page: the value in force (a module, pedal or unit off the desk reads as its defaults in
conditions, facts and the request; a desk read back from its autosave sends what it sent), the parts' contract, the
translation keys, reading the real fields back from what each value sends, and saved records of older shapes.

## The planted mistakes

`tools/test-validator.ts` makes one plausible mistake at a time on a fresh copy of `ui-next` and runs the type check and
the validator on it, and for a mistake marked so the request comparison or the core's tests. It holds the mistakes of
every review: the first review's 40, the second's 30 with its five wrong sends, and the third's fourteen that passed
every check then (a Shift on Auto sending 0, the scheduler transform's Beta and Composite written the wrong way round, a
schedule id sent as another, a wrong condition on the guidance scale, the trigger word and the lead sheet, a default
off by a decision, HeartMuLa's adapter never sent, two group scales exchanged, a misspelt undo name and option label,
the engine switch toast counting twice or one adapter too many, and the Power group shown on Cosine). Every one must
be caught, except those this format made impossible, which must pass and say why. Add a mistake when one gets through.

Copies are checked several at a time (one per two cores, eight at most; `--serial` for one at a time): the 99 mistakes
take about three minutes on a twelve-core machine; the live comparison after them about six and a half.

## The fixtures

`tools/fixtures/real-app.json` is what the real application declares, read from its source by `tools/make-fixtures.ts`:
`GENERATION_DEFAULTS`, the Create form's content and fields, each backend's extensions and features, the fields each
backend reads, the plugin ids, the model buckets, the existing UI's sliders, the values a field takes, each backend's
batch and length limits, and the Create intent's compose fields. Run `npm run fixtures` after the server changes and
commit the result. `existing-requests.json` is the existing page's requests (`npm run record`); `concept.json` and
`concept-tiers.json` the concept page's behaviour, recorded before the cutover; `shipped.json` what a release shipped; `request-differences.json` and
`concept-differences.json` the accepted differences.
