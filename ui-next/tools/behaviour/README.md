# Behaviour suite

Checks what the page does, where `tools/visual-baseline` checks how it looks. It drives the test build
(`dist-test/index.html`) through its handlers, as a user would (real mouse clicks, drags and keys, HTML5 drag and drop,
files chosen and dropped, reloads), and checks the result of each step: 3626 checks in 466
scenarios, in about 4 minutes 20 at 6 workers (the scenarios run longest first, by their times in `out/durations.json` from the last run).

Needs Playwright with Chromium: `npm install` (it is a devDependency), or point `PLAYWRIGHT_MODULE` at an existing
`node_modules/playwright` folder; and the test build, `npm run build:test`.

```
node tools/behaviour/run.mjs                        # the acceptance: every scenario; exit 1 if a check fails
node tools/behaviour/run.mjs --only "desk|picker"   # scenarios whose id matches
node tools/behaviour/run.mjs --list                 # every scenario id
node tools/behaviour/run.mjs --workers 6            # parallel pages (default: half the machine's threads)
node tools/behaviour/run.mjs --page some.html       # another build, for example one made from an older commit
```

(`--slice`, the acceptance's name while the rebuild was built package by package, is still taken and changes nothing.)

Three checks end every scenario, whatever it tested: **every change to the desk was an undo step** (the test layer
writes to `vsStray` any change to what the desk holds that no undo or redo recorded, outside a scene: a control that
changes the desk without a step fails here as soon as a scenario uses it), **every undo step is named in the page's
words** (no internal key such as `lm.temp`, no bare "On on → off", none of the vague names of earlier rounds), and **no
console error or warning**. `t.do` sets the scene: a change it makes through the test layer (`S`, then `render()`) is
taken as where things stand, not as a person's step.

Each scenario prints `ok` or `FAIL`, and every failed check is listed at the end with what the page gave and what was
expected. A click on an element that something else covers (a toast over a button) fails too, saying what covers it, so
a scenario never passes by clicking the wrong thing.

## The acceptance files

`acceptance/<area>.json` lists the scenarios of one area of the page (`slice`: the shell, the desk, the pure-data
modules and the board; `1-controls` to `11-notes-logs`: each area as the rebuild was built; `fix-*` and `fold`: the
review rounds' fixes and Quick's fold), each with one sentence on what it exercises:

```
{ "id": "35-models/choosing-a-model-from-its-list", "why": "The model list (rows, quant chips, the role and the card in its head), ..." }
```

Every scenario of `scenarios/` is in an acceptance file: `run.mjs` refuses to run while one is in none (a scenario that
nobody runs is a hole, not a choice), or while a file lists one that does not exist. A new scenario goes into the file
of the area it tests; the switches use the files to try a feature's own area first.

## Waiting scenarios: right, but not passable until later work lands

An entry may say it waits on later work, when what it tests is right but the page cannot pass it until that work lands
(a placeholder that work will replace changes what the scenario sees):

```
{ "id": "...", "why": "...", "waiting": { "package": "the real application's services", "because": "One sentence: what that work draws, and why the scenario needs it." } }
```

`run.mjs` still runs it and prints it as `wait <id> (waiting on ...)`, lists it with its reason and its failing checks
after the summary, and does not count it as a failure. If it passes, the run fails ("waiting on ..., but it passes"):
whoever lands the work removes the `waiting` field, so every run keeps it passing from then on. `switches.mjs` ignores
waiting scenarios, so a feature switched off is caught only by a scenario that passes with the feature on. Keep the list
short: a scenario that fails for any other reason is a failure, not a wait. None waits today.

A scenario may carry `pending: { finding, right }`: the review finding it was written for while the behaviour was still
wrong, and the sentence that says what is right. It is run like any other; the field is a note of where it comes from.

## Switches: every feature is caught by a scenario

Each area lists its features in `switches/<area>.json`, each with one small change to its source that switches it off
(`{ feature, file, find, replace }`; the test build applies it when `HS_SWITCH` names the feature, `vite.config.ts`,
into `dist-switch/`). `switches.mjs` proves the acceptance has teeth: each feature switched off must make at least one
scenario fail that passes with the feature on.

```
npm run switches:apply                               # no builds: each switch's text is found exactly once (a second)
npm run switches                                     # every switch, built and tried (below)
npm run switches -- --package "^(slice|5-models)$"   # the switches of some areas (switches/<name>.json)
npm run switches -- --only "^cancel"                 # the switches whose feature matches
npm run switches -- --whole                          # the whole acceptance for every switch
```

**`--applies`** (`npm run switches:apply`, in `npm run check`) builds nothing: for every switch, and every planted
defect of `tools/visual-baseline/plants/`, it applies the changes as the build does (in order, with the file's line
endings) and checks that each `find` is found exactly once in its file and changes something. A merge breaks this more
often than anything else (a switched line reworded by other work), and the check takes under a second; run it after
every merge. It cannot see a file the build does not include; the full run can.

**The full run** builds the page once per switch (about a second) and runs scenarios on it, which is where the time
goes. A switch need not be tried against the whole acceptance, only against the scenarios that can catch it, so each is
tried in stages, stopping at the first that catches it:

1. **recorded**: the scenarios that caught it the last time (`caught/<area>.json`, which every run writes);
2. **own**: the rest of its own area's acceptance (`acceptance/<area>.json`);
3. **rest**: the rest of the whole acceptance, one other area's acceptance file at a time.

A scenario counts as catching a switch only if it fails with the feature off **and passes with it on**: each scenario
that fails on a switched build is run once on the test build (`dist-test/index.html`, built first when missing or
older than the sources) to know, and one that fails there too is named on the switch's line and not counted. A
scenario the harness could not run (the browser gone, the page not loaded) is run again; if it does not run twice it
counts neither way, and a switch nothing else catches is reported as not known, which fails the run. Waiting
scenarios catch nothing. A stage that runs fewer scenarios can only miss a catch, never invent one, and a miss sends the
switch on to the next stage, so a switch is reported caught exactly when the whole acceptance would catch it; what
differs from `--whole` is only how many of its catching scenarios are listed. Every switch gets a line (caught or not,
at which stage, by which scenarios, how many were run, how long it took), the summary counts the stages, and
`out/switches-last.json` keeps every switch's full result. `--scenarios <regex>` limits every stage to some scenarios
(and then nothing is recorded); `--no-record` leaves `caught/` as it is; `--workers` is passed to `run.mjs` (default:
half the machine's threads, 6 on a 12-thread machine; lower it to leave more for other work).

`caught/<area>.json` is written only when what caught a switch changes; commit it with the work. An area that adds
switches adds nothing else: a switch with no record starts at its own area's acceptance. With `caught/` filled a switch
costs about 9 seconds at 6 workers; all of them take about an hour.

## Proving a refactor changes nothing

```
node tools/behaviour/run.mjs --record base     # on the commit you start from
node tools/behaviour/run.mjs --compare base    # after the change: "trace compared with base: identical"
```

`--record` keeps, after every step of every scenario, a digest of the page: a hash of each top-level element's markup,
the state object `S`, every `localStorage` value, the focused element, the title, the root's classes and the scroll
position. `--compare` runs the same scenarios and reports, per scenario, the first step where anything differs and what
(the state, a storage key, focus, or the markup of which element). Add `--dump` to keep the full markup in the trace, to
diff two runs. Recordings go to `tools/behaviour/out/<name>/` (git-ignored).

Three things are left out of the comparison, because they are not behaviour: the order of `S`'s top-level keys (and a
key that holds `null` or an empty object counts as absent), the order of keys inside stored JSON (arrays keep their
order), and the inline position of the toast and the dropdown (the page re-places both on scroll events, which the
browser sends on its own frames; their content is compared, and where they sit is the visual baseline's job).

## Determinism

No check passes or fails by timing. Every scenario runs in a fresh context at a fixed viewport, colour scheme and
motion preference, with:

- Playwright's fake clock, installed paused: timers and animation frames run only when the harness advances it, by the
  same amounts on every run;
- `Math.random` seeded, smooth scrolling made instant, anything not served locally blocked (the web fonts), the page
  served over local HTTP as UTF-8;
- the animation timeline frozen (DevTools `Animation.setPlaybackRate(0)`): an animation stays at its start until the
  harness finishes it, so code that measures the page mid-animation measures the same thing every time;
- after each slice of fake time, a wait for two real rendering frames, so ResizeObserver, IntersectionObserver and
  scroll events are delivered however loaded the machine is; and before a click, the same wait after the page scrolls
  the target into view;
- at the end of each settle, a mouse move to where the mouse already is: when what lies under a still mouse changes (an
  overlay closes), the browser owes a mouseout and a mouseover and sends them on its own clock; this makes them arrive
  at once, on every run.

Two runs of the suite, at different parallelism and under load, give the same results. Their traces are identical but
at the steps where a tier changes or the page reloads: the tier change is a View Transition, which runs on the
browser's own clock, and the desk's autosave after it, the markup at that step and a reload's scroll can then differ
between runs (measured at the cutover: 25 of 466 scenarios differ from an earlier run of the same build there). Read a
trace difference at such a step before taking it for a change.

## What it covers

`scenarios/*.mjs`, one file per area or round. The files numbered below 30 are the page's oldest scenarios, one per
area or review round of the design (each file's own first lines say what it holds): `01-desk` the tiers, the engine
panel, adding, removing, moving and undoing; `02-dnd-panels` drags and files and the side panels; `03-workflows`
workflows, revisions, saving, renaming, importing; `04-controls` the controls and their keys; `05-modules` each
module's own handlers; `06-prompt-generate` the prompt tools, Generate, the player and the song menu;
`07-overlays-keys` the picker, the notes, toasts, the theme, the other pages, focus and tooltips;
`08-persistence-cables` reloads, what is kept and the pedalboard's cables; `09` to `23` the review rounds after them.
The files from 30 on are the rebuild's own, one per area as it was built (`30-slice` to `36-notes-logs`), its fix
rounds (`37-fix-*`), Quick's fold (`38-fold`) and motion (`38-motion`, which reads the page at moments of each motion
with the timeline held).

## Adding a scenario

Add to a file in `scenarios/` (or a new one) an entry `{ id, w, h, scheme, reduced, touch, storage, run: async t => {...} }`
(all but `id` and `run` optional; `storage` seeds `localStorage` before the page loads), and list it in the acceptance
file of its area. `t` drives the page: `t.click(sel)`, `t.dblclick(sel)`, `t.hover(sel)`, `t.press(key)`,
`t.type(text)`, `t.focus(sel)`, `t.mouseDrag(sel, [[dx, dy], ...])`, `t.drag(src, dst, { at, drop })`,
`t.dropFile(sel, name, type)`, `t.file(sel, name, mime, text)`, `t.resize(w, h)`, `t.reload()`, `t.do(label, fn)` (a
scene set in the page), `t.js(fn)` (a question), and checks with `t.ok(label, fn)` (true, or `{ ok: true }`; anything
else is shown as the detail) and `t.eq(label, fn, want)`. Each action settles the page (fake time, finished
animations, real frames) before the next. What a scenario reaches into the page for is the test build's compatibility
layer (`src/app/testing/compat.ts` and `testing/compat/*.ts`).

## How it was shown to fail

PLANTED
