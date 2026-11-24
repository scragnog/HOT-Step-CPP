# Behaviour suite

Checks what the page does, where `tools/visual-baseline` checks how it looks. It drives `hotstep-tiers.html` through
every handler, as a user would (real mouse clicks, drags and keys, HTML5 drag and drop, files chosen and dropped,
reloads), and checks the result of each step: about 600 checks in 62 scenarios, in about a minute.

Needs Playwright with Chromium: `npm install` (it is a devDependency), or point `PLAYWRIGHT_MODULE` at an existing
`node_modules/playwright` folder.

```
node tools/behaviour/run.mjs                        # run every scenario; exit 1 if a check fails
node tools/behaviour/run.mjs --only "desk|picker"   # scenarios whose id matches
node tools/behaviour/run.mjs --list                 # every scenario id
node tools/behaviour/run.mjs --coverage             # also: which click hooks the page has that no scenario clicked
node tools/behaviour/run.mjs --workers 8            # parallel pages (default 4)
node tools/behaviour/run.mjs --page some.html       # another copy of the page, for example one built from an older commit
node tools/behaviour/run.mjs --pending              # only the pending scenarios (behaviours known to be wrong; below)
```

Two checks end every scenario, whatever it tested: **every change to the desk was an undo step** (the page writes to
`vsStray` any change it finds between gestures that is not one it makes itself, or that no step can name: a control
that changes the desk without a step fails here as soon as a scenario uses it), and **every undo step is named in the
page's words** (no internal key such as `lm.temp`, no bare "On on → off", none of the vague names of earlier rounds).
`t.do` sets the scene: the change it makes from inside the page starts a gesture of its own and is taken as where things
stand, not as a person's step (events it dispatches are still gestures, and are recorded as such).

Each scenario prints `ok` or `FAIL`, and every failed check is listed at the end with what the page gave and what was
expected. A page error or console warning fails its scenario. A click on an element that something else covers (a toast
over a button) fails too, saying what covers it, so a scenario never passes by clicking the wrong thing.

## Pending scenarios: behaviours known to be wrong

Some behaviours of the page are known to be wrong and are not fixed in it (the page is the reference for a rebuild, and
their fix would be script the rebuild throws away). Each is stated as a **pending scenario**: written like any other, with
real input and exact checks, and marked with the finding it comes from and one sentence saying what is right:

```
{ id: 'tier-keys-wait-under-a-modal', pending: { finding: 'F-08', right: 'No page shortcut acts under a modal layer ...' },
  run: async t => { await t.click('#notesBtn'); await t.press('Alt+3'); await t.eq('Alt+3 under the notes leaves the tier', () => S.tier, 'quick'); } },
```

```
node tools/behaviour/run.mjs                 # leaves them out; ends with "N pending scenarios not run"
node tools/behaviour/run.mjs --pending       # runs only them: each "still failing" (with what failed) or "NOW PASSES"
node tools/behaviour/run.mjs --pending --list
```

A normal run's exit code is decided by the other scenarios alone. `--pending` exits 1 while any pending scenario still
fails. Each pending scenario must fail on today's page for the reason its finding gives (run it and read the failure: a
check that passes today, or fails because a selector is wrong, defines nothing). One that passes is the signal to
promote it: remove its `pending` field, and every normal run keeps it passing from then on.

**For whoever rebuilds the page:** the pending scenarios are the behaviours known to be wrong in this reference page.
The rebuild is expected to make them pass (and then promote them), as well as every normal scenario. Their `right`
sentence is the specification; the checks say exactly how it is measured.

## Every concept scenario is in a list

`run.mjs --slice` refuses to run while a scenario of the concept's suite (the files numbered below 30) is in no
acceptance file's `scenarios` and no file's `excluded`: each is run, or left out with the reason that is true today (a
deviation, a hook or a global of the concept's script the test layer does not give, a defect named with who owns it).
A reason that names a package or a finding is re-read when that lands (`acceptance/fix-core.json` holds the round
after the merge's re-reading).

## Waiting scenarios: right, but not passable until a later package lands

In the rebuild's acceptance (`acceptance/*.json`, `run.mjs --slice`), an entry may say it is waiting on a later work
package, when what it tests is right but a placeholder that package will replace changes what the scenario sees (a
tile's height decided by parts not drawn yet):

```
{ "id": "10-desk-modules/undo-covers-structure", "why": "...",
  "waiting": { "package": 3, "because": "One sentence: what that package draws, and why the scenario needs it." } }
```

`run.mjs --slice` still runs it and prints it as `wait <id> (waiting on package N ...)`, lists it with its reason and its
failing checks after the summary, and does not count it as a failure. If it passes, the run fails ("waiting on package
N, but it passes"): whoever lands that package removes the `waiting` field, so every run keeps it passing from then on.
`switches.mjs` ignores waiting scenarios altogether, so a feature switched off is caught only by a scenario that passes
with the feature on. Keep the list short: a scenario that fails for any other reason is a failure, not a wait.

## Switches: every feature is caught by a scenario

In the rebuild, each work package lists its features in `switches/<package>.json`, each with one small change to its
source that switches it off (`{ feature, file, find, replace }`; the test build applies it when `HS_SWITCH` names the
feature, `vite.config.ts`, into `dist-switch/`). `switches.mjs` proves the acceptance has teeth: each feature switched
off must make at least one scenario fail that passes with the feature on.

```
npm run switches:apply                               # no builds: each switch's text is found exactly once (a second)
npm run switches                                     # every switch, built and tried (below)
npm run switches -- --package "^(slice|5-models)$"   # the switches of some packages (switches/<name>.json)
npm run switches -- --only "^cancel"                 # the switches whose feature matches
npm run switches -- --whole                          # the old way: the whole acceptance for every switch
```

**`--applies`** (`npm run switches:apply`) builds nothing: for every switch, and every planted defect of
`tools/visual-baseline/plants/`, it applies the changes as the build does (in order, with the file's line endings) and
checks that each `find` is found exactly once in its file and changes something. A merge breaks this more often than
anything else (a switched line reworded by another package), and the check takes under a second; run it after every
merge. It cannot see a file the build does not include; the full run can.

**The full run** builds the page once per switch (about a second) and runs scenarios on it, which is where the time
goes. A switch need not be tried against the whole acceptance, only against the scenarios that can catch it, so each is
tried in stages, stopping at the first that catches it:

1. **recorded**: the scenarios that caught it the last time (`caught/<package>.json`, which every run writes);
2. **own**: the rest of its own package's acceptance (`acceptance/<package>.json`);
3. **rest**: the rest of the whole acceptance, one other package's acceptance file at a time.

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
(and then nothing is recorded); `--no-record` leaves `caught/` as it is; `--workers` is passed to `run.mjs` (default: half the machine's threads, 6 on a 12-thread machine; lower it to leave more
for other work).

`caught/<package>.json` is written only when what caught a switch changes; commit it with the package's work. A package
that adds switches adds nothing else: a switch with no record starts at its own package's acceptance.

Measured on the slice's and package 5's 28 switches (24 November 2026): with `caught/` filled, 4 min 20 s at 6 workers
(about 9 s a switch, 2.9 scenarios run for each); starting from each package's own acceptance (`--fresh`), 7 min 25 s
at 6 workers, 16 min 20 s at 2. The old way took 6 to 7 minutes a switch at 2 workers (70 min for the slice's 11), and
every scenario the staged run named as a catcher was one the old way named too. For all 387 switches that is about an
hour once `caught/` is filled, against about 18 hours for the old way at 6 workers (a whole acceptance is about
2 min 40 there).

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

Two runs of the suite, at different parallelism and under load, give identical traces.

## What it covers

`scenarios/*.mjs`, one file per area:

- `01-desk`: tiers; the engine panel (cards, arrow keys, Escape, outside click); engine switches (takes clamped, Repaint
  removed); add, remove and re-add in place, Undo from the toast, the Undo button and Ctrl+Z (and not in a text field);
  adding by Enter and Space; moving tiles by keyboard; Lab modules offered from Studio; Quick's six-module cap; parked
  modules; the empty desk's starters; Show changes; every engine in every tier with every module (no collapsed or
  overlapping tile).
- `02-dnd-panels`: dragging cards, tiles, picker items, pedals and side units onto the desk (before and after a tile,
  cancelled drags); files dropped on and chosen for every drop zone (audio, latents, wrong kinds, outside a zone,
  clearing with Undo, recent references); the side panels collapsed and kept, as an overlay at 1280, as drawers at 1000.
- `03-workflows`: loading workflows (click, Enter, Space, open in its tier, Undo); revisions (load, details, Enter,
  Revert, export as JSON); saving a revision and a workflow (Enter, Replace, factory names refused, the follow-up
  action, Escape, the toggle); Update, Revert and Save as on changed rows; renaming and deleting from the row menu
  (Enter, Escape, blur, names in use, the two-step delete, Undo); importing files (cleaned values, Overwrite, bad
  JSON); older storage shapes migrated; corrupt storage.
- `04-controls`: knobs (drag, Home, End, arrows, Shift, PageDown, double-click reset, leaving auto); LED toggles,
  segments, AUTO buttons, range inputs, checkboxes; number boxes (typing, Enter, arrows, PageUp, Shift, words and
  out-of-range values refused, blur, Escape, scrubbing); rulers, faders, scrub tokens; XY pads (drag, arrows,
  double-click); the CFG interval handles; the look-back window; every dropdown variant with its keyboard (arrows,
  typing into the filter, Enter, Escape, Home, End, Tab, no match) and the cycle buttons; the language stamp's recent
  list; typed-field checks; Insert example.
- `05-modules`: each module's own handlers: tap tempo, the key wheel (click and every key), beat I/O, auto-trim, tempo
  snippets; the seed manager (roll, random, typing, Enter, Escape, blur, save, update, cancel, star, load, search, delete
  with Undo, from saved, the LM seed link); takes (reels, hover preview, Use N takes); adapters (load, stack, section
  masks by pointer and keyboard, phase windows, removal with Undo, planner adapter, Quick strength, trigger word); planner
  and guidance (Lua plugins, guidance modes, flags, inline tokens, chain of thought, score preview, Quick rows); YuE2
  plugins, re-planning, Plan CFG; the lead sheet (planned and pasted previews, focus trap, retry, download, play, render,
  lyric timing); Build by section (plan, takes, keep, pick, render, re-plan, strip); Continue from recording and Repaint
  (files, arrows, drags, the song menu); lyrics (tags, the strip, Instrumental, wildcards, gender, engines without
  lyrics); the Pedalboard (foot switches, the chain switch, side units, pedal knobs, re-adding, removal with Undo, the last
  pedal, cover art).
- `06-prompt-generate`: suggestions (move to lyrics with Undo, set the BPM), the placeholder; wildcards, Presets &
  genres, Write it for me, Song info; Compose caption, the caption source, the locked prompt, dataset tags; Generate (empty
  prompt, takes, Ctrl+Enter, the queue, cancelling one and all, finishing); each engine's own run (MiniMax refusing and
  Retry, Dismiss, ACE-Step's passes, YuE2's preview, SA3 and HeartMuLa needing a description, latents, wildcards); the
  player (play, pause, seek by keys and drag, Space, stop) and the song menu (each action, toggle, Escape, outside click).
- `07-overlays-keys`: the picker (button, F9, filter, Enter, Escape, categories, arrows, Home, End, Tab, Space, Lab
  modules, a click on one already on the desk, Ctrl+Enter blocked, ×, backdrop); Concept notes (Tab trap, Escape, scrim,
  ×); the engine log; Escape closing only the top layer; drafts (save, named, load on another engine with Undo, delete
  with Undo, the thirty-draft limit); toasts (hover holds, actions, ×, replaced messages); the theme kept over a reload;
  the other pages; the keyboard focus ring, tooltips by keyboard and by touch, held keys.
- `08-persistence-cables`: a reload keeps every part of the desk and session; songs still rendering are not saved, and
  `pagehide` and a hidden tab save; restored state is cleaned field by field; an older shape is kept aside and said; the
  pedalboard's cables land on their jacks and pass behind no pedal through adds, removals, bypass, seven widths, tiers and
  engines.
- `09` to `20`: later rounds, one file each (undo and storage, desk modules, engines, desk layout, Quick and Studio,
  the Lab desk, other engines, sizes and input modes, the undo net, tips and touch, round 5's desk and Lab).
- `21-chrome-round5`: the go-bar player names its song in Quick; the picker's wall fits 1920x1080 to 1440x900 and scrolls
  by column at 1280x720; its category bar; names that differ only by spaces; stored data that can't be used is kept
  aside; the Workflows list's long names and "yours" mark; model steps named by role; Latest's cards and count; target
  sizes; the stub pages' way back; the Lab desk on a short window. Then the pending scenarios of review 5 (F-07, F-08,
  F-13, F-17 to F-22, F-25, F-27 to F-29, F-36, F-37).
- `22-engines-round5`: MiniMax-Music3 and YuE2 (review 3, E-): side units on a narrow board, YuE2's adapter strength, Tempo
  at Quick, the lyrics language, the caption dataset, the engine-switch toast, the lead sheet with Chain of Thought off and
  its chords, composer CFG on auto, Plans & endings, Build's locked values and its Quick line, the planner seed, Reuse,
  the top-p off rail. Then the pending scenarios of review 3 (E-03, E-06, E-09, E-23, E-24, E-46, E-47).

`--coverage` reads the click hooks from the page's own script (every `d.x` and `'x' in d` that names a `data-`
attribute of the page) and lists any no scenario clicked; today it reports all of them clicked.

- `23-sa3-hm-round5`: Stable Audio 3 and HeartMuLa, narrow sizes and input modes (review 4, S-): one word for the main box per
  engine, the sampler's settings staying with their engine, a revision without adapters unloading them, the Tiled decoder's
  idle state, the changes count, HeartMuLa's Adapters width, reduced-motion arrival rings, touch targets and words, the
  workflow tree's Right key. Then the pending scenarios of review 4 (S-03, S-06, S-07, S-16, S-17, S-18, S-28, S-31, S-36, S-37).

## Adding a scenario

Add to a file in `scenarios/` (or a new one) an entry `{ id, w, h, scheme, reduced, touch, storage, run: async t => {...} }`
(all but `id` and `run` optional; `storage` seeds `localStorage` before the page loads). `t` drives the page:
`t.click(sel)`, `t.dblclick(sel)`, `t.hover(sel)`, `t.press(key)`, `t.type(text)`, `t.focus(sel)`, `t.mouseDrag(sel, [[dx,
dy], ...])`, `t.drag(src, dst, { at, drop })`, `t.dropFile(sel, name, type)`, `t.file(sel, name, mime, text)`,
`t.resize(w, h)`, `t.reload()`, `t.do(label, fn)` (a change made in the page), `t.js(fn)` (a question), and checks with
`t.ok(label, fn)` (true, or `{ ok: true }`; anything else is shown as the detail) and `t.eq(label, fn, want)`. Each action
settles the page (fake time, finished animations, real frames) before the next.

## How it was shown to fail

On the starting commit, breaking one handler (the tier buttons' click) made 7 checks fail in 3 scenarios and the
trace differ in 5, starting at the first tier click; reverting it brought both back clean.
