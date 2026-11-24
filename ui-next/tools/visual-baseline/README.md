# Visual baseline

Proves that a change to `src` changes nothing a user can see, or shows exactly what it changes. It drives the test
build (`dist-test/index.html`) through the states of `slice-states/` and records, for each state, the **computed style
of every element and of its `::before`, `::after` and `::placeholder`** (every longhand and every custom property the
page uses), keyed by a stable path, with forced `:hover`, `:focus`, `:focus-visible`, `:focus-within` and `:active`;
**every element's role, accessible name, ARIA states, title, own words and attributes**; and **screenshots** of the
states that ask for one. Then it compares that recording with the **approved recording** of the same states, value by
value.

The approved recordings are the page as it was last looked at and approved, one per visual set, committed in
`approved/<set>/`. The rebuild is its own reference: where the concept page once was the reference, the tag
`rebuild-parity` marks the last commit proved against it.

Needs Playwright with Chromium: `npm install` (it is a devDependency; the build itself needs nothing), or point
`PLAYWRIGHT_MODULE` at an existing `node_modules/playwright` folder.

## Use

```
npm run visual:all                                    # every set against its approved recording; exit 1 on a difference
npm run visual:all -- --only "^(models|overlays)$"    # some sets, by name
npm run visual:all -- --list                          # the sets, their states, scopes and when each was approved
npm run visual:all -- --page some.html                # another build (default dist-test/index.html, built first when stale)
npm run visual:all -- --workers 3                     # pages at once (default: half the machine's threads, 6 on 12)

npm run visual:approve -- --only "^models$"           # record the current build as the set's approved recording
```

For each set, `visual-all.mjs` captures the build in the set's states, with the approved recording's plan of forced
states, and compares the two inside the set's scope. It prints one line per set (its states, and `identical` or the
number of differences, with the first few and the log's path) and a verdict. A set fails when it differs, when it has
no approved recording, or when its states are not the ones approved (a state added, removed or renamed). Logs go to
`out/all-logs/`, the current capture of each set to `out/now-<set>/` (both git-ignored).

**After a deliberate change** (a module added, a part restyled): run `visual:all`, read every difference it lists for
the sets you meant to change and none other, look at the screenshots in `out/now-<set>/shots/`, then approve those sets
alone, `npm run visual:approve -- --only "^(set|other)$"`, and commit `approved/` with the change. Approving prints the
setup notes of any state whose setup failed (a click that found nothing): an approved state is approved as it is, so
read them. `--approve` with no `--only` approves every set; do that only when every difference is meant (a change to the
page's base styles).

**A new set** is a file of `slice-states/` with its states, the regions they compare (`SCOPES`) and, last, its set:

```
export const VISUAL_SET = { name: 'models', only: '^models-', scope: 'models' };
```

`only` picks the set's states by id (from any file: the slice's set takes `01-parts.mjs`'s `slice-parts-` states too)
and `scope` is optional (a set without one compares every element of its states, or their parts). Approve it once
(`visual:approve -- --only "^models$"`). A state of `slice-states/` that belongs to no set, or to two, fails the run at
once, before anything is captured, naming the states.

`vb.mjs` is the recorder under it, for work on one area:

```
node tools/visual-baseline/vb.mjs capture <name> [--page some.html] [--only <regex>] [--scope <name>] [--workers 6] [--plan-from <name|dir>] [--no-shots] [--slim] [--classmap map.json]
node tools/visual-baseline/vb.mjs compare <a> <b> [--only <regex>] [--scope <name>] [--max 200]
node tools/visual-baseline/vb.mjs check <base> [--only <regex>]    # capture "<base>-now" with base's plan, then compare
node tools/visual-baseline/vb.mjs list                             # every state id
```

A recording is named (`out/<name>/`, git-ignored) or given as a folder (`tools/visual-baseline/approved/models`). A full
recording keeps, for analysis, which elements every selector of the stylesheet matched (`matches`), the class lists seen
on elements (`classSets`) and, with `--classmap`, which named groups of classes were seen inside which (`nesting`): what
you need to argue that two rules can never meet on one element. `--slim` leaves those out and keeps only the elements
inside the scope, which is what an approved recording holds.

## The tolerance

Computed styles, roles, names, words and attributes are compared exactly: two captures of one build are identical.
Screenshots have one tolerance, for rasterising, which is not bit-exact between runs (anti-aliasing on an edge, a level
or two on a blurred shadow or a gradient): a pixel counts as different when a channel moves by more than 8 of 255
(`PIXEL_DELTA` in `vb.mjs`), and a screenshot differs when more than 64 pixels do (`NOISE_PIXELS`); fewer are printed
as noise. A one-line change to a stylesheet that anything in a set shows changes a computed style, and fails.

## Why the approved recordings are committed

`approved/` holds, per set, one `<state>.json.gz` per state (the slim record: the scope's elements, their semantics, the
parts recorded on their own, the forced states and their plan) and `shots/*.png`: about 16 MB in all (5 MB of it screenshots, the slice's set 10 MB), which git keeps without
trouble, and a re-approval rewrites only the sets it approves. Committed, the baseline travels with the code it
describes: a branch that changes the page carries its new approved recordings, a merge of two such branches shows a
conflict in exactly the sets both changed (approve them again on the merged tree), and anyone can run the check without
first recording a reference from an older commit.

## What the states cover

The states are the files of `slice-states/`, one per area, each with its set: the desk at every engine and tier in
both themes, at the desktop and narrow widths, with reduced motion and on a touch screen (`00-slice.mjs`, `01-parts.mjs`);
the shared controls, the containers and read-outs, the instruments and the cartridges as parts on their own (`10-`,
`20-`, `30-`, `31-`, `32-`); Quick's fold (`37-`); the prompt card and the lyric sheet (`40-`); the engine panel, the
models and the VRAM meter (`50-`); the go bar, Latest, the player, the song menu and the toasts (`60-`); the Workflows
panel, its rows, menus and drafts (`70-`); the picker, the popover, the tip and drags in progress (`80-`); the notes
and the engine log (`110-`). `npm run visual:all -- --list` gives the counts.

For states marked `pseudo`, every selector that uses `:hover`, `:focus`, `:focus-visible`, `:focus-within` or `:active`
gets one element forced into that state (Chrome DevTools Protocol `CSS.forcePseudoState`), and the element's subtree,
following siblings and ancestors are recorded again. Those records read the properties a state rule can change: every
longhand a state rule sets, every property of a rule that reads a custom property a state rule sets, and box geometry.
Targets whose regions cannot touch are forced together, one style pass per batch. The plan of which elements are
forced is the approved recording's, so a check forces exactly what was approved.

## Determinism

Every state runs in a fresh context with a fixed viewport, colour scheme and motion preference. The page is served over
local HTTP as UTF-8. The page's clock is Playwright's fake clock, installed paused and only advanced by the harness,
and frame callbacks get its time; `Math.random` is seeded; smooth scrolling is made instant; anything not served
locally (the web fonts) is blocked, so the fallback fonts always apply. Finite animations and transitions are finished,
and infinite ones paused at time 0, before fake time runs and before each reading, so code that measures the page
measures the settled layout. Fake time always moves by the same amounts. Between steps of fake time, and before every
reading, the harness waits for two real rendering updates of the browser (a fresh `ResizeObserver` reports in the next
one), so what the browser delivers on its own frames (`ResizeObserver`, `IntersectionObserver`, hover after a layout
change) has always run, however busy the machine is; no step sleeps for a fixed time. Clicks, fills and hovers in a
state's setup have a 2-minute deadline, which only a broken state reaches. An open dropdown is placed once more after
the page settles.

Before every reading, every box is laid out again from the root (`vb.relayout()` in `page-lib.js`: a constructable
stylesheet setting `contain-intrinsic-size` on every element, added and removed, which changes nothing the page shows).
Chromium answers `getComputedStyle` for an `auto` margin from the margin it stored at the box's last layout, and some
incremental layouts store `auto` as `0px` while leaving the box where it was; without this step the same layout read
`margin-left: 12px` in one run and `0px` in another. If the page ever sets `contain-intrinsic-size` itself, the reading
stops with an error saying so.

The screenshot is taken right after the unforced record, before the forced-state pass (which runs fake time). Two
captures of the same build compare as identical, also with 6 workers on a busy machine.

## Planted defects

`plants.mjs` plants each defect of `plants/*.json` in the test build, one at a time (`vite.config.ts`, `HS_SWITCH`, into
`dist-switch/`), captures the slice's states that show it, compares them with the slice's approved recording, and runs
the parts gallery on it: each must be caught by one or the other. It first checks that the clean build matches the
approved recording on those states, so a stale approval cannot pass for a catch.

```
node tools/visual-baseline/plants.mjs [--only <regex>] [--workers n]
```

## Every part on every surface

`gallery.mjs` draws the test build's parts gallery (`src/app/testing/gallery.ts`: every built part in a tile's body, a
row, a rack, a well, an LCD, a tile's foot, a labelled pairing, on a pedal and on a side unit, and the containers' seven
surfaces) in both themes at every tier, its values changed, and checks each part where it stands: drawn inside its
container, its words at 3:1, the page's caps label, the changed mark, a choice that says which is chosen, a new part's
markup taking the shared label and mark, and every part's look as it was recorded (`tools/fixtures/gallery.json`;
`--record` writes it, after the screenshots in `out/gallery/` have been looked at).

## Limits

- `::-webkit-scrollbar`, `::selection` and `::marker` cannot be read through `getComputedStyle`; changes there show only in
  screenshots, and only where a scrollbar is visible.
- Screenshots are of the viewport (or of one element): a full-page screenshot resizes the viewport, the page reacts to
  the resize, and the result is not repeatable. Everything below the fold is still covered by the computed-style records.
- The states drive the page through the test build's compatibility layer (`src/app/testing/compat.ts`: `S`, `render`,
  `setVal`, `addItem`, ...). If it renames one, update the states; a state whose setup fails says so in the capture
  output, and a setup that fails in the same way in both recordings compares as identical.
- Elements inside `<head>` are left out of the comparison: they are never rendered.
- A state the matrix does not reach is not checked; a region outside a set's scope is not compared in that set.
