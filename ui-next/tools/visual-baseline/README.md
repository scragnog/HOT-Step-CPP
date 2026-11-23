# Visual baseline

Proves that a change to `src/css` (or anything else) changes nothing a user can see. It drives `hotstep-tiers.html`
through a matrix of states and records, for each state, the **computed style of every element and of its `::before`,
`::after` and `::placeholder`** (every longhand and every custom property the page uses), keyed by a stable path; then it
compares two recordings value by value. Screenshots of 47 main states are a second, coarser check, compared pixel by
pixel.

Needs Playwright with Chromium: `npm install` (it is a devDependency; the build itself needs nothing), or point
`PLAYWRIGHT_MODULE` at an existing `node_modules/playwright` folder.

## Use

```
# on the commit you start from (or pass --page with a copy of its hotstep-tiers.html):
node build.mjs
node tools/visual-baseline/vb.mjs capture base        # the reference recording (8 to 10 minutes with 8 workers)

...make your change...
node build.mjs
node tools/visual-baseline/vb.mjs check base          # records "base-now" with base's plan and compares
```

`check` prints `RESULT: identical`, or every difference as `state: element-key property: before -> after`, then a count
by property. Other commands:

```
node tools/visual-baseline/vb.mjs capture <name> [--page some.html] [--only <regex>] [--workers 8] [--plan-from <name>] [--no-shots] [--classmap map.json]
node tools/visual-baseline/vb.mjs compare <a> <b> [--only <regex>] [--max 200]
node tools/visual-baseline/vb.mjs list                # every state id
```

`--only` limits a run to states whose id matches, which is how to iterate quickly on one area (for example
`--only "pedal|board|w1000"`); run the full matrix before you merge. `--page` records another copy of the page, for example
one built from an older commit.

Recordings go to `tools/visual-baseline/out/<name>/` (git-ignored): one `<state>.json.gz` per state and `shots/*.png`.
Each record also keeps, for analysis, which elements every selector of the stylesheet matched (`matches`), the class
lists seen on elements (`classSets`) and, with `--classmap`, which named groups of classes were seen inside which
(`nesting`). Those are what you need to argue that two rules can never meet on one element.

## What it covers

`states.mjs` is the matrix; add a state there when you add a screen or a state the matrix cannot reach. Today it has:

- every engine (ACE-Step, MiniMax-Music3, YuE2, Stable Audio 3, HeartMuLa) in every tier, with every module the engine
  supports on the desk and every pedal and side unit on the board, in light and dark;
- the same at widths 1920, 1600, 1440, 1280, 1000, 760 and 390 (media and container queries), Lab in dark at each;
- reduced motion; a touch screen (`hover: none`, coarse pointer);
- modules the engine does not use (the parked tray), the empty desk, Repaint with a source song, hidden Lab modules;
- every overlay open, in light and dark, several at 1000 and 390: engine panel, model dropdown, engine log, notes drawer,
  picker (filtered, by category), popover, every dropdown kind (plugin, schedule, LCD, chip, stamp, model), song menu,
  score modal, row menu, toasts with and without an action, the save row, drafts (empty and with drafts);
- populated states: adapters loaded, the seed manager with saved seeds and the save form, lyrics with sections, a run in
  progress, a finished run, an error card, the player, a modified workflow and revision, a revision's details, a rename in
  progress, user workflows, changed values, the undo button, keyboard focus with its tooltip, song info, wildcards,
  MiniMax's composed caption, YuE's dataset tags and Continue from recording;
- collapsed side panels, the Modules panel as an overlay at 1280, both drawers at 1000, 760 and 390;
- a drag in progress: a card onto the desk, a tile being moved, from the picker, a pedal.

For states marked `pseudo`, every selector that uses `:hover`, `:focus`, `:focus-visible`, `:focus-within` or `:active`
gets one element forced into that state (Chrome DevTools Protocol `CSS.forcePseudoState`), and the element's subtree,
following siblings and ancestors are recorded again. Those records read the properties a state rule can change: every
longhand a state rule sets, every property of a rule that reads a custom property a state rule sets, and box geometry.
Targets whose regions cannot touch are forced together, one style pass per batch.

## Determinism

Every state runs in a fresh context with a fixed viewport, colour scheme and motion preference. The page is served over
local HTTP as UTF-8 (it was written without a `<meta charset>`, and from `file://` Chromium's guess at the encoding varied). The page's
clock is Playwright's fake clock, installed paused and only advanced by the harness, and frame callbacks get its time;
`Math.random` is seeded; smooth scrolling is made instant; anything not served locally (the web fonts) is blocked, so the
fallback fonts always apply. Finite animations and transitions are finished, and infinite ones paused at time 0, before
fake time runs and before each reading, so code that measures the page measures the settled layout. Fake time always moves
by the same amounts. Between steps of fake time, and before every reading, the harness waits for two real rendering updates
of the browser (a fresh `ResizeObserver` reports in the next one), so what the browser delivers on its own frames
(`ResizeObserver`, `IntersectionObserver`, hover after a layout change) has always run, however busy the machine is; no step
sleeps for a fixed time. Clicks, fills and hovers in a state's setup have a 2-minute deadline, which only a broken state
reaches (a 5-second deadline was missed on a busy machine, and the state was then recorded without its click). An open dropdown is placed once more after the page settles.

Before every reading, every box is laid out again from the root (`vb.relayout()` in `page-lib.js`: a constructable
stylesheet setting `contain-intrinsic-size` on every element, added and removed, which changes nothing the page shows).
Chromium answers `getComputedStyle` for an `auto` margin from the margin it stored at the box's last layout, and some
incremental layouts store `auto` as `0px` while leaving the box where it was; without this step the same layout read
`margin-left: 12px` in one run and `0px` in another, depending on what earlier steps and frames had done. If the page
ever sets `contain-intrinsic-size` itself, the reading stops with an error saying so.

The screenshot is taken right after the unforced record, before the forced-state pass (which runs fake time). Two
captures of the same page compare as identical, also with 8 workers on a busy machine. Recordings made before these
changes (November 2026) should be captured again before they are compared with new ones: their forced records and their
screenshots of `pseudo` states were taken differently.

## Limits

- `::-webkit-scrollbar`, `::selection` and `::marker` cannot be read through `getComputedStyle`; changes there show only in
  screenshots, and only where a scrollbar is visible.
- Screenshots are of the viewport: a full-page screenshot resizes the viewport, the page reacts to the resize, and the
  result is not repeatable. Everything below the fold is still covered by the computed-style records. Rasterising is not
  bit-exact between runs, so a pixel counts only when a channel moves by more than 8 of 255, and a screenshot with 64 such
  pixels or fewer is listed as noise, not as a difference.
- The states call the page's own globals (`S`, `render`, `toast`, `setVal`, `openScore`, `loadPreset`, `placeDD`, ...). If
  the JS renames them, update `states.mjs`; a state whose setup fails says so in the capture output.
- Elements inside `<head>` are left out of the comparison: they are never rendered, and adding one (the
  `<meta charset>`) renumbers the others.
- A state the matrix does not reach is not checked. When you move CSS, also check by reading that each selector you move
  cannot meet another one that sets the same property with the same specificity on the same element.
