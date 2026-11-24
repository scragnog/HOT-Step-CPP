# Open questions for the maintainer

Each question says what a user would see either way, what choosing the other way costs, and what the definitions do
meanwhile. Where the new page would send the server something the existing page does not, the request comparison
records it as an accepted difference (`tools/fixtures/request-differences.json`), and every accepted difference names
the question here that decides it. Until a question is answered, the rule is to send what the existing page sends; the
questions below say where the definitions follow the concept instead, and why.

## 1. Could the server supply the parameter lists?

Mostly, it already does. Four of the five engines declare their knobs in `capabilities().extensions`
(`server/src/services/backends/<engine>/index.ts`, served by `GET /api/capabilities`), each with `key`, `type`,
`label`, `hint`, `min`, `max`, `step`, `default`, `options` and `visible_when`. MiniMax-Music3 declares 27, YuE2 86,
Stable Audio 3 5, HeartMuLa 14.

Meanwhile the validator reads a fixture of those declarations (`tools/fixtures/real-app.json`, made by
`npm run fixtures` from the server's source) and fails when a definition's `ext:` wire names a key its backend neither
declares nor reads, or states a range, default or option the server does not have, without a note saying why. So a
definition cannot drift from the server without someone writing down that it does.

What is missing on the server, judged from `server/src` as it is:

1. **ACE-Step has no manifest.** Its knobs are `GENERATION_DEFAULTS`: defaults only, no ranges. Their ranges live in the
   existing UI's sliders, which the fixture also reads; after the rebuild, the definitions would be the only statement of
   them. One table of `{ key, type, min, max, step, default, options }` that `GENERATION_DEFAULTS`, the resolver and the UI
   all read would end that.
2. **Fields read but not declared.** MiniMax-Music3 reads its adapter, mode, scales, merge switch, trigger switch,
   renderer and precision (`mm3LmAdapter*`, `mm3LmMergeGpu`, `mm3DitBackend`, `mm3DitPrecision`). A compact intent's
   `backendParams` cannot carry them. (YuE2's lead sheet, `yue2Abc`, is declared.)
3. **Engine state is not a parameter.** YuE2's adapter dials are per slot in `POST /api/backends/models`; HeartMuLa's and
   MiniMax-Music3's models are buckets.
4. **A few ranges are computed at run time** (YuE2's longest length, noise variations and batch size).

If the server supplied them, a definition would keep only presentation (part, words, tier, format, Quick's choices,
`owner`, `stored`) and take range, default and options from the server at run time and from the fixture in tests.

Decision needed: whether to add the ACE-Step manifest and declare the undeclared fields, and whether the definitions
should then drop `range`, `default` and `options` for declared keys.

## 2. Keys: the concept's or the real ones?

The definitions keep the concept's keys and say the real names in `wire`. One control can be five real fields (Steps),
so some key of the desk's own is needed; for the 1:1 cases (`pp.ss.str` is always `stableStepStrength`) the real name
could be the key. Nothing has been saved in this format yet, and `tools/fixtures/shipped.json` is empty until the first
release (`npm run ship` records it then), so either answer costs nothing before release.

## 3. The mismatches

Each needs a decision; [04-parameter-names.md](04-parameter-names.md) lists them with what the real app does.
Meanwhile the definitions keep the concept's design and say so in the wire.

- `den.on`, `lss.on`: the switches stay; each gates its strength's wire, sending 0 while off. Dropping them would leave
  the strength alone deciding, as the real app has it.
- `pp.vst.on`: the concept's chain switch has no server counterpart. Keep it as a UI bypass that disables every plugin?
- Build by section: the concept's takes per section and section mode do not exist in the real `Yue2Build`, and the real
  plan strength default is 0.6 where the concept says 1.
- Takes on ACE-Step: the concept allows 9, the server's `batch.max` is 8. The definitions stop at 8 (the existing page
  sends 8 for 9).
- APG on MiniMax-Music3, Stable Audio 3 and HeartMuLa (entries `apg-mom-without-control`, `apg-norm-without-control`):
  the concept shows the momentum and threshold there; the existing page shows them only on ACE-Step and sends their
  defaults (0.75, 2.5) elsewhere. A user who sets them on the new page sends their values; on the existing page they
  cannot. Only MiniMax-Music3 reads the threshold. Following the existing page is two `engines` lists.
- The concept's ranges and steps against the existing UI's (the stack's combined strength to 2 against 4, and nine
  steps): the definitions follow the concept, and each wire's note gives the UI's.
- The real parameters with no concept control ([04](04-parameter-names.md), "The real app has it"): which belong on the
  desk, and at which tier?

## 4. Where values live in the rebuild

The definitions assume a store keyed by definition key, with engine stashes driven by `owner`, and build the request by
handing the server's own resolver its input ([05](05-what-the-definition-drives.md), "What is sent").

Two pieces of the existing frontend are still used by the definitions' side: `createContentParams()`
(`ui/src/components/create/createContent.ts`) and the app settings every request carries (`DEFAULT_SETTINGS` in
`ui/src/components/settings/SettingsPanel.tsx`). Before `ui/` is removed they have to move with the page (the request
comparison then keeps running on its recording, [07](07-checking.md)). Also open: whether the other studios keep
reading `globalParamsStore` until they move, and how saved profiles and "Reuse settings" come in (each wire reads its
real field back, so both can be derived).

## 5. Option values for server-listed choices

Solvers, schedules, guidance modes and models come from the server. The definitions store the server's ids and use
`''` for "Native" (the engine's own sampling) where plugins are optional, and for "Engine default" on the model rows.
Is `''` right for both, and are the plugin ids stable enough to save?

Meanwhile (package 5): a model the engine loads (its options' `loaded`) left at `''` is drawn as the file the server
has selected for its bucket, which is what the request sends; choosing that file stores `''`, so the page shows no
"Engine default" row while the server lists the selected file, and nothing is marked changed on it.

## 6. Engine state and undo

Picking a YuE2 adapter or a model is a `POST` that evicts and reloads weights. The concept makes it an undo step like any
other. Should undoing it post again (and pay the reload), or should engine-state changes be outside undo?

## 7. Translations

The core returns what to say as a message id and its slots, and English is one table among the others
([05](05-what-the-definition-drives.md), "Strings and translation"). Should the extraction write the definitions' strings
and the core's messages into `ui/src/i18n/locales/*.json` directly, or into a separate namespace file?

## 8. Thumbnails

A new pedal's drawing can be made from its definition (`node tools/thumb.ts <pedal>`); a new module or unit still needs
one by hand. Is a generated fallback wanted for modules added before someone draws theirs?

## 9. HeartMuLa's scalar decoder

Entry `hm-scalar-decoder`. The existing page sends HeartMuLa's selected scalar decoder with every request, so a queued
job keeps it; the concept's engine panel has no row for it, so the new page sends none and a queued job uses the decoder
selected when it runs. A user notices only if they change the decoder while a job waits. Following the existing page is
one parameter in `models.ts` with the wire `['model:heartmula.scalar_decoder', 'body:heartmulaScalarDecoderModel']`.

## 10. The defaults the concept and the existing page start on differently

- **Shift on ACE-Step** (entry `shift-auto`). The concept starts on Auto (-1). On Auto the engine works out a shift from
  the song's length and the step count (`engine/tools/hot-step-server.cpp`, about line 1707: 3 times a length factor and
  a steps factor, each 0.8 to 1.4, so between about 1.9 and 5.9; 3.18 for a 60-second song at 12 steps, 4.13 for a
  3-minute one), on every model. The existing page starts at a fixed 3.0. So every default ACE-Step render differs a
  little, whether or not the Sampler is on the desk. Starting at 3.0 is one line (`auto: false` on ACE-Step's line,
  default 3). Meanwhile the definitions follow the concept, because a fixed shift and an adaptive one are both
  defensible and the concept's choice was reviewed; this is the one place the default is not the existing page's.
- **Lua plugins on MiniMax-Music3, Stable Audio 3 and HeartMuLa, the solver, schedule and guidance untouched**
  (entries `native-*`). The concept starts on Native (the engine's own loop), the existing page on euler, linear and APG,
  which moves the render onto those plugins. A user who turns plugins on gets a different render.
- **Guidance on Stable Audio 3 with Lua plugins on** (entries `sa3-guidance-*`). The concept starts at 1 (no extra
  guidance) and lets it be set; the existing page has no control there and sends ACE-Step's 9.
- **Expand wildcards on YuE2** (entry `yue2-autoexpand`). The concept offers it, the server honours it; the existing page
  hides it on YuE2. Following the existing page is one line (YuE2 off the parameter's engines).
- **MiniMax-Music3's renderer** (entry `mm3-precision-on-auto`). The definitions add Auto, the server's own choice when
  nothing is set, and start there, and apply the TensorRT precision on Auto; the existing page shows the precision only
  on TensorRT. The concept starts on GGML.

## 11. What a module, pedal or unit off the desk sends

Entry `off-the-desk`. On the new page a definition that is not on the desk reads as its defaults everywhere: its tile
says nothing of it, conditions read its defaults, and the request sends nothing of it, so the engine's defaults apply.
Its values are kept for when it comes back. This is the concept's rule (its `generate()` reads Takes and Tempo only while
they are on the desk, and its meter counts only what is on the desk as "changes"); a module hidden by the tier is still
on the desk, and its settings still apply.

The existing page has no desk: once a user has set 40 steps, it sends 40 steps until they set something else. So a
user who sets 40 steps and then removes the Sampler sends 12 on the new page, and would see that the Sampler is gone
and the meter says "the rest on defaults". The alternative, a module off the desk that keeps applying, is one branch in
`valueOf()` and needs the go bar to say what is applied unseen.

A pedal's switch defaults to off (the validator checks it), so a pedal off the board is off on both readings. One side
unit defaults to on: Lyric timestamps. Meanwhile a desk without it sends nothing of it, so the engine makes timestamps,
as the existing page does unless told not to. The concept's factory desk "Songwriter" puts the unit on the board, which
suggests the concept means "no unit, no timestamps"; following that is to send its off value when it is off the board,
one line in `realState()`.

## 12. A trigger word typed with no adapter loaded

Entries `trigger-without-adapter*`. The concept shows the trigger word only while an adapter (or a planner adapter) is
loaded, so a word typed for an adapter that is gone is not put in the caption. The existing page prepends a typed
trigger word with or without an adapter, which also serves a merged fine-tune or a style prefix. Following the existing
page is removing the parameter's `when`.

## 13. Settings sent where the engine does not read them

The two pages disagree about whether to send a setting in a state where the engine ignores it. No render differs; the
request does, and a user who reads the request (the queue's details, a saved song's settings) sees the value. Each is
one entry; following the existing page in each is a `when` on the parameter or the wire.

- The server hides a declared knob by `visible_when`, the existing page follows it, and the concept shows the knob and
  the definitions send it once set: MiniMax-Music3's repetition mode and window under a penalty of 1
  (`mm3-rep-*-gated`); YuE2's Plan CFG range, rescale and level match under a Plan CFG of 1
  (`yue2-plan-cfg-*-gated`), CADS's range under a strength of 0 (`yue2-cads-*-gated`) and the lyric timing with the
  planner writing nothing (`yue2-lyric-timing-gated`); HeartMuLa's flow shift with plugins off (`hm-flow-shift-gated`,
  the existing page keeps sending a value set before); Stable Audio 3's loudness target with normalize off
  (`sa3-lufs-target-gated`); MiniMax-Music3's plan name while no plan is saved (`mm3-plan-name-gated`).
- An adapter's dials set with no adapter loaded (`mm3-dial-without-adapter-*`, `hm-dial-without-adapter-*`): the concept
  keeps them settable, the existing page shows them only with an adapter. MiniMax-Music3's "Add trigger to caption" is
  offered by the existing page only for an adapter whose metadata has a trigger (`mm3-trigger-to-caption`).
- The sampler picks set while Lua plugins are off (`picks-without-plugins-*`, `apg-without-plugins-*`,
  `solver-setting-without-plugins-*`): the existing page keeps sending them, with the picked solver's own settings; the
  definitions send none and the server's resolver fills in its defaults.
- The Beat I/O bars with Beat I/O off (`beat-bars-without-beat-io`), the retake seed with no retake variance
  (`retake-seed-without-variance`), the stack's alignment timing with no per-section directives
  (`section-align-without-directives`), the refiner's overlap with a window of 0 (`refiner-overlap-without-window`), and
  the seed while it is random (`seed-while-random`).

## 14. MiniMax-Music3's schedule shift with Lua plugins off

Entry `mm3-low-step-shift`. The concept shows the shift only with plugins on. The server also reads it with plugins off,
for low-step compensation under 30 steps (`mapMinimaxSamplerPlugins`). The existing page sends the shift set there; the
new page leaves the server's automatic one, and a user who set the shift with plugins on, turned them off and renders
under 30 steps hears another shift. Also, the low-step shift cannot be set without turning plugins on. Following the
existing page is widening the parameter's `when` on MiniMax-Music3.

## 15. The tiled decoder on Stable Audio 3

Entry `sa3-tiled-decoder`. The concept offers the tiled decoder pedal on Stable Audio 3; the existing page shows it only
on engines with ACE-Step's plugin registry, so on Stable Audio 3 it cannot be turned on. Whether the server runs it there
was not established. Following the existing page is taking `sa3` off the pedal's engines.

## 16. Schedule shapes on MiniMax-Music3, Stable Audio 3 and HeartMuLa

Entries `schedule-shape-*`. On those engines the existing page's plugin picker sends the bare schedule id ("power",
"beta", "composite") and has no shape sliders; the definitions pack the concept's shape into it ("power:2.00"), as
ACE-Step's page does. If the engine reads a bare id at the same defaults, the render is the same until a shape is set.
Following the existing page is taking those engines off the shape parameters and the transform's wire.

## 17. Which engines a pedal or unit is on

The validator now checks the subsystems the server declares (`capabilities().features`) against the definitions'
engines: StableStep, Whisper lyrics, Lyric timestamps and Forced alignment. One disagrees: the server declares Whisper
on Stable Audio 3 and the existing page shows it there; the definitions keep the concept's list (ACE-Step,
MiniMax-Music3, YuE2). Separately, HeartMuLa runs the shared post-processing chain with every job parameter but PP-VAE
and the spectral lifter (`heartmula/generate.ts`, `runPostProcessingChain`), and the existing page shows the Vocal
naturalizer and the Quality evaluator there; the definitions, following the concept, leave both off HeartMuLa, and
the naturalizer off Stable Audio 3. A user of the new page cannot turn them on there. Following the server is adding the
engines (and a line in the validator's `FEATURE_ASKED` goes).

## 18. Choices made meanwhile

These were the rebuild's to decide rather than the concept's, so they were decided in the way that is cheapest to undo;
none changes what is sent for a state the existing page can reach.

- **A number whose step is whole is stored whole.** A saved 12.5 steps loads as 13.
- **A range's two ends that arrive crossed are swapped.** The concept keeps them crossed.
- **Text that fails its check is not kept when a record loads**; while it is typed the field keeps it and says what is
  in force.
- **A saved revision holds every value it covers, defaults included**; factory revisions list only what they set.
- **A number carried from another engine is sent as it was set**, clamped to the engine's range but not put on its step
  grid (a 47-second length set on Stable Audio 3 reads 47 on ACE-Step, whose ruler steps by 5), as the existing page
  sends it. The grid applies when a value is set there.
- **Seeds are kept to 2^64-1**, as the application keeps them; ACE-Step, MiniMax-Music3 and Stable Audio 3 are sent a
  number, as the existing page sends them (`Number(seed)`, the nearest number past 2^53), YuE2 and HeartMuLa the decimal
  text.

## 19. Fields the existing page never sends

A parameter the existing page has no control for, whose field the existing page never sends on that engine (a new
parameter, such as YuE2's plan token limit if it were added), is sent by the definitions once it is changed. The request
comparison lists each such field and the parameter that sends it ("new fields the existing page never sends") rather than
failing; a field the existing page does send, sent wrongly, still fails. Today the list is empty: every field the
definitions send that the existing page does not is an exact entry naming its own question. A user sees the new control
and gets what it says. Following the existing page strictly would mean a new parameter is not sent until this is
answered, so each new knob would ship inert.

## 20. The song menu where nothing has room for it

On a short desk at 1920x1080 (and at 1536 and 1280 with the page at its end) a song card sits under the go bar with
neither 340px above it (the go bar is in the way) nor below it (the window ends), and the page cannot move. The concept
then puts the menu beside its card, over the go bar, and shows all seven items. Its own later rule (F-28, pending there)
says the menu never covers the go bar, so the rebuild stands it beside its card between the go bar and the window's
foot, 293px of 340, and its last item scrolls in (design decision P6-4, RC-04). Covering the go bar for the moment the menu
is open would show everything at once; keeping the go bar clear keeps Generate and the queue in sight. Meanwhile: the
go bar stays clear.

## 21. Saved seeds and parameter profiles: the page's or the server's?

The existing frontend keeps saved seeds on the server (`ui/src/components/global-bar/SeedManagerDrawer.tsx`: `/seeds`,
with favourites and a random pick) and parameter profiles per backend (`ui/src/services/api.ts`: `/profiles`). The
rebuild keeps saved seeds as the Seed module's state (`stars`, a list kept with the desk in this browser, every change
an undo step: design decision 32 in [09](09-architecture.md)), and has no profiles: its revisions (`hotstep.create.workflows`)
play their part, also in this browser.

- Following the server: the seeds and the revisions (or profiles) become services of the boundary (`services/index.ts`),
  each change a request, so they are shared between browsers and with the existing pages; Undo of a saved seed then
  posts again (as question 6 asks for engine state), or saved seeds leave Undo.
- Keeping the page's model: nothing to wire in package 10, but a seed saved on the existing page is not seen here, and
  one saved here is lost with the browser's storage.

Meanwhile the page keeps both in the browser, as the concept does. Moving the seeds is one service and the Seed part's
list read from it (the `stars` state then holds nothing, as `recentRefs` does for the reference tracks); moving the
revisions is the workflows store's reads and writes (`store/stored.ts`) behind a service.

## 22. The Concept notes in the real application

The Notes button opens the concept's design notes (`src/app/i18n/words/notes.ts`, about 14 KB: what the page does and
why, the review's remarks). They are right for the concept's published page, where they explain the redesign; in the
real application they are text to translate into five languages that a user of HOT-Step does not need. Choices: drop
the button and the drawer in package 10 (one `fill` and one file), keep them behind a developer setting, or rewrite
them as the page's own help. Meanwhile they ship as the concept's.

## 23. The page's language

The application has five locales and a language detector (`ui/src/i18n/index.ts`); the rebuild runs i18next on an
instance of its own (so it never takes over the application's) fixed to English, and every word it says is a message
(question 7 is where the tables go). Package 10 needs one line to follow the application's language
(`i18n.changeLanguage` when the application's changes), once the tables exist: which language the page follows is the
application's choice, and whether a missing table falls back to English or to the application's own words is yours.
