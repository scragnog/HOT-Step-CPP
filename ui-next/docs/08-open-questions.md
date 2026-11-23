# Open questions for the maintainer

Each question says what the definitions do meanwhile, so the format works whichever way it is decided, and what would
change.

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
   renderer and precision (`mm3LmAdapter*`, `mm3LmMergeGpu`, `mm3DitBackend`, `mm3DitPrecision`); YuE2 reads `yue2Abc`.
   A compact intent's `backendParams` cannot carry them.
3. **Engine state is not a parameter.** YuE2's adapter dials are per slot in `POST /api/backends/models`; HeartMuLa's and
   MiniMax-Music3's models are buckets.
4. **A few ranges are computed at run time** (YuE2's longest length, noise variations and batch size).

If the server supplied them, a definition would keep only presentation (part, words, tier, format, Quick's choices,
`owner`, `stored`) and take range, default and options from the server at run time and from the fixture in tests. What
would change: `range`, `default` and `options` become optional for an `ext:` parameter, and the validator's fixture
checks become the source. Nothing else in the format.

Decision needed: whether to add the ACE-Step manifest and declare the undeclared fields, and whether the definitions
should then drop `range`, `default` and `options` for declared keys.

## 2. Keys: the concept's or the real ones?

The definitions keep the concept's keys and say the real names in `wire`. One control can be five real fields (Steps),
so some key of the desk's own is needed; for the 1:1 cases (`pp.ss.str` is always `stableStepStrength`) the real name
could be the key. Meanwhile the concept's keys stay, which keeps the concept usable as a side-by-side reference. Either
answer is cheap to apply before anything is saved, and after that too: a renamed key lists its old one in `was`, and
saved records keep loading.

## 3. The mismatches

Each needs a decision; [04-parameter-names.md](04-parameter-names.md) lists them with what the real app does.
Meanwhile the definitions keep the concept's design and say so in the wire.

- `den.on`, `lss.on`: the switches stay; each gates its strength's wire, sending 0 while off. Dropping them would leave the
  strength alone deciding, as the real app has it.
- `pp.vst.on`: the concept's chain switch has no server counterpart. Keep it as a UI bypass that disables every plugin?
- Build by section: the concept's takes per section and section mode do not exist in the real `Yue2Build`, and the real
  plan strength default is 0.6 where the concept says 1.
- Takes on ACE-Step: the concept allows 9, the server's `batch.max` is 8. The definitions stop at 8 (the existing page
  sends 8 for 9). Following the concept is one number, and the server would need to take 9.
- APG on Stable Audio 3 and HeartMuLa: the concept shows the momentum (and on Stable Audio 3 the norm threshold); nothing
  those engines run reads them. Keep the controls (and add the server side), or take those engines off the two
  parameters (two lines).
- The concept's ranges and steps against the existing UI's (the stack's combined strength to 2 against 4, and nine steps):
  the definitions follow the concept, and each wire's note gives the UI's. Following the UI is one number per parameter.
- The real parameters with no concept control ([04](04-parameter-names.md), "The real app has it"): which belong on the
  desk, and at which tier? [06-how-to.md](06-how-to.md) shows the planner's minimum-P and the song-ending controls as
  worked examples.

## 4. Where values live in the rebuild

Today `globalParamsStore.ts` holds one field per real name, scopes five fields per backend (`BACKEND_SCOPED_FIELDS`) and
assembles the request in `getGlobalParams()`, a hand-kept mirror of the server's `resolveGenerationParams()`. The
definitions assume a store keyed by definition key, with engine stashes driven by `owner`, and they build the request by
handing the server's own resolver its input ([05](05-what-the-definition-drives.md), "What is sent"), so the rebuild does
not restate `getGlobalParams()`'s emission rules at all.

What remains to decide is whether the other studios (Cover, Repaint, Lyric Studio, the assistant actions) keep reading
`globalParamsStore`, which would mean two stores holding the post-processing chain until they move, and how existing
saved profiles (`ui/src/utils/paramProfiles.ts`, raw store fields) and "Reuse settings" come in: each wire reads its real
field back (and each transform has a `read`), so both can be derived rather than written as a third mapping. Meanwhile
nothing in `ui-next` depends on the answer.

## 5. Option values for server-listed choices

Solvers, schedules, guidance modes and models come from the server. The definitions store the server's ids (`euler`,
`apg`, `dpm3m`) and use `''` for "Native" (the engine's own sampling) on engines where plugins are optional, and for
"Engine default" on the model rows, where the concept shows the first installed file instead. Is `''` right for both, and
are the plugin ids stable enough to save in revisions? If an id changes, `was` on an `include` option keeps old records
loading; for an id only the server lists, the validator's fixture shows the change on the next `npm run fixtures`.

## 6. Engine state and undo

Picking a YuE2 adapter or a model is a `POST` that evicts and reloads weights. The concept makes it an undo step like any
other. Should undoing it post again (and pay the reload), or should engine-state changes be outside undo?

## 7. Translations

The key scheme in [05](05-what-the-definition-drives.md) derives keys from definitions and keys node words by their text.
Is that acceptable next to the app's hand-written keys, and should the extraction write into `en.json` directly or into a
separate namespace file?

## 8. Thumbnails

The drawings were captured from the concept. A new pedal's can be drawn from its definition (`node tools/thumb.ts
<pedal>`); a new module still needs a drawing by hand, and a new unit too (its object is its own). Is a generated fallback
for modules wanted (a frame with one knob per number) for modules added before someone draws theirs?

## 9. HeartMuLa's scalar decoder

The existing UI sends HeartMuLa's scalar decoder choice with every request, so a queued job keeps it; the concept's
engine panel has no row for it, so no definition carries it, and the request comparison shows that one difference.
Meanwhile the job uses the scalar decoder selected when it runs. Adding a row is one parameter in `models.ts` with the
wire `['model:heartmula.scalar_decoder', 'body:heartmulaScalarDecoderModel']`.

## 10. The defaults the concept and the existing page start on differently

The request comparison lists each with its reason (`tools/fixtures/request-differences.json`); these need a decision:

- **Shift on ACE-Step.** The concept starts on Auto, which is -1, the existing slider's own Auto: the engine then uses its
  shift, 3.0 on turbo models and 1.0 on base ones. The existing page starts at 3.0, so on a base model the default
  render differs. Meanwhile the definitions follow the concept. Starting at 3.0 is one line (`auto: false` on ACE-Step's
  line, default 3).
- **Expand wildcards on YuE2.** The concept offers it there, and the server honours it; the existing page hides it on
  YuE2 and sends it off. Meanwhile the definitions send it. Following the existing page is one line (YuE2 off the
  parameter's engines).
- **Lua plugins on MiniMax-Music3, Stable Audio 3 and HeartMuLa**, solver, schedule and guidance untouched: the concept
  starts on Native (the engine's own loop), the existing page on ACE-Step's euler, linear and APG. Meanwhile Native.
- **Guidance on Stable Audio 3 with Lua plugins on**: the concept's 1, the existing page's 9 carried from ACE-Step.
- **MiniMax-Music3's renderer**: the definitions add Auto and start on it, as the server does when nothing is chosen; the
  concept starts on GGML.

## 11. Choices made meanwhile

These are the rebuild's to decide rather than the concept's, so they were decided, in the way that is cheapest to undo:

- **A module not on the desk sends nothing** (its values are kept), so the engine's defaults apply. **A pedal or side
  unit not on the board is off**: its switch sends its off value. For every pedal and unit but one the off value is the
  real default, so nothing changes; for Lyric timestamps, which is on by default, it means a desk without the unit asks
  for no timestamps (`skipLrc: true`), where the existing page makes them unless told not to. The alternative for the
  board is one branch in `realState`.
- **A number whose step is whole is stored whole.** A saved 12.5 steps loads as 13. The alternative, an explicit field
  for whole numbers, is one field.
- **A range's two ends that arrive crossed are swapped.** The concept keeps them crossed.
- **Text that fails its check is not kept when a record loads**, as the concept does; while it is typed the field keeps
  it and says what is in force.
- **A saved revision holds every value it covers, defaults included**; factory revisions list only what they set. The
  alternative, saving only changed values as the concept does, is one filter where a revision is written.
- **Untouched values equal to the real default are not sent**, so an untouched desk sends what the existing UI sends.
  Sending every value is one line in `realState`.
