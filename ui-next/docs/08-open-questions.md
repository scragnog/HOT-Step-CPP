# Open questions for the maintainer

## 1. Could the server supply the parameter list?

Mostly, it already does. Four of the five engines declare their knobs in `capabilities().extensions`
(`server/src/services/backends/<engine>/index.ts`, served by `GET /api/capabilities`), each with `key`, `type`, `label`,
`hint`, `min`, `max`, `step`, `default`, `options` and `visible_when`, plus a `group` and `section`. MiniMax-Music3 declares
27, YuE2 86, Stable Audio 3 5, HeartMuLa 14. That is the shape a definition's parameter needs, minus presentation.

Compared with the definitions here, 126 parameter-engine pairs map to a declared extension. Of those:

- 123 agree on range, step, default and options once the definitions store real values (this comparison found two option
  lists, YuE2's lie curve and overdrive shape, and two steps, MiniMax-Music3's window hop and YuE2's composer CFG rescale,
  which the definitions now follow).
- 3 differ because the server works them out at run time: `yue2MaxMinutes` has no static maximum, and `yue2Variations` and
  `yue2BatchSize` are declared with a maximum of 1 that the engine raises.

What is missing, judged from `server/src` as it is:

1. **ACE-Step has no manifest.** Its knobs (`inferenceSteps`, `guidanceScale`, the LM, latent, adapter and
   post-processing fields, about 100 of them) are `GENERATION_DEFAULTS` in `contracts/generationControls.ts`: defaults
   only, every field typed `z.unknown()` on purpose, no ranges. Their ranges live in the UI's sliders. Giving them the
   extension shape would mean one table of `{ key, type, min, max, step, default, options }` that `GENERATION_DEFAULTS`,
   `resolveGenerationParams()` and the UI all read, instead of the three hand-kept copies the repository's own docs warn
   about. The post-processing fields (shared by every engine) would want the same table.
2. **Fields read but not declared.** `backends/minimax/generate.ts` reads `mm3LmAdapterMode`, `mm3LmAdapterScale` and
   its five group dials, `mm3LmAdapterScalePlan`, `mm3LmMergeGpu`, `mm3LmAdapterTrigger`, `mm3DitBackend` and
   `mm3DitPrecision`; YuE2 reads `yue2LmAdapterTrigger`. None is in a manifest, so a compact intent cannot carry them
   (`docs/dev/frontend-create-controls.md` says as much for the adapter keys).
3. **Engine state is not a parameter.** YuE2's adapter dials are per slot in `POST /api/backends/models`, not request
   fields; HeartMuLa's and MiniMax-Music3's models are buckets. A manifest of those (bucket, value type, range) would let
   the Models and Adapters definitions be checked the same way.
4. **A few ranges are computed.** Those three would need the server to report the engine's current limit
   (`batch.max` in `core` already does this for one of them).

With that, a definition would keep only presentation: which part, its words beyond the server's label, its tier, its
format, Quick's choices, `owner` and `stored`. The validator would read a captured `GET /api/capabilities` (or a test
fixture of it) and fail when a definition's `wire` names a key the server does not declare, or states a range or default
that differs. A new engine parameter could then not be out of step with the UI: it would either have a definition or be
listed as undrawn.

Decision needed: whether to add the ACE-Step manifest and declare the undeclared fields, and whether the definitions
should then drop `range`, `default` and `options` for declared keys (taking them from the server at run time and from a
fixture in tests), or keep them and let the validator compare.

## 2. Keys: the concept's or the real ones?

The definitions keep the concept's keys and record the real names in `wire`. One control can be five real fields (Steps),
so some UI key is needed; but for the 1:1 cases (`pp.ss.str` is always `stableStepStrength`) the real name could be the
key. Keeping the concept's keys keeps the concept usable as a side-by-side reference and its behaviour suite comparable.
Which do you want the rebuild's store to use?

## 3. The mismatches

Each needs a decision, listed in [04-parameter-names.md](04-parameter-names.md):

- `den.on`, `lss.on`: keep a switch in the UI that sends strength 0 when off, or drop the switch as the real app has?
- `pp.vst.on`: the concept's chain switch has no server counterpart. Keep it as a UI bypass that disables every plugin?
- Build by section: the concept's takes per section and section mode do not exist in the real `Yue2Build`, and the real
  AR strength default is 0.6 where the concept says 1. Which is the design?
- Takes on ACE-Step: the concept allows 9, the server's `batch.max` is 8.
- `ad.trigcap` (MiniMax-Music3 "add trigger to caption"): no switch found. Dropped, or is it somewhere not searched?
- The 28 real parameters with no concept control (ACE-Step's planner adapter strength, embedding model, independent LM
  seed, solver substeps, StableStep's own sampler, YuE2's ending and token-budget controls): which belong on the desk, and
  at which tier?

## 4. Where values live in the rebuild

Today `globalParamsStore.ts` holds one field per real name, scopes five fields per backend
(`BACKEND_SCOPED_FIELDS`) and assembles the request in `getGlobalParams()`. The definitions assume a store keyed by
definition key, with engine stashes driven by `owner`, and a wire layer that turns it into the request. Is a new store
acceptable for `ui-next`, keeping `getGlobalParams()`'s output identical (the wire layer must reproduce its conditional
emission rules, such as sending `cfgCutoffRatio` only below 1), or should the rebuild write into the existing store?

## 5. Option values for server-listed choices

Solvers, schedules, guidance modes and models come from the server. The definitions store the server's ids
(`euler`, `apg`) and use `''` for "Native" (the engine's own sampling) on engines where plugins are optional. Is `''`
right for Native everywhere, and are the plugin ids stable enough to save in revisions?

## 6. Engine state and undo

Picking a YuE2 adapter or a model is a `POST` that evicts and reloads weights. The concept makes it an undo step like any
other. Should undoing it post again (and pay the reload), or should engine-state changes be outside undo?

## 7. Translations

The key scheme in [05](05-what-the-definition-drives.md) derives keys from definitions and keys node words by their text.
Is that acceptable next to the app's hand-written keys, and should the extraction write into `en.json` directly or into a
separate namespace file?

## 8. Thumbnails

The drawings were captured from the concept as SVG files. A new module needs a drawing; there is no automatic one. Is a
generated fallback (a frame with one knob per number) wanted for modules added before someone draws theirs?
