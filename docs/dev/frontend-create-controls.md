# Create's control dictionary

This page documents every control the Create workflow (`POST /api/generate`,
plain text2music; the UI calls the studio Custom-Gen) sends, for the four
backends: `ace`, `yue2`, `minimax-m3` and `sa3`. It covers defaults, backend
applicability, the omitted vs explicit wire idiom, and task modes. Auto-Gen,
Cover, Repaint, Stem Builder's layer render and Lyric Studio batches are
documented on [frontend-studios.md](frontend-studios.md); YuE2 preset
effective-application policy is on [frontend-presets.md](frontend-presets.md).
Arbitrary backend and plugin parameters (`backendParams`, `pluginParams`)
stay free-form: see "Extension channels" below rather than an enumerated field
list.

## Defaults discovery

`GET /api/generation-controls/defaults` returns the shared defaults object as
JSON. It requires the same authentication as other Create endpoints. The bundled
UI imports the synchronous constant from `server/src/contracts/generationControls.ts`;
the intent resolver uses that same object. Explicit intent parameters override
these values, and backend-specific conditional emission still applies. Existing
locally saved UI values take precedence over the defaults.

## Two submission shapes

`POST /api/generate` accepts two body shapes, both ending up at the same
`submitGeneration` (`routes/generate.ts:344`), which the durable audio queue
also calls (`services/audioQueue/intentQueue.ts:40`) rather than
re-implementing submission. Either shape may carry `backendOverride`, which
runs that one job on another registered backend without changing the active
one ([Generate request](api-contracts.md#generate-request)). "The backend"
below means that one, or the active backend when there is none.

- **Compact `generation-intent/1`** (`contracts/generation.ts:20-29`,
  `generationIntentSchema`): `{ contract, backendOverride?, params, input?,
  settings? }`. `routes/generate.ts:392-416` detects this with
  `isGenerationIntent()` (`contracts/generation.ts:34-37`) and expands it
  through `resolveGenerationIntent()` (`services/generation/intent.ts:18-53`),
  which merges `params` over `GENERATION_DEFAULTS`
  (`contracts/generationControls.ts`) and runs
  `resolveGenerationParams()` (`defaults.ts:155-422`), the conditional
  emission layer described below. That layer returns a fixed set of keys, so
  a `params` key it does not emit is dropped. `input` is then laid over the
  result unchanged: the request's content (caption, lyrics, duration, the
  metadata fields, `taskType`, title) and any other field this layer does not
  emit belong in `input`.
- **Legacy flat body** (no `contract` key; the Create UI's actual wire
  format): skips `GENERATION_DEFAULTS` entirely and goes straight to
  `buildEnvelope()` (`services/generation/envelope.ts:99-157`). The bundled
  UI builds the engine params in the browser with `getGlobalParams()`
  (`ui/src/stores/globalParamsStore.ts`, a hand-kept mirror of
  `resolveGenerationParams()`), adds Create's content fields, and posts the
  result as a Create intent to `POST /api/resolve/preview`. There
  `resolveCreateIntent()`
  (`services/generation/resolve/resolveIntent.ts:76-145`) applies the
  caption source, compose helpers and the duration rule, and the UI submits
  the returned `request` unchanged
  ([Caption and content resolution](api-contracts.md#caption-and-content-resolution)).

On ACE, `translateParams.ts` then maps either shape (camelCase UI params to
the snake_case `AceRequest` the engine expects); YuE2, MM3 and SA3 have their
own mappers (`backends/yue2/generate.ts`,
`backends/minimax/generate.ts`, `backends/sa3/generate.ts`).

Both shapes are accepted today and this slice does not change that:
`generationIntentSchema.params` stays `z.record(z.string(), z.unknown())`
(generation.ts) and the new `generationControlsSchema`
(`contracts/generationControls.ts`) only documents the shape — every field
optional, unknown keys pass through — so it never rejects anything the old
schema already accepted. See `generationControls.test.ts` for the parity
check against the real golden fixture set
(`services/generation/fixtures/batch1-intent-golden.json`, 31 recorded
Create/Written-song intents across `ace`, `yue2` and `minimax-m3`; none for
`sa3`).

## Backend applicability (core capabilities)

From each backend's `capabilities().core`
(`backends/types.ts:28-47`, `BackendCoreCapabilities`):

| Core control | ace | yue2 | minimax-m3 | sa3 |
|---|---|---|---|---|
| `duration` | `{ max: 600, auto: false }`: a target, not a ceiling | `{ max: 360, auto: true, editable: false }`: AR/NAR stop on their own terminator | `{ max: 300, auto: true, editable: false }`: accepted on the wire but advisory; the planner LM ends the render itself | `{ max: 600, auto: false }`: the output length exactly; a missing value or one under 1 renders 30 seconds |
| `bpm` | `true`: first-class field | `true`, but composed into the style caption's trained tail (unless the `yue2CaptionBpmKey` extension is `false`), not a discrete field | `false`: no structured tempo field; lives in the Structured Caption prose | `false` |
| `keyscale` | `true` | `true`, same caption-composition path as `bpm` | `false` | `false` |
| `negativePrompt` | `true` | `false` | `false`: no uncond prompt slot | `false`: no CFG uncond branch |
| `batch.max` | `8` | the engine's `max_lm_batch` (1 when it reports none); the songs per job come from the `yue2BatchSize` extension, falling back to `batchSize` | `1`: batching is exposed as the `mm3Takes` extension (1-4 takes) instead of `batchSize` | `1` |
| `seed` | `true` | `true` | `true` | `true` |

A backend's `operations` list (below) gates which operations it runs at all;
everything else in the dictionary is either read or ignored
per-field, never validated against backend identity by the shared code
(`backends/types.ts:135-137`: shared code must only branch on declared
capability flags, never on a backend id, outside that backend's own module).

## Task modes

`operation` (wire key `taskType`) is the task-mode axis.
`GenerationOperation` (`backends/types.ts:226-228`) is
`'text2music' | 'cover' | 'repaint' | 'lego' | 'extract' | 'complete' | string`.
Each backend declares which it accepts:

- **ace**: `text2music, cover, cover-nofsq, repaint, lego, extract, complete`.
- **yue2**: `text2music` only (a YuE2 "cover" is a `yue2Cover`/`yue2Abc`
  marker object carried inside a `text2music` request, not a separate
  operation).
- **minimax-m3**: `text2music` only.
- **sa3**: `text2music` only.

`buildEnvelope()` (`envelope.ts:86-91`) throws `unsupported_operation` when
the resolved operation isn't in the backend's list, so a `repaint`
submitted against MM3 is rejected at submit. The YuE2 and SA3 resolvers
ignore `taskType` and always resolve `text2music`: without `backendOverride`,
a `repaint` sent to either renders as plain text2music. With
`backendOverride`, a `taskType` other than the resolved operation is refused
with the same error (`envelope.ts:131-139`).

For Create specifically, `resolveCreateIntent()` does not set `taskType`
itself, unlike the Written Song path (`resolveWrittenSongIntent`, same file),
which forces `params.taskType = 'text2music'`. The bundled UI already puts
`taskType: 'text2music'` in Create's content params (`createContentParams`,
`ui/src/components/create/createContent.ts`). A body without one relies on
each backend's own `resolveRequest()`: `taskType` absent or empty falls back
to `'text2music'` (`backends/ace/index.ts:154-155`,
`backends/minimax/index.ts:126`), and YuE2 and SA3 never read it. In practice
every Create request is `text2music`. The other operations come from other
studios: Cover (`cover`, `cover-nofsq`), Repaint and Song Builder's later
sections (`repaint`) and Stem Builder (`lego`), documented in
[frontend-studios.md](frontend-studios.md) and
[frontend-library.md](frontend-library.md). Stem Separator's Extract sends
`extract` to the engine directly, not through `/api/generate`.

**Representative resolution:** an ACE submission with no `taskType` key
resolves to `operation: 'text2music'` (`backends/ace/index.ts:154-155`), and
`translateParams.ts:155` (`if (params.taskType) req.task_type =
params.taskType;`) sends no `task_type`, so the engine's own default task
type applies. Create's own `taskType: 'text2music'` reaches the engine as
`task_type: 'text2music'`.

## Omitted vs explicit: two different idioms

**1. Declared backend extension knobs** (`backendParams`, compact-intent path
only). `resolveGenerationIntent()` (`intent.ts:30-42`) validates each key
against the backend's `capabilities().extensions`
(`backends/types.ts:148-158`, `BackendExtensionParam`) and fills any
*declared* key the caller omitted with that extension's own `default`. An
unknown key throws (`intent.ts:34`), and so does a value of the wrong kind
(`intent.ts:35`): a toggle takes a boolean, a text knob a string, a select
one of its options and a number a finite value inside `min`/`max`. Both are
400. Legacy flat bodies have no equivalent check, and whatever the backend's
own mapper reads simply applies its own default. YuE2 declares 86 such
knobs, MM3 24 and SA3 5; ACE declares none, because its solver, scheduler,
guidance and adapter knobs come from `/api/plugins`. See each backend's
`extensions` array for the live, authoritative list rather than duplicating
it here (that is exactly the kind of drift this slice exists to avoid).

**2. Node-side conditional emission** (`resolveGenerationParams()`,
`defaults.ts:224-421`, compact-intent path; mirrored by hand in the
browser's `getGlobalParams()` for the legacy body, while `translateParams.ts`
forwards whatever arrives without these checks). A field is written to the
outgoing request only when it departs from the engine's own default, or when
its governing toggle is on. Otherwise it is omitted, so the engine applies
its own default instead of Node re-asserting it:

- `cfgCutoffRatio` only when `< 1.0` (`defaults.ts:283`).
- `cacheRatio` only when `> 0` (`defaults.ts:285`).
- `denoiseStrength`/`lssStrength` only when `> 0` (`defaults.ts:333,336`).
- Every `stableStep*` field gated on `postProcessingEnabled && stableStepOn`
  (`defaults.ts:354-401`).
- `apgMomentum`/`apgNormThreshold` only when `guidanceMode === 'apg'`
  (`defaults.ts:294-295`); `storkSubsteps` only for the `stork2`/`stork4`
  solvers (`defaults.ts:290`), and `beatStability`/`frequencyDamping`/
  `temporalSmoothing` only for `jkass_fast` (`defaults.ts:291-293`).

Opt-out toggles use `!== false` so an *absent* value means ON — the UI never
sends a field for an untouched control, so "missing" must mean "default
behavior," which for these specific fields is enabled:
`stableStepPreserveDynamics`/`stableStepSeedFollowsDit`
(`defaults.ts:364,379`), MM3's `mm3ReuseAr`/`mm3RequireEnding`/`mm3MinLength`
(each read with `!== false` in `backends/minimax/generate.ts`).

**Representative resolution (duration, the clearest cross-layer example):**
`GENERATION_DEFAULTS` has no raw `duration` key at all — only
`durationBuffer: 15` and `autoTrimEnabled: false` (`defaults.ts:69-70`), used
only when auto-trim is on. In the legacy ACE path,
`translateParams.ts:65-68` checks `params.duration > 0` (not truthy —
the UI's "Auto" sentinel is `-1`, itself truthy, and a `> 0` guard is what
stops "Auto" plus the 15s trim buffer from asking the engine for a
14-second song). `resolveCreateIntent()` additionally forces
`params.duration = -1` whenever the target engine is MM3
(`resolveIntent.ts:124-127`), unconditionally overriding whatever numeric
value the UI held: MM3's duration is advisory only (see the core
capability table above), so Create never lets a stale UI value reach it. The
bundled UI also sends `-1` on YuE2, whose duration control is hidden. SA3
takes the duration as the output length and renders 30 seconds for `-1`.

## The control dictionary

Grouped by area; types, defaults and the emission rule for each live in
`contracts/generationControls.ts` (schema) and `services/generation/defaults.ts`
(defaults + emission). "Backends" lists where the field is actually read —
derived by checking which fields each backend's `generate.ts`/mapper
consumes, not just where the field is declared.

### Model routing and adapters

| Field | Default | Backends | Note |
|---|---|---|---|
| `ditModel`, `lmModel`, `vaeModel`, `embeddingModel` | `''` (engine default) | ace | `vaeModel` dropped if it ends `.onnx` (stale profile) |
| `adapter` / `loraPath`, `adapterScale` / `loraScale` | `''`, `1.0` | ace | single-slot (Simple mode) |
| `adapterStack` | `[]` | ace | multi-entry (Advanced mode); supersedes `adapter` when non-empty and `advancedAdapters` is true. Sent as `loraStack`, a one-entry stack in Simple mode |
| `adapterStackMode` | `'blend'` | ace | `'blend'` normalizes stack scales to `adapterStackBudget`; `'sum'` sends them as-is |
| `adapterStackBudget` | `0.75` | ace | blend-mode only, 2+ adapters |
| `adapterSectionAlignAt`, `adapterSectionIsolation` | `0.55`, `0.5` | ace | per-section masking, 2+ adapter stack only |
| `adapterMode` | `'runtime'` | ace | `'merge' \| 'runtime' \| 'runtime_lowrank'`; `'merge'` is sent when no adapter is set. Forced to `'runtime'` server-side when timestep windows are present, or lyric section directives with a 2+ adapter stack |
| `adapterRuntimeQuant` | `'bf16'` | ace | only sent in a runtime mode, or when timestep windows force one |
| `adapterRuntimeDora` | `false` | ace | sent only when `true`, as `adapter_runtime_dora` |
| `adapterMergeLowVram` | `false` | ace | meant for merge mode; sent only when it is on, an adapter is set and `adapterMode` is not `'runtime'` |
| `adapterGroupScales` | all `1.0` except `time_embed`/`proj_in` `0.0` | ace | sent whenever an adapter is set |
| `rebaseSource`, `rebaseBeta` | `''`, `0.75` | ace | only with an adapter loaded and a source chosen |
| `noAdapterRender` | `false` | ace | reference render without the adapter, meaningful only with one loaded |
| `lmAdapter`, `lmAdapterScale` | `''`, `1.0` | ace | planner-LM LoRA; independent of the DiT adapter stack. YuE2 does not read it: its AR/NAR adapter pair is the backend's model selection (`POST /api/backends/models`), overlaid per request by `yue2Pick`. MM3's is `mm3LmAdapter` |
| `advancedAdapters`, `adapterFolder`, `lmAdapterFolder`, `adaptersOpen` | `false`, `''`, `''`, `false` | none | form state: `advancedAdapters` picks the stack over `adapter`; the others are never sent |

**Representative resolution:** a Create request with one adapter set in
Simple mode (`adapter: 'foo.safetensors'`, no `adapterStack`) resolves
`primary = 'foo.safetensors'` and sends `loraPath`/`loraScale` with the
selected `adapterMode` (default `'runtime'`); `'merge'` goes out only when
no adapter is set (`defaults.ts:188,254`).

### Core DiT sampling

| Field | Default | Backends | Note |
|---|---|---|---|
| `inferenceSteps` | `12` | ace | the step count elsewhere is a backend-declared extension: `yue2OdeSteps` (default `32`), `mm3Steps` (default `30`), `sa3Steps` (default `8`) |
| `guidanceScale` | `9.0` | ace; sa3 with `samplerPluginsEnabled`, above 1 only | MM3's analogue is `mm3CfgFlow` (default `1.7`) |
| `shift` | `3.0` | ace | MM3's is the `mm3FlowShift` extension |
| `inferMethod` | `'euler'` | ace; minimax-m3 and sa3 with `samplerPluginsEnabled` | solver choice; gates `storkSubsteps`/`beatStability`/`frequencyDamping`/`temporalSmoothing` |
| `scheduler` | `'linear'` | ace; minimax-m3 and sa3 with `samplerPluginsEnabled` | YuE2's NAR solver and scheduler are its own `yue2NarSolver`/`yue2NarScheduler` extensions |
| `guidanceMode` | `'apg'` | ace; minimax-m3 and sa3 with `samplerPluginsEnabled` | gates `apgMomentum`/`apgNormThreshold` |
| `storkSubsteps` | `10` | ace | sent only for the `stork2`/`stork4` solvers |
| `beatStability`, `frequencyDamping`, `temporalSmoothing` | `0.25`, `0.4`, `0.13` | ace | sent only for the `jkass_fast` solver |
| `apgMomentum`, `apgNormThreshold` | `0.75`, `2.5` | ace; minimax-m3 reads `apgNormThreshold` with `samplerPluginsEnabled` | sent only when `guidanceMode === 'apg'` |
| `cfgCutoffRatio`, `lmCfgCutoffRatio` | `1.0` (no-op) | ace | sent only when `< 1.0` |
| `cacheRatio` | `0` (off) | ace | sent only when `> 0` |
| `ditSlidingWindow` | unset (the model's own `128`) | ace | `0` = full attention on all layers. The compact path never emits it (the feature is parked), so only a legacy body carries it |

### Latent, denoise, DCW and auto-trim (ACE-only)

None of these are read by `backends/yue2/generate.ts`,
`backends/minimax/generate.ts` or `backends/sa3/generate.ts` (checked by grep
against all three files): they are ACE-specific DiT-latent/VAE-level knobs
with no cross-backend analogue.
MM3 explicitly does not apply the auto-trim buffer concept at all
(`backends/minimax/generate.ts:416-417`: *"The ACE duration buffer
(autoTrimEnabled + durationBuffer) is deliberately NOT added: nothing trims
it back off on this path."*).

| Field | Default | Note |
|---|---|---|
| `dcwEnabled` | `false` | sent unconditionally; gates the rest |
| `dcwMode` | `'double'` | `'low' \| 'double' \| 'high' \| 'pix'`, sent only when `dcwEnabled` |
| `dcwLowScaler`, `dcwHighScaler` | `0.2`, `0.2` | combined into a single `dcwScaler` wire value (`dcwMode === 'high'` uses the high scaler × 0.02, otherwise the low scaler × 0.05); `dcwHighScaler` is also sent on its own, but only in `'double'` mode |
| `latentShift` | `0.0` | sent only when `!== 0` |
| `latentRescale` | `1.0` | sent only when `!== 1` |
| `customTimesteps` | `''` | sent only when non-empty |
| `denoiseStrength` | `0.0` | post-VAE spectral denoiser; sent only when `> 0` |
| `denoiseSmoothing`, `denoiseMix` | `0.7`, `0.25` | sent only when `denoiseStrength > 0` |
| `lssStrength` | `0.0` | Latent Spectral Suppressor (pre-VAE latent channel gate); sent only when `> 0` |
| `lssVarThresh`, `lssDcRemove` | `0.15`, `true` | sent only when `lssStrength > 0` |
| `autoTrimEnabled` | `false` | sent only when true |
| `durationBuffer`, `autoTrimFadeMs` | `15`, `2000` | sent only when `autoTrimEnabled` |

### Seeding and batching

| Field | Default | Backends | Note |
|---|---|---|---|
| `seed` | `42` | ace, yue2, minimax-m3, sa3 | overridden by a fresh random value when `randomSeed` is true |
| `randomSeed` | `true` | ace, yue2, minimax-m3, sa3 | |
| `lmSeed`, `lmSeedFollowsDit` | `42`, `true` | ace | when following, `lm_seed` is left unset entirely so the engine's own DiT-seed fallback ties it, including to a *randomized* DiT seed |
| `batchSize` | `1` | ace; yue2 when `yue2BatchSize` is absent | MM3 and SA3 render one track per job; MM3's batching control is `mm3Takes` (1-4) |
| `seedB`, `seedBlend`, `seedBlendEnd` | unset | ace, sa3 | seed breeding, not a Create control. ACE sends `seed_b` when `seedB` is 0 or more, with `seed_blend` (clamped to 0-1) when given and `seed_blend_end` for a blend that changes over the render. SA3 reads `seedB` and `seedBlend`, which defaults to 0.5 when `seedB` is set |

### Cover/repaint axis (task-mode-gated, ACE-only on Create's wire path)

| Field | Default | Backends | Note |
|---|---|---|---|
| `taskType` | unset → `'text2music'` | ace, minimax-m3 | see Task modes above; Create sends `'text2music'`, and YuE2 and SA3 do not read it |
| `audioCoverStrength` | derived from `lmCodesMode`/`lmCodesStrength`/`lmCodesSteps` | ace | only sent when `!skipLm && < 1.0` |
| `lmCodesStrength`, `lmCodesMode`, `lmCodesSteps` | `1.0`, `'ratio'`, `6` | ace | never sent themselves: `'ratio'` mode uses the strength, `'steps'` converts the step count against `inferenceSteps` (1.0 at or above it) |
| `coverNoiseStrength`, `coverNoiseMethod` | unset | ace | cover/repaint operations only |
| `repaintingStart`, `repaintingEnd`, `seedStrength` | unset | ace | repaint operation only |
| `evictLm`, `vaeChunk`, `batchCfg` | unset | ace | engine switches: `evict_lm: true` and `vae_chunk` when truthy, `batch_cfg` as `1` or `0` when present |
| `trackName` | unset | ace | stem/layer target for lego/extract |

### LM / planner sampling

| Field | Default | Backends | Note |
|---|---|---|---|
| `skipLm` | `false` | ace | skips the planner LM stage entirely |
| `useCotCaption` | `true` | ace | chain-of-thought caption expansion |
| `skipLrc` | `false` | ace | sent only when true |
| `lmTemperature`, `lmCfgScale`, `lmTopK`, `lmTopP` | `0.8`, `2.2`, `0`, `0.92` | ace | `lmTopK: 0` is a meaningful value (disables top-k), not "unset" — the parity test covers this explicitly |
| `lmRepPenalty`, `lmRepWindow`, `lmRepMode` | `1.1`, `64`, `'presence'` | ace | all three sent only when `lmRepPenalty > 1.0`. With an LM adapter loaded and no penalty on the wire (one of 1.0 or less is dropped by that rule), ACE uses `1.05` (`translateParams.ts:185`) |
| `lmDryBase`, `lmDryMinLen` | `1.75`, `3` | ace | only with `lmRepMode === 'dry'` |
| `lmMinP`, `lmCfgMinP`, `lmCfgLogprob`, `lmCfgRescale` | `0`, `0`, `false`, `0` | ace | code-phase min-p and CFG refinements (engine `lm-cfg-extras.h`); each sent only when non-default, and all four key the LM cache |
| `lmNegativePrompt` | `'NO USER INPUT'` | ace | |

### Post-processing chain

All but the last two rows are gated on `postProcessingEnabled` (default
`true`), and each stage has its own enable toggle gating its sub-fields.
The `sa3` backend runs the same chain with the Spectral Lifter, PP-VAE and
StableStep forced off, and treats every render as instrumental:

| Field | Default | Backends | Note |
|---|---|---|---|
| `spectralLifterEnabled` + `slDenoiseStrength`, `slNoiseFloor`, `slHfMix`, `slTransientBoost`, `slShimmerReduction` | `false`; `0.3`, `0.1`, `0.0`, `0.0`, `6.0` | ace | |
| `masteringEnabled` + `masteringReference`, `timbreReference`, `timbreAudioPath` | `false`; `''`, `false`, `''` | ace, yue2, minimax-m3, sa3 (shared post-process, per `features.postProcess`) | reads the WAV's own sample rate, not tied to one backend's native rate. `timbreAudioPath` is folded INTO the `timbreReference` wire value (sent as that string) rather than its own key; without it, `timbreReference` is sent as `true` only when mastering is also on and a reference is set |
| `postprocessEnabled` + `postprocessPlugin` | `false`; `''` | ace; sa3 with `samplerPluginsEnabled` | replaces the built-in tiled VAE decoder with a Lua plugin; sent only when both are set AND `postProcessingEnabled`. YuE2 drops it. Note: the legacy ACE path (`translateParams.ts:341-343`) sends `postprocessPlugin` whenever it is non-empty, without checking `postprocessEnabled`: a documented idiom difference between the two submission shapes, not a bug introduced here |
| `vocalNaturalizerEnabled` + `naturalizeAmount`, `natVibratoRate`, `natVibratoDepth`, `natFormantStrength`, `natMetallicReduction`, `natQuantizationMask`, `natTransitionSmooth` | `false`; `0.5`, `4.5`, `1.0`, `1.0`, `1.0`, `0.0`, `1.0` | ace, yue2, minimax-m3 | skipped for an instrumental render |
| `gainOffsetDb` | `0` | ace, yue2, minimax-m3, sa3 | the pre-VST gain stage; sent whenever it is non-zero, independent of the Vocal Naturalizer |
| `ppVaeReencode`, `ppVaeBlend` | `false`, `0.0` | ace | ACE-VAE-coupled; no analogue on backends without that VAE |
| `coverArtEnabled`, `coverArtSubject` | `false`, `''` | ace, yue2, minimax-m3 | SA3 does not run it |
| `qualityEvalEnabled`, `qualityEvalTarget` | `false`, `'unmastered'` | ace, yue2, minimax-m3, sa3 | `qualityEvalTarget` is `'unmastered' \| 'mastered' \| 'both'` |
| `lufsEnabled`, `lufsPreset`, `lufsTarget`, `lufsCeilingDb` | `false`, `'spotify'`, `-14`, `-1` | ace, yue2, minimax-m3, sa3 | the final normalizer, independent of `masteringEnabled`. `lufsPreset` is form state and is not sent; the wire carries `lufsTarget` and `lufsCeilingDb` |
| `whisperLyricsEnabled` + `whisperModel`, `whisperLanguage`, `whisperBeamSize`, `whisperIsolateVocals` | `false`; `''`, `'auto'`, `5`, `false` | ace, yue2, minimax-m3 (`features.whisper`) | backend-agnostic: runs on the rendered file, not the model. `whisperLyricsEnabled` is sent whether or not post-processing is on. SA3 declares `features.whisper` but its runner never reads these fields |
| `yue2AlignLyrics` | `false` | yue2 only (`features.forcedAlignment`) | sent whenever it is true, whatever the backend; only YuE2 reads it. YuE2's DiT has no cross-attention for live lyric timestamps, so it force-aligns after the fact instead (`features.lyricTimestamps: false`, `forcedAlignment: true`) |

### StableStep (SA3) refinement

Gated on `postProcessingEnabled && stableStepOn` (`postProcessingEnabled`
defaults ON, `stableStepOn` defaults OFF, `defaults.ts:92,114`); SA3 is
natively 44.1kHz and backend-agnostic (`features.stableStep`, declared by
ace, yue2 and minimax-m3; the `sa3` backend does not run it).

| Field | Default | Note |
|---|---|---|
| `stableStepStrength` | `0.3` | |
| `stableStepBackend` | `'auto'` | `'gguf'` sent only when explicitly chosen |
| `stableStepAdapters` | `[]` | filtered to enabled, non-zero-scale entries |
| `stableStepPreserveDynamics`, `stableStepSeedFollowsDit` | `true`, `true` | opt-out idiom: `!== false` |
| `stableStepVocalPpVae`, `stableStepVocalTrimDb` | `false`, `0` | opt-in only — omitted means "leave the vocals alone" |
| `stableStepBlendMode` | `'off'` | gates `stableStepCrossoverHz`/`stableStepCrossoverWidthHz` (`250`, `200`; `'crossover'`) or `stableStepMix` (`1.0`; `'mix'`) |
| `stableStepSeed` | `4242` | only sent when `stableStepSeedFollowsDit === false` |
| `stableStepSteps`, `stableStepSolver`, `stableStepScheduler`, `stableStepGuidanceMode`, `stableStepGuidanceScale` | `8`, `''`, `''`, `''`, `1.0` | only sent when each departs from its own default (SA3 sampler routing: an untouched StableStep stays byte-identical to the pre-feature pingpong/euler path); `stableStepGuidanceScale` also needs a guidance mode and a value above 1. With a solver, scheduler or guidance mode chosen, the shared `pluginParams` map rides along as `stableStepPluginParams` |

### Metadata (legacy flat-body names)

On the compact path these fields, like the caption and lyrics, travel in
`input`.

| Field | Default | Backends | Note |
|---|---|---|---|
| `bpm` | unset | ace, yue2 (composed into caption) | |
| `keyScale` | unset | ace, yue2 (composed into caption) | |
| `timeSignature` | unset | ace | numerator only sent (`'3/4'` → `'3'`) |
| `vocalLanguage` | `'en'` (every UI path defaults it) | ace | always sent, normalized to an ISO code; `'unknown'` sent explicitly preserves "let the LM decide" |
| `negative_prompt` | unset | ace | snake_case on the wire (the dictionary's `negativePrompt` is the UI's name for it); Create sends it only when non-empty |

## Extension channels (arbitrary, stay extensible)

Two free-form maps carry everything not in the shared dictionary above.
Neither is enumerated here because enumerating them is what each backend's
own manifest is for, and copying the list into this page would drift the
first time a backend adds a knob:

- **`backendParams`** (compact-intent path) / backend-prefixed top-level
  keys like `yue2*`/`mm3*`/`sa3*` (legacy path), plus the unprefixed
  `samplerPluginsEnabled` that MM3 and SA3 declare: the backend's declared
  `capabilities().extensions`. `intent.ts:30-42` validates each key against
  that manifest and fills declared defaults for omitted ones; an unknown key
  throws. The live list is `GET /api/capabilities?backend=<id>` →
  `extensions`, not a static table: YuE2 currently declares 86 such keys,
  MM3 24 and SA3 5. A backend field that is not declared there, such as
  MM3's `mm3LmAdapter*` keys (which the bundled UI keeps beside the declared
  ones), cannot ride in a compact intent's `backendParams`; send it in
  `input`.
- **`pluginParams`**: keyed `"<plugin>:<key>"`, read generically by the Lua
  sampler plugin registry (`features.samplerPlugins`). The same map backs
  ACE's native plugin dropdown and MM3's generic one, and feeds StableStep's
  plugin picks. SA3 forwards only the keys of the plugins it is using, and
  YuE2 forwards its NAR plugins' keys but none for the global postprocess
  plugin.

`contracts/generationControls.ts` models both as `z.record(...)` so neither
channel requires a schema change when a backend or plugin adds a field.

## Verification

- `server/src/contracts/generationControls.test.ts`: the 31-branch golden
  fixture set (`services/generation/fixtures/batch1-intent-golden.json`,
  covering Create and Written Song across ace/yue2/minimax-m3) parses under
  both `generationIntentSchema` (existing) and `generationControlsSchema`
  (new) — parity, not a tightening. Separately covers the zero/false values
  that must survive (`lmTopK: 0`, `skipLm`/`dcwEnabled`/`masteringEnabled`/
  `mm3ReuseAr: false`, `yue2NarCacheRatio: 0`) and unknown/future keys
  passing through uncut.
- `npx tsc --noEmit` and `node --import tsx --test "src/**/*.test.ts"` from
  `server/`.
- `node tools/docs/check-docs.mjs` from the repo root.
