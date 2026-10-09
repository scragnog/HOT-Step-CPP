# Create's control dictionary

This page documents every control the Create workflow (`POST /api/generate`,
plain text2music) sends, for the three backends: `ace`, `yue2`,
`minimax-m3`. It covers defaults, backend applicability, the omitted vs
explicit wire idiom, and task modes. Insta-Gen, Cover, Repaint, Lego and
Lyric Studio batches are documented on
[frontend-studios.md](frontend-studios.md); YuE2 preset effective-application
policy is on [frontend-presets.md](frontend-presets.md). Arbitrary backend
and plugin parameters (`backendParams`, `pluginParams`) stay free-form —
see "Extension channels" below rather than an enumerated field list.

## Two submission shapes

`POST /api/generate` accepts two body shapes, both ending up at the same
`submitGeneration` (`routes/generate.ts:323`), which the durable audio queue
also calls (`services/audioQueue/intentQueue.ts:40`) rather than
re-implementing submission:

- **Compact `generation-intent/1`** (`contracts/generation.ts:4-12`,
  `generationIntentSchema`): `{ contract, params, input?, settings? }`.
  `routes/generate.ts:356-365` detects this with `isGenerationIntent()`
  (`contracts/generation.ts:16-19`) and expands it through
  `resolveGenerationIntent()` (`services/generation/intent.ts:18-45`), which
  merges `params` over `GENERATION_DEFAULTS`
  (`services/generation/defaults.ts:4-144`, 144 fields) and runs
  `resolveGenerationParams()` (`defaults.ts:151-413`) — the conditional
  emission layer described below.
- **Legacy flat body** (no `contract` key — the Create UI's actual wire
  format, built by `resolveCreateIntent()`,
  `services/generation/resolve/resolveIntent.ts:76-142`): skips
  `GENERATION_DEFAULTS` entirely and goes straight to
  `buildEnvelope()` (`services/generation/envelope.ts:98-133`). For the ACE
  backend this body is shaped by `translateParams.ts` (camelCase UI params to
  the snake_case `AceRequest` the engine expects); YuE2 and MM3 have their
  own mappers (`backends/yue2/generate.ts`, `backends/minimax/generate.ts`).

Both shapes are accepted today and this slice does not change that:
`generationIntentSchema.params` stays `z.record(z.string(), z.unknown())`
(generation.ts) and the new `generationControlsSchema`
(`contracts/generationControls.ts`) only documents the shape — every field
optional, unknown keys pass through — so it never rejects anything the old
schema already accepted. See `generationControls.test.ts` for the parity
check against the real golden fixture set
(`services/generation/fixtures/batch1-intent-golden.json`, 31 recorded
Create/Written-song intents across all three backends).

## Backend applicability (core capabilities)

From each backend's `capabilities().core`
(`backends/types.ts:28-47`, `BackendCoreCapabilities`):

| Core control | ace | yue2 | minimax-m3 |
|---|---|---|---|
| `duration` | `{ max: 600, auto: false }` — a target, not a ceiling | `{ max: 360, auto: true, editable: false }` — AR/NAR stop on their own terminator | `{ max: 300, auto: true, editable: false }` — accepted on the wire but advisory; the planner LM ends the render itself |
| `bpm` | `true` — first-class field | `true`, but composed into the style caption's trained tail, not a discrete field | `false` — no structured tempo field; lives in the Structured Caption prose |
| `keyscale` | `true` | `true`, same caption-composition path as `bpm` | `false` |
| `negativePrompt` | `true` | not declared | `false` — no uncond prompt slot |
| `batch.max` | multi-batch | — | `1` — batching is exposed as the `mm3Takes` extension (1-4 takes) instead of `batchSize` |
| `seed` | `true` | `true` | `true` |

A backend's `operations` list (below) gates which `taskType` values it
accepts at all; everything else in the dictionary is either read or ignored
per-field, never validated against backend identity by the shared code
(`backends/types.ts:135-137`: shared code must only branch on declared
capability flags, never on a backend id, outside that backend's own module).

## Task modes

`operation` (wire key `taskType`) is the task-mode axis.
`GenerationOperation` (`backends/types.ts:225-227`) is
`'text2music' | 'cover' | 'repaint' | 'lego' | 'extract' | 'complete' | string`.
Each backend declares which it accepts:

- **ace**: `text2music, cover, cover-nofsq, repaint, lego, extract, complete`.
- **yue2**: `text2music` only (a YuE2 "cover" is a `yue2Cover`/`yue2Abc`
  marker object carried inside a `text2music` request, not a separate
  operation).
- **minimax-m3**: `text2music` only.

`buildEnvelope()` (`envelope.ts:85-90`) throws `unsupported_operation` when
the resolved operation isn't in the active backend's list — a `repaint`
submitted against YuE2 or MM3 is rejected at submit, not silently coerced.

For Create specifically, `resolveCreateIntent()` never sets `taskType` —
unlike the Written Song path (`resolveWrittenSongIntent`, same file), which
forces `params.taskType = 'text2music'`. Create relies on each backend's own
`resolveRequest()` default: `taskType` absent or empty falls back to
`'text2music'` (`backends/ace/index.ts:154-155`,
`backends/minimax/index.ts:126`, same pattern in `backends/yue2/index.ts`).
In practice every Create request is `text2music`; cover/repaint/lego/extract
are reachable only through the separate Insta-Gen/Cover/Repaint/Lego studios
documented in frontend-studios.md.

**Representative resolution:** an ACE Create submission with no `taskType`
key resolves to `operation: 'text2music'`
(`backends/ace/index.ts:154-155`) and `translateParams.ts:137`
(`if (params.taskType) req.task_type = params.taskType;`) sends nothing to
the engine — the engine's own default task type applies.

## Omitted vs explicit: two different idioms

**1. Declared backend extension knobs** (`backendParams`, compact-intent path
only). `resolveGenerationIntent()` (`intent.ts:30-42`) validates each key
against the active backend's `capabilities().extensions`
(`backends/types.ts:148-158`, `BackendExtensionParam`) and fills any
*declared* key the caller omitted with that extension's own `default`. An
unknown key throws (`intent.ts:34`); legacy flat bodies have no equivalent
check and whatever the backend's own mapper reads simply applies its own
default. YuE2 declares ~54 such knobs, MM3 ~24 — see each backend's
`extensions` array for the live, authoritative list rather than duplicating
it here (that is exactly the kind of drift this slice exists to avoid).

**2. Node-side conditional emission** (`resolveGenerationParams()`,
`defaults.ts:220-412`, compact-intent path; mirrored by hand in
`translateParams.ts` for the legacy ACE path). A field is written to the
outgoing request only when it departs from the engine's own default, or when
its governing toggle is on — otherwise it is omitted so the engine applies
its own default instead of Node re-asserting it:

- `cfgCutoffRatio` only when `< 1.0` (`defaults.ts:278`).
- `cacheRatio` only when `> 0` (`defaults.ts:280`).
- `denoiseStrength`/`lssStrength` only when `> 0` (`defaults.ts:324,327`).
- Every `stableStep*` field gated on `postProcessingEnabled && stableStepOn`
  (`defaults.ts:345-392`).
- `apgMomentum`/`apgNormThreshold` only when `guidanceMode === 'apg'`
  (`defaults.ts:289-290`); `storkSubsteps` only for the `stork2`/`stork4`
  solvers (`defaults.ts:285`).

Opt-out toggles use `!== false` so an *absent* value means ON — the UI never
sends a field for an untouched control, so "missing" must mean "default
behavior," which for these specific fields is enabled:
`stableStepPreserveDynamics`/`stableStepSeedFollowsDit`
(`defaults.ts:355,370`), MM3's `mm3ReuseAr`/`mm3RequireEnding`/`mm3MinLength`
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
(`resolveIntent.ts:121-125`), unconditionally overriding whatever numeric
value the UI held — MM3's duration is advisory only (see the core
capability table above), so Create never lets a stale UI value reach it.

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
| `adapterStack` | `[]` | ace | multi-entry (Advanced mode); supersedes `adapter` when non-empty and `advancedAdapters` is true |
| `adapterStackMode` | `'blend'` | ace | `'blend'` normalizes stack scales to `adapterStackBudget`; `'sum'` sends them as-is |
| `adapterStackBudget` | `0.75` | ace | blend-mode only, 2+ adapters |
| `adapterSectionAlignAt`, `adapterSectionIsolation` | `0.55`, `0.5` | ace | per-section masking, 2+ adapter stack only |
| `adapterMode` | `'runtime'` | ace | `'merge' \| 'runtime' \| 'runtime_lowrank'`; forced to `'runtime'` server-side when timestep windows are present |
| `adapterRuntimeQuant` | `'bf16'` | ace | only sent in a runtime mode, or when timestep windows force one |
| `adapterMergeLowVram` | `false` | ace | merge mode only |
| `adapterGroupScales` | all `1.0` except `time_embed`/`proj_in` `0.0` | ace | single-adapter only |
| `rebaseSource`, `rebaseBeta` | `''`, `0.75` | ace | only with an adapter loaded and a source chosen |
| `noAdapterRender` | `false` | ace | reference render without the adapter, meaningful only with one loaded |
| `lmAdapter`, `lmAdapterScale` | `''`, `1.0` | ace, yue2 | planner-LM LoRA; independent of the DiT adapter stack. YuE2 holds this as engine state (a model-manager POST), not a per-request field |

**Representative resolution:** a Create request with one adapter set in
Simple mode (`adapter: 'foo.safetensors'`, no `adapterStack`) resolves
`primary = 'foo.safetensors'` and sends `loraPath`/`loraScale` with
`adapterMode: 'merge'` unless a 2+ adapter stack or timestep window forces
runtime (`defaults.ts:184,250`).

### Core DiT sampling

| Field | Default | Backends | Note |
|---|---|---|---|
| `inferenceSteps` | `12` | ace, yue2 (via extensions), minimax-m3 (via `mm3Steps`, default `30`) | core knob on ace, backend-declared extension elsewhere |
| `guidanceScale` | `9.0` | ace | MM3's analogue is `mm3CfgFlow` (default `1.7`) |
| `shift` | `3.0` | ace, yue2, minimax-m3 | |
| `inferMethod` | `'euler'` | ace, minimax-m3 | solver choice; gates `storkSubsteps`/`beatStability`/`frequencyDamping`/`temporalSmoothing` |
| `scheduler` | `'linear'` | ace, yue2, minimax-m3 | |
| `guidanceMode` | `'apg'` | ace, minimax-m3 | gates `apgMomentum`/`apgNormThreshold` |
| `cfgCutoffRatio`, `lmCfgCutoffRatio` | `1.0` (no-op) | ace | sent only when `< 1.0` |
| `cacheRatio` | `0` (off) | ace | sent only when `> 0` |
| `ditSlidingWindow` | unset (full attention) | ace | `0` = full attention on all layers; omitted uses the model's own `128` |

### Seeding and batching

| Field | Default | Backends | Note |
|---|---|---|---|
| `seed` | `42` | ace, yue2, minimax-m3 | overridden by a fresh random value when `randomSeed` is true |
| `randomSeed` | `true` | ace, yue2, minimax-m3 | |
| `lmSeed`, `lmSeedFollowsDit` | `42`, `true` | ace | when following, `lm_seed` is left unset entirely so the engine's own DiT-seed fallback ties it, including to a *randomized* DiT seed |
| `batchSize` | `1` | ace, yue2, minimax-m3 | MM3 instead exposes `mm3Takes` (1-4) as its batching control |

### Cover/repaint axis (task-mode-gated, ACE-only on Create's wire path)

| Field | Default | Backends | Note |
|---|---|---|---|
| `taskType` | unset → `'text2music'` | ace, yue2, minimax-m3 | see Task modes above |
| `audioCoverStrength` | derived from `lmCodesMode`/`lmCodesStrength`/`lmCodesSteps` | ace | only sent when `!skipLm && < 1.0` |
| `coverNoiseStrength`, `coverNoiseMethod` | unset | ace | cover/repaint operations only |
| `repaintingStart`, `repaintingEnd`, `seedStrength` | unset | ace | repaint operation only |
| `trackName` | unset | ace | stem/layer target for lego/extract |

### LM / planner sampling

| Field | Default | Backends | Note |
|---|---|---|---|
| `skipLm` | `false` | ace | skips the planner LM stage entirely |
| `useCotCaption` | `true` | ace | chain-of-thought caption expansion |
| `skipLrc` | `false` | ace | sent only when true |
| `lmTemperature`, `lmCfgScale`, `lmTopK`, `lmTopP` | `0.8`, `2.2`, `0`, `0.92` | ace | `lmTopK: 0` is a meaningful value (disables top-k), not "unset" — the parity test covers this explicitly |
| `lmRepPenalty`, `lmRepWindow`, `lmRepMode` | `1.1`, `64`, `'presence'` | ace | window/mode sent only when `lmRepPenalty > 1.0`; auto-bumped to `1.05` when an LM adapter is loaded and the caller didn't set one (`translateParams.ts:161-167`) |
| `lmDryBase`, `lmDryMinLen` | `1.75`, `3` | ace | only with `lmRepMode === 'dry'` |
| `lmNegativePrompt` | `'NO USER INPUT'` | ace | |

### Post-processing chain

All gated on `postProcessingEnabled` (default `true`), and each stage has
its own enable toggle gating its sub-fields:

| Field | Default | Backends | Note |
|---|---|---|---|
| `spectralLifterEnabled` + `slDenoiseStrength`, `slNoiseFloor`, `slHfMix`, `slTransientBoost`, `slShimmerReduction` | `false`; `0.3`, `0.1`, `0.0`, `0.0`, `6.0` | ace | |
| `masteringEnabled` + `masteringReference`, `timbreReference` | `false`; `''`, `false` | ace, yue2, minimax-m3 (shared post-process, per `features.postProcess`) | reads the WAV's own sample rate, not tied to one backend's native rate |
| `vocalNaturalizerEnabled` + `gainOffsetDb`, `naturalizeAmount`, `natVibratoRate`, `natVibratoDepth`, `natFormantStrength`, `natMetallicReduction`, `natQuantizationMask`, `natTransitionSmooth` | `false`; `0`, `0.5`, `4.5`, `1.0`, `1.0`, `1.0`, `0.0`, `1.0` | ace, yue2, minimax-m3 | |
| `ppVaeReencode`, `ppVaeBlend` | `false`, `0.0` | ace | ACE-VAE-coupled; no analogue on backends without that VAE |
| `coverArtEnabled`, `coverArtSubject` | `false`, `''` | ace, yue2, minimax-m3 | |
| `qualityEvalEnabled`, `qualityEvalTarget` | `false`, `'unmastered'` | ace, yue2, minimax-m3 | |
| `lufsEnabled`, `lufsPreset`, `lufsTarget`, `lufsCeilingDb` | `false`, `'spotify'`, `-14`, `-1` | ace, yue2, minimax-m3 | the final normalizer, independent of `masteringEnabled` |
| `whisperLyricsEnabled` + `whisperModel`, `whisperLanguage`, `whisperBeamSize`, `whisperIsolateVocals` | `false`; `''`, `'auto'`, `5`, `false` | ace, yue2, minimax-m3 (`features.whisper`) | backend-agnostic — runs on the rendered file, not the model |
| `yue2AlignLyrics` | `false` | yue2 only (`features.forcedAlignment`) | sent unconditionally on every backend; only YuE2 reads it. YuE2's DiT has no cross-attention for live lyric timestamps, so it force-aligns after the fact instead (`features.lyricTimestamps: false`, `forcedAlignment: true`) |

### StableStep (SA3) refinement

Gated on `postProcessingEnabled && stableStepOn` (both default
`false`/`false`); SA3 is natively 44.1kHz and backend-agnostic
(`features.stableStep`).

| Field | Default | Note |
|---|---|---|
| `stableStepStrength` | `0.3` | |
| `stableStepBackend` | `'auto'` | `'gguf'` sent only when explicitly chosen |
| `stableStepAdapters` | `[]` | filtered to enabled, non-zero-scale entries |
| `stableStepPreserveDynamics`, `stableStepSeedFollowsDit` | `true`, `true` | opt-out idiom: `!== false` |
| `stableStepVocalPpVae`, `stableStepVocalTrimDb` | `false`, `0` | opt-in only — omitted means "leave the vocals alone" |
| `stableStepBlendMode` | `'off'` | gates `stableStepCrossoverHz`/`stableStepCrossoverWidthHz` (`'crossover'`) or `stableStepMix` (`'mix'`) |
| `stableStepSeed` | `4242` | only sent when `stableStepSeedFollowsDit === false` |
| `stableStepSteps`, `stableStepSolver`, `stableStepScheduler`, `stableStepGuidanceMode`, `stableStepGuidanceScale` | `8`, `''`, `''`, `''`, `1.0` | only sent when each departs from its own default (SA3 sampler routing — an untouched StableStep stays byte-identical to the pre-feature pingpong/euler path) |

### Metadata (legacy flat-body names)

| Field | Default | Backends | Note |
|---|---|---|---|
| `bpm` | unset | ace, yue2 (composed into caption) | |
| `keyScale` | unset | ace, yue2 (composed into caption) | |
| `timeSignature` | unset | ace | numerator only sent (`'3/4'` → `'3'`) |
| `vocalLanguage` | `'en'` (every UI path defaults it) | ace | always sent, normalized to an ISO code; `'unknown'` sent explicitly preserves "let the LM decide" |

## Extension channels (arbitrary, stay extensible)

Two free-form maps carry everything not in the shared dictionary above —
neither is enumerated here because enumerating them is what each backend's
own manifest is for, and copying the list into this page would drift the
first time a backend adds a knob:

- **`backendParams`** (compact-intent path) / backend-prefixed top-level
  keys like `yue2*`/`mm3*` (legacy path) — the active backend's declared
  `capabilities().extensions`. `intent.ts:30-42` validates each key against
  that manifest and fills declared defaults for omitted ones; an unknown key
  throws. The live list is `GET /api/capabilities` → `extensions`, not a
  static table — YuE2 currently declares roughly 54 such keys, MM3 roughly
  24.
- **`pluginParams`** — keyed `"<plugin>:<key>"`, read generically by the Lua
  sampler plugin registry (`features.samplerPlugins`); the same map backs
  both ACE's native plugin dropdown and MM3's generic one.

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
