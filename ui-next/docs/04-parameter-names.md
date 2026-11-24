# Parameter names: the concept and the real application

The concept invented its own keys (`pp.ss.str`, `lm.temp`). The rebuild talks to the real server, so every parameter's
`wire` says where its value goes in the real application, per engine. This page explains where the names were found,
gives the lists of mismatches, and ends with the full table. The counts, the first list and the table are generated
from the definitions by `npm run docs`, and the validator fails when they are out of date.

## Where the names come from

| Channel | Where the real app keeps it | How it travels |
| --- | --- | --- |
| `param:` | `GENERATION_DEFAULTS` (`server/src/contracts/generationControls.ts`), held by the existing UI in `ui/src/stores/globalParamsStore.ts` | The input of `resolveGenerationParams()`, which makes the request's engine fields |
| `ext:` | Each backend's `capabilities().extensions` (`server/src/services/backends/<engine>/index.ts`), with key, type, range, default and options; or a field the backend's code reads without declaring it | `backendParams`, flattened into the request |
| `content:` | `CreateContent` in `ui/src/components/create/createContent.ts` (caption, lyrics, bpm, keyScale, duration...) | The input of `createContentParams()` |
| `body:` | Fields the existing UI sets on the request itself (HeartMuLa's frozen model choices) | Top-level request fields |
| `model:` | `GET/POST /api/backends/models` buckets per backend (ACE-Step's `ditModel` and friends are `param:`) | Engine state, not part of the request |
| `intent:` | The Create intent's `compose` (auto-expand, trigger word, Beat I/O) and `captionSource` (`server/src/contracts/resolution.ts`) | Beside the intent's params; the server runs the caption steps |
| `ui:` | Form state the browser keeps (`hs-*` keys), such as the LUFS preset or vocal gender | Not sent |

The engine ids in definitions are the server's: `ace`, `minimax-m3`, `yue2`, `sa3`, `heartmula` (`engines.ts`). The
concept's `mm3`, `yue` and `hm` are not used.

What each backend declares and reads is read from the server's source into `tools/fixtures/real-app.json` by
`npm run fixtures`: the extensions (MiniMax-Music3 declares 27, YuE2 86, Stable Audio 3 5, HeartMuLa 14), the
backend-prefixed fields each backend's code reads, the `GENERATION_DEFAULTS` fields each engine's code reads (directly,
or through a field `resolveGenerationParams()` makes from them), the plugin ids, the model buckets, and the existing
UI's slider for each `GENERATION_DEFAULTS` field. The validator checks every wire against it ([07](07-checking.md)).

## Counts

<!-- generated: counts (npm run docs) -->
- 40 definitions: 25 modules, 10 pedals, 5 side units, on 5 engines.
- 284 parameters; 515 parameter-engine pairs. Of those, 500 say where the value goes (content 41, intent 16, ui 11, model 19, body 4, param 281, ext 138): 438 as a plain name, 41 through wire fields (invert, scale, divide, map, auto, when, several fields), 21 through a named transform. 15 pairs, on 9 parameters, have no counterpart in the real app.
- 13 state entries; 73 parts in the inventory, 73 used; 8 transforms; 7 factory workflows and 11 factory revisions.
<!-- end generated -->

## The concept has it, the real app does not

<!-- generated: none (npm run docs) -->
| Key | Engines | What the real app does instead |
| --- | --- | --- |
| `ad.trigger` | yue2, hm | YuE2 composes its adapter's own trigger (yue2LmAdapterTrigger); the server drops a typed trigger word there. |
| `cap.ds` | mm3 | MiniMax-Music3 takes its caption tracks from a Lyric Studio handoff, not from a dataset dropdown. |
| `s0` | yue2 | On YuE2 the AR slot's strength is its two dials, Plan and Compose (ad.arp, ad.arc). |
| `den.on` | ace | No switch in the real app: the denoiser runs while denoiseStrength is above 0. This switch gates den.str's wire. |
| `lss.on` | ace | No switch in the real app: LSS runs while lssStrength is above 0. This switch gates lss.str's wire. |
| `yue.btakes` | yue2 | The real Build by section (Yue2Build) has no takes-per-section setting; takes come one at a time from the build session. |
| `yue.bmode` | yue2 | No section mode in the real Build by section. |
| `takes.n` | sa3, hm | One song per run. |
| `pp.vst.on` | ace, mm3, yue2, sa3, hm | No chain switch in the real app: the VST3 chain runs while any plugin in it is enabled (vstChainStore chainEnabled). |
<!-- end generated -->

### Close, but not the same

These have a real counterpart that differs in a way the request handles. Each is in its `wire`, and the note says why.

- **Units and scale**: StableStep strength and the refiner mix are percent in the concept and 0..1 in the real app
  (`divide: 100`); YuE2's longest length is seconds in the concept and minutes in the real app (`divide: 60`); the
  auto-trim fade is seconds against milliseconds (`scale: 1000`).
- **Inverted**: Plan the song first is `skipLm`; Keep this seed is `randomSeed`; Lyric timestamps is `skipLrc`.
- **Mapped**: the stack choice (`ad.adv`) is the boolean `advancedAdapters`; the merge VRAM choice (`ad.mq`) is the
  boolean `adapterMergeLowVram`; the normalizer's presets send their target (`lufsTarget`), the preset itself being form
  state.
- **Gated**: the denoiser and LSS switches send their strength while on and 0 while off.
- **Packed**: the schedule shapes (`sch.*`) and the schedule are one scheduler string, `'beta:0.50:0.70'`,
  `'power:2.00'`, `'composite:<a>+<b>:<x>:<split>'` (transform `scheduler`); the recording and its span are
  `yue2Continuation` (transform `continuation`).
- **Different type**: the retake seed and StableStep's fixed seed are numbers in the real app and text in the concept
  (transform `int`); YuE2's composer CFG is a text knob where empty means auto (transform `text`).
- **Different step or range**: the concept's step or range is kept and the wire's note gives the existing UI's: the
  stack's combined strength (UI to 4, concept to 2), the steps of the LM temperature, CFG rescale, latent shift and
  rescale, PP-VAE blend, the lifter's denoise and transient, and the naturalizer's amount; the codes steps (the UI's
  slider stops at the step count). MiniMax-Music3's window hop and YuE2's composer CFG rescale follow the server. YuE2's
  noise variations and batch size are declared with a maximum of 1 in the static manifest and raised by the engine.
  ACE-Step's takes go to 9 where `batch.max` is 8.
- **Read but not declared**: MiniMax-Music3's adapter (`mm3LmAdapter`), its mode, scales and merge switch, the trigger
  switch, the renderer and its TensorRT precision. A compact intent's `backendParams`
  cannot carry these ([08](08-open-questions.md) question 1).
- **Not sent, by design** (`ui:`): vocal gender (Compose caption writes it), the LUFS preset, and Build by section's
  condition and plan strengths (`Yue2Build` settings; the real plan strength default is 0.6, the concept's 1).
- **In the intent** (`intent:`): Expand wildcards, the trigger word and Beat I/O are the intent's `compose`; the caption
  source and dataset its `captionSource`, through the `mm3CaptionSource` and `yue2CaptionSource` transforms.
- **Read nowhere on some engines**: the APG momentum on Stable Audio 3 and HeartMuLa, and the APG norm threshold on
  Stable Audio 3; the concept shows them, nothing those engines run reads them.
- **Stood back, not hidden**: MiniMax-Music3's manifest hides the repetition mode and window while the penalty is 1,
  and YuE2's hides the Plan CFG and CADS sub-controls while they are off (`visible_when`); the definitions stand them
  back with `dim` instead, as the concept does. The server's `visible_when` is the one condition language the
  definitions could be checked against.
- **Per slot**: YuE2's adapter dials are kept per slot (AR and NAR, each with its own six) as engine state; the concept
  has one bank for both.

## The real app has it, the concept does not

From `GENERATION_DEFAULTS` (20): `lmAdapterScale` (the ACE-Step planner adapter's strength; the concept loads a planner
adapter but gives it no strength), `embeddingModel`, `adapterSectionIsolation`, `adapterFolder`, `lmAdapterFolder`,
`adaptersOpen` (form state), `pluginParams` and `backendParams` (the two free-form channels themselves), `lmSeed` (an
independent planner seed; the concept has only "LM seed follows"), `storkSubsteps`, `beatStability`,
`frequencyDamping`, `temporalSmoothing` (solver-specific, sent only for STORK and JKASS), and StableStep's own sampler:
`stableStepBackend`, `stableStepAdapters`, `stableStepSolver`, `stableStepScheduler`, `stableStepGuidanceMode`,
`stableStepGuidanceScale`.

From the backends' extensions (6, all YuE2): `yue2PlanMaxTokens`, `yue2SemMinTokens`, `yue2EndThreshold`,
`yue2EndBias`, `yue2EndBiasFrom`, `yue2EndBiasRamp`.

The Create form's `sourceLatentUrl` is the concept's Import latent: the Prompt's `latent` state.

From the model buckets (3): MiniMax-Music3's `cond` and `voc`, HeartMuLa's `scalar_decoder`. The existing UI sends
HeartMuLa's scalar decoder with the request ([08](08-open-questions.md) question 9).

From the app's settings (2): `triggerPlacement` and `triggerUseFilename` (`ace-settings`), which decide how adapter
trigger words are placed. They are settings of the app, not of the desk, and reach the resolver as its `settings`.

Not counted: the parameters of other studios that the backend mappers also read (MiniMax-Music3's `mm3A2a*`,
`mm3Guide*` and `mm3Force*` cover controls, YuE2's lyric schedule sub-knobs, the cover and repaint fields), which are
outside the Create page.

## The table

A real name is shown once when it is the same on every engine the parameter is on.

<!-- generated: names (npm run docs) -->
### Prompt (`prompt`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `prompt.text` | text | all | ace: `content:caption`<br>mm3: `content:caption` + `intent:captionSource` (via mm3CaptionSource)<br>yue2: `content:caption`<br>sa3: `content:caption`<br>hm: `content:caption` |
| `wild.auto` | boolean | ace, mm3, yue2, sa3 | `intent:compose.autoExpand` |
| `song.artist` | text | all | `content:artist` |
| `song.title` | text | all | `content:title` |
| `song.subject` | text | all | `content:subject` |
| `ad.trigger` | text | ace, mm3, yue2, hm | ace: `intent:compose.loraTrigger`. The server puts it at the start of the caption (composeCreateCaption).<br>mm3: `intent:compose.loraTrigger`. The server puts it at the start of the caption (composeCreateCaption).<br>yue2: none: YuE2 composes its adapter's own trigger (yue2LmAdapterTrigger); the server drops a typed trigger word there.<br>hm: none: HeartMuLa adds its adapter's own trigger (withHeartmulaTrigger); the existing Create page sends no typed trigger word there. |
| `cap.src` | choice | mm3, yue2 | mm3: `intent:captionSource` (via mm3CaptionSource)<br>yue2: `intent:captionSource` (via yue2CaptionSource) |
| `cap.ds` | choice | mm3, yue2 | mm3: none: MiniMax-Music3 takes its caption tracks from a Lyric Studio handoff, not from a dataset dropdown.<br>yue2: `intent:captionSource` (via yue2CaptionSource) |

### Lyrics (`lyrics`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `lyrics.text` | text | ace, mm3, yue2, hm | `content:lyrics` |
| `lyrics.instr` | boolean | ace, mm3, yue2, hm | `content:instrumental` |
| `lyrics.lang` | choice | ace, mm3 | ace: `content:vocalLanguage`<br>mm3: `content:vocalLanguage`. createContentParams sends it on every engine; only ACE-Step reads it. Compose caption writes it into Vocal Details on MiniMax-Music3. |
| `lyrics.gender` | choice | mm3 | `ui:hs-vocalGender`. Not sent: it reaches the model only through Compose caption writing it into Vocal Details (createContent.ts). |

### Models (`models`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `model.var` | choice | hm | `model:heartmula.variant` + `body:heartmulaVariant`. Sent with the request so a queued job keeps the models it was queued with. |
| `model.dit` | choice | all | ace: `param:ditModel`<br>mm3: `model:minimax-m3.dit`<br>yue2: `model:yue2.lm`<br>sa3: `model:sa3.dit`<br>hm: `model:heartmula.global` + `body:heartmulaGlobalModel`. Sent with the request so a queued job keeps the models it was queued with. |
| `model.lm` | choice | ace, mm3, hm | ace: `param:lmModel`<br>mm3: `model:minimax-m3.lm`<br>hm: `model:heartmula.local` + `body:heartmulaLocalModel`. Sent with the request so a queued job keeps the models it was queued with. |
| `model.x` | choice | mm3, hm | mm3: `model:minimax-m3.depth`<br>hm: `model:heartmula.flow` + `body:heartmulaFlowModel`. Sent with the request so a queued job keeps the models it was queued with. |
| `model.vae` | choice | ace, yue2 | ace: `param:vaeModel`<br>yue2: `model:yue2.vae` |

### Repaint (`repaint`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `rp.mode` | choice | ace | `ui:repaint.mode`. A field of the Repaint studio (studioWorkflows), not of the Create form. |
| `rp.xfade` | number | ace | `ui:repaint.crossfadeFrames`. A field of the Repaint studio (studioWorkflows), not of the Create form. |
| `rp.lyrics` | text | ace | `ui:repaint.lyrics`. The Repaint studio's own lyrics field (studioWorkflows). |

### Tempo, key & length (`tempo`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `tempo.bpm` | number | ace, mm3, yue2 | `content:bpm` (auto: 0). Auto is 0. createContentParams sends it on every engine: MiniMax-Music3 reads it only through Compose caption, YuE2 composes it into the caption tail unless yue2CaptionBpmKey is false. |
| `tempo.key` | choice | ace, mm3, yue2 | `content:keyScale`. Auto is the option "". Real values are "C major" style. Sent on every engine, as tempo.bpm. |
| `tempo.sig` | choice | ace | `content:timeSignature`. Auto is the option "". Only the numerator reaches the engine. |
| `tempo.dur` | number | ace, sa3, hm | `content:duration` (auto: -1). Auto is -1. Stable Audio 3 renders 30 s for -1. |
| `tempo.max` | number | yue2 | `ext:yue2MaxMinutes` (÷ 60). Real is minutes, step 0.5. |
| `tempo.incap` | boolean | yue2 | `ext:yue2CaptionBpmKey` |
| `tempo.beatio` | boolean | ace, mm3, sa3 | `intent:compose.beatIntro` |
| `tempo.beatbars` | choice | ace, mm3, sa3 | `intent:compose.introBars` |
| `tempo.trim` | boolean | ace | `param:autoTrimEnabled` |
| `tempo.trimbuf` | number | ace | `param:durationBuffer` |
| `tempo.trimfade` | number | ace | `param:autoTrimFadeMs` (× 1000). Real is milliseconds. |

### Avoid (`avoid`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `avoid.text` | text | ace | `content:negativePrompt`. Sent as negative_prompt, only when not empty. |

### Timbre reference (`timbre`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `timbre.str` | number | ace | `param:timbreReferenceStrength` |

### Adapters (`adapters`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `s0` | number | ace, mm3, yue2, hm | ace: `param:adapterScale` + `param:adapterStack[0].scale`<br>mm3: `ext:mm3LmAdapterScale`. Read by backends/minimax/generate.ts; not declared as an extension.<br>yue2: none: On YuE2 the AR slot's strength is its two dials, Plan and Compose (ad.arp, ad.arc).<br>hm: `ext:heartmulaAdapterStrength` |
| `s1` | number | ace, yue2 | ace: `param:adapterStack[1].scale`<br>yue2: `model:yue2.nar.scales.global` |
| `s2` | number | ace | `param:adapterStack[2].scale` |
| `ad.mode` | choice | ace, mm3 | ace: `param:adapterMode`. Real values merge, runtime, runtime_lowrank.<br>mm3: `ext:mm3LmAdapterMode`. Read by backends/minimax/generate.ts; not declared as an extension. |
| `ad.planner` | choice | ace | `param:lmAdapter` |
| `ad.dora` | boolean | ace | `param:adapterRuntimeDora` |
| `ag.self` | number | ace | `param:adapterGroupScales.self_attn` |
| `ag.cross` | number | ace | `param:adapterGroupScales.cross_attn` |
| `ag.mlp` | number | ace | `param:adapterGroupScales.mlp` |
| `ag.cond` | number | ace | `param:adapterGroupScales.cond_embed` |
| `ag.time` | number | ace | `param:adapterGroupScales.time_embed` |
| `ag.proj` | number | ace | `param:adapterGroupScales.proj_in` |
| `ad.att` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleAttn`. Read by backends/minimax/generate.ts; not declared as an extension.<br>yue2: `model:yue2.ar.scales.attn`. Engine state; each YuE2 slot (ar, nar) has its own six dials. The concept has one bank for both.<br>hm: `ext:heartmulaAdapterScaleAttn` |
| `ad.mlp` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleMlp`. Read by backends/minimax/generate.ts; not declared as an extension.<br>yue2: `model:yue2.ar.scales.mlp`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleMlp` |
| `ad.early` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleEarly`. Read by backends/minimax/generate.ts; not declared as an extension.<br>yue2: `model:yue2.ar.scales.early`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleEarly` |
| `ad.mid` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleMid`. Read by backends/minimax/generate.ts; not declared as an extension.<br>yue2: `model:yue2.ar.scales.mid`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleMid` |
| `ad.late` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleLate`. Read by backends/minimax/generate.ts; not declared as an extension.<br>yue2: `model:yue2.ar.scales.late`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleLate` |
| `ad.plan` | number | mm3 | `ext:mm3LmAdapterScalePlan`. Read by backends/minimax/generate.ts; not declared as an extension. |
| `ad.adv` | choice | ace | `param:advancedAdapters` (map simple → false, advanced → true) |
| `ad.stack` | choice | ace | `param:adapterStackMode` |
| `ad.sigma` | number | ace | `param:adapterStackBudget`. The existing UI's slider goes to 4; the concept stops at 2. |
| `ad.rq` | choice | ace | `param:adapterRuntimeQuant` |
| `ad.mq` | choice | ace | `param:adapterMergeLowVram` (map hq → false, low → true) |
| `ad.home` | choice | ace | `param:rebaseSource` |
| `ad.beta` | number | ace, yue2, hm | ace: `param:rebaseBeta`<br>yue2: `model:yue2.ar.scales.rebase`<br>hm: `ext:heartmulaAdapterRebase` |
| `ad.ref` | boolean | ace | `param:noAdapterRender` |
| `ad.align` | number | ace | `param:adapterSectionAlignAt` |
| `ad.arp` | number | yue2 | `model:yue2.ar.scales.plan`. The AR slot's planner strength (engine state, POST /api/backends/models). |
| `ad.arc` | number | yue2 | `model:yue2.ar.scales.global`. The AR slot's composer strength (engine state). |
| `ad.trigcap` | boolean | mm3 | `ext:mm3LmAdapterTrigger`. Read with !== false by backends/minimax/generate.ts: on puts the adapter's own trigger into the caption. Not declared as an extension. |
| `ad.gpu` | boolean | mm3 | `ext:mm3LmMergeGpu`. Read by backends/minimax/generate.ts; not declared as an extension. |

### Planner (`planner`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `lm.on` | boolean | ace | `param:skipLm` (inverted). skipLm is the opposite of Plan the song first. |
| `lm.cot` | boolean | ace | `param:useCotCaption` |
| `lm.cotmode` | choice | yue2 | `ext:yue2Cot` |
| `lm.preview` | boolean | yue2 | `ext:yue2PreviewScore` |
| `lm.temp` | number | ace | `param:lmTemperature`. The existing UI steps by 0.01; the concept by 0.05. |
| `lm.cfg` | number | ace | `param:lmCfgScale` |
| `lm.topk` | number | ace | `param:lmTopK` |
| `lm.topp` | number | ace | `param:lmTopP` |
| `lm.minp` | number | ace | `param:lmMinP` |
| `lm.plaus` | number | ace, yue2 | ace: `param:lmCfgMinP`<br>yue2: `ext:yue2CfgMinP` |
| `lm.resc` | number | ace, yue2 | ace: `param:lmCfgRescale`. The existing UI steps by 0.05; the concept by 0.01.<br>yue2: `ext:yue2CfgRescale`. The server declares steps of 0.05; the concept steps by 0.01 on both engines (RD-12). |
| `lm.logp` | boolean | ace, yue2 | ace: `param:lmCfgLogprob`<br>yue2: `ext:yue2CfgLogprob` |
| `lm.codes` | number | ace | `param:lmCodesStrength` |
| `lm.codesmode` | choice | ace | `param:lmCodesMode` |
| `lm.codesteps` | number | ace | `param:lmCodesSteps`. The existing UI's slider stops at the step count in force; the concept at 300. |
| `lm.neg` | text | ace | `param:lmNegativePrompt` |
| `lm.seedlock` | boolean | ace | `param:lmSeedFollowsDit` |
| `mm.temp` | number | mm3 | `ext:mm3LmTemperature` |
| `mm.topk` | number | mm3 | `ext:mm3LmTopK` |
| `mm.topp` | number | mm3 | `ext:mm3LmTopP` |
| `mm.seed` | text | mm3 | `ext:mm3ArSeed` |
| `yp.temp` | number | yue2 | `ext:yue2PlanTemperature` |
| `yp.topp` | number | yue2 | `ext:yue2PlanTopP` |
| `yp.topk` | number | yue2 | `ext:yue2PlanTopK` |
| `yc.temp` | number | yue2 | `ext:yue2SemTemperature` |
| `yc.topp` | number | yue2 | `ext:yue2SemTopP` |
| `yc.topk` | number | yue2 | `ext:yue2SemTopK` |
| `yc.minp` | number | yue2 | `ext:yue2SemMinP` |
| `yue.cfg` | number | yue2 | `ext:yue2CfgScale` (via text; auto: ""). A text knob in the real app: empty means the checkpoint's own. |
| `hm.temp` | number | hm | `ext:heartmulaTemperature` |
| `hm.topk` | number | hm | `ext:heartmulaTopK` |
| `hm.cfg` | number | hm | `ext:heartmulaCfgScale` |

### Repetition guard (`repetition`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `lm.rep` | number | ace | `param:lmRepPenalty` |
| `lm.repmode` | choice | ace | `param:lmRepMode` |
| `lm.win` | number | ace | `param:lmRepWindow` |
| `lm.drybase` | number | ace | `param:lmDryBase` |
| `lm.drymin` | number | ace | `param:lmDryMinLen` |
| `mm.rep` | number | mm3 | `ext:mm3LmRepPenalty` |
| `mm.repmode` | choice | mm3 | `ext:mm3LmRepMode` |
| `mm.win` | number | mm3 | `ext:mm3LmRepWindow` |
| `yp.rep` | number | yue2 | `ext:yue2PlanRepPenalty` |
| `yp.win` | number | yue2 | `ext:yue2PlanRepWindow` |
| `yc.rep` | number | yue2 | `ext:yue2SemRepPenalty` |
| `yc.win` | number | yue2 | `ext:yue2SemRepWindow` |

### Sampler (`sampler`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `samp.steps` | number | all | ace: `param:inferenceSteps`<br>mm3: `ext:mm3Steps`<br>yue2: `ext:yue2OdeSteps`<br>sa3: `ext:sa3Steps`<br>hm: `ext:heartmulaCodecSteps` |
| `samp.cfg` | number | ace, mm3, sa3, hm | ace: `param:guidanceScale`<br>mm3: `ext:mm3CfgFlow`<br>sa3: `param:guidanceScale`. Read only with samplerPluginsEnabled and above 1: the default 1 adds no guidance, where the existing page carries ACE-Step's 9 here.<br>hm: `ext:heartmulaCodecGuidance` |
| `samp.shift` | number | ace, mm3, hm | ace: `param:shift` (auto: -1). Auto is -1, as the existing slider's Auto button sets it: the engine then works out a shift from the song's length and the step count (hot-step-server.cpp: 3 times a length factor and a steps factor, each 0.8 to 1.4, so about 1.9 to 5.9; 3.18 at 12 steps and 60 s). The concept starts on Auto; the existing page starts at 3.0 (docs/08, question 10).<br>mm3: `ext:mm3FlowShift`<br>hm: `ext:heartmulaFlowShift` |
| `samp.solver` | choice | all | ace: `param:inferMethod`<br>mm3: `param:inferMethod`. Native ("") sends no pick, so the engine runs its own loop; the existing page carries ACE-Step's euler here.<br>yue2: `ext:yue2NarSolver`<br>sa3: `param:inferMethod`. Native ("") sends no pick, so the engine runs its own loop; the existing page carries ACE-Step's euler here.<br>hm: `param:inferMethod`. Native ("") sends no pick, so the engine runs its own loop; the existing page carries ACE-Step's euler here. |
| `samp.sched` | choice | all | ace: `param:scheduler` (via scheduler). The schedule and its shape, packed: "beta:0.50:0.70".<br>mm3: `param:scheduler` (via scheduler). The schedule and its shape, packed: "beta:0.50:0.70".<br>yue2: `ext:yue2NarScheduler`<br>sa3: `param:scheduler` (via scheduler). The schedule and its shape, packed: "beta:0.50:0.70".<br>hm: `param:scheduler` (via scheduler). The schedule and its shape, packed: "beta:0.50:0.70". |
| `samp.guide` | choice | ace, mm3, sa3, hm | ace: `param:guidanceMode`<br>mm3: `param:guidanceMode`. Native ("") sends no pick, so the engine runs its own loop; the existing page carries ACE-Step's APG here.<br>sa3: `param:guidanceMode`. Native ("") sends no pick, so the engine runs its own loop; the existing page carries ACE-Step's APG here.<br>hm: `param:guidanceMode`. Native ("") sends no pick, so the engine runs its own loop; the existing page carries ACE-Step's APG here. |
| `samp.plugins` | boolean | mm3, sa3, hm | `ext:samplerPluginsEnabled` |
| `sa3.sampler` | choice | sa3 | `ext:sa3Sampler` |
| `samp.cs` | number | ace | `param:cfgIntervalStart` |
| `samp.ce` | number | ace | `param:cfgIntervalEnd` |
| `samp.retake` | number | ace | `param:retakeVariance` |
| `samp.rseed` | text | ace | `param:retakeSeed` (via int). -1 is a fresh one each run. |
| `sa3.lufs` | boolean | sa3 | `ext:sa3LufsEnabled` |
| `sa3.lufst` | number | sa3 | `ext:sa3LufsTarget` |
| `sch.alpha` | number | ace | `param:scheduler` (via scheduler) |
| `sch.beta` | number | ace | `param:scheduler` (via scheduler) |
| `sch.exp` | number | ace | `param:scheduler` (via scheduler) |
| `sch.a` | choice | ace | `param:scheduler` (via scheduler) |
| `sch.b` | choice | ace | `param:scheduler` (via scheduler) |
| `sch.x` | number | ace | `param:scheduler` (via scheduler) |
| `sch.split` | number | ace | `param:scheduler` (via scheduler) |
| `yn.wassersteinflow.0` | number | yue2 | `ext:yue2WassTau` |
| `yn.wassersteinflow.1` | number | yue2 | `ext:yue2WassSpectral` |
| `yn.wassersteinflow.2` | number | yue2 | `ext:yue2WassRmsWeight` |
| `yn.wassersteinflow.3` | number | yue2 | `ext:yue2WassRmsTarget` |
| `yn.wassersteinflow.4` | number | yue2 | `ext:yue2WassGate` |
| `yn.wassersteinflow.5` | number | yue2 | `ext:yue2WassIterations` |
| `yn.wassersteinflow.6` | boolean | yue2 | `ext:yue2WassProject` |
| `yn.wassersteinflow.7` | number | yue2 | `ext:yue2WassLatentRms` |
| `yn.latentchanneleq.0` | number | yue2 | `ext:yue2EqGroup` |
| `yn.latentchanneleq.1` | number | yue2 | `ext:yue2EqGain` |
| `yn.latentchanneleq.2` | number | yue2 | `ext:yue2EqT1` |
| `yn.latentchanneleq.3` | number | yue2 | `ext:yue2EqT2` |
| `yn.latentchanneleq.4` | text | yue2 | `ext:yue2EqGains` |
| `yn.latentchanneleq.5` | text | yue2 | `ext:yue2EqSection` |
| `yn.timesteplie.0` | number | yue2 | `ext:yue2LieAmount` |
| `yn.timesteplie.1` | choice | yue2 | `ext:yue2LieCurve` |
| `yn.timesteplie.2` | number | yue2 | `ext:yue2LieT1` |
| `yn.timesteplie.3` | number | yue2 | `ext:yue2LieT2` |
| `yn.timesteplie.4` | text | yue2 | `ext:yue2LieSections` |
| `yn.planoverdrive.0` | number | yue2 | `ext:yue2OdDrive` |
| `yn.planoverdrive.1` | choice | yue2 | `ext:yue2OdShape` |
| `yn.planoverdrive.2` | number | yue2 | `ext:yue2OdBase` |
| `yn.planoverdrive.3` | number | yue2 | `ext:yue2OdPeriod` |
| `yn.htv3.0` | number | yue2 | `ext:yue2HtKinetic` |
| `yn.htv3.1` | number | yue2 | `ext:yue2HtDamping` |
| `yn.htv3.2` | number | yue2 | `ext:yue2HtCritical` |
| `yn.htv3.3` | number | yue2 | `ext:yue2HtIntensity` |
| `yn.htv3.4` | number | yue2 | `ext:yue2HtWell` |
| `yn.htv3.5` | number | yue2 | `ext:yue2HtFloor` |
| `yn.htv3.6` | boolean | yue2 | `ext:yue2HtSnr` |
| `yn.htv3.7` | number | yue2 | `ext:yue2HtPoly` |
| `yn.htv3.8` | number | yue2 | `ext:yue2HtBlend` |
| `yn.htv3.9` | number | yue2 | `ext:yue2HtSmooth` |
| `yn.htv3.10` | number | yue2 | `ext:yue2HtDense` |
| `yn.htv3.11` | number | yue2 | `ext:yue2HtShift` |

### Guidance plugin (`guidance`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `apg.mom` | number | ace, sa3, hm | ace: `param:apgMomentum`<br>sa3: `param:apgMomentum`. The concept has it here; nothing this engine runs reads it today, so it changes nothing until the server does (docs/08, question 3).<br>hm: `param:apgMomentum`. The concept has it here; nothing this engine runs reads it today, so it changes nothing until the server does (docs/08, question 3). |
| `apg.norm` | number | ace, mm3, sa3, hm | ace: `param:apgNormThreshold`<br>mm3: `param:apgNormThreshold`<br>sa3: `param:apgNormThreshold`. The concept has it here; nothing this engine runs reads it today, so it changes nothing until the server does (docs/08, question 3).<br>hm: `param:apgNormThreshold` |

### Performance trades (`perf`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `perf.cfgcut` | number | ace | `param:cfgCutoffRatio` |
| `perf.lmcut` | number | ace | `param:lmCfgCutoffRatio` |
| `perf.cache` | number | ace | `param:cacheRatio` |

### Latent & correction (`latent`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `lat.ts` | text | ace | `param:customTimesteps` |
| `lat.shift` | number | ace | `param:latentShift`. The existing UI steps by 0.01; the concept by 0.05. |
| `lat.rescale` | number | ace | `param:latentRescale`. The existing UI steps by 0.01; the concept by 0.05. |
| `dcw.on` | boolean | ace | `param:dcwEnabled` |
| `dcw.mode` | choice | ace | `param:dcwMode` |
| `den.on` | boolean | ace | none: No switch in the real app: the denoiser runs while denoiseStrength is above 0. This switch gates den.str's wire. |
| `lss.on` | boolean | ace | none: No switch in the real app: LSS runs while lssStrength is above 0. This switch gates lss.str's wire. |
| `dcw.low` | number | ace | `param:dcwLowScaler` (while {"param":"dcw.mode","not":"high"}) + `param:dcwHighScaler` (while {"param":"dcw.mode","is":"high"}) |
| `dcw.high` | number | ace | `param:dcwHighScaler` |
| `den.str` | number | ace | `param:denoiseStrength` (while {"param":"den.on","is":true}, else 0). The denoiser runs while denoiseStrength is above 0, so its switch off sends 0. |
| `den.smooth` | number | ace | `param:denoiseSmoothing` |
| `den.mix` | number | ace | `param:denoiseMix` |
| `lss.str` | number | ace | `param:lssStrength` (while {"param":"lss.on","is":true}, else 0). LSS runs while lssStrength is above 0, so its switch off sends 0. |
| `lss.var` | number | ace | `param:lssVarThresh` |
| `lss.dc` | boolean | ace | `param:lssDcRemove` |

### Lead sheet (`score`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `yue.abc` | text | yue2 | `ext:yue2Abc`. Declared by the YuE2 backend and read by backends/yue2/generate.ts as a lead sheet marker. |
| `yue.timing` | choice | yue2 | `ext:yue2LyricSchedule` |

### Build by section (`build`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `yue.btakes` | number | yue2 | none: The real Build by section (Yue2Build) has no takes-per-section setting; takes come one at a time from the build session. |
| `yue.bmode` | choice | yue2 | none: No section mode in the real Build by section. |
| `yue.bcond` | number | yue2 | `ui:Yue2Build.cfgScale`. BuildSettings.cfgScale, "Condition strength", 0.1 to 2. |
| `yue.bar` | number | yue2 | `ui:Yue2Build.arStrength`. BuildSettings.arStrength; the real default is 0.6 (COVER_AR_DEFAULT), the concept's 1. |

### Flow renderer (`flow`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `mm.hop` | number | mm3 | `ext:mm3FlowHop`. The concept stepped by 5; the server declares 10, which this follows. |
| `mm.dcache` | number | mm3 | `ext:mm3FlowUncondInterval` |
| `mm.lowstep` | boolean | mm3 | `ext:mm3AutoLowStep` |
| `mm.fused` | boolean | mm3 | `ext:mm3DepthFused` |
| `mm.rend` | choice | mm3 | `ext:mm3DitBackend`. Read by backends/minimax/generate.ts but no longer declared as an extension (its control was removed). |
| `mm.prec` | choice | mm3 | `ext:mm3DitPrecision`. Read by backends/minimax/generate.ts; not declared as an extension. |
| `mm.stream` | boolean | mm3 | `ext:mm3Stream` |

### Plans & endings (`plans`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `mm.reuse` | boolean | mm3 | `ext:mm3ReuseAr` |
| `mm.savear` | boolean | mm3 | `ext:mm3SaveArCodes` |
| `mm.plank` | choice | mm3 | `ext:mm3PlankPath` |
| `mm.savehid` | boolean | mm3 | `ext:mm3SaveHiddens` |
| `mm.hidname` | text | mm3 | `ext:mm3HiddensName` |
| `mm.hidpath` | choice | mm3 | `ext:mm3HiddensPath` |
| `mm.reqend` | boolean | mm3 | `ext:mm3RequireEnding` |
| `mm.minlen` | boolean | mm3 | `ext:mm3MinLength` |
| `mm.basehand` | boolean | mm3 | `ext:mm3BaseHandoff` |
| `mm.kvq8` | boolean | mm3 | `ext:mm3KvQ8` |

### Re-planning (`replan`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `yue.replan` | boolean | yue2 | `ext:yue2AutoReplan` |
| `yue.replanflag` | boolean | yue2 | `ext:yue2ReplanFlags` |
| `yue.tries` | number | yue2 | `ext:yue2ReplanAttempts` |
| `yue.recompose` | number | yue2 | `ext:yue2ComposeRetries` |

### Plan adherence (`adhere`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `yue.pws` | number | yue2 | `ext:yue2PlanWStart` |
| `yue.pwe` | number | yue2 | `ext:yue2PlanWEnd` |
| `yue.pwc` | choice | yue2 | `ext:yue2PlanWCurve` |
| `yue.pcfg` | number | yue2 | `ext:yue2PlanCfg` |
| `yue.pcfg1` | number | yue2 | `ext:yue2PlanCfgT1` |
| `yue.pcfg2` | number | yue2 | `ext:yue2PlanCfgT2` |
| `yue.pcfgr` | number | yue2 | `ext:yue2PlanCfgRescale` |
| `yue.pcfglm` | boolean | yue2 | `ext:yue2PlanCfgLevelMatch` |
| `yue.cads` | number | yue2 | `ext:yue2CadsStrength` |
| `yue.cads1` | number | yue2 | `ext:yue2CadsT1` |
| `yue.cads2` | number | yue2 | `ext:yue2CadsT2` |
| `yue.cadsr` | boolean | yue2 | `ext:yue2CadsRescale` |

### Throughput (`yuebatch`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `yue.coalesce` | boolean | yue2 | `ext:yue2Coalesce` |
| `yue.noisevar` | number | yue2 | `ext:yue2Variations`. Declared with max 1 in the static manifest; the engine raises it. |
| `yue.narcache` | number | yue2 | `ext:yue2NarCacheRatio` |
| `yue.chunk` | number | yue2 | `ext:yue2NarChunkSeconds` |

### Seed (`seed`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `seed.lock` | boolean | all | `param:randomSeed` (inverted). randomSeed is the opposite of Keep this seed. |

### Takes (`takes`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `takes.n` | number | all | ace: `param:batchSize`. The server takes 8 at most (batch.max); the existing slider goes to 9 and sends 8.<br>mm3: `ext:mm3Takes`<br>yue2: `ext:yue2BatchSize`. Declared with a maximum of 1 in the static manifest; the engine raises it to its batch limit.<br>sa3: none: One song per run.<br>hm: none: One song per run. |

### Pedalboard (`post`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.on` | boolean | all | `param:postProcessingEnabled` |

### Tiled decoder (`tile`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.tile.on` | boolean | ace, sa3 | `param:postprocessEnabled` |
| `pp.tile.plugin` | choice | ace, sa3 | `param:postprocessPlugin`. The concept starts on V2.1, so turning the decoder on uses it; the real app starts with none chosen. |

### MM3 latent refiner (`refiner`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.refiner.on` | boolean | all | `param:mm3RefinerEnabled` |
| `pp.refiner.steps` | number | all | `param:mm3RefinerSteps` |
| `pp.refiner.win` | number | all | `param:mm3RefinerWindowSeconds`. The existing UI takes any length from 0, by 0.1; the concept 0 to 60 by 1. |
| `pp.refiner.ov` | number | all | `param:mm3RefinerOverlapSeconds`. The existing UI keeps it below the window; the concept 0 to 10 by 0.5. |
| `pp.refiner.mix` | number | all | `param:mm3RefinerMix` (÷ 100). Real is 0..1. |

### StableStep (`ss`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.ss.on` | boolean | ace, mm3, yue2, hm | `param:stableStepOn` |
| `pp.ss.str` | number | ace, mm3, yue2, hm | `param:stableStepStrength` (÷ 100). Real is 0..1, so the knob divides by 100; the existing UI steps by 5%, the concept by 1%. |
| `pp.ss.vox` | number | ace, mm3, yue2, hm | `param:stableStepVocalTrimDb` |
| `pp.ss.steps` | number | ace, mm3, yue2, hm | `param:stableStepSteps` |
| `pp.ss.dyn` | boolean | ace, mm3, yue2, hm | `param:stableStepPreserveDynamics` |
| `pp.ss.ppv` | boolean | ace, mm3, yue2, hm | `param:stableStepVocalPpVae` |
| `pp.ss.blend` | choice | ace, mm3, yue2, hm | `param:stableStepBlendMode` |
| `pp.ss.xhz` | number | ace, mm3, yue2, hm | `param:stableStepCrossoverHz` |
| `pp.ss.xw` | number | ace, mm3, yue2, hm | `param:stableStepCrossoverWidthHz` |
| `pp.ss.mix` | number | ace, mm3, yue2, hm | `param:stableStepMix` |
| `pp.ss.seedf` | boolean | ace, mm3, yue2, hm | `param:stableStepSeedFollowsDit` |
| `pp.ss.seed` | text | ace, mm3, yue2, hm | `param:stableStepSeed` (via int) |

### PP-VAE (`ppvae`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.ppvae.on` | boolean | ace | `param:ppVaeReencode` |
| `pp.ppvae.blend` | number | ace | `param:ppVaeBlend`. The existing UI steps by 0.01; the concept by 0.05. |

### Spectral lifter (`lift`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.lift.on` | boolean | ace | `param:spectralLifterEnabled` |
| `pp.lift.den` | number | ace | `param:slDenoiseStrength`. The existing UI steps by 0.01; the concept by 0.05. |
| `pp.lift.floor` | number | ace | `param:slNoiseFloor` |
| `pp.lift.hf` | number | ace | `param:slHfMix` |
| `pp.lift.tr` | number | ace | `param:slTransientBoost`. The existing UI steps by 0.01; the concept by 0.05. |
| `pp.lift.shim` | number | ace | `param:slShimmerReduction` |

### Vocal naturalizer (`nat`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.nat.on` | boolean | ace, mm3, yue2 | `param:vocalNaturalizerEnabled` |
| `pp.nat.amt` | number | ace, mm3, yue2 | `param:naturalizeAmount`. The existing UI steps by 0.01; the concept by 0.05. |
| `pp.nat.vr` | number | ace, mm3, yue2 | `param:natVibratoRate` |
| `pp.nat.vd` | number | ace, mm3, yue2 | `param:natVibratoDepth` |
| `pp.nat.fm` | number | ace, mm3, yue2 | `param:natFormantStrength` |
| `pp.nat.mt` | number | ace, mm3, yue2 | `param:natMetallicReduction` |
| `pp.nat.qm` | number | ace, mm3, yue2 | `param:natQuantizationMask` |
| `pp.nat.ts` | number | ace, mm3, yue2 | `param:natTransitionSmooth` |

### Pre-VST gain (`gain`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.gain` | number | all | `param:gainOffsetDb` |

### VST3 chain (`vst`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.vst.on` | boolean | all | none: No chain switch in the real app: the VST3 chain runs while any plugin in it is enabled (vstChainStore chainEnabled). |

### Mastering (`master`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.master.on` | boolean | all | `param:masteringEnabled` |
| `pp.master.timbre` | boolean | ace | `param:timbreReference` |

### Final normalizer (`norm`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.norm.on` | boolean | all | `param:lufsEnabled` |
| `pp.norm.target` | choice | all | `ui:hs-lufsPreset`. Form state: the real app keeps the preset to show it and sends only the target. + `param:lufsTarget` (map spotify → -14, apple → -16, ebu → -23, club → -8). Custom sends the Custom knob's value instead. |
| `pp.norm.custom` | number | all | `param:lufsTarget` |
| `pp.norm.ceil` | number | all | `param:lufsCeilingDb` |

### Lyric timestamps (`lrc`, unit)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.lrc.on` | boolean | ace, mm3 | `param:skipLrc` (inverted). skipLrc is the opposite. |

### Forced alignment (`fa`, unit)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.fa.on` | boolean | yue2 | `param:yue2AlignLyrics` |

### Whisper lyrics (`wh`, unit)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.wh.on` | boolean | ace, mm3, yue2 | `param:whisperLyricsEnabled` |
| `pp.wh.model` | choice | ace, mm3, yue2 | `param:whisperModel` |
| `pp.wh.beam` | number | ace, mm3, yue2 | `param:whisperBeamSize` |
| `pp.wh.iso` | boolean | ace, mm3, yue2 | `param:whisperIsolateVocals` |
| `pp.wh.lang` | choice | ace, mm3, yue2 | `param:whisperLanguage` |

### Cover art (`art`, unit)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.art.on` | boolean | ace, mm3, yue2 | `param:coverArtEnabled` |
| `pp.art.subject` | text | ace, mm3, yue2 | `param:coverArtSubject` |

### Quality evaluator (`q`, unit)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.q.on` | boolean | ace, mm3, yue2, sa3 | `param:qualityEvalEnabled` |
| `pp.q.eval` | choice | ace, mm3, yue2, sa3 | `param:qualityEvalTarget` |
<!-- end generated -->
