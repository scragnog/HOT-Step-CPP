# Parameter names: the concept and the real application

The concept invented its own keys (`pp.ss.str`, `lm.temp`). The rebuild has to talk to the real server, so every
parameter's `wire` records what the real application calls it, per engine. This page explains where the names were
found, gives the two lists of mismatches, and ends with the full table (printed from the definitions; see the end).

## Where the names come from

| Channel | Where the real app keeps it | How it travels |
| --- | --- | --- |
| `param:` | `ui/src/stores/globalParamsStore.ts`, defaults in `GENERATION_DEFAULTS` (`server/src/contracts/generationControls.ts`) | Fields of the Create request body, assembled by `getGlobalParams()` |
| `ext:` | Each backend's `capabilities().extensions` (`server/src/services/backends/<engine>/index.ts`), with key, type, range, default and options | `backendParams`, per backend, rendered generically by the existing UI |
| `content:` | `CreateContent` in `ui/src/components/create/createContent.ts` (caption, lyrics, bpm, keyScale, duration...) | Per-song fields of the request |
| `model:` | `GET/POST /api/backends/models` buckets per backend (ACE's ditModel and friends are `param:`) | Engine state, not part of the request |
| `ui:` | Form state the browser keeps (`hs-*` keys) and uses itself, such as the auto-expand switch or the caption source | Not sent as such |
| `plugin:` | Lua plugin parameters, `pluginParams` as `"<plugin>:<key>"` | Not used by any concept parameter today (see below) |

The engine ids in definitions are the server's: `ace`, `minimax-m3`, `yue2`, `sa3`, `heartmula`. The concept's `mm3`,
`yue` and `hm` are not used.

The extension lists were read from the backends' source as it is on `ui-next` (MiniMax-Music3 declares 27, YuE2 86,
Stable Audio 3 5, HeartMuLa 14). `docs/dev/frontend-create-controls.md` in the repository is the narrative for the
`param:` fields and was checked against `generationControls.ts`.

## Counts

- 284 parameters, 515 parameter-engine pairs.
- 501 pairs have a real name: `param` 276, `ext` 139, `content` 49, `model` 19, `ui` 18.
- 275 of the 284 parameters have a real name on every engine they are on.
- 14 pairs on 9 parameters have none.

## The concept has it, the real app does not (9 parameters)

| Key | Engines | What the real app does instead |
| --- | --- | --- |
| `den.on` | ace | No switch: the spectral denoiser runs while `denoiseStrength` is above 0. The concept's Off has to send 0. |
| `lss.on` | ace | No switch: LSS runs while `lssStrength` is above 0. |
| `pp.vst.on` | all | No chain switch: the VST3 chain runs while any plugin in it is enabled (`vstChainStore.chainEnabled`). |
| `ad.trigcap` | mm3 | No "add trigger to caption" switch was found; the Create form always puts a set LoRA trigger at the start of the caption (`composeCaption`). |
| `yue.btakes` | yue2 | The real Build by section (`Yue2Build`) has no takes-per-section setting; takes come one at a time from the build session. |
| `yue.bmode` | yue2 | No section mode in the real Build by section. |
| `cap.ds` | mm3 | MiniMax-Music3 takes caption tracks from a Lyric Studio handoff, not a dataset dropdown (YuE2 has `hs-yue2CaptionDataset`). |
| `s0` | yue2 | On YuE2 the AR adapter's strength is two dials, plan and compose (`ad.arp`, `ad.arc`); there is no single strength. |
| `takes.n` | sa3, hm | One song per run; there is nothing to send. |

### Close, but not the same

These have a real counterpart that differs in a way the rebuild must handle. Each is in its `wire` note.

- **Units and scale**: StableStep strength and the refiner mix are percent in the concept and 0..1 in the real app;
  YuE2's longest length is seconds in the concept and minutes (`yue2MaxMinutes`) in the real app; the auto-trim fade is
  seconds against milliseconds (`autoTrimFadeMs`).
- **Inverted**: Plan the song first is `skipLm`; Keep this seed is `randomSeed`; Lyric timestamps is `skipLrc`.
- **Packed**: the schedule shapes (`sch.alpha`, `sch.beta`, `sch.exp`, `sch.a`, `sch.b`, `sch.x`, `sch.split`) are
  packed into the scheduler string, `'beta:0.50:0.70'`, `'power:2.00'`, `'composite:<a>+<b>:<x>:<split>'`.
- **Different type**: the merge VRAM choice (`ad.mq`) is the boolean `adapterMergeLowVram`; the retake seed and
  StableStep's fixed seed are numbers in the real app and text in the concept; YuE2's composer CFG is a text knob where
  empty means auto.
- **Different step or range**: MiniMax-Music3's window hop steps by 10 in the real app and 5 in the concept, and YuE2's
  composer CFG rescale by 0.05 against 0.01; the definitions follow the server. ACE-Step's batch maximum is 8
  (`batch.max`), the concept's takes go to 9 (kept, as a question). YuE2's noise variations and batch size are declared
  with a maximum of 1 in the static manifest and raised by the engine at run time. YuE2's planner lie curve and overdrive
  shape options are stored as the server's values (`constant`, `late`).
- **Read but not declared**: MiniMax-Music3's renderer and TensorRT precision (`mm3DitBackend`, `mm3DitPrecision`) are
  read by `backends/minimax/generate.ts` but are no longer declared as extensions (the control was removed).
- **Not sent, by design** (`ui:`): vocal gender, the BPM and key on MiniMax-Music3 (read by Compose caption), the language
  on MiniMax-Music3, the LUFS preset (the wire carries the target), the caption source and dataset, the auto-expand
  switch, and Build by section's condition and AR strengths (`Yue2Build` settings; the real AR default is 0.6, the
  concept's 1).
- **Per slot**: YuE2's adapter dials are kept per slot (AR and NAR, each with its own six) as engine state; the concept has
  one bank for both.

## The real app has it, the concept does not (28)

From `GENERATION_DEFAULTS` (21):

- Adapter plumbing that the cartridges would carry: `adapter`, `adapterScale` (Simple mode's single slot, which the
  concept's slot 1 maps onto), `adapterSectionIsolation`, `adapterFolder`, `lmAdapterFolder`, `adaptersOpen` (form state).
- `lmAdapterScale`: the ACE-Step planner adapter's strength. The concept loads a planner adapter but gives it no strength.
- `embeddingModel`: ACE-Step's embedding model choice.
- `lmSeed`: an independent planner seed for ACE-Step (the concept has only "LM seed follows").
- `storkSubsteps`, `beatStability`, `frequencyDamping`, `temporalSmoothing`: solver-specific knobs, sent only for STORK
  and JKASS. The concept has no per-solver controls on ACE-Step; in the real app they belong with `pluginParams`, which
  the Lua plugins declare.
- `pluginParams`, `backendParams`: the two free-form channels themselves.
- StableStep's own sampler: `stableStepBackend`, `stableStepAdapters`, `stableStepSolver`, `stableStepScheduler`,
  `stableStepGuidanceMode`, `stableStepGuidanceScale`.

From the backends' extensions (6, all YuE2): `yue2PlanMaxTokens`, `yue2SemMinTokens`, `yue2EndThreshold`, `yue2EndBias`,
`yue2EndBiasFrom`, `yue2EndBiasRamp` (the plan's token budget, the composer's minimum length, and the song-ending
controls). [06-how-to.md](06-how-to.md) uses two of these as its worked examples.

From the Create form (1): `sourceLatentUrl`.

Not counted: the parameters of other studios that the backend mappers also read (MiniMax-Music3's `mm3A2a*`, `mm3Guide*`
and `mm3Force*` cover controls, YuE2's lyric schedule sub-knobs, the cover and repaint fields), which are outside the
Create page.

## The table

Printed from the definitions with `node tools/validate-definitions.ts --names`; print it again after changing a `wire`
rather than editing rows here. A real name is shown once when it is the same on every engine the parameter is on.

### Prompt (`prompt`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `prompt.text` | text | all | `content:caption` |
| `wild.auto` | boolean | ace, mm3, yue2, sa3 | `ui:hs-main-auto-expand` |
| `song.artist` | text | all | `content:artist` |
| `song.title` | text | all | `content:title` |
| `song.subject` | text | all | `content:subject` |
| `cap.src` | choice | mm3, yue2 | mm3: `ui:mm3CaptionSources.mode`. The Send-to-Create handoff in localStorage; values custom, auto, track:<title>.<br>yue2: `ui:yue2CaptionSelection.mode`. CreatePanel state; values custom, auto, track. |
| `cap.ds` | choice | mm3, yue2 | mm3: none: MiniMax-Music3 takes its caption tracks from a Lyric Studio handoff, not from a dataset dropdown.<br>yue2: `ui:hs-yue2CaptionDataset` |

### Lyrics (`lyrics`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `lyrics.text` | text | ace, mm3, yue2, hm | `content:lyrics` |
| `lyrics.instr` | boolean | ace, mm3, yue2, hm | `content:instrumental` |
| `lyrics.lang` | choice | ace, mm3 | ace: `content:vocalLanguage`<br>mm3: `ui:hs-vocalLanguage`. Sent on every engine and read only by ACE-Step; Compose caption writes it into Vocal Details on MiniMax-Music3. |
| `lyrics.gender` | choice | mm3 | `ui:hs-vocalGender`. Deliberately not sent: it reaches the model only through Compose caption writing it into Vocal Details (createContent.ts). Real values are "", male, female, duet. |

### Models (`models`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `model.var` | choice | hm | `model:heartmula.variant` |
| `model.dit` | choice | all | ace: `param:ditModel`<br>mm3: `model:minimax-m3.dit`<br>yue2: `model:yue2.lm`<br>sa3: `model:sa3.dit`<br>hm: `model:heartmula.global` |
| `model.lm` | choice | ace, mm3, hm | ace: `param:lmModel`<br>mm3: `model:minimax-m3.lm`<br>hm: `model:heartmula.local` |
| `model.x` | choice | mm3, hm | mm3: `model:minimax-m3.depth`<br>hm: `model:heartmula.flow` |
| `model.vae` | choice | ace, yue2 | ace: `param:vaeModel`<br>yue2: `model:yue2.vae` |

### Repaint (`repaint`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `rp.mode` | choice | ace | `content:repaintMode` |
| `rp.xfade` | number | ace | `content:crossfadeFrames` |
| `rp.lyrics` | text | ace | `content:lyrics`. The Repaint studio's own lyrics field (studioWorkflows repaint fields). |

### Tempo, key & length (`tempo`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `tempo.bpm` | number | ace, mm3, yue2 | ace: `content:bpm` (auto: 0). Auto is 0.<br>mm3: `ui:hs-bpm` (auto: 0). Sent on every engine, read by MiniMax-Music3 only through Compose caption.<br>yue2: `content:bpm` (auto: 0). Composed into the caption tail unless yue2CaptionBpmKey is false. |
| `tempo.key` | choice | ace, mm3, yue2 | ace: `content:keyScale` (auto: ""). Auto is "". Real values are "C major" style.<br>mm3: `ui:hs-keyScale` (auto: ""). Read only by Compose caption.<br>yue2: `content:keyScale` (auto: ""). Composed into the caption tail. |
| `tempo.sig` | choice | ace | `content:timeSignature` (auto: ""). Only the numerator reaches the engine. |
| `tempo.dur` | number | ace, sa3, hm | `content:duration` (auto: -1). Auto is -1. SA3 renders 30 s for -1. |
| `tempo.max` | number | yue2 | `ext:yue2MaxMinutes` (× 0.0167). Real is minutes, step 0.5. |
| `tempo.incap` | boolean | yue2 | `ext:yue2CaptionBpmKey` |
| `tempo.beatio` | boolean | ace, mm3, sa3 | `content:beatIntro` |
| `tempo.beatbars` | choice | ace, mm3, sa3 | `content:introBars` |
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
| `s0` | number | ace, mm3, yue2, hm | ace: `param:adapterStack[0].scale`. adapterScale in Simple mode.<br>mm3: `ext:mm3LmAdapterScale`<br>yue2: none: On YuE2 the AR slot's strength is its two dials, Plan and Compose (ad.arp, ad.arc).<br>hm: `ext:heartmulaAdapterStrength` |
| `s1` | number | ace, yue2 | ace: `param:adapterStack[1].scale`<br>yue2: `model:yue2.nar.scales.global` |
| `s2` | number | ace | `param:adapterStack[2].scale` |
| `ad.mode` | choice | ace, mm3 | ace: `param:adapterMode`. Real values merge, runtime, runtime_lowrank.<br>mm3: `ext:mm3LmAdapterMode` |
| `ad.planner` | choice | ace | `param:lmAdapter` |
| `ad.dora` | boolean | ace | `param:adapterRuntimeDora` |
| `ag.self` | number | ace | `param:adapterGroupScales.self_attn` |
| `ag.cross` | number | ace | `param:adapterGroupScales.cross_attn` |
| `ag.mlp` | number | ace | `param:adapterGroupScales.mlp` |
| `ag.cond` | number | ace | `param:adapterGroupScales.cond_embed` |
| `ag.time` | number | ace | `param:adapterGroupScales.time_embed` |
| `ag.proj` | number | ace | `param:adapterGroupScales.proj_in` |
| `ad.att` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleAttn`<br>yue2: `model:yue2.ar.scales.attn`. Engine state; each YuE2 slot (ar, nar) has its own six dials. The concept has one bank for both.<br>hm: `ext:heartmulaAdapterScaleAttn` |
| `ad.mlp` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleMlp`<br>yue2: `model:yue2.ar.scales.mlp`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleMlp` |
| `ad.early` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleEarly`<br>yue2: `model:yue2.ar.scales.early`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleEarly` |
| `ad.mid` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleMid`<br>yue2: `model:yue2.ar.scales.mid`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleMid` |
| `ad.late` | number | mm3, yue2, hm | mm3: `ext:mm3LmAdapterScaleLate`<br>yue2: `model:yue2.ar.scales.late`. Per slot in the real app.<br>hm: `ext:heartmulaAdapterScaleLate` |
| `ad.plan` | number | mm3 | `ext:mm3LmAdapterScalePlan` |
| `ad.adv` | choice | ace | `param:advancedAdapters`. Simple is false, Advanced is true. |
| `ad.stack` | choice | ace | `param:adapterStackMode` |
| `ad.sigma` | number | ace | `param:adapterStackBudget` |
| `ad.rq` | choice | ace | `param:adapterRuntimeQuant` |
| `ad.mq` | choice | ace | `param:adapterMergeLowVram`. A boolean in the real app: HQ is false, Low is true. |
| `ad.home` | choice | ace | `param:rebaseSource` |
| `ad.beta` | number | ace, yue2, hm | ace: `param:rebaseBeta`<br>yue2: `model:yue2.ar.scales.rebase`<br>hm: `ext:heartmulaAdapterRebase` |
| `ad.ref` | boolean | ace | `param:noAdapterRender` |
| `ad.align` | number | ace | `param:adapterSectionAlignAt` |
| `ad.arp` | number | yue2 | `model:yue2.ar.scales.plan`. The AR slot's planner strength (engine state, POST /api/backends/models). |
| `ad.arc` | number | yue2 | `model:yue2.ar.scales.global`. The AR slot's composer strength (engine state). |
| `ad.trigcap` | boolean | mm3 | none: No such switch found: the Create form always puts the LoRA trigger at the start of the caption when one is set (composeCaption). |
| `ad.gpu` | boolean | mm3 | `ext:mm3LmMergeGpu` |
| `ad.trigger` | text | ace, mm3, yue2, hm | ace: `content:loraTrigger`<br>mm3: `ext:mm3LmAdapterTrigger`<br>yue2: `ext:yue2LmAdapterTrigger`<br>hm: `content:loraTrigger` |

### Planner (`planner`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `lm.on` | boolean | ace | `param:skipLm` (inverted). skipLm is the opposite of Plan the song first. |
| `lm.cot` | boolean | ace | `param:useCotCaption` |
| `lm.cotmode` | choice | yue2 | `ext:yue2Cot` |
| `lm.preview` | boolean | yue2 | `ext:yue2PreviewScore` |
| `lm.temp` | number | ace | `param:lmTemperature` |
| `lm.cfg` | number | ace | `param:lmCfgScale` |
| `lm.topk` | number | ace | `param:lmTopK` |
| `lm.topp` | number | ace | `param:lmTopP` |
| `lm.minp` | number | ace | `param:lmMinP` |
| `lm.plaus` | number | ace, yue2 | ace: `param:lmCfgMinP`<br>yue2: `ext:yue2CfgMinP` |
| `lm.resc` | number | ace, yue2 | ace: `param:lmCfgRescale`<br>yue2: `ext:yue2CfgRescale` |
| `lm.logp` | boolean | ace, yue2 | ace: `param:lmCfgLogprob`<br>yue2: `ext:yue2CfgLogprob` |
| `lm.codes` | number | ace | `param:lmCodesStrength` |
| `lm.codesmode` | choice | ace | `param:lmCodesMode` |
| `lm.codesteps` | number | ace | `param:lmCodesSteps` |
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
| `yue.cfg` | number | yue2 | `ext:yue2CfgScale` (auto: ""). A text knob in the real app: empty means the checkpoint default. |
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
| `samp.cfg` | number | ace, mm3, sa3, hm | ace: `param:guidanceScale`<br>mm3: `ext:mm3CfgFlow`<br>sa3: `param:guidanceScale`. Read only with samplerPluginsEnabled and above 1.<br>hm: `ext:heartmulaCodecGuidance` |
| `samp.shift` | number | ace, mm3, hm | ace: `param:shift`<br>mm3: `ext:mm3FlowShift`<br>hm: `ext:heartmulaFlowShift` |
| `samp.solver` | choice | all | ace: `param:inferMethod`<br>mm3: `param:inferMethod`<br>yue2: `ext:yue2NarSolver`<br>sa3: `param:inferMethod`<br>hm: `param:inferMethod` |
| `samp.sched` | choice | all | ace: `param:scheduler`<br>mm3: `param:scheduler`<br>yue2: `ext:yue2NarScheduler`<br>sa3: `param:scheduler`<br>hm: `param:scheduler` |
| `samp.guide` | choice | ace, mm3, sa3, hm | `param:guidanceMode` |
| `samp.plugins` | boolean | mm3, sa3, hm | `ext:samplerPluginsEnabled` |
| `sa3.sampler` | choice | sa3 | `ext:sa3Sampler` |
| `samp.cs` | number | ace | `param:cfgIntervalStart` |
| `samp.ce` | number | ace | `param:cfgIntervalEnd` |
| `samp.retake` | number | ace | `param:retakeVariance` |
| `samp.rseed` | text | ace | `param:retakeSeed`. A number in the real app (-1 is a fresh one each run); the concept keeps it as text. |
| `sa3.lufs` | boolean | sa3 | `ext:sa3LufsEnabled` |
| `sa3.lufst` | number | sa3 | `ext:sa3LufsTarget` |
| `sch.alpha` | number | ace | `param:scheduler`. Packed into the scheduler string: "beta:<alpha>:<beta>". |
| `sch.beta` | number | ace | `param:scheduler`. Packed into the scheduler string: "beta:<alpha>:<beta>". |
| `sch.exp` | number | ace | `param:scheduler`. Packed into the scheduler string: "power:<exponent>". |
| `sch.a` | choice | ace | `param:scheduler`. Packed into the scheduler string: "composite:<a>+<b>:<crossover>:<split>"; stage names are plugin ids (bong_tangent, linear). |
| `sch.b` | choice | ace | `param:scheduler`. Packed into the scheduler string (see sch.a). |
| `sch.x` | number | ace | `param:scheduler`. Packed into the scheduler string (see sch.a). |
| `sch.split` | number | ace | `param:scheduler`. Packed into the scheduler string (see sch.a). |
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
| `apg.mom` | number | ace, sa3, hm | `param:apgMomentum` |
| `apg.norm` | number | ace, mm3, sa3, hm | `param:apgNormThreshold` |

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
| `lat.shift` | number | ace | `param:latentShift` |
| `lat.rescale` | number | ace | `param:latentRescale` |
| `dcw.on` | boolean | ace | `param:dcwEnabled` |
| `dcw.mode` | choice | ace | `param:dcwMode` |
| `den.on` | boolean | ace | none: No switch in the real app: the denoiser runs while denoiseStrength is above 0. Off has to send denoiseStrength 0. |
| `lss.on` | boolean | ace | none: No switch in the real app: LSS runs while lssStrength is above 0. Off has to send lssStrength 0. |
| `dcw.low` | number | ace | `param:dcwLowScaler` |
| `dcw.high` | number | ace | `param:dcwHighScaler` |
| `den.str` | number | ace | `param:denoiseStrength` |
| `den.smooth` | number | ace | `param:denoiseSmoothing` |
| `den.mix` | number | ace | `param:denoiseMix` |
| `lss.str` | number | ace | `param:lssStrength` |
| `lss.var` | number | ace | `param:lssVarThresh` |
| `lss.dc` | boolean | ace | `param:lssDcRemove` |

### Lead sheet (`score`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `yue.abc` | text | yue2 | `ext:yue2Abc` |
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
| `takes.n` | number | all | ace: `param:batchSize`<br>mm3: `ext:mm3Takes`<br>yue2: `ext:yue2BatchSize`<br>sa3: none: One song per run.<br>hm: none: One song per run. |

### Pedalboard (`post`, module)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.on` | boolean | all | `param:postProcessingEnabled` |

### Tiled decoder (`tile`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.tile.on` | boolean | ace, sa3 | `param:postprocessEnabled` |
| `pp.tile.plugin` | choice | ace, sa3 | `param:postprocessPlugin` |

### MM3 latent refiner (`refiner`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.refiner.on` | boolean | all | `param:mm3RefinerEnabled` |
| `pp.refiner.steps` | number | all | `param:mm3RefinerSteps` |
| `pp.refiner.win` | number | all | `param:mm3RefinerWindowSeconds` |
| `pp.refiner.ov` | number | all | `param:mm3RefinerOverlapSeconds` |
| `pp.refiner.mix` | number | all | `param:mm3RefinerMix` (× 0.01). Real is 0..1. |

### StableStep (`ss`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.ss.on` | boolean | ace, mm3, yue2, hm | `param:stableStepOn` |
| `pp.ss.str` | number | ace, mm3, yue2, hm | `param:stableStepStrength` (× 0.01). Real is 0..1. |
| `pp.ss.vox` | number | ace, mm3, yue2, hm | `param:stableStepVocalTrimDb` |
| `pp.ss.steps` | number | ace, mm3, yue2, hm | `param:stableStepSteps` |
| `pp.ss.dyn` | boolean | ace, mm3, yue2, hm | `param:stableStepPreserveDynamics` |
| `pp.ss.ppv` | boolean | ace, mm3, yue2, hm | `param:stableStepVocalPpVae` |
| `pp.ss.blend` | choice | ace, mm3, yue2, hm | `param:stableStepBlendMode` |
| `pp.ss.xhz` | number | ace, mm3, yue2, hm | `param:stableStepCrossoverHz` |
| `pp.ss.xw` | number | ace, mm3, yue2, hm | `param:stableStepCrossoverWidthHz` |
| `pp.ss.mix` | number | ace, mm3, yue2, hm | `param:stableStepMix` |
| `pp.ss.seedf` | boolean | ace, mm3, yue2, hm | `param:stableStepSeedFollowsDit` |
| `pp.ss.seed` | text | ace, mm3, yue2, hm | `param:stableStepSeed`. A number in the real app. |

### PP-VAE (`ppvae`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.ppvae.on` | boolean | ace | `param:ppVaeReencode` |
| `pp.ppvae.blend` | number | ace | `param:ppVaeBlend` |

### Spectral lifter (`lift`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.lift.on` | boolean | ace | `param:spectralLifterEnabled` |
| `pp.lift.den` | number | ace | `param:slDenoiseStrength` |
| `pp.lift.floor` | number | ace | `param:slNoiseFloor` |
| `pp.lift.hf` | number | ace | `param:slHfMix` |
| `pp.lift.tr` | number | ace | `param:slTransientBoost` |
| `pp.lift.shim` | number | ace | `param:slShimmerReduction` |

### Vocal naturalizer (`nat`, pedal)

| Concept key | Type | Engines | Real name |
| --- | --- | --- | --- |
| `pp.nat.on` | boolean | ace, mm3, yue2 | `param:vocalNaturalizerEnabled` |
| `pp.nat.amt` | number | ace, mm3, yue2 | `param:naturalizeAmount` |
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
| `pp.norm.target` | choice | all | `ui:lufsPreset`. Form state; the wire carries lufsTarget from the preset. |
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
