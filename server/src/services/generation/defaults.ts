// Generation policy moved from globalParamsStore.getGlobalParams.
// Keep legacy flat submissions unchanged until every caller uses generation-intent/1.

export const GENERATION_DEFAULTS = {
  ditModel: '',
  lmModel: '',
  vaeModel: '',
  lmAdapter: '',
  lmAdapterScale: 1.0,
  embeddingModel: '',
  adapter: '',
  adapterScale: 1.0,
  adapterStack: [] as { path: string; scale: number }[],
  adapterStackMode: 'blend',
  adapterStackBudget: 0.75,
  adapterSectionAlignAt: 0.55,
  adapterSectionIsolation: 0.5,
  adapterMode: 'runtime',
  adapterRuntimeQuant: 'bf16',
  adapterMergeLowVram: false,
  adapterGroupScales: {
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0, time_embed: 0.0, proj_in: 0.0,
  },
  rebaseSource: '',
  rebaseBeta: 0.75,
  adapterFolder: '',
  lmAdapterFolder: '',
  advancedAdapters: false,
  noAdapterRender: false,
  adaptersOpen: false,
  inferenceSteps: 12,
  guidanceScale: 9.0,
  cfgCutoffRatio: 1.0,
  lmCfgCutoffRatio: 1.0,
  cacheRatio: 0,
  shift: 3.0,
  // Backend-scoped (see BACKEND_SCOPED_FIELDS below).
  inferMethod: 'euler',
  scheduler: 'linear',
  guidanceMode: 'apg',
  pluginParams: {} as Record<string, string>,
  backendParams: {} as Record<string, unknown>,
  seed: 42,
  randomSeed: true,
  lmSeed: 42,
  lmSeedFollowsDit: true,
  batchSize: 1,
  storkSubsteps: 10,
  beatStability: 0.25,
  frequencyDamping: 0.4,
  temporalSmoothing: 0.13,
  apgMomentum: 0.75,
  apgNormThreshold: 2.5,
  dcwEnabled: false,
  dcwMode: 'double',
  // Legacy fallback chain lives at the hs-dcwLowScaler readKey call below; this
  // is the value that chain ultimately falls back to.
  dcwLowScaler: 0.2,
  dcwHighScaler: 0.2,
  latentShift: 0.0,
  latentRescale: 1.0,
  customTimesteps: '',
  denoiseStrength: 0.0,
  denoiseSmoothing: 0.7,
  denoiseMix: 0.25,
  lssStrength: 0.0,
  lssVarThresh: 0.15,
  lssDcRemove: true,
  autoTrimEnabled: false,
  durationBuffer: 15,
  autoTrimFadeMs: 2000,
  skipLm: false,
  skipLrc: false,
  useCotCaption: true,
  lmTemperature: 0.8,
  lmCfgScale: 2.2,
  lmTopK: 0,
  lmTopP: 0.92,
  lmRepPenalty: 1.1,
  lmRepWindow: 64,
  lmRepMode: 'presence',
  lmDryBase: 1.75,
  lmDryMinLen: 3,
  lmNegativePrompt: 'NO USER INPUT',
  lmCodesStrength: 1.0,
  lmCodesMode: 'ratio' as 'ratio' | 'steps',
  lmCodesSteps: 6,
  postProcessingEnabled: true,
  spectralLifterEnabled: false,
  slDenoiseStrength: 0.3,
  slNoiseFloor: 0.1,
  slHfMix: 0.0,
  slTransientBoost: 0.0,
  slShimmerReduction: 6.0,
  masteringEnabled: false,
  masteringReference: '',
  timbreReference: false,
  timbreAudioPath: '',
  vocalNaturalizerEnabled: false,
  gainOffsetDb: 0,
  naturalizeAmount: 0.5,
  natVibratoRate: 4.5,
  natVibratoDepth: 1.0,
  natFormantStrength: 1.0,
  natMetallicReduction: 1.0,
  natQuantizationMask: 0.0,
  natTransitionSmooth: 1.0,
  ppVaeReencode: false,
  ppVaeBlend: 0.0,
  stableStepOn: false,
  stableStepStrength: 0.3,
  stableStepBackend: 'auto',
  stableStepAdapters: [] as Array<{ name: string; scale: number; enabled: boolean }>,
  stableStepPreserveDynamics: true,
  stableStepVocalPpVae: false,
  stableStepVocalTrimDb: 0,
  stableStepBlendMode: 'off',
  stableStepCrossoverHz: 250,
  stableStepCrossoverWidthHz: 200,
  stableStepMix: 1.0,
  stableStepSeed: 4242,
  stableStepSeedFollowsDit: true,
  stableStepSteps: 8,
  stableStepSolver: '',
  stableStepScheduler: '',
  stableStepGuidanceMode: '',
  stableStepGuidanceScale: 1.0,
  coverArtEnabled: false,
  coverArtSubject: '',
  qualityEvalEnabled: false,
  qualityEvalTarget: 'unmastered',
  whisperLyricsEnabled: false,
  whisperModel: '',
  whisperLanguage: 'auto',
  whisperBeamSize: 5,
  whisperIsolateVocals: false,
  yue2AlignLyrics: false,
  postprocessEnabled: false,
  postprocessPlugin: '',
  lufsEnabled: false,
  lufsPreset: 'spotify',
  lufsTarget: -14,
  lufsCeilingDb: -1,
};

export interface GenerationResolverOptions {
  backendId: string;
  settings?: Record<string, unknown>;
}

export function resolveGenerationParams(
  params: Record<string, unknown>, options: GenerationResolverOptions,
): Record<string, unknown> {
    const s: any = { ...GENERATION_DEFAULTS, ...params };
    const settings: any = options.settings ?? {};

    // Effective adapter stack: the multi-adapter list when in Advanced mode,
    // otherwise the single `adapter` folded into a one-element stack. `primary`
    // drives the single-adapter features (trigger word, basin re-base, group
    // scales) which remain keyed on the first adapter. Gated on advancedAdapters
    // to match the badge/UI — a stack persisted from a previous Advanced session
    // must not override the Simple-mode selection.
    const isStack = !!(s.advancedAdapters && s.adapterStack && s.adapterStack.length > 0);
    const rawStack: { path: string; scale: number; stepStart?: number; stepEnd?: number }[] = isStack
      ? s.adapterStack
      : (s.adapter ? [{ path: s.adapter, scale: s.adapterScale }] : []);

    // Blend mode (multi-adapter stacks only): treat each entry's `scale` as a
    // relative weight and normalise so the effective scales sum to the budget,
    // keeping combined strength constant regardless of how many are stacked.
    // Sum mode (and the single-adapter fallback) sends the raw scales as-is.
    // Blend only applies with 2+ adapters — a single adapter's strength is just
    // its own scale, sent as-is.
    let stack = rawStack;
    if (isStack && s.adapterStackMode === 'blend' && rawStack.length >= 2) {
      const budget = s.adapterStackBudget ?? 0.75;
      const sumW = rawStack.reduce((acc, e) => acc + (e.scale || 0), 0);
      // Spread each entry so per-adapter extras (stepStart/stepEnd timestep
      // window, etc.) survive the blend re-scale.
      stack = sumW > 0
        ? rawStack.map(e => ({ ...e, scale: +(budget * (e.scale || 0) / sumW).toFixed(4) }))
        : rawStack.map(e => ({ ...e, scale: +(budget / rawStack.length).toFixed(4) }));
    }
    const primary = stack[0]?.path || '';
    // Timestep windows force runtime mode server-side regardless of the selected
    // adapter mode — runtime-only knobs must flow whenever they're present.
    const hasStepWindows = stack.some(
      (e: { stepStart?: number; stepEnd?: number }) => e.stepStart !== undefined || e.stepEnd !== undefined,
    );

    // Trigger words. The server resolves each adapter's EMBEDDED trigger from
    // its safetensors metadata — that is what the adapter was actually trained
    // with, so it wins and needs nothing from us. What we send is the filename
    // fallback for adapters that carry no embedded trigger, tagged with the
    // adapter path so the server can match it per-adapter rather than applying
    // it to the whole stack.
    // docs/plans/2026-07-28-adapter-trigger-embedding.md T6
    const triggerPlacement = (settings.triggerPlacement || 'prepend') as 'prepend' | 'append' | 'replace';
    const triggerSpecs = settings.triggerUseFilename
      ? stack
          .map(e => ({
            word: e.path.split(/[\\\/]/).pop()?.replace(/\.safetensors$/i, '') || '',
            placement: triggerPlacement,
            source: 'filename' as const,
            path: e.path,
          }))
          .filter(s => s.word)
      : [];
    const triggerWords: string[] = triggerSpecs.map(s => s.word);
    const triggerWord = triggerWords.join(', ');

    // LM codes window → audio_cover_strength fraction. Steps mode converts the
    // absolute count against the current step budget; +0.5 lands mid-bin so the
    // engine's floor(num_steps * strength) yields exactly that many steps
    // despite f32 rounding. At or above the budget → 1.0 (no silence switch).
    const lmCodesEff = s.lmCodesMode === 'steps'
      ? (s.lmCodesSteps >= s.inferenceSteps ? 1.0 : Math.max(0, s.lmCodesSteps + 0.5) / s.inferenceSteps)
      : s.lmCodesStrength;

    return {
      // Backend-declared knobs (capabilities().extensions), flattened alongside
      // the core params. Spread FIRST on purpose: later properties win in an
      // object literal, so a backend can never shadow a core field by picking a
      // colliding key.
      ...(s.backendParams || {}),
      // Multi-backend: which engine backend this request targets. Read directly
      // from backendStore (not a globalParamsStore field) so it's always the
      // live active id; defaults to 'ace' for installs with no second backend
      // registered (docs/plans/multi-backend-architecture.md §4.5).
      backend: options.backendId || 'ace',
      ditModel: s.ditModel, lmModel: s.lmModel, vaeModel: s.vaeModel, embeddingModel: s.embeddingModel,
      // Planner-LM adapter (runtime LoRA on the 5Hz LM) — aceReq fields, so
      // they survive the LM-echo synth rebuild by construction.
      // lmAdapterScale is emitted UNCONDITIONALLY: like the DiT loraScale, the
      // global strength governs album-preset planner adapters too (the preset
      // supplies only the path).
      lmAdapter: s.lmAdapter || undefined,
      lmAdapterScale: s.lmAdapterScale,
      loraPath: primary, loraScale: stack[0]?.scale ?? 1.0,
      // Multi-adapter stack (>1 entry) — sent alongside loraPath; the engine
      // prefers the stack and applies each adapter with its own scale.
      loraStack: stack.length > 0 ? stack : undefined,
      // Stack scaling mode + budget — reused for per-section masking transforms.
      adapterStackMode: s.adapterStackMode,
      adapterStackBudget: s.adapterStackBudget,
      // Per-section masking tuning (only meaningful with a 2+ adapter stack).
      adapterSectionAlignAt: stack.length >= 2 ? s.adapterSectionAlignAt : undefined,
      adapterSectionIsolation: stack.length >= 2 ? s.adapterSectionIsolation : undefined,
      adapterGroupScales: primary ? s.adapterGroupScales : undefined,
      adapterMode: primary ? s.adapterMode : 'merge',
      // Runtime delta quantization (VRAM saver) — relevant in both runtime modes
      // (lowrank still stores full-size re-base corrections / Conv1d fallbacks).
      // Also sent when timestep windows are active: they force runtime mode
      // server-side even from Merge, and gating on the *selected* mode silently
      // killed the knob there (full BF16 deltas, 2×8 GB — the 32 GB bug).
      adapterRuntimeQuant: (primary && (s.adapterMode === 'runtime' || s.adapterMode === 'runtime_lowrank' || hasStepWindows))
        ? s.adapterRuntimeQuant : undefined,
      // Merge low-VRAM storage (native-quant re-encode) — only relevant in merge mode.
      adapterMergeLowVram: (primary && s.adapterMode !== 'runtime' && s.adapterMergeLowVram) ? true : undefined,
      // No-adapter reference render — only meaningful with a DiT adapter loaded.
      noAdapterRender: (primary && s.noAdapterRender) ? true : undefined,
      // Basin re-base: only sent with an adapter and a chosen source. Works in
      // both merge and runtime modes (runtime folds the nudge into the delta sum);
      // the engine skips it on the per-section masking path.
      rebaseSource: (primary && s.rebaseSource) ? s.rebaseSource : undefined,
      rebaseBeta: (primary && s.rebaseSource) ? s.rebaseBeta : undefined,
      triggerSpecs: triggerSpecs.length ? triggerSpecs : undefined,
      triggerWord: triggerWord || undefined,
      triggerWords: triggerWords.length ? triggerWords : undefined,
      // '|| prepend' fallback matches the queue path (audioGenQueueStore):
      // an ace-settings object saved before triggerPlacement existed has the
      // key undefined, and translateParams skips injection entirely without a
      // placement — which silently dropped trigger words on Create/custom-gen.
      triggerPlacement: triggerWords.length ? triggerPlacement : undefined,
      inferenceSteps: s.inferenceSteps, guidanceScale: s.guidanceScale, shift: s.shift,
      // ditSlidingWindow deliberately not sent while the feature is parked, so
      // requests are byte-identical to pre-feature generations.
      cfgCutoffRatio: s.cfgCutoffRatio < 1.0 ? s.cfgCutoffRatio : undefined,
      lmCfgCutoffRatio: s.lmCfgCutoffRatio < 1.0 ? s.lmCfgCutoffRatio : undefined,
      cacheRatio: s.cacheRatio > 0 ? s.cacheRatio : undefined,
      inferMethod: s.inferMethod, scheduler: s.scheduler, guidanceMode: s.guidanceMode,
      seed: s.seed, randomSeed: s.randomSeed,
      lmSeed: s.lmSeed, lmSeedFollowsDit: s.lmSeedFollowsDit,
      batchSize: s.batchSize,
      storkSubsteps: (s.inferMethod === 'stork2' || s.inferMethod === 'stork4') ? s.storkSubsteps : undefined,
      beatStability: s.inferMethod === 'jkass_fast' ? s.beatStability : undefined,
      frequencyDamping: s.inferMethod === 'jkass_fast' ? s.frequencyDamping : undefined,
      temporalSmoothing: s.inferMethod === 'jkass_fast' ? s.temporalSmoothing : undefined,
      apgMomentum: s.guidanceMode === 'apg' ? s.apgMomentum : undefined,
      apgNormThreshold: s.guidanceMode === 'apg' ? s.apgNormThreshold : undefined,
      skipLm: s.skipLm, useCotCaption: s.useCotCaption,
      skipLrc: s.skipLrc || undefined,
      lmTemperature: s.lmTemperature, lmCfgScale: s.lmCfgScale,
      lmTopK: s.lmTopK, lmTopP: s.lmTopP, lmNegativePrompt: s.lmNegativePrompt,
      lmRepPenalty: s.lmRepPenalty > 1.0 ? s.lmRepPenalty : undefined,
      lmRepWindow: s.lmRepPenalty > 1.0 ? s.lmRepWindow : undefined,
      lmRepMode: s.lmRepPenalty > 1.0 ? s.lmRepMode : undefined,
      lmDryBase: (s.lmRepPenalty > 1.0 && s.lmRepMode === 'dry') ? s.lmDryBase : undefined,
      lmDryMinLen: (s.lmRepPenalty > 1.0 && s.lmRepMode === 'dry') ? s.lmDryMinLen : undefined,
      audioCoverStrength: (!s.skipLm && lmCodesEff < 1.0) ? lmCodesEff : undefined,
      postProcessingEnabled: s.postProcessingEnabled,
      spectralLifterEnabled: s.postProcessingEnabled ? s.spectralLifterEnabled : false,
      slDenoiseStrength: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slDenoiseStrength : undefined,
      slNoiseFloor: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slNoiseFloor : undefined,
      slHfMix: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slHfMix : undefined,
      slTransientBoost: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slTransientBoost : undefined,
      slShimmerReduction: (s.postProcessingEnabled && s.spectralLifterEnabled) ? s.slShimmerReduction : undefined,
      masteringEnabled: s.postProcessingEnabled ? s.masteringEnabled : false,
      masteringReference: (s.postProcessingEnabled && s.masteringEnabled) ? s.masteringReference : undefined,
      timbreReference: s.timbreAudioPath
        ? s.timbreAudioPath
        : (s.postProcessingEnabled && s.masteringEnabled && s.timbreReference && s.masteringReference) ? true : undefined,
      dcwEnabled: s.dcwEnabled,
      dcwMode: s.dcwEnabled ? s.dcwMode : undefined,
      // Route the correct scaler to dcw_scaler based on mode:
      // low/double/pix use dcwLowScaler, high uses dcwHighScaler
      dcwScaler: s.dcwEnabled
        ? (s.dcwMode === 'high' ? s.dcwHighScaler * 0.02 : s.dcwLowScaler * 0.05)
        : undefined,
      dcwHighScaler: (s.dcwEnabled && s.dcwMode === 'double') ? s.dcwHighScaler * 0.02 : undefined,
      latentShift: s.latentShift !== 0 ? s.latentShift : undefined,
      latentRescale: s.latentRescale !== 1 ? s.latentRescale : undefined,
      customTimesteps: s.customTimesteps || undefined,
      denoiseStrength: s.denoiseStrength > 0 ? s.denoiseStrength : undefined,
      denoiseSmoothing: s.denoiseStrength > 0 ? s.denoiseSmoothing : undefined,
      denoiseMix: s.denoiseStrength > 0 ? s.denoiseMix : undefined,
      lssStrength: s.lssStrength > 0 ? s.lssStrength : undefined,
      lssVarThresh: s.lssStrength > 0 ? s.lssVarThresh : undefined,
      lssDcRemove: s.lssStrength > 0 ? s.lssDcRemove : undefined,
      pluginParams: Object.keys(s.pluginParams).length > 0 ? s.pluginParams : undefined,
      autoTrimEnabled: s.autoTrimEnabled || undefined,
      durationBuffer: s.autoTrimEnabled ? s.durationBuffer : undefined,
      autoTrimFadeMs: s.autoTrimEnabled ? s.autoTrimFadeMs : undefined,
      vocalNaturalizerEnabled: s.postProcessingEnabled ? s.vocalNaturalizerEnabled : false,
      gainOffsetDb: (s.postProcessingEnabled && s.gainOffsetDb !== 0) ? s.gainOffsetDb : undefined,
      naturalizeAmount: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.naturalizeAmount : undefined,
      natVibratoRate: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natVibratoRate : undefined,
      natVibratoDepth: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natVibratoDepth : undefined,
      natFormantStrength: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natFormantStrength : undefined,
      natMetallicReduction: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natMetallicReduction : undefined,
      natQuantizationMask: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natQuantizationMask : undefined,
      natTransitionSmooth: (s.postProcessingEnabled && s.vocalNaturalizerEnabled) ? s.natTransitionSmooth : undefined,
      ppVaeReencode: (s.postProcessingEnabled && s.ppVaeReencode) || undefined,
      ppVaeBlend: (s.postProcessingEnabled && s.ppVaeReencode && s.ppVaeBlend > 0) ? s.ppVaeBlend : undefined,
      stableStepOn: (s.postProcessingEnabled && s.stableStepOn) || undefined,
      stableStepStrength: (s.postProcessingEnabled && s.stableStepOn) ? s.stableStepStrength : undefined,
      stableStepBackend: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBackend === 'gguf')
        ? 'gguf' : undefined,
      stableStepAdapters: (s.postProcessingEnabled && s.stableStepOn)
        ? (s.stableStepAdapters ?? [])
            .filter((a: any) => a.enabled && a.scale !== 0)
            .map((a: any) => ({ name: a.name, scale: a.scale }))
        : undefined,
      stableStepPreserveDynamics: (s.postProcessingEnabled && s.stableStepOn)
        ? s.stableStepPreserveDynamics !== false : undefined,
      // Opt-in only — omitted means "leave the AS1.5 vocals alone"
      stableStepVocalPpVae: (s.postProcessingEnabled && s.stableStepOn && s.stableStepVocalPpVae)
        || undefined,
      stableStepVocalTrimDb: (s.postProcessingEnabled && s.stableStepOn && s.stableStepVocalTrimDb !== 0)
        ? s.stableStepVocalTrimDb : undefined,
      stableStepBlendMode: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode !== 'off')
        ? s.stableStepBlendMode : undefined,
      stableStepCrossoverHz: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode === 'crossover')
        ? s.stableStepCrossoverHz : undefined,
      stableStepCrossoverWidthHz: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode === 'crossover')
        ? s.stableStepCrossoverWidthHz : undefined,
      stableStepMix: (s.postProcessingEnabled && s.stableStepOn && s.stableStepBlendMode === 'mix')
        ? s.stableStepMix : undefined,
      stableStepSeedFollowsDit: (s.postProcessingEnabled && s.stableStepOn)
        ? s.stableStepSeedFollowsDit !== false : undefined,
      stableStepSeed: (s.postProcessingEnabled && s.stableStepOn && s.stableStepSeedFollowsDit === false)
        ? s.stableStepSeed : undefined,
      // SA3 sampler routing. Each field is emitted only when it departs from the
      // engine default, so an untouched StableStep keeps the original
      // pingpong/euler path with nothing extra on the wire.
      stableStepSteps: (s.postProcessingEnabled && s.stableStepOn && s.stableStepSteps !== 8)
        ? s.stableStepSteps : undefined,
      stableStepSolver: (s.postProcessingEnabled && s.stableStepOn && s.stableStepSolver)
        ? s.stableStepSolver : undefined,
      stableStepScheduler: (s.postProcessingEnabled && s.stableStepOn && s.stableStepScheduler)
        ? s.stableStepScheduler : undefined,
      stableStepGuidanceMode: (s.postProcessingEnabled && s.stableStepOn && s.stableStepGuidanceMode)
        ? s.stableStepGuidanceMode : undefined,
      stableStepGuidanceScale: (s.postProcessingEnabled && s.stableStepOn && s.stableStepGuidanceMode
                                && s.stableStepGuidanceScale > 1)
        ? s.stableStepGuidanceScale : undefined,
      // Lua plugins read their declared params out of the shared pluginParams
      // map (keyed "<plugin>:<key>"), the same one the generation dropdowns
      // populate — so a plugin's knobs work identically in both places.
      stableStepPluginParams: (s.postProcessingEnabled && s.stableStepOn
                               && (s.stableStepSolver || s.stableStepScheduler || s.stableStepGuidanceMode))
        ? s.pluginParams : undefined,
      coverArtEnabled: (s.postProcessingEnabled && s.coverArtEnabled) || undefined,
      coverArtSubject: (s.postProcessingEnabled && s.coverArtEnabled && s.coverArtSubject) ? s.coverArtSubject : undefined,
      qualityEvalEnabled: (s.postProcessingEnabled && s.qualityEvalEnabled) || undefined,
      qualityEvalTarget: (s.postProcessingEnabled && s.qualityEvalEnabled) ? s.qualityEvalTarget : undefined,
      postprocessPlugin: (s.postProcessingEnabled && s.postprocessEnabled && s.postprocessPlugin) ? s.postprocessPlugin : undefined,
      // Independent of masteringEnabled: the Final Normalizer is the last stage
      // in the chain whether or not the reference-mastering stage is on.
      lufsEnabled: (s.postProcessingEnabled && s.lufsEnabled) || undefined,
      lufsTarget: (s.postProcessingEnabled && s.lufsEnabled) ? s.lufsTarget : undefined,
      lufsCeilingDb: (s.postProcessingEnabled && s.lufsEnabled) ? s.lufsCeilingDb : undefined,
      whisperLyricsEnabled: s.whisperLyricsEnabled,
      whisperModel: s.whisperLyricsEnabled ? s.whisperModel : undefined,
      whisperLanguage: s.whisperLyricsEnabled ? s.whisperLanguage : undefined,
      whisperBeamSize: s.whisperLyricsEnabled ? s.whisperBeamSize : undefined,
      whisperIsolateVocals: s.whisperLyricsEnabled ? s.whisperIsolateVocals : undefined,
      // Sent unconditionally: the backend that ignores it does not read it,
      // and gating it on the active backend here would drop it on a render
      // queued while the manifest is still loading.
      yue2AlignLyrics: s.yue2AlignLyrics || undefined,
    };
}
