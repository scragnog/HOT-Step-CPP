// Sa3SamplerControls.tsx — solver / scheduler / guidance pickers for the
// StableStep (SA3) refine.
//
// Same registry, same PluginControls renderer and same shared pluginParams map
// as the generation-side pickers — but bound to its OWN store fields
// (stableStepSolver/Scheduler/GuidanceMode). StableStep is a post-processing
// refine, not the generation sampler; sharing gp.inferMethod would mean picking
// a solver here silently changed the DiT's solver too.
//
// Engine side: engine/src/sa3-refine.h (Sa3PluginParams), reached through the
// /sa3-refine query string. Leaving every picker on Native keeps the original
// pingpong path, bit-identical.
//
// DIFFERENCES FROM SamplerPluginControls (the MM3 one) — both are real:
//   • owns_loop solvers are still listed here, unlike MM3, but they never run:
//     sa3-refine.h swaps them for native pingpong on the refine's pingpong
//     path, because a full-loop solver skips the per-step re-noise the
//     distilled SA3 model needs.
//   • Guidance is close to decorative. SA3 was trained at cfg=1 and has no
//     unconditional branch, so the engine passes the cond velocity as both
//     predictions. APG-family modes see diff = 0 and pass through unchanged;
//     only plugins that do cond-side work have any effect. The warning below
//     says so rather than letting the dropdown imply more than it delivers.

import React from 'react';
import { useGlobalParams } from '@/context/GlobalParamsContext';
import { usePluginRegistry } from '@/hooks/usePluginRegistry';
import { PluginControls } from './PluginControls';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { EditableSlider } from '@/components/shared/controls/EditableSlider';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { GLOBAL_PARAM_DEFAULTS } from '@/stores/globalParamsStore';
import { isPresent } from '@/utils/guards';

// Registered plugin names for the SA3 Clean preset (verified against plugins/).
const SA3_CLEAN_SOLVER    = 'md_wasserstein_flow_v121';
const SA3_CLEAN_SCHEDULER = 'md_psychoacoustic';
const SA3_CLEAN_GUIDANCE  = 'md_phase_inversion';
const SA3_CLEAN_STEPS     = 24;
// Engine-registered name for Bucket Brigade guidance.
const BUCKET_BRIGADE_NAME = 'md_bucket_brigade';

// This panel's own colour, matched to the StableStep Accordion's
// accentColor="sky" in PostProcessingDropdown.tsx (the panel this mounts
// inside). Every StyledSelect here shares it.
const ACCENT = 'sky' as const;

const NATIVE = '';

export const Sa3SamplerControls: React.FC = () => {
  const gp = useGlobalParams() as any;
  const { registry, findSolver, findScheduler, findGuidance } = usePluginRegistry();

  const solvers = registry.solvers ?? [];
  const schedulers = registry.schedulers ?? [];
  const guidance = registry.guidance ?? [];

  const solverMeta = findSolver(gp.stableStepSolver);
  const schedMeta = findScheduler(gp.stableStepScheduler);
  const guideMeta = findGuidance(gp.stableStepGuidanceMode);

  const singleNfe = solvers.filter((s: any) => (s.nfe ?? 1) === 1);
  const multiNfe = solvers.filter((s: any) => (s.nfe ?? 1) > 1);

  // A picker's hover card is the selected plugin's own description plus
  // whatever caveat that selection earns. These used to be stacked paragraphs
  // under the <select> — three of them at once, in the deepest nested panel in
  // the app.
  const join = (...parts: (string | false | undefined)[]) =>
    parts.filter(isPresent).join(' ') || undefined;

  const solverInfo = join(
    solverMeta?.description,
    gp.stableStepSolver && 'A solver replaces the ping-pong re-noise rather than stacking with it, so the refine becomes deterministic in that stage.',
    (solverMeta?.nfe ?? 1) > 1 && 'Multi-evaluation solvers run extra forward passes per step; the refine scales roughly with NFE.',
  );

  const schedInfo = join(
    schedMeta?.description,
    gp.stableStepScheduler && 'Schedulers are written for full denoising, starting at 1.0. The refine only partly re-noises, so the curve is rescaled onto the refine strength — its shape is kept, its starting point is not.',
  );

  const guideInfo = join(
    guideMeta?.description,
    gp.stableStepGuidanceMode && 'SA3 has no unconditional branch (it was trained without CFG), so guidance runs with the conditional prediction as both inputs. APG-style modes pass straight through unchanged; only plugins that enhance the conditional prediction on its own do anything here.',
  );

  const applyCleanPreset = () => {
    gp.setStableStepSolver(SA3_CLEAN_SOLVER);
    gp.setStableStepScheduler(SA3_CLEAN_SCHEDULER);
    gp.setStableStepGuidanceMode(SA3_CLEAN_GUIDANCE);
    gp.setStableStepSteps(SA3_CLEAN_STEPS);
  };

  const bucketBrigadeActive = gp.stableStepGuidanceMode === BUCKET_BRIGADE_NAME;

  return (
    <div className="space-y-3 pt-1">
      {/* SA3 Clean preset */}
      <div className="flex items-center justify-between">
        <ParamLabel label="Preset" underline={false}
          className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider"
          info="SA3 Clean: Wasserstein Flow solver, Psychoacoustic scheduler, Phase Inversion guidance, 24 steps. A tested combination that sounds clean on SA3 refine." />
        <button
          type="button"
          onClick={applyCleanPreset}
          className="text-[10px] font-semibold px-2 py-1 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-400 border border-sky-500/20 transition-colors"
        >
          SA3 Clean
        </button>
      </div>

      <EditableSlider
        label="Refine steps"
        value={gp.stableStepSteps}
        min={1} max={64} step={1}
        onChange={gp.setStableStepSteps}
        defaultValue={GLOBAL_PARAM_DEFAULTS.stableStepSteps}
        tooltip="Sampler steps for the refine. 8 is the tuned default; more steps cost time roughly linearly."
      />

      <ParamLabel label="Refine Sampler" underline={false}
        className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider"
        info={"The same solver, scheduler and guidance plugins the generation sampler uses, "
          + "applied to the StableStep refine. Leaving a picker on Native keeps the original "
          + "tested path. These picks are separate from the Generation dropdown's."} />

      {/* Solver */}
      <div>
        <ParamLabel label="Refine solver" info={solverInfo} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
          onReset={gp.stableStepSolver !== GLOBAL_PARAM_DEFAULTS.stableStepSolver ? () => gp.setStableStepSolver(GLOBAL_PARAM_DEFAULTS.stableStepSolver) : undefined} />
        <StyledSelect
          accent={ACCENT}
          className="w-full"
          value={gp.stableStepSolver ?? NATIVE}
          onChange={(v: string) => gp.setStableStepSolver(v)}
          options={[
            { value: NATIVE, label: 'Native (ping-pong)' },
            ...singleNfe.map((s: any) => ({ value: s.name, label: s.display, hint: '1 NFE' })),
            ...multiNfe.map((s: any) => ({ value: s.name, label: s.display, hint: `${s.nfe} NFE` })),
          ]}
        />
      </div>

      {solverMeta && solverMeta.params?.length > 0 && (
        <PluginControls
          pluginName={solverMeta.name}
          displayName={solverMeta.display}
          accent={solverMeta.accent}
          params={solverMeta.params}
          values={gp.pluginParams}
          onChange={gp.setPluginParam}
          onReset={() => gp.resetPluginParams(solverMeta.name)}
        />
      )}

      {/* Scheduler */}
      <div>
        <ParamLabel label="Refine schedule" info={schedInfo} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
          onReset={gp.stableStepScheduler !== GLOBAL_PARAM_DEFAULTS.stableStepScheduler ? () => gp.setStableStepScheduler(GLOBAL_PARAM_DEFAULTS.stableStepScheduler) : undefined} />
        <StyledSelect
          accent={ACCENT}
          className="w-full"
          value={gp.stableStepScheduler ?? NATIVE}
          onChange={(v: string) => gp.setStableStepScheduler(v)}
          options={[
            { value: NATIVE, label: 'Native (SA3 LogSNR)' },
            ...schedulers.map((s: any) => ({ value: s.name, label: s.display })),
          ]}
        />
      </div>

      {schedMeta && schedMeta.params?.length > 0 && (
        <PluginControls
          pluginName={schedMeta.name}
          displayName={schedMeta.display}
          accent={schedMeta.accent}
          params={schedMeta.params}
          values={gp.pluginParams}
          onChange={gp.setPluginParam}
          onReset={() => gp.resetPluginParams(schedMeta.name)}
        />
      )}

      {/* Guidance */}
      <div>
        <ParamLabel label="Refine guidance" info={guideInfo} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
          onReset={gp.stableStepGuidanceMode !== GLOBAL_PARAM_DEFAULTS.stableStepGuidanceMode ? () => gp.setStableStepGuidanceMode(GLOBAL_PARAM_DEFAULTS.stableStepGuidanceMode) : undefined} />
        <StyledSelect
          accent={ACCENT}
          className="w-full"
          value={gp.stableStepGuidanceMode ?? NATIVE}
          onChange={(v: string) => gp.setStableStepGuidanceMode(v)}
          options={[
            { value: NATIVE, label: 'None' },
            ...guidance.map((g: any) => ({ value: g.name, label: g.display })),
          ]}
        />
      </div>

      {bucketBrigadeActive && (
        <p className="text-[10px] text-amber-400 leading-snug px-1">
          Bucket Brigade tends to distort on SA3.
        </p>
      )}

      {gp.stableStepGuidanceMode && (
        <EditableSlider
          label="Refine guidance scale"
          value={gp.stableStepGuidanceScale}
          min={1.0} max={10.0} step={0.1}
          onChange={gp.setStableStepGuidanceScale}
          defaultValue={GLOBAL_PARAM_DEFAULTS.stableStepGuidanceScale}
          tooltip="Passed to the guidance plugin. 1.0 disables guidance entirely."
        />
      )}

      {guideMeta && guideMeta.params?.length > 0 && (
        <PluginControls
          pluginName={guideMeta.name}
          displayName={guideMeta.display}
          accent={guideMeta.accent}
          params={guideMeta.params}
          values={gp.pluginParams}
          onChange={gp.setPluginParam}
          onReset={() => gp.resetPluginParams(guideMeta.name)}
        />
      )}
    </div>
  );
};
