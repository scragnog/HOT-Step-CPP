// SamplerPluginControls.tsx — solver / scheduler / guidance pickers for a
// backend that RUNS the Lua sampler plugins but does not render ACE's
// GenerationDropdown.
//
// Why this exists rather than reusing GenerationDropdown's sections: that
// component is one 700-line block mixing the plugin pickers with a pile of
// ACE-only knobs (DCW, latent shift, LSS, timbre references, auto-trim, the
// LM cache). Lifting the three pickers out of it would be a large, risky
// refactor of the app's busiest control surface. This is the same registry,
// the same store fields and the same PluginControls renderer — just the subset
// a second backend can actually use.
//
// Gated on capabilities.features.samplerPlugins (server: backends/types.ts).
// Currently that means MiniMax-Music3, whose flow DiT runs these plugins via
// the convention bridge in engine minimax/mm3-plugins.h.
//
// DELIBERATELY OMITTED, because the MM3 bridge does not implement them:
//   • the synthetic "beta / power / composite" scheduler entries. Parameterised
//     names survive (scheduler_lookup prefix-matches "power:4.00" → "power"),
//     but `composite:A+B:...` is built by sampler-schedule.h from ACE's own
//     param globals and has no MM3 path — it would silently fall back to the
//     native schedule.
//   • full-loop (owns_loop) solvers, filtered out below. The engine also
//     refuses them, but a dropdown that offers a choice the engine drops is
//     worse than one that never offered it.

import React from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePluginRegistry } from '../../hooks/usePluginRegistry';
import { PluginControls } from './PluginControls';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';

const NATIVE = '';

// This picker cluster lives inside BarSection's dropdown panel, which already
// carries data-hovercard-boundary — no need to add it here.
const ACCENT = 'sky';

export const SamplerPluginControls: React.FC = () => {
  const gp = useGlobalParams() as any;
  const { registry, findSolver, findScheduler, findGuidance } = usePluginRegistry();

  // owns_loop solvers drive their own iteration and would bypass MM3's
  // per-step overlap blend, breaking every window seam.
  const solvers = (registry.solvers ?? []).filter(s => !s.owns_loop);
  const schedulers = registry.schedulers ?? [];
  const guidance = registry.guidance ?? [];

  const solverMeta = findSolver(gp.inferMethod);
  const schedMeta = findScheduler(gp.scheduler);
  const guideMeta = findGuidance(gp.guidanceMode);

  // Each picker's hover card is the selected plugin's own description plus any
  // caveat that selection earns.
  const join = (...parts: (string | false | undefined)[]) =>
    parts.filter(Boolean).join(' ') || undefined;

  const solverInfo = join(
    solverMeta?.description,
    (solverMeta?.nfe ?? 1) > 1 && 'Multi-evaluation solvers run extra forward passes per step. On this backend the flow stage is already the bulk of the render time, so expect it to scale roughly with NFE.',
  ) ?? 'Decides how each denoising step moves from noise toward audio. Solvers are grouped by network evaluations (NFE) per step: a 2-NFE solver at 20 steps costs about what a 1-NFE solver costs at 40. Native (Euler) is this backend\'s own solver.';

  const schedInfo = schedMeta?.description
    ?? 'Decides where the sampling steps fall across the noise range. Native keeps this backend\'s own default schedule.';

  const guideInfo = guideMeta?.description
    ?? 'Decides how the conditional and unconditional predictions are combined each step, using the Guidance Scale. Native (plain CFG) is classifier-free guidance with no extra shaping.';

  const solverOptions = [
    { value: NATIVE, label: 'Native (Euler)' },
    ...solvers
      .filter((s: any) => (s.nfe ?? 1) === 1)
      .map((s: any) => ({ value: s.name as string, label: s.display as string })),
    ...solvers
      .filter((s: any) => (s.nfe ?? 1) > 1)
      .map((s: any) => ({
        value: s.name as string,
        label: `${s.display} (${s.nfe} NFE)`,
        hint: 'Multi-evaluation: extra forward passes per step.',
      })),
  ];

  const schedulerOptions = [
    { value: NATIVE, label: 'Native' },
    ...schedulers.map((s: any) => ({ value: s.name as string, label: s.display as string })),
  ];

  const guidanceOptions = [
    { value: NATIVE, label: 'Native (plain CFG)' },
    ...guidance.map((g: any) => ({ value: g.name as string, label: g.display as string })),
  ];

  return (
    <div className="space-y-3">
      <ParamLabel label="Sampler Plugins" underline={false}
        className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider"
        info={"The same solver, scheduler and guidance plugins the ACE-Step backend uses. "
          + "Leaving a picker on Native keeps this backend's own, parity-tested sampling for that stage."} />

      {/* Solver */}
      <div>
        <ParamLabel label="Solver" info={solverInfo} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <StyledSelect
          accent={ACCENT}
          value={gp.inferMethod ?? NATIVE}
          onChange={v => gp.setInferMethod(v)}
          options={solverOptions}
          className="w-full"
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
        <ParamLabel label="Schedule" info={schedInfo} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <StyledSelect
          accent={ACCENT}
          value={gp.scheduler ?? NATIVE}
          onChange={v => gp.setScheduler(v)}
          options={schedulerOptions}
          className="w-full"
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
        <ParamLabel label="Guidance" info={guideInfo} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <StyledSelect
          accent={ACCENT}
          value={gp.guidanceMode ?? NATIVE}
          onChange={v => gp.setGuidanceMode(v)}
          options={guidanceOptions}
          className="w-full"
        />
      </div>

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
