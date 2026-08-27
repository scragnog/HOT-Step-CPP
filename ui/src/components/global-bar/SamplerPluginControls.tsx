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

const selectClasses =
  'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 ' +
  'text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 ' +
  'outline-none transition-colors cursor-pointer';

const NATIVE = '';

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
  );

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
        <select
          className={selectClasses}
          value={gp.inferMethod ?? NATIVE}
          onChange={e => gp.setInferMethod(e.target.value)}
        >
          <option value={NATIVE}>Native (Euler)</option>
          {solvers.filter((s: any) => (s.nfe ?? 1) === 1).length > 0 && (
            <optgroup label="── Single Evaluation (1 NFE) ──">
              {solvers.filter((s: any) => (s.nfe ?? 1) === 1).map((s: any) => (
                <option key={s.name} value={s.name}>{s.display}</option>
              ))}
            </optgroup>
          )}
          {solvers.filter((s: any) => (s.nfe ?? 1) > 1).length > 0 && (
            <optgroup label="── Multi Evaluation ──">
              {solvers.filter((s: any) => (s.nfe ?? 1) > 1).map((s: any) => (
                <option key={s.name} value={s.name}>{s.display} ({s.nfe} NFE)</option>
              ))}
            </optgroup>
          )}
        </select>
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
        <ParamLabel label="Schedule" info={schedMeta?.description} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <select
          className={selectClasses}
          value={gp.scheduler ?? NATIVE}
          onChange={e => gp.setScheduler(e.target.value)}
        >
          <option value={NATIVE}>Native</option>
          {schedulers.map((s: any) => (
            <option key={s.name} value={s.name}>{s.display}</option>
          ))}
        </select>
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
        <ParamLabel label="Guidance" info={guideMeta?.description} rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <select
          className={selectClasses}
          value={gp.guidanceMode ?? NATIVE}
          onChange={e => gp.setGuidanceMode(e.target.value)}
        >
          <option value={NATIVE}>Native (plain CFG)</option>
          {guidance.map((g: any) => (
            <option key={g.name} value={g.name}>{g.display}</option>
          ))}
        </select>
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
