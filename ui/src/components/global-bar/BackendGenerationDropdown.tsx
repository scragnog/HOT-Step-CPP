// BackendGenerationDropdown.tsx — Generation cluster for backends without the
// ACE plugin system.
//
// ACE's GenerationDropdown is built around the Lua plugin registry (solvers,
// schedulers, guidance) plus a pile of ACE-specific sampler knobs, none of
// which exist for another backend. But "Generation" is not only plugins: the
// SEED lives here, and seed is backend-agnostic — hiding the whole cluster left
// MiniMax-Music3 with no way to change it, so every render of a given prompt
// came back identical (the persisted hs-randomSeed / hs-seed pair was being
// used with no control to reach it).
//
// This renders the parts that are genuinely backend-agnostic:
//   - the shared SeedControl (same component ACE uses)
//   - whatever knobs the backend DECLARES via capabilities().extensions,
//     minus the ones tagged for another cluster
//
// The backend's planner/LM knobs used to pile up in here too, which put its
// sampling controls in a panel called Generation while the LM tab — the place
// an ACE user looks for exactly those — sat hidden. They now declare
// `group: 'lm'` and render in BackendLmDropdown instead.
//
// Nothing here is MM3-specific; a future backend that declares extensions gets
// them rendered without touching this file.

import React from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { useCapabilities } from '../../hooks/useCapabilities';
import { SeedControl } from './SeedControl';
import { SamplerPluginControls } from './SamplerPluginControls';
import {
  BackendExtensionControls,
  useBackendExtensions,
  useBackendExtensionSummary,
  backendInputClasses as inputClasses,
} from './BackendExtensionControls';

export const BackendGenerationDropdown: React.FC = () => {
  const gp = useGlobalParams() as any;
  const { capabilities } = useCapabilities();

  const extensions = useBackendExtensions('generation');
  const supportsSeed = capabilities?.core?.seed !== false;

  return (
    <div className="space-y-3">
      {supportsSeed && (
        <SeedControl
          inputClasses={inputClasses}
          hint="Drives the whole render — the token plan and the flow sampling. Turn Random off to reproduce a take exactly."
        />
      )}

      <BackendExtensionControls group="generation" accentColor="sky" />

      {/* Shared Lua sampler plugins, for a backend that runs them but does not
          render ACE's GenerationDropdown. Two conditions, both necessary: the
          backend must claim the capability, AND the user must have switched on
          the declared `samplerPluginsEnabled` knob above — the picks are shared
          global state, so showing live pickers that are not being sent would be
          a lie in the other direction. */}
      {capabilities?.features?.samplerPlugins && !!gp.backendParams?.samplerPluginsEnabled && (
        <div className="pt-2 border-t border-zinc-300 dark:border-white/10">
          <SamplerPluginControls />
        </div>
      )}

      {extensions.length === 0 && !supportsSeed && (
        <p className="text-[11px] text-zinc-500 leading-relaxed">
          The active backend exposes no generation controls.
        </p>
      )}
    </div>
  );
};

/** Summary badge — seed mode plus any declared knob moved off its default. */
export const BackendGenerationBadge: React.FC = () => {
  const gp = useGlobalParams() as any;
  const changed = useBackendExtensionSummary('generation');
  const parts = [gp.randomSeed ? 'Rnd' : `Seed ${gp.seed}`, ...changed];
  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">{parts.join(' · ')}</span>
  );
};
