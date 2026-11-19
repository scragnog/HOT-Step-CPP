// BackendLmDropdown.tsx — LM cluster for backends without ACE's CoT LM stage.
//
// ACE's LmThinkingDropdown is built around its chain-of-thought metadata LM:
// thinking budget, CoT visibility, the skip-LM switch. None of that exists for
// a backend whose LM is an autoregressive PLANNER — a stage that is not
// optional, has no chain of thought to show, and cannot be skipped because
// there is no render without it.
//
// What such a backend does have is the sampling that drives that planner, plus
// whatever is done with its output (reuse, save, replay). Those knobs were
// living in the Generation panel purely because it was the only generic
// cluster, so the tab an ACE user reaches for to find "temperature" was hidden
// in MM3 mode while the knob itself sat two tabs away. They now declare
// `group: 'lm'` and land here.
//
// No global on/off toggle: GlobalParamBar only hangs one on the section when
// `features.lm` is true (ACE), because only there does skipping the LM mean
// anything.
//
// Nothing here is MM3-specific — it renders declared schema.

import React from 'react';
import { useTranslation } from 'react-i18next';
import { useBackendStore } from '@/stores/backendStore';
import {
  BackendExtensionControls,
  useBackendExtensions,
  useBackendExtensionSummary,
} from './BackendExtensionControls';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { useGlobalParamsStore } from '@/stores/globalParamsStore';

/** YuE2: unlimited composer top-k with temperature above 1.0 breaks the song.
 *  The server clamps temperature to 1.0 then; this tells the user up front. */
const Yue2SemanticWarning: React.FC = () => {
  const { t } = useTranslation();
  const bp = useGlobalParamsStore((s: any) => s.backendParams as Record<string, unknown>) ?? {};
  // Checkpoint's own composer temperature (the control's declared default),
  // which is what the engine uses when the slider is untouched.
  const ckptDefault = useBackendExtensions('lm').find(p => p.key === 'yue2SemTemperature')?.default;
  const set = (v: unknown) => v !== undefined && v !== null && v !== '';
  // The engine truncates top_k to int, so 0.5 is unlimited too.
  if (!set(bp.yue2SemTopK) || Math.trunc(Number(bp.yue2SemTopK)) !== 0) return null;
  const temp = set(bp.yue2SemTemperature) ? Number(bp.yue2SemTemperature)
    : (typeof ckptDefault === 'number' && Number.isFinite(ckptDefault) ? ckptDefault : 1.0);
  if (typeof temp !== 'number' || !Number.isFinite(temp) || !(temp > 1.0)) return null;
  return (
    <p className="text-[11px] text-amber-400 leading-relaxed border border-amber-500/40 rounded px-2 py-1">
      {t('globalBar.yue2SemanticWarning', { temp: temp.toFixed(2) })}
    </p>
  );
};

export const BackendLmDropdown: React.FC = () => {
  const displayName = useBackendStore(s =>
    s.backends.find(b => b.id === s.activeBackendId)?.displayName);
  const params = useBackendExtensions('lm');
  const activeBackendId = useBackendStore(s => s.activeBackendId);

  if (params.length === 0) {
    return (
      <p className="text-[11px] text-zinc-500 leading-relaxed">
        The active backend exposes no language-model controls.
      </p>
    );
  }

  // A backend that folds its knobs into sections titles them itself; a
  // second "Planner" heading above two accordions was one label too many.
  const sectioned = params.some((p) => p.section);
  return (
    <div className="space-y-3">
      {!sectioned && <ParamLabel label="Planner" underline={false}
        className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider"
        info={`${displayName ?? 'This backend'} plans the whole song as a token stream before a `
          + `single sample is rendered — these are that planner's controls. It always runs; `
          + `there is no render without it.`} />}
      {activeBackendId === 'yue2' && <Yue2SemanticWarning />}
      <BackendExtensionControls group="lm" accentColor="purple" />
    </div>
  );
};

/** Summary badge — the planner knobs moved off their defaults, or a plain
 *  "default" when none have been. */
export const BackendLmBadge: React.FC = () => {
  const changed = useBackendExtensionSummary('lm');
  return (
    <span className="text-[10px] text-zinc-500 font-mono truncate">
      {changed.length ? changed.join(' · ') : 'default'}
    </span>
  );
};
