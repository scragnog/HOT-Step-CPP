// BackendExtensionControls.tsx — generic renderer for backend-declared knobs
//
// A backend without ACE's Lua plugin registry still has knobs, and it declares
// them as schema (capabilities().extensions) rather than as components. This
// renders that schema: one control per declared param, written straight into
// globalParams.backendParams under the param's own key.
//
// Split out of BackendGenerationDropdown when the knobs stopped belonging to a
// single cluster. Each param carries a `group`, and each dropdown asks for its
// own group — so a new knob lands in the right panel by declaring where it
// goes, with no UI change at all.

import React from 'react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { useCapabilities } from '../../hooks/useCapabilities';
import { Slider } from '../shared/Slider';
import { ToggleSwitch } from './BarSection';
import type { BackendExtensionGroup, BackendExtensionParam } from '../../stores/backendStore';

export const backendInputClasses =
  'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 ' +
  'text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20';

/** The declared knobs belonging to one cluster. An untagged param is a
 *  'generation' param — that is where every knob lived before groups. */
export function useBackendExtensions(group: BackendExtensionGroup): BackendExtensionParam[] {
  const { capabilities } = useCapabilities();
  return (capabilities?.extensions ?? []).filter(p => (p.group ?? 'generation') === group);
}

type Accent = 'pink' | 'emerald' | 'sky' | 'purple' | 'amber' | 'teal';

export const BackendExtensionControls: React.FC<{
  group: BackendExtensionGroup;
  /** Cluster accent, so a declared toggle matches the section it lives in
   *  (the same one BarSection is given in GlobalParamBar). */
  accentColor?: Accent;
}> = ({ group, accentColor = 'sky' }) => {
  const gp = useGlobalParams() as any;
  const params = useBackendExtensions(group);

  return (
    <>
      {params.map((p) => {
        const value = gp.backendParams?.[p.key] ?? p.default;
        if (p.type === 'slider') {
          return (
            <div key={p.key}>
              <Slider
                label={p.label}
                value={typeof value === 'number' ? value : Number(p.default ?? 0)}
                onChange={(v: number) => gp.setBackendParam?.(p.key, v)}
                min={p.min ?? 0}
                max={p.max ?? 1}
                step={p.step ?? 0.1}
              />
              {p.hint && <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">{p.hint}</p>}
            </div>
          );
        }
        if (p.type === 'toggle') {
          // The app's toggle, not a raw checkbox — and the hint goes UNDER the
          // label the way every other param type gets one. A declared toggle
          // used to render neither, so knobs like Low-Step Compensation and
          // Play While Rendering shipped their whole explanation to nobody.
          return (
            <div key={p.key} className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <span className="block text-xs font-medium text-zinc-500 uppercase tracking-wider">
                  {p.label}
                </span>
                {p.hint && <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">{p.hint}</p>}
              </div>
              <div className="pt-0.5">
                <ToggleSwitch
                  checked={!!value}
                  onChange={(on) => gp.setBackendParam?.(p.key, on)}
                  accentColor={accentColor}
                />
              </div>
            </div>
          );
        }
        if (p.type === 'select') {
          return (
            <div key={p.key}>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
                {p.label}
              </label>
              <select
                className={backendInputClasses}
                value={String(value ?? '')}
                onChange={(e) => gp.setBackendParam?.(p.key, e.target.value)}
              >
                {(p.options ?? []).map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              {p.hint && <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">{p.hint}</p>}
            </div>
          );
        }
        return (
          <div key={p.key}>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
              {p.label}
            </label>
            <input
              className={backendInputClasses}
              value={String(value ?? '')}
              onChange={(e) => gp.setBackendParam?.(p.key, e.target.value)}
            />
            {p.hint && <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">{p.hint}</p>}
          </div>
        );
      })}
    </>
  );
};

/** Compact "Label: value" list for a cluster's badge. Only knobs that are
 *  actually set are worth the width, so a param sitting on its declared
 *  default is skipped — a badge that always reads the same tells you nothing. */
export const useBackendExtensionSummary = (group: BackendExtensionGroup): string[] => {
  const gp = useGlobalParams() as any;
  const params = useBackendExtensions(group);
  const parts: string[] = [];
  for (const p of params) {
    const v = gp.backendParams?.[p.key];
    if (v === undefined || v === null || v === '' || v === p.default) continue;
    parts.push(`${p.label}: ${typeof v === 'boolean' ? (v ? 'on' : 'off') : v}`);
  }
  return parts;
};
