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

import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { useCapabilities } from '../../hooks/useCapabilities';
import { Slider } from '../shared/Slider';
import { ParamLabel } from '../shared/ParamLabel';
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
  const declared = useBackendExtensions(group);

  // `visible_when`, same schema field the Lua plugin renderer honours: a param
  // that only makes sense once another one is on (a name field under a save
  // toggle) declares its dependency rather than the UI hardcoding the pair.
  //
  // Compared as STRINGS because that is what the schema can carry — a declared
  // `equals: 'true'` has to match a real boolean `true` from backendParams.
  // The dependency is looked up across ALL extensions, not just this group's,
  // so a control can depend on one that renders in another cluster.
  const { capabilities } = useCapabilities();
  const all = capabilities?.extensions ?? [];
  const valueOf = (key: string): unknown => {
    const dep = all.find((d) => d.key === key);
    return gp.backendParams?.[key] ?? dep?.default;
  };
  const params = declared.filter((p) => {
    const vw = p.visible_when as { key: string; equals?: string; not_equals?: string } | undefined;
    if (!vw) return true;
    const cur = String(valueOf(vw.key) ?? '');
    if (vw.not_equals !== undefined) return Number(cur) !== Number(vw.not_equals) && cur !== vw.not_equals;
    return cur === vw.equals;
  });

  const renderParam = (p: BackendExtensionParam): React.ReactNode => {
        const value = gp.backendParams?.[p.key] ?? p.default;
        if (p.type === 'slider') {
          return (
            <div key={p.key}>
              <Slider
                label={p.label}
                info={p.hint}
                value={typeof value === 'number' ? value : Number(p.default ?? 0)}
                onChange={(v: number) => gp.setBackendParam?.(p.key, v)}
                min={p.min ?? 0}
                max={p.max ?? 1}
                step={p.step ?? 0.1}
              />
            </div>
          );
        }
        if (p.type === 'toggle') {
          // The app's toggle, not a raw checkbox — and the hint is reachable
          // the way every other param type's is. A declared toggle used to
          // render neither, so knobs like Low-Step Compensation and Play While
          // Rendering shipped their whole explanation to nobody.
          return (
            <div key={p.key} className="flex items-start justify-between gap-3">
              <div className="flex-1 min-w-0">
                <ParamLabel label={p.label} info={p.hint}
                  className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
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
              <ParamLabel label={p.label} info={p.hint} rootClassName="flex mb-1.5"
                className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
              <select
                className={backendInputClasses}
                value={String(value ?? '')}
                onChange={(e) => gp.setBackendParam?.(p.key, e.target.value)}
              >
                {(p.options ?? []).map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
          );
        }
        return (
          <div key={p.key}>
            <ParamLabel label={p.label} info={p.hint} rootClassName="flex mb-1.5"
              className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
            <input
              className={backendInputClasses}
              value={String(value ?? '')}
              onChange={(e) => gp.setBackendParam?.(p.key, e.target.value)}
            />
          </div>
        );
  };

  // Sections (`section` on the schema): knobs without one render flat, as
  // before; the rest fold under one collapsible header per section, collapsed
  // by default so a backend with many planner knobs shows a couple of
  // headers rather than the whole list. The header counts how many of its
  // knobs sit off their default, so a folded section still says whether it
  // is doing anything.
  const flat = params.filter((p) => !p.section);
  const sections: string[] = [];
  for (const p of params) if (p.section && !sections.includes(p.section)) sections.push(p.section);
  // Open/closed state survives reloads (per backend + cluster + section).
  const backendId = capabilities?.backend ?? 'backend';
  const storageKey = `hs-ext-sections:${backendId}:${group}`;
  const [open, setOpenState] = useState<Record<string, boolean>>(() => {
    try { return JSON.parse(localStorage.getItem(storageKey) || '{}') as Record<string, boolean>; } catch { return {}; }
  });
  const setOpen = (next: Record<string, boolean>) => {
    setOpenState(next);
    try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* private mode etc. */ }
  };
  const hintOf = (name: string) => params.find((p) => p.section === name && p.section_hint)?.section_hint;
  const changedIn = (name: string) => params.filter((p) => p.section === name && (() => {
    const v = gp.backendParams?.[p.key];
    return !(v === undefined || v === null || v === '' || v === p.default);
  })()).length;

  return (
    <>
      {flat.map(renderParam)}
      {sections.map((name) => {
        const isOpen = !!open[name];
        const changed = changedIn(name);
        return (
          <div key={name} className="rounded-xl border border-white/10 bg-white/[0.03] overflow-hidden">
            <button
              type="button"
              onClick={() => setOpen({ ...open, [name]: !isOpen })}
              className="w-full flex items-center justify-between px-3 py-2 hover:bg-white/5 transition-colors"
            >
              <div className="flex items-center gap-2">
                <ChevronDown size={12} className={`text-zinc-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
                <div onClick={(e) => e.stopPropagation()}>
                  <ParamLabel label={name} underline={false} info={hintOf(name)}
                    className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider" />
                </div>
              </div>
              <span className="text-[10px] text-zinc-500 font-mono">{changed ? `${changed} changed` : 'default'}</span>
            </button>
            {isOpen && (
              <div className="px-3 pb-3 space-y-3">
                {params.filter((p) => p.section === name).map(renderParam)}
              </div>
            )}
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
