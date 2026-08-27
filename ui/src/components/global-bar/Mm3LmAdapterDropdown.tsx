// Mm3LmAdapterDropdown.tsx — Adapters cluster for a backend whose adapters are
// RUNTIME LM LoRAs rather than ACE's DiT adapter stack.
//
// Rendered instead of AdaptersDropdown when the active backend reports
// `features.lmAdapters` and not `features.adapters` (today: MiniMax-Music3).
// Gated on the capability, never on a backend id — see
// docs/plans/multi-backend-architecture.md §2 principle 2.
//
// What the knobs are, and why these defaults (docs/plans/2026-08-17-mm3-lm-route.md,
// the ck800 ablation grid Rob eared through on 2026-08-20):
//   - Attention 1.0 / MLP 0.5 ("mlp-half") is the production dial. Attention-only
//     is clean but loses the artist; MLP-only keeps the voice and drifts genre.
//   - The depth thirds are ADVANCED and default to 1.0 across the board. The late
//     third is load-bearing for sequence termination: halving it produced fades
//     and early/never-EOS; early-off produced chaos + runaway tempo.
//   - The adapter's own sidecar may recommend different scales; picking an
//     adapter prefills from it, so the ear-validated numbers travel with the file.
//
// Values are written into `backendParams` (globalParamsStore), which
// getGlobalParams() spreads into the request — so the keys here must match the
// ones server/src/services/backends/minimax/generate.ts reads. That bag is the
// generic extension channel, so no per-knob store field is needed; it also
// persists (hs-backendParams@<backendId> — the bag is per backend), which is
// why a pick survives a reload.

import React, { useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, ChevronDown, ChevronRight, RotateCcw } from 'lucide-react';
import { useGlobalParamsStore } from '../../stores/globalParamsStore';
import { Slider } from '../shared/Slider';

const inputClasses =
  'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 ' +
  'text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20';

/** Request-param keys, in the order the panel presents them. */
const PARAM = {
  adapter: 'mm3LmAdapter',
  mode:    'mm3LmAdapterMode',
  scale:   'mm3LmAdapterScale',
  attn:    'mm3LmAdapterScaleAttn',
  mlp:     'mm3LmAdapterScaleMlp',
  early:   'mm3LmAdapterScaleEarly',
  mid:     'mm3LmAdapterScaleMid',
  late:    'mm3LmAdapterScaleLate',
  trigger: 'mm3LmAdapterTrigger',
} as const;

interface Mm3LmAdapterScales {
  scale: number;
  scaleAttn: number;
  scaleMlp: number;
  scaleEarly: number;
  scaleMid: number;
  scaleLate: number;
}

interface Mm3LmAdapterEntry {
  /** Path relative to the mm3-lm-adapters root — what the request carries. */
  file: string;
  name?: string;
  trigger?: string;
  /** false when the run recorded a trigger it never trained. Absent on every
   *  sidecar written before the field existed, and absent means trained. */
  triggerPrepend?: boolean;
  rank?: number;
  dataset?: string;
  encoder?: string;
  trainedSteps?: number;
  recommendedScales?: Partial<Mm3LmAdapterScales>;
  notes?: string;
}

interface Mm3LmAdapterCatalogue {
  adapters: Mm3LmAdapterEntry[];
  defaultScales: Mm3LmAdapterScales;
  dir: string;
}

/** Matches MM3_LM_ADAPTER_DEFAULT_SCALES server-side; used only until the
 *  catalogue lands (or if it fails to), so the sliders are never NaN. */
const FALLBACK_SCALES: Mm3LmAdapterScales = {
  scale: 1.0, scaleAttn: 1.0, scaleMlp: 1.0, scaleEarly: 1.0, scaleMid: 1.0, scaleLate: 1.0,
};

/** Fetched once per mount; the list only changes when files are installed. */
function useMm3LmAdapters() {
  const [data, setData] = useState<Mm3LmAdapterCatalogue | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/mm3/lm-adapters')
      .then(r => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: Mm3LmAdapterCatalogue) => { if (!cancelled) setData(d); })
      .catch(err => {
        // Advisory: an unreachable catalogue means "no picker", not a broken
        // generation — the adapter is optional by construction.
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => { cancelled = true; };
  }, []);

  return { data, error };
}

export const Mm3LmAdapterDropdown: React.FC = () => {
  const { t } = useTranslation();
  const backendParams = useGlobalParamsStore(s => s.backendParams) as Record<string, unknown> | undefined;
  const setBackendParam = useGlobalParamsStore(s => s.setBackendParam) as (key: string, v: unknown) => void;
  const { data, error } = useMm3LmAdapters();
  const [showDepth, setShowDepth] = useState(false);

  const params = backendParams ?? {};
  const selected = String(params[PARAM.adapter] ?? '');
  const defaults = data?.defaultScales ?? FALLBACK_SCALES;
  const entry = data?.adapters.find(a => a.file === selected);

  const num = (key: string, fallback: number): number => {
    const v = Number(params[key]);
    return Number.isFinite(v) ? v : fallback;
  };

  /** Write all six dials at once — used on pick and on reset, so the sliders
   *  can never sit at a mix of one adapter's recommendation and another's. */
  const applyScales = useCallback((s: Mm3LmAdapterScales) => {
    setBackendParam(PARAM.scale, s.scale);
    setBackendParam(PARAM.attn,  s.scaleAttn);
    setBackendParam(PARAM.mlp,   s.scaleMlp);
    setBackendParam(PARAM.early, s.scaleEarly);
    setBackendParam(PARAM.mid,   s.scaleMid);
    setBackendParam(PARAM.late,  s.scaleLate);
  }, [setBackendParam]);

  const recommendedFor = useCallback((file: string): Mm3LmAdapterScales => ({
    ...defaults,
    ...(data?.adapters.find(a => a.file === file)?.recommendedScales ?? {}),
  }), [data, defaults]);

  const onPick = (file: string) => {
    setBackendParam(PARAM.adapter, file);
    // Prefill from the sidecar on every pick (including "None", which resets
    // to the house defaults) — the ear-validated numbers ship with the file.
    applyScales(recommendedFor(file));
  };

  if (error || (data && data.adapters.length === 0)) {
    return (
      <div className="space-y-2">
        {!error && (
          <p className="text-[11px] text-zinc-500 leading-relaxed">
            {t('globalBar.mm3LmNone',
              'No MiniMax-Music3 LM adapters installed. Drop a PEFT .safetensors (and an optional .json sidecar) into the folder below and reopen this panel.')}
          </p>
        )}
        {data?.dir && (
          <p className="text-[10px] text-zinc-600 font-mono break-all">{data.dir}</p>
        )}
        {error && (
          <p className="text-[10px] text-amber-500/80">{t('globalBar.mm3LmListFailed', 'Adapter list unavailable')}: {error}</p>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* ── Picker ── */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          {t('globalBar.mm3LmAdapter', 'LM Adapter')}
        </label>
        <select
          className={inputClasses}
          value={selected}
          onChange={e => onPick(e.target.value)}
        >
          <option value="">{t('globalBar.mm3LmAdapterNone', 'None — base model')}</option>
          {(data?.adapters ?? []).map(a => (
            <option key={a.file} value={a.file}>{a.name || a.file}</option>
          ))}
        </select>
        <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
          {t('globalBar.mm3LmAdapterHint',
            'A LoRA on the MiniMax-Music3 planner LM — artist identity lives here. Applied at runtime, so these strengths are live per generation. An adapter that fails to load fails the job rather than silently rendering the base model.')}
        </p>
      </div>

      {/* ── What this adapter is ── */}
      {entry && (
        <div className="px-3 py-2 rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 space-y-1">
          {entry.trigger && (
            <div className="flex items-baseline gap-2">
              <span className="text-[10px] uppercase tracking-wider text-zinc-500">
                {t('globalBar.mm3LmTrigger', 'Trigger')}
              </span>
              <span className="text-xs font-mono text-emerald-500">{entry.trigger}</span>
            </div>
          )}
          <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
            {[
              entry.dataset,
              entry.rank ? `rank ${entry.rank}` : '',
              entry.trainedSteps ? `${entry.trainedSteps} steps` : '',
            ].filter(Boolean).join(' · ')}
          </p>
          {entry.trigger && entry.triggerPrepend === false && (
            <p className="text-[10px] text-amber-600 dark:text-amber-500/80 leading-relaxed">
              {t('globalBar.mm3LmTriggerUntrained',
                'This run recorded a trigger but never trained it, so it is not added and typing it would only feed the model an unseen phrase.')}
            </p>
          )}
          {entry.trigger && entry.triggerPrepend !== false && (
            <label className="flex items-start gap-2 pt-0.5 cursor-pointer">
              <input
                type="checkbox"
                checked={params[PARAM.trigger] !== false}
                onChange={e => setBackendParam(PARAM.trigger, e.target.checked)}
                className="mt-0.5 accent-emerald-500"
              />
              <span className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
                {t('globalBar.mm3LmTriggerAuto',
                  'Add the trigger to the caption automatically, in the position it was trained in. The identity is bound to it, so leave this on unless you are placing it yourself. A caption that already opens with the trigger is left alone.')}
              </span>
            </label>
          )}
          {entry.notes && (
            <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed italic">{entry.notes}</p>
          )}
        </div>
      )}

      {/* ── Application mode ── */}
      {selected && (
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
            {t('globalBar.mm3LmMode', 'Application')}
          </label>
          <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10">
            {(['runtime', 'merge'] as const).map(m => {
              const active = String(params[PARAM.mode] ?? 'runtime') === m;
              return (
                <button
                  key={m}
                  type="button"
                  onClick={() => setBackendParam(PARAM.mode, m)}
                  className={
                    'flex-1 px-3 py-1.5 text-xs transition-colors ' +
                    (active
                      ? 'bg-emerald-500/20 text-emerald-500 font-medium'
                      : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300')
                  }
                >
                  {m === 'runtime'
                    ? t('globalBar.mm3LmModeRuntime', 'Runtime')
                    : t('globalBar.mm3LmModeMerge', 'Merge')}
                </button>
              );
            })}
          </div>
          <p className="text-[10px] text-zinc-500 mt-1 leading-relaxed">
            {String(params[PARAM.mode] ?? 'runtime') === 'merge'
              ? t('globalBar.mm3LmModeMergeHint',
                  'Folded into the model weights once per adapter+dial combination — planning runs at full base-model speed. Changing any dial re-merges (a few seconds). On a quantized LM the merge re-quantizes; ear-check against Runtime if in doubt.')
              : t('globalBar.mm3LmModeRuntimeHint',
                  'Applied as live low-rank deltas every step — dial changes cost nothing, but planning runs ~25% slower at rank 256. Pick Merge when the dials are settled.')}
          </p>
        </div>
      )}

      {/* ── Strength dials ── */}
      {selected && (
        <>
          <Slider
            label={t('globalBar.mm3LmStrength', 'Strength')}
            value={num(PARAM.scale, defaults.scale)}
            onChange={v => setBackendParam(PARAM.scale, v)}
            min={0} max={2} step={0.05} showInput
          />
          <Slider
            label={t('globalBar.mm3LmAttention', 'Attention')}
            value={num(PARAM.attn, defaults.scaleAttn)}
            onChange={v => setBackendParam(PARAM.attn, v)}
            min={0} max={2} step={0.05} showInput
          />
          <Slider
            label={t('globalBar.mm3LmMlp', 'MLP')}
            value={num(PARAM.mlp, defaults.scaleMlp)}
            onChange={v => setBackendParam(PARAM.mlp, v)}
            min={0} max={2} step={0.05} showInput
          />
          <p className="text-[10px] text-zinc-500 leading-relaxed">
            {t('globalBar.mm3LmGroupsHint',
              'Attention 1.0 with MLP 0.5 is the tested dial: the artist\'s timbre lives largely in the MLP projections, and halving them keeps the identity while holding the genre steady. Attention alone renders clean but generic; MLP alone keeps the voice and drifts genre.')}
          </p>

          <div className="flex items-center justify-between gap-2 pt-1">
            <button
              type="button"
              onClick={() => setShowDepth(v => !v)}
              className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              {showDepth ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {t('globalBar.mm3LmDepth', 'Depth thirds (advanced)')}
            </button>
            <button
              type="button"
              onClick={() => applyScales(recommendedFor(selected))}
              title={t('globalBar.mm3LmResetHint', 'Back to this adapter\'s recommended scales') as string}
              className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              <RotateCcw size={11} />
              {t('globalBar.mm3LmReset', 'Recommended')}
            </button>
          </div>

          {showDepth && (
            <div className="space-y-3 pl-3 border-l-2 border-amber-500/30">
              <div className="flex gap-2 items-start">
                <AlertTriangle size={12} className="text-amber-500 flex-shrink-0 mt-0.5" />
                <p className="text-[10px] text-amber-500/90 leading-relaxed">
                  {t('globalBar.mm3LmDepthWarning',
                    'Leave these at 1.0 for production. The late third carries sequence termination — halving it produced fades and songs that end early or never; dialling the early third down produced runaway tempo and chaos. Useful for probing what the adapter learned, not for renders you want to keep.')}
                </p>
              </div>
              <Slider
                label={t('globalBar.mm3LmEarly', 'Early third')}
                value={num(PARAM.early, defaults.scaleEarly)}
                onChange={v => setBackendParam(PARAM.early, v)}
                min={0} max={2} step={0.05} showInput
              />
              <Slider
                label={t('globalBar.mm3LmMid', 'Middle third')}
                value={num(PARAM.mid, defaults.scaleMid)}
                onChange={v => setBackendParam(PARAM.mid, v)}
                min={0} max={2} step={0.05} showInput
              />
              <Slider
                label={t('globalBar.mm3LmLate', 'Late third')}
                value={num(PARAM.late, defaults.scaleLate)}
                onChange={v => setBackendParam(PARAM.late, v)}
                min={0} max={2} step={0.05} showInput
              />
            </div>
          )}
        </>
      )}
    </div>
  );
};

/** Summary badge — adapter short name plus the two dials that matter. */
export const Mm3LmAdapterBadge: React.FC = () => {
  const backendParams = useGlobalParamsStore(s => s.backendParams) as Record<string, unknown> | undefined;
  const params = backendParams ?? {};
  const selected = String(params[PARAM.adapter] ?? '');

  if (!selected) return <span className="text-[10px] text-zinc-600">None</span>;

  const short = selected.split(/[\\/]/).slice(-2).join('/').replace(/\.safetensors$/i, '');
  const dial = (key: string, fallback: number) => {
    const v = Number(params[key]);
    return (Number.isFinite(v) ? v : fallback).toFixed(2);
  };

  return (
    <div className="flex items-center gap-1.5 min-w-0">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
      <span className="text-[10px] text-emerald-400 font-mono truncate max-w-[120px]" title={selected}>{short}</span>
      <span className="text-[10px] text-zinc-600 flex-shrink-0">
        a{dial(PARAM.attn, 1.0)}/m{dial(PARAM.mlp, 0.5)}
      </span>
    </div>
  );
};
