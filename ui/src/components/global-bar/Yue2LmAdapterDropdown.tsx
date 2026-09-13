// Yue2LmAdapterDropdown.tsx — Adapters cluster for a backend whose LM adapter
// is ENGINE STATE rather than a per-request parameter.
//
// The sibling of Mm3LmAdapterDropdown, and the difference between them is the
// whole reason there are two: MM3 applies its LoRA per generation, so its
// picker writes to the request bag and nothing happens until Generate. YuE2
// MERGES the delta into the resident LM at load (engine/src/yue2/yue2-adapter.h)
// — there is no way to change it in place, and merging again on top of an
// already-merged model would double it — so picking one is a POST that tears
// the model down and the next generation pays a reload.
//
// That makes this a model picker wearing the Adapters cluster's clothes: it
// reads the same catalogue BackendModelsDropdown does (GET /api/backends/models)
// and commits through the same POST, using the `lmAdapter` bucket. Gated on
// `features.lmAdapterSelectable`, never on a backend id (multi-backend plan §2
// principle 2).
//
// The trigger word is not decoration. A YuE2 NAR adapter is addressed by the
// word it was trained under, and a style prompt that does not lead with it gets
// a model that has been nudged and not much else — so the trigger is shown
// large, copyable, and called out when an adapter has none.
//
// LICENCE: YuE2 weights are CC BY-NC 4.0 and a trained adapter inherits that.
// The text is YUE2_LICENSE_NOTICE in the server's backends/yue2/index.ts and
// arrives here as `capabilities.license` — rendered verbatim, never retyped.

import React, { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { useBackendStore } from '../../stores/backendStore';
import { useCapabilities } from '../../hooks/useCapabilities';
import { ParamLabel } from '../shared/ParamLabel';

const inputClasses =
  'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 ' +
  'text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 ' +
  'disabled:opacity-50';

function formatSize(bytes?: number): string {
  if (!bytes || !Number.isFinite(bytes) || bytes <= 0) return '';
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

/** Last two path segments, extension dropped — enough to tell two runs apart
 *  without spilling an absolute path across the badge. */
function shortName(p: string): string {
  return p.split(/[\\/]/).slice(-2).join('/').replace(/\.safetensors$/i, '');
}

/** The catalogue rows the picker renders, plus what the engine says is
 *  actually merged right now. Both come from the one models catalogue, which
 *  the store refetches after every successful selection. */
function useYue2Adapters() {
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const catalogue = useBackendStore(s => s.models[s.activeBackendId] ?? null);
  const fetchModels = useBackendStore(s => s.fetchModels);

  useEffect(() => {
    if (!activeBackendId) return;
    void fetchModels(activeBackendId);
  }, [activeBackendId, fetchModels]);

  const paths = useMemo(() => catalogue?.lmAdapters ?? [], [catalogue]);
  const meta = catalogue?.lmAdapterMeta ?? {};
  const defaults = (catalogue?.defaults ?? {}) as Record<string, unknown>;

  return {
    activeBackendId,
    catalogue,
    paths,
    meta,
    selected: String(defaults.lmAdapter ?? ''),
    /** '' until the next warm/synth after a pick — the merge happens at load,
     *  so "chosen" and "in force" genuinely differ for that window. */
    merged: String(defaults.lmAdapterMerged ?? ''),
    inForce: defaults.lmAdapterInForce === true,
  };
}

export const Yue2LmAdapterDropdown: React.FC = () => {
  const { t } = useTranslation();
  const { capabilities } = useCapabilities();
  const selectModels = useBackendStore(s => s.selectModels);
  const { activeBackendId, catalogue, paths, meta, selected, merged, inForce } = useYue2Adapters();
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const license = typeof capabilities?.license === 'string' ? capabilities.license : undefined;

  const onPick = async (value: string) => {
    setBusy(true);
    setNote(null);
    // Only the adapter bucket: the server resolves the LM quant and the VAE
    // variant from the persisted pick, and the engine leaves an omitted field
    // alone, so a partial body here cannot disturb either of them.
    const ok = await selectModels({ lmAdapter: value }, activeBackendId);
    setBusy(false);
    setNote(ok
      ? (value
        ? t('globalBar.yue2AdapterSwitched', 'Selected — the next generation reloads the model and merges it (adds a warm-up).')
        : t('globalBar.yue2AdapterCleared', 'Cleared — the next generation reloads the base model.'))
      : t('globalBar.yue2AdapterFailed', 'Could not select that adapter; the previous one is still in force.'));
  };

  if (!catalogue) {
    return <p className="text-[11px] text-zinc-500">{t('globalBar.yue2AdapterLoading', 'Loading adapters…')}</p>;
  }

  const entry = selected ? meta[selected] : undefined;

  return (
    <div className="space-y-3">
      {paths.length === 0 && (
        <p className="text-[11px] text-zinc-500 leading-relaxed">
          {t('globalBar.yue2AdapterNone',
            'No YuE2 adapters trained yet. Train one in the Training Studio (YuE2 NAR LoRA) and it appears here — no restart needed.')}
        </p>
      )}
      {/* The select survives an empty catalogue whenever something IS picked:
          an adapter whose file has gone still has to be clearable. */}
      {(paths.length > 0 || selected) && (
        <div>
          <ParamLabel
            label={t('globalBar.yue2Adapter', 'NAR Adapter')}
            info={t('globalBar.yue2AdapterHint',
              'A LoRA on YuE2\'s NAR stack, merged into the model weights when it loads. Switching evicts the resident model, so the next generation pays a reload. The adapter is addressed by its trigger word: unless that word leads the style prompt, you mostly get the base model.')}
            rootClassName="flex mb-1.5"
            className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
          <select
            className={inputClasses}
            value={selected}
            disabled={busy || catalogue.selectable === false}
            onChange={e => void onPick(e.target.value)}
          >
            <option value="">{t('globalBar.yue2AdapterBase', 'None — base model')}</option>
            {paths.map(p => (
              <option key={p} value={p}>{meta[p]?.label || shortName(p)}</option>
            ))}
            {/* A pick whose file has since moved or been deleted still has to
                be visible: a native select silently shows blank for a value
                that is not among its options, which reads as "nothing is
                selected" when something very much is. */}
            {selected && !paths.includes(selected) && (
              <option value={selected}>
                {shortName(selected)} {t('globalBar.yue2AdapterMissing', '(file not found)')}
              </option>
            )}
          </select>
        </div>
      )}

      {/* ── What this adapter is ── */}
      {selected && (
        <div className="px-3 py-2 rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 space-y-1">
          {entry?.trigger ? (
            <div className="flex items-baseline gap-2">
              <span className="text-[10px] uppercase tracking-wider text-zinc-500">
                {t('globalBar.yue2AdapterTrigger', 'Trigger')}
              </span>
              <span className="text-xs font-mono text-emerald-500 select-all break-all">{entry.trigger}</span>
            </div>
          ) : (
            <div className="flex gap-2 items-start">
              <AlertTriangle size={12} className="text-amber-500 flex-shrink-0 mt-0.5" />
              <p className="text-[10px] text-amber-500/90 leading-relaxed">
                {t('globalBar.yue2AdapterNoTrigger',
                  'This adapter records no trigger word, so there is no phrase that reliably calls its style up. It was trained without one.')}
              </p>
            </div>
          )}
          {entry?.trigger && (
            <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
              {t('globalBar.yue2AdapterTriggerUse',
                'Start the style prompt with this word — it is how the adapter is addressed.')}
            </p>
          )}
          <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
            {[
              entry?.dataset,
              entry?.rank ? `rank ${entry.rank}` : '',
              entry?.steps ? `${entry.steps} steps` : '',
              entry?.final ? 'final export' : '',
              formatSize(entry?.bytes),
            ].filter(Boolean).join(' · ')}
          </p>
          <p className="text-[10px] text-zinc-500 font-mono break-all">{selected}</p>
          {/* Chosen vs merged: the delta is baked in at load, so a fresh pick
              is not in force until the next generation warms the model. Saying
              "merged" while nothing is merged would be a lie for that window. */}
          <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
            {inForce && merged
              ? t('globalBar.yue2AdapterInForce', 'Merged into the resident model now.')
              : t('globalBar.yue2AdapterPending', 'Merges when the model next loads.')}
          </p>
        </div>
      )}

      {busy && (
        <p className="text-[10px] text-zinc-500 flex items-center gap-1.5">
          <Loader2 size={11} className="animate-spin" />
          {t('globalBar.yue2AdapterBusy', 'Switching…')}
        </p>
      )}
      {note && <p className="text-[10px] text-zinc-500 leading-relaxed">{note}</p>}

      {license && (
        <p className="text-[10px] text-amber-600 dark:text-amber-500/80 leading-relaxed border-t border-zinc-200 dark:border-white/5 pt-2">
          {license}{' '}
          {t('globalBar.yue2AdapterLicenseDerivative',
            'An adapter trained on these weights is a derivative and inherits the same restriction.')}
        </p>
      )}
    </div>
  );
};

/** Summary badge — the adapter in force, or the base model. */
export const Yue2LmAdapterBadge: React.FC = () => {
  const catalogue = useBackendStore(s => s.models[s.activeBackendId] ?? null);
  const defaults = (catalogue?.defaults ?? {}) as Record<string, unknown>;
  const selected = String(defaults.lmAdapter ?? '');
  if (!selected) return <span className="text-[10px] text-zinc-600">None</span>;

  const meta = catalogue?.lmAdapterMeta?.[selected];
  const label = meta?.trigger || meta?.runName || shortName(selected);
  return (
    <div className="flex items-center gap-1.5 min-w-0">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
      <span className="text-[10px] text-emerald-400 font-mono truncate max-w-[140px]" title={selected}>{label}</span>
    </div>
  );
};
