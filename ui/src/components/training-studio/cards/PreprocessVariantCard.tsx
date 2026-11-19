// PreprocessVariantCard.tsx — one cached tensor variant
//
// A variant is one (dataset × DiT base) pair: different bases never share a
// directory, so several cards can coexist. Stale/missing badges are advisory —
// re-running preprocess with "Re-encode existing" off tops the cache up.

import React, { useState } from 'react';
import { AlertTriangle, CheckCircle2, Layers, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { PreprocessVariantStatus } from '@/services/trainingApi';
import { ConfirmDialog } from '@/components/shared/overlays/ConfirmDialog';
import { formatBytes } from '@/utils/format/format';
import { CARD } from '@/components/training-studio/lib/yue2CardUi';

const PILL = 'text-[10px] font-semibold px-1.5 py-0.5 rounded-full border';

/** Locale-aware relative time — no translation key needed. */
function formatAgo(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const seconds = Math.round((d.getTime() - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  const steps: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['second', 60], ['minute', 60], ['hour', 24], ['day', 30], ['month', 12],
  ];
  let value = seconds;
  for (const [unit, span] of steps) {
    if (Math.abs(value) < span) return rtf.format(Math.round(value), unit);
    value /= span;
  }
  return rtf.format(Math.round(value), 'year');
}

interface Props {
  variant: PreprocessVariantStatus;
  onDelete: (variantKey: string) => void;
  busy?: boolean;
}

export const PreprocessVariantCard: React.FC<Props> = ({ variant, onDelete, busy }) => {
  const { t } = useTranslation();
  const [confirming, setConfirming] = useState(false);

  const clean = variant.staleCount === 0 && variant.missingCount === 0 && variant.cachedCount > 0;

  return (
    <div className={`${CARD} flex flex-col gap-2`}>
      <div className="flex items-start gap-2">
        <Layers size={15} className="text-amber-500 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-bold text-zinc-900 dark:text-white truncate" title={variant.modelVariant || variant.variantKey}>
            {variant.modelVariant || variant.variantKey}
          </div>
          <div className="text-[11px] font-mono text-zinc-500 truncate" title={variant.outputDir}>
            {variant.outputDir}
          </div>
        </div>
        <button
          onClick={() => setConfirming(true)}
          disabled={!!busy}
          title={t('trainingStudio.preprocess.delete')}
          className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-red-500/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex-shrink-0"
        >
          <Trash2 size={13} />
        </button>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {variant.compat && (
          <span className={`${PILL} text-zinc-500 bg-zinc-500/10 border-zinc-500/20`}>{variant.compat}</span>
        )}
        {variant.dtype && (
          <span className={`${PILL} text-sky-500 bg-sky-500/10 border-sky-500/20`}>{variant.dtype}</span>
        )}
        {clean && (
          <span className={`${PILL} flex items-center gap-1 text-emerald-500 bg-emerald-500/10 border-emerald-500/20`}>
            <CheckCircle2 size={10} /> {variant.cachedCount}
          </span>
        )}
        {variant.staleCount > 0 && (
          <span className={`${PILL} flex items-center gap-1 text-amber-500 bg-amber-500/10 border-amber-500/20`}>
            <AlertTriangle size={10} /> {t('trainingStudio.preprocess.stale', { count: variant.staleCount })}
          </span>
        )}
        {variant.missingCount > 0 && (
          <span className={`${PILL} text-sky-500 bg-sky-500/10 border-sky-500/20`}>
            {t('trainingStudio.preprocess.missing', { count: variant.missingCount })}
          </span>
        )}
      </div>

      <div className="flex items-center gap-3 flex-wrap text-[11px] text-zinc-600 dark:text-zinc-400">
        <span className="tabular-nums">{variant.processed} / {variant.total}</span>
        <span className="text-zinc-400 dark:text-zinc-600">·</span>
        <span className="tabular-nums">{variant.cachedCount} × .safetensors</span>
        <span className="text-zinc-400 dark:text-zinc-600">·</span>
        <span className="tabular-nums">{formatBytes(variant.bytes)}</span>
        {variant.failed > 0 && (
          <>
            <span className="text-zinc-400 dark:text-zinc-600">·</span>
            <span className="text-red-500">{t('trainingStudio.job.failedCount', { count: variant.failed })}</span>
          </>
        )}
        {variant.createdAt && (
          <>
            <span className="text-zinc-400 dark:text-zinc-600">·</span>
            <span title={variant.createdAt}>{formatAgo(variant.createdAt)}</span>
          </>
        )}
      </div>

      <ConfirmDialog
        isOpen={confirming}
        title={t('trainingStudio.preprocess.delete')}
        message={t('trainingStudio.preprocess.deleteConfirm', { variant: variant.modelVariant || variant.variantKey })}
        danger
        onConfirm={() => { setConfirming(false); onDelete(variant.variantKey); }}
        onCancel={() => setConfirming(false)}
      />
    </div>
  );
};

export default PreprocessVariantCard;
