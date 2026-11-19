// CapabilityBanner.tsx — engine / model / Essentia degradation notices
//
// Never blocks the studio: labeling with Essentia alone still works while the
// engine is down, so each condition renders its own advisory row.

import React from 'react';
import { AlertTriangle, Clock, WifiOff } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '@/stores/trainingStore';

export const CapabilityBanner: React.FC = () => {
  const { t } = useTranslation();
  const caps = useTrainingStore(s => s.capabilities);
  if (!caps) return null;

  const rows: React.ReactNode[] = [];

  if (!caps.engine.up) {
    rows.push(
      <div key="engine" className="flex items-start gap-2 px-3 py-2 rounded-xl border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">
        <WifiOff size={14} className="mt-0.5 flex-shrink-0" />
        <span>{t('trainingStudio.caps.engineDown')}</span>
      </div>
    );
  } else if (!caps.engine.understandSupported) {
    rows.push(
      <div key="models" className="flex items-start gap-2 px-3 py-2 rounded-xl border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
        <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
        <span>
          {t('trainingStudio.caps.modelsMissing', { models: caps.engine.missingModels.join(', ') || '—' })}{' '}
          <a href="/settings" className="underline hover:no-underline">Settings</a>
        </span>
      </div>
    );
  }

  if (!caps.essentia.available) {
    rows.push(
      <div key="essentia" className="flex items-start gap-2 px-3 py-2 rounded-xl border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
        <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
        <span>{t('trainingStudio.caps.essentiaMissing')}</span>
      </div>
    );
  }

  if (caps.engine.up && caps.engine.queueDepth > 0) {
    rows.push(
      <div key="queue" className="flex items-start gap-2 px-3 py-2 rounded-xl border border-zinc-200 dark:border-white/10 bg-zinc-100/60 dark:bg-white/5 text-xs text-zinc-600 dark:text-zinc-400">
        <Clock size={14} className="mt-0.5 flex-shrink-0" />
        <span>{t('trainingStudio.caps.queueDepth', { count: caps.engine.queueDepth })}</span>
      </div>
    );
  }

  if (rows.length === 0) return null;
  return <div className="flex flex-col gap-2">{rows}</div>;
};

export default CapabilityBanner;
