import { Database, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { HeartmulaPrepareRequest, HeartmulaPrepareStatus } from '../../../../server/src/contracts/heartmulaTraining';
import { Toggle } from '../shared/Toggle';
import { Yue2StageCard } from './Yue2StageCard';
import { JobProgress } from './JobProgress';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';

export interface HeartmulaPrepareCardProps {
  datasetId?: string;
  status: HeartmulaPrepareStatus;
  busy: boolean;
  preparing: boolean;
  blockedReason?: string;
  onPrepare: (request: HeartmulaPrepareRequest) => void;
}

export function HeartmulaPrepareCard({ datasetId, status, busy, preparing, blockedReason, onPrepare }: HeartmulaPrepareCardProps) {
  const { t } = useTranslation();
  const [force, setForce] = useState(false);
  const job = useActiveTrainingJob();
  const mine = job?.kind === 'heartmula-prepare' && (!datasetId || job.datasetId === datasetId);
  const ready = status.cacheValid && status.staleSamples === 0 && status.preparedSamples > 0;
  return <Yue2StageCard icon={<Database size={16} className="text-amber-500" />}
    title={t('trainingStudio.heartmula.prepare', 'Prepare HeartMuLa audio')}
    done={ready}
    action={<div className="flex items-center gap-3 flex-wrap">
      <p className="text-xs text-zinc-500" aria-live="polite">
        {t('trainingStudio.heartmula.prepareCounts', '{{prepared}} prepared · {{stale}} stale · {{skipped}} skipped', {
          prepared: status.preparedSamples, stale: status.staleSamples, skipped: status.skipped.length,
        })}
      </p>
      {blockedReason && <p className="text-xs text-amber-600 dark:text-amber-400">{blockedReason}</p>}
      <button type="button" className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 disabled:opacity-40 transition-colors flex items-center gap-1.5"
        disabled={busy || !!blockedReason} onClick={() => onPrepare({ force })}>
        {preparing && <Loader2 size={12} className="animate-spin" />}
        {preparing ? t('trainingStudio.heartmula.preparing', 'Preparing…') : t('trainingStudio.heartmula.prepareStart', 'Prepare audio')}
      </button>
    </div>}
    footer={mine && <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/10"><JobProgress /></div>}>
    <p className="text-xs text-zinc-500 mb-3">{t('trainingStudio.heartmula.prepareInfo',
      'Build the dataset first. Preparation encodes audio into eight HeartMuLa codebooks and checks tags, lyrics and the model context limit. Changed samples must be prepared again before training.')}</p>
    <Toggle accent="amber" checked={force} disabled={busy} onChange={setForce}
      label={t('trainingStudio.heartmula.force', 'Rebuild all samples')}
      info={t('trainingStudio.heartmula.forceInfo', 'Re-encode every sample, including valid cached samples. Leave off to reuse unchanged preparation.')}
      defaultValue={false} />
    {status.contextLimit !== null && <p className="mt-3 text-xs text-zinc-500">{t('trainingStudio.heartmula.contextLimit', 'Context limit: {{count}} tokens', { count: status.contextLimit })}</p>}
    {status.skipped.length > 0 && <ul className="mt-3 max-h-48 overflow-auto text-xs text-zinc-500 space-y-1">
      {status.skipped.map((sample, index) => <li key={`${sample.sampleId}:${index}`}>
        {sample.sampleId}: {t(`trainingStudio.heartmula.skip.${sample.reason}`, { defaultValue: sample.reason })} · {sample.detail}
      </li>)}
    </ul>}
  </Yue2StageCard>;
}
