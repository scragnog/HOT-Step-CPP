// Mm3CodesCard.tsx — Training Studio phase 2, MiniMax-Music3 branch.
//
// FOR MM3, CODES *ARE* THE PREPROCESS STEP. ACE's phase 2 encodes audio into a
// per-song tensor cache keyed by the DiT model it was produced with; MM3 has no
// such cache and no such key — it needs RVQ codes, which are model-independent
// in the same sense (they depend on the ENCODER, not the DiT). So this replaces
// the ACE preprocess UI in MM3 mode rather than sitting next to it. Showing
// ACE's variant list here while MM3 is active was the bug: those caches belong
// to a different pipeline and can never be used by an MM3 run.
//
// Shared job machinery below the button, as everywhere else in this studio.

import React, { useState } from 'react';
import { AlertTriangle, FileCode2, Loader2, PauseCircle, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
import { useMm3Status } from './useMm3Status';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

export const Mm3CodesCard: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startMm3Codes = useTrainingStore(s => s.startMm3Codes);
  const storeError = useTrainingStore(s => s.error);
  const { status, error } = useMm3Status(datasetId);
  const [busy, setBusy] = useState(false);
  // The launder gate. Off by default, and off means the export is the plain
  // mm3-codes path, byte-identical to before the gate existed. The laundered
  // export writes a SEPARATE sibling cache, so the two coexist.
  const [launder, setLaunder] = useState(false);

  const kind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const running = jobStatus === 'queued' || jobStatus === 'running';
  const mine = kind === 'mm3-codes';

  if (!status && !error) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  const hasCodes = (status?.codes ?? 0) > 0;
  const hasLaundered = (status?.codesLaundered ?? 0) > 0;
  const missingNow = (launder ? status?.missingForLaunder : status?.missingForCodes) ?? [];
  const blocked = missingNow.length > 0;
  const shown = error || storeError;

  const run = async () => {
    setBusy(true);
    try { await startMm3Codes(launder ? { launder: true } : undefined); } finally { setBusy(false); }
  };

  return (
    <div className="flex flex-col gap-4">
      {shown && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{shown}</span>
        </div>
      )}

      <div className={CARD}>
        <div className="flex items-center gap-2 mb-2">
          <FileCode2 size={15} className="text-amber-500" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.mm3.codesTitle', 'RVQ codes')}
          </h3>
        </div>
        <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
          {t('trainingStudio.mm3.codesBlurb',
            'MiniMax-Music3 ships no audio-to-code encoder, so the tokens the LM learns from are produced '
            + 'here, natively, by the adopted community encoder. Run this once per dataset; re-running '
            + 'overwrites with whichever encoder is installed now.')}
        </p>
        <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
          {t('trainingStudio.mm3.notTensors',
            'This replaces the ACE-Step tensor cache: the two pipelines share nothing, and an ACE variant '
            + 'can never be used by a MiniMax-Music3 run.')}
        </p>

        {blocked ? (
          <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t('trainingStudio.mm3.missing', 'Missing model files')}: {missingNow.join(', ')}
            </span>
          </div>
        ) : (
          <>
            <label className="flex items-start gap-2 mb-3 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={launder}
                onChange={e => setLaunder(e.target.checked)}
                className="mt-0.5 accent-amber-500"
              />
              <span className="text-[11px] text-zinc-600 dark:text-zinc-400 leading-relaxed">
                <span className="font-semibold text-zinc-800 dark:text-zinc-200">
                  {t('trainingStudio.mm3.launderLabel', 'Cover-launder the codes (dense mixes)')}
                </span>
                {' — '}
                {t('trainingStudio.mm3.launderBlurb',
                  'renders each track back through the model (rec7 states → flow DiT) before encoding, so '
                  + 'buried vocals reach the code targets. Slower (~3 min/track, capped at 6 min of audio), '
                  + 'writes a separate cache, and the training form chooses which cache a run uses.')}
              </span>
            </label>
            <div className="flex items-center gap-3 flex-wrap">
              <button
                onClick={() => void run()}
                disabled={busy || running}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 disabled:opacity-40 transition-colors"
              >
                {launder
                  ? (hasLaundered
                    ? t('trainingStudio.mm3.reexportLaundered', 'Re-export laundered codes')
                    : t('trainingStudio.mm3.exportLaundered', 'Export laundered codes'))
                  : (hasCodes
                    ? t('trainingStudio.mm3.reexport', 'Re-export codes')
                    : t('trainingStudio.mm3.export', 'Export codes'))}
              </button>
              <span className="text-xs text-zinc-500">
                {hasCodes
                  ? `${status?.codes} ${t('trainingStudio.mm3.codesReady', 'tracks encoded')}`
                    + (status?.encoder ? ` · ${status.encoder}` : '')
                  : t('trainingStudio.mm3.noCodes', 'no codes yet')}
                {hasLaundered
                  ? ` · ${status?.codesLaundered} ${t('trainingStudio.mm3.launderedReady', 'laundered')}`
                  : ''}
              </span>
            </div>
            <p className="text-[11px] text-zinc-500 flex items-center gap-1.5 mt-3">
              <PauseCircle size={12} className="flex-shrink-0" />
              {t('trainingStudio.mm3.enginePaused',
                'The engine is paused while this runs and restarted afterwards.')}
            </p>
          </>
        )}
      </div>

      {mine && activeJob && (
        <div className={CARD}>
          <JobProgress />
        </div>
      )}
    </div>
  );
};

export default Mm3CodesCard;
