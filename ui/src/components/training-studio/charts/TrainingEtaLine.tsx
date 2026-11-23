import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { formatDuration, type TrainingRunEta } from '@/components/training-studio/lib/trainingRunEta';
import { formatTimeOfDay } from '@/utils/format/dateUtils';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';

/** The one ETA line every training card renders, so the text reads the same everywhere. */
export const TrainingEtaLine: React.FC<{ eta: TrainingRunEta }> = ({ eta }) => {
  const { t } = useTranslation();
  // The finish time has to move as the clock does; reading it in render would
  // make the component impure and freeze the estimate until the next poll.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);
  return <ParamLabel
    label={t('trainingStudio.heartmula.eta', 'ETA')}
    info={eta.previewsLeft
      ? t('trainingStudio.heartmula.etaPreviewsInfo', 'Includes {{n}} preview(s) at {{each}} each, measured from this run.', { n: eta.previewsLeft, each: eta.previewMs ? formatDuration(eta.previewMs) : '?' })
      : t('trainingStudio.heartmula.etaInfo', 'Estimated time remaining, measured from the progress so far.')}
  >
    <span>
      {t('trainingStudio.heartmula.eta', 'ETA')}: {formatDuration(eta.ms)}
      {eta.previewsLeft > 0 && ` (${t('trainingStudio.heartmula.etaPreviews', 'incl. {{n}} preview(s)', { n: eta.previewsLeft })})`}
      {' · '}{t('trainingStudio.heartmula.etaDone', 'done ≈ {{time}}', { time: formatTimeOfDay(now + eta.ms) })}
    </span>
  </ParamLabel>;
};
