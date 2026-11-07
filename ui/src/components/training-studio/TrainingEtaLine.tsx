import React from 'react';
import { useTranslation } from 'react-i18next';
import { formatDuration, type TrainingRunEta } from './trainingRunEta';

/** The one ETA line every training card renders, so the text reads the same everywhere. */
export const TrainingEtaLine: React.FC<{ eta: TrainingRunEta }> = ({ eta }) => {
  const { t } = useTranslation();
  return <span title={eta.previewsLeft
    ? t('trainingStudio.heartmula.etaPreviewsInfo', 'Includes {{n}} preview(s) at {{each}} each, measured from this run.', { n: eta.previewsLeft, each: eta.previewMs ? formatDuration(eta.previewMs) : '?' })
    : undefined}>
    {t('trainingStudio.heartmula.eta', 'ETA')}: {formatDuration(eta.ms)}
    {eta.previewsLeft > 0 && ` (${t('trainingStudio.heartmula.etaPreviews', 'incl. {{n}} preview(s)', { n: eta.previewsLeft })})`}
    {' · '}{t('trainingStudio.heartmula.etaDone', 'done ≈ {{time}}', { time: new Date(Date.now() + eta.ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) })}
  </span>;
};
