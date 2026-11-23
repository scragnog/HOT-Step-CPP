import { useTranslation } from 'react-i18next';
import type { HeartmulaPreviewListResponse, HeartmulaPreviewRenderRequest, HeartmulaRunSummary } from '../../../../../server/src/contracts/heartmulaTraining';
import { TrainingCheckpointLadder, TrainingPreviewLadderRows, TrainingPreviewRung, TrainingRunPicker } from '@/components/training-studio/panels/TrainingPreviewPanel';

export interface HeartmulaPreviewCardProps {
  runs?: HeartmulaRunSummary[];
  selectedRun?: string;
  onSelectRun?: (run: string) => void;
  previews: HeartmulaPreviewListResponse | null;
  busy: boolean;
  error?: string | null;
  commandError?: string | null;
  onRender?: (request: HeartmulaPreviewRenderRequest) => void;
}

/** Preview orchestration and retries belong to the server integration. */
export function HeartmulaPreviewCard({ runs = [], selectedRun = '', onSelectRun, previews, busy, error, commandError, onRender }: HeartmulaPreviewCardProps) {
  const { t } = useTranslation();
  return <TrainingCheckpointLadder>
    <h3 className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.ladderTitle', 'Checkpoint ladder')}</h3>
    {runs.length > 0 && onSelectRun && <div className="mt-2 flex flex-col gap-1 max-w-md">
      <TrainingRunPicker label={t('trainingStudio.yue2.method.ladderRun', 'Run')}
        info={t('trainingStudio.heartmula.reviewRunInfo', 'Choose a training run to hear its saved checkpoint previews.')}
        value={selectedRun} onChange={onSelectRun}
        options={runs.map(run => ({ value: run.run, label: `${run.run} · ${run.step} · ${run.state}` }))} />
    </div>}
    <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.heartmula.reviewInfo', 'Compare the base model, saved checkpoints and final adapter with the same tags, lyrics, seed and sampler. Training pauses while each preview renders.')}</p>
    {error && <p role="alert" className="text-xs text-red-500">{error}</p>}
    {commandError && <p role="alert" className="text-xs text-red-500">{commandError}</p>}
    {!previews?.previews.length && <p className="text-xs text-zinc-500">{t('trainingStudio.heartmula.noPreviews', 'No checkpoint previews yet.')}</p>}
    <TrainingPreviewLadderRows>{previews?.previews.map((record, index) => {
      const label = record.kind === 'base' ? t('trainingStudio.heartmula.base', 'Base model')
        : record.kind === 'final' ? t('trainingStudio.heartmula.final', 'Final · step {{step}}', { step: record.step })
          : t('trainingStudio.heartmula.step', 'Step {{step}}', { step: record.step });
      const state = t(`trainingStudio.heartmula.previewState.${record.status}`, { defaultValue: record.status });
      return <TrainingPreviewRung key={`${record.run}:${record.kind}:${record.step}:${record.createdAt}:${index}`} title={label}
        takes={[{ id: String(index), label, src: record.status === 'done' ? record.audioUrl : null,
          sublabel: t('trainingStudio.heartmula.takeDetails', 'up to {{seconds}} s · seed {{seed}} · temperature {{temperature}} · top-k {{topk}} · CFG {{cfg}} · codec {{codecSteps}} / {{codecGuidance}}', {
            seconds: record.settings.seconds, seed: record.settings.seed, temperature: record.settings.temperature,
            topk: record.settings.topk, cfg: record.settings.cfgScale, codecSteps: record.settings.codecSteps, codecGuidance: record.settings.codecGuidance,
          }), text: `${label} · ${state}${record.error ? ` · ${record.error}` : ''}` }]}
        actions={onRender && (record.status === 'failed' || record.status === 'skipped') && <button type="button"
          className="text-xs text-amber-600 dark:text-amber-400 disabled:opacity-40" disabled={busy}
          onClick={() => onRender({ run: record.run, step: record.step, settings: record.settings })}>
          {t('trainingStudio.heartmula.retryPreview', 'Retry preview')}
        </button>}>
        <details className="mt-2 text-xs text-zinc-500">
          <summary className="cursor-pointer">{t('trainingStudio.heartmula.conditioning', 'Tags and lyrics')}</summary>
          {record.source && <p className="mt-1 text-[11px] text-zinc-500">{[
            t('trainingStudio.heartmula.sourceTags', 'Tags from {{from}}', { from: record.source.tags === 'override' ? t('trainingStudio.heartmula.sourceOverride', 'the override') : record.source.sampleName }),
            t('trainingStudio.heartmula.sourceLyrics', 'Lyrics from {{from}}', { from: record.source.lyrics === 'override' ? t('trainingStudio.heartmula.sourceOverride', 'the override')
              : record.source.lyrics === 'lyric-studio' ? t('trainingStudio.heartmula.sourceLyricStudio', 'Lyric Studio generation #{{id}} {{title}}', { id: record.source.generationId, title: record.source.generationTitle ?? '' }).trim()
                : record.source.sampleName }),
          ].join(' · ')}</p>}
          <p className="mt-1 whitespace-pre-wrap break-words">{record.settings.tags}</p>
          <p className="mt-1 whitespace-pre-wrap break-words">{record.settings.lyrics}</p>
        </details>
      </TrainingPreviewRung>;
    })}</TrainingPreviewLadderRows>
  </TrainingCheckpointLadder>;
}
