import { queryOptions, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, RefreshCw } from 'lucide-react';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { usePersistedState } from '../../hooks/usePersistedState';
import { pollWhile } from '../../queries/contract';
import { stableStepAdaptersOptions } from '../../queries/global-models/options';
import { scopeOf, serverKey } from '../../queries/keys';
import { manualRenderRequest, renderSa3PreviewCommand, sa3PreviewOptions } from '../../queries/sa3-previews/queries';
import { generationStatusOptions } from '../../queries/shell-queue/queue';
import { queryManager } from '../../queries/invalidation';
import { generateApi } from '../../services/api';
import type { Sa3PreviewRecord } from '../../services/sa3PreviewApi';
import { getSa3TrainingStatus, type Sa3Recipe, type Sa3TrainingStatus, type StartSa3TrainingRequest, type TrainingJobSummary } from '../../services/trainingApi';
import { useGlobalParamsStore } from '../../stores/globalParamsStore';
import { useTrainingStore } from '../../stores/trainingStore';
import type { GenerationJob, GenerationParams } from '../../types';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { TrainingChart } from './TrainingChart';
import { PREVIEW_INPUT, PreviewField, TrainingPreviewRung, TrainingPreviewSettings, type PreviewTake } from './TrainingPreviewPanel';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'w-full rounded-lg border border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-black/20 px-3 py-2 text-sm text-zinc-900 dark:text-white disabled:opacity-50';
const BUTTON = 'rounded-lg bg-amber-500/15 border border-amber-500/25 px-3 py-2 text-xs font-semibold text-amber-700 dark:text-amber-400 disabled:opacity-40';
const active = (job: TrainingJobSummary | null | undefined) => job?.status === 'queued' || job?.status === 'running';

const sa3StatusOptions = (worker: string | null, datasetId: string) => queryOptions({
  queryKey: serverKey('training-data/sa3-status', { worker, id: datasetId }),
  queryFn: ({ queryKey, signal }) => getSa3TrainingStatus(datasetId, { worker: scopeOf(queryKey)!.worker, signal }),
  staleTime: 0,
  refetchInterval: pollWhile<Sa3TrainingStatus>(data => active(data?.activeJob)),
});

const RECIPE_FIELDS: Array<{ key: keyof Sa3Recipe; min: number; max: number; step?: number }> = [
  { key: 'steps', min: 1, max: 1_000_000 },
  { key: 'rank', min: 1, max: 256 },
  { key: 'alpha', min: 0.001, max: 1024, step: 0.001 },
  { key: 'learningRate', min: 1e-8, max: 0.1, step: 0.00001 },
  { key: 'duration', min: 1, max: 384, step: 0.1 },
  { key: 'seed', min: 0, max: 0x7fffffff },
  { key: 'checkpointEvery', min: 1, max: 1_000_000 },
];

/** Preview form state, kept in this browser like the YuE2 card's. Cadence is
 *  null until edited, then it follows the recipe's snapshot interval. */
interface PreviewForm { enabled: boolean; everySteps: number | null; seconds: number; seedsText: string; promptsText: string }
const PREVIEW_DEFAULTS: PreviewForm = { enabled: true, everySteps: null, seconds: 30, seedsText: '424242', promptsText: '' };
const parseSeeds = (text: string) => text.split(/[\s,]+/).filter(Boolean).map(Number);
const parsePrompts = (text: string) => text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
const seedsValid = (seeds: number[]) => seeds.length > 0 && seeds.length <= 4 && seeds.every(n => Number.isInteger(n) && n >= 0 && n <= 0x7fffffff);

/** Latest run first: run ids end in a timestamp (`<adapter>/<timestamp>`). */
const newestRun = (runs: Array<{ run: string }>) => [...runs].sort((a, b) => b.run.split('/').pop()!.localeCompare(a.run.split('/').pop()!))[0]?.run ?? '';

/** One step's records as the shared take rows. */
function previewTakes(records: Sa3PreviewRecord[], t: (key: string, fallback: string, options?: Record<string, unknown>) => string): PreviewTake[] {
  return records.map(p => {
    const label = t('trainingStudio.sa3.preview.take', 'Prompt {{n}} · seed {{seed}}', { n: p.promptIndex + 1, seed: p.seed });
    const id = `${p.step}-${p.promptIndex}-${p.seed}`;
    if (p.status === 'done' && p.audioUrl) return { id, src: p.audioUrl, label, sublabel: `${p.seconds} s · ${p.prompt}`, downloadName: `sa3_step${p.step}_p${p.promptIndex + 1}_seed${p.seed}.wav` };
    return { id, label, text: `${label}: ${t(`trainingStudio.sa3.preview.status.${p.status}`, p.status)}${p.error ? ` — ${p.error}` : ''}` };
  });
}

function AuditionResult({ jobId, label }: { jobId: string; label: string }) {
  const { t } = useTranslation();
  const query = useQuery({ ...generationStatusOptions(jobId),
    refetchInterval: pollWhile<GenerationJob>(data => !data || !['succeeded', 'failed', 'cancelled'].includes(data.status)),
  });
  const cancel = useMutation({ mutationFn: () => generateApi.cancel(jobId) });
  const job = query.data;
  return <div className="space-y-2 rounded-lg border border-zinc-200 dark:border-white/10 p-3">
    <p className="text-xs">{label} · {job ? t(`trainingStudio.sa3.render.${job.status}`) : t('trainingStudio.sa3.loading')}</p>
    {(query.error || job?.error || cancel.error) && <p role="alert" className="text-xs text-red-500">{query.error?.message || job?.error || cancel.error?.message}</p>}
    {job?.result?.audioUrls.map(url => <audio key={url} controls src={url} className="w-full" />)}
    {job && !['succeeded', 'failed', 'cancelled'].includes(job.status) && <button className={BUTTON} disabled={cancel.isPending || cancel.isSuccess} onClick={() => cancel.mutate()}>{t('trainingStudio.sa3.cancelAudition')}</button>}
  </div>;
}

export const Sa3TrainCard: React.FC<{ datasetId: string; slug: string }> = ({ datasetId, slug }) => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const worker = useTrainingStore(s => s.trainingWorker);
  const storedJob = useActiveTrainingJob();
  const error = useTrainingStore(s => s.error);
  const start = useTrainingStore(s => s.startSa3Train);
  const cancel = useTrainingStore(s => s.cancelJob);
  const steps = useTrainingStore(s => s.trainStepSeries);
  const logs = useTrainingStore(s => s.jobLog);
  const statusQuery = useQuery(sa3StatusOptions(worker, datasetId));
  const status = statusQuery.data;
  const job = storedJob?.datasetId === datasetId ? storedJob : status?.activeJob;
  const busy = active(job) || active(status?.activeJob);
  const mine = job?.kind === 'sa3-train';
  const [adapterName, setAdapterName] = useState(slug);
  const [edits, setEdits] = useState<Partial<Sa3Recipe>>({});
  const [resumeRun, setResumeRun] = useState('');
  const [resumeSteps, setResumeSteps] = useState<number | null>(null);
  const run = status?.runs.find(r => r.run === resumeRun);
  const recipe = resumeRun ? run?.recipe : status ? { ...status.defaults, ...edits } : undefined;
  const targetSteps = resumeRun ? resumeSteps ?? run?.recipe.steps ?? 0 : recipe?.steps ?? 0;
  const invalidRecipe = !recipe || RECIPE_FIELDS.some(field => {
    const value = field.key === 'steps' ? targetSteps : recipe[field.key];
    return !Number.isFinite(value) || value < field.min || value > field.max || (!field.step && !Number.isInteger(value));
  });
  const startMutation = useMutation({ mutationFn: async () => {
    // ponytail: `preview` rides on the start body as an extra key until the frozen start request carries it (server P3).
    const body: StartSa3TrainingRequest & { preview?: unknown } = resumeRun ? { resumeRun, steps: targetSteps } : {
      ...recipe, adapterName, exportEvery: preview.enabled ? everySteps : 0,
      preview: { everySteps: preview.enabled ? everySteps : 0, prompts, seconds: preview.seconds, seeds },
    };
    await start(body);
    await queryManager.invalidate(client, sa3StatusOptions(worker, datasetId).queryKey);
  } });
  const cancelMutation = useMutation({ mutationFn: cancel });
  const [form, setForm] = usePersistedState<PreviewForm>('hs-sa3-preview', PREVIEW_DEFAULTS);
  const setPreview = (patch: Partial<PreviewForm>) => setForm(previous => ({ ...PREVIEW_DEFAULTS, ...previous, ...patch }));
  const preview = { ...PREVIEW_DEFAULTS, ...form };
  const everySteps = preview.everySteps ?? (status?.defaults.exportEvery || status?.defaults.checkpointEvery || 0);
  const prompts = parsePrompts(preview.promptsText);
  const seeds = parseSeeds(preview.seedsText);
  const previewInvalid = preview.enabled && !resumeRun && (!Number.isInteger(everySteps) || everySteps < 1 || everySteps > 1_000_000
    || !prompts.length || !seedsValid(seeds) || !Number.isFinite(preview.seconds) || preview.seconds < 1 || preview.seconds > 384);
  const [pickedRun, setPickedRun] = useState('');
  const viewRun = pickedRun && status?.runs.some(r => r.run === pickedRun) ? pickedRun : newestRun(status?.runs ?? []);
  const previewsQuery = useQuery({ ...sa3PreviewOptions(worker, datasetId, viewRun, busy), enabled: !!viewRun });
  const [rendering, setRendering] = useState<number | null>(null);
  const renderMutation = useMutation({
    mutationFn: async (step: number) => {
      setRendering(step);
      try {
        await renderSa3PreviewCommand(client, worker, datasetId, manualRenderRequest(viewRun, step));
      } finally { setRendering(null); }
    },
  });
  const previewSteps = [...new Set((previewsQuery.data?.previews ?? []).map(p => p.step))].sort((a, b) => a - b);
  const adaptersQuery = useQuery({ ...stableStepAdaptersOptions(), enabled: worker === null });
  const adapters = adaptersQuery.data?.adapters ?? [];
  const [pickedAdapter, setPickedAdapter] = useState('');
  const selectedAdapter = pickedAdapter || status?.runs.find(r => r.published && adapters.some(a => a.name === r.published))?.published || '';
  const [useAdapter, setUseAdapter] = useState(true);
  const [strength, setStrength] = useState(1);
  const [caption, setCaption] = useState('');
  const [auditionSeed, setAuditionSeed] = useState(42);
  const [auditionSeconds, setAuditionSeconds] = useState(30);
  const [auditions, setAuditions] = useState<Array<{ jobId: string; label: string }>>([]);
  const auditionMutation = useMutation({ mutationFn: async () => {
    const params = useGlobalParamsStore.getState().getGlobalParams();
    const scale = useAdapter ? strength : 0;
    const res = await generateApi.submit({ ...params, backend: 'sa3', backendOverride: 'sa3', expectedBackend: 'sa3', caption: caption.trim(), lyrics: '',
      instrumental: true, duration: auditionSeconds, seed: auditionSeed, randomSeed: false, batchSize: 1,
      postProcessingEnabled: false, samplerPluginsEnabled: false, sa3Steps: 8, sa3Sampler: 'md_pingpong_blend',
      sa3Adapters: scale === 0 ? [] : [{ name: selectedAdapter, scale }],
      sa3LufsEnabled: false,
      seedB: undefined, seedBlend: undefined,
    } as GenerationParams);
    return { jobId: res.jobId, label: t('trainingStudio.sa3.auditionLabel', { adapter: scale === 0 ? t('trainingStudio.sa3.base') : selectedAdapter, scale, seed: auditionSeed }) };
  }, onSuccess: result => setAuditions(current => [...current, result]) });

  return <div className="flex flex-col gap-4 text-zinc-700 dark:text-zinc-300">
    <section className={`${CARD} space-y-3`}>
      <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.sa3.title')}</h3>
      <p className="text-xs text-zinc-500">{t('trainingStudio.sa3.scope')}</p>
      <p className="text-xs">{t('trainingStudio.sa3.preparation')}</p>
      <p className="text-xs">{t('trainingStudio.sa3.enginePause')}</p>
      <div className="flex items-center gap-2 text-xs">
        {statusQuery.isPending && <Loader2 size={14} className="animate-spin" />}
        {status ? t(status.available ? 'trainingStudio.sa3.ready' : 'trainingStudio.sa3.unavailable', { backend: status.backend }) : t('trainingStudio.sa3.loading')}
        <button className={BUTTON} aria-label={t('trainingStudio.sa3.refresh')} disabled={statusQuery.isFetching} onClick={() => { void statusQuery.refetch(); if (worker === null) void adaptersQuery.refetch(); }}><RefreshCw size={12} /></button>
      </div>
      {!!status?.missing.length && <p role="alert" className="text-xs text-amber-700 dark:text-amber-400">{t('trainingStudio.sa3.missing', { files: status.missing.join(', ') })}</p>}
      {(statusQuery.error || error || startMutation.error || cancelMutation.error) && <p role="alert" className="text-xs text-red-500">{statusQuery.error?.message || error || startMutation.error?.message || cancelMutation.error?.message}</p>}
      {status && <>
        <div className="space-y-1">
          <ParamLabel label={t('trainingStudio.sa3.run')} info={t('trainingStudio.sa3.runInfo')} />
          <StyledSelect accent="amber" value={resumeRun} disabled={busy || startMutation.isPending} onChange={value => { setResumeRun(value); setResumeSteps(null); }} options={[
            { value: '', label: t('trainingStudio.sa3.newRun') },
            ...status.runs.filter(r => r.resumable).map(r => ({ value: r.run, label: r.run })),
          ]} />
        </div>
        {!resumeRun && <label className="block space-y-1">
          <ParamLabel label={t('trainingStudio.sa3.adapterName')} info={t('trainingStudio.sa3.adapterNameInfo')} />
          <input className={INPUT} value={adapterName} disabled={busy || startMutation.isPending} maxLength={64} onChange={e => setAdapterName(e.target.value)} />
        </label>}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {recipe && RECIPE_FIELDS.map(field => <label key={field.key} className="space-y-1">
            <ParamLabel label={t(`trainingStudio.sa3.fields.${field.key}`)} info={t(`trainingStudio.sa3.fields.${field.key}Info`)} />
            <input type="number" className={INPUT} value={field.key === 'steps' ? targetSteps : recipe[field.key]}
              min={field.key === 'steps' && run ? run.recipe.steps : field.min} max={field.max} step={field.step ?? 1}
              disabled={busy || startMutation.isPending || (!!resumeRun && field.key !== 'steps')}
              onChange={e => resumeRun ? setResumeSteps(e.target.valueAsNumber) : setEdits(current => ({ ...current, [field.key]: e.target.valueAsNumber }))} />
          </label>)}
        </div>
        <TrainingPreviewSettings
          title={t('trainingStudio.sa3.preview.title', 'Checkpoint previews (optional)')}
          enabled={preview.enabled}
          disabled={busy || startMutation.isPending || !!resumeRun}
          onEnabledChange={enabled => setPreview({ enabled })}
          enableLabel={t('trainingStudio.sa3.preview.enable', 'Render previews at saved checkpoints')}
          enableInfo={t('trainingStudio.sa3.preview.enableInfo', 'Each cadence step saves an inference snapshot and renders every prompt and seed below, one at a time, after the trainer exits and before it resumes. Off: no snapshots or previews.')}
          enableDefault={PREVIEW_DEFAULTS.enabled}
          afterToggle={resumeRun && <p className="mt-2 text-[11px] text-zinc-500">{t('trainingStudio.sa3.preview.resumeKeeps', 'A resumed run keeps the preview settings it started with.')}</p>}
        >
          {preview.enabled && <>
            <PreviewField label={t('trainingStudio.sa3.preview.everySteps', 'Preview every (steps)')} info={t('trainingStudio.sa3.preview.everyStepsInfo', 'Render previews every this many training steps. Starts from the recipe snapshot interval.')}
              onReset={preview.everySteps !== null && !busy ? () => setPreview({ everySteps: null }) : undefined}>
              <input className={PREVIEW_INPUT} type="number" min={1} max={1_000_000} step={1} value={everySteps} disabled={busy || !!resumeRun}
                onChange={e => setPreview({ everySteps: e.target.valueAsNumber })} />
            </PreviewField>
            <PreviewField label={t('trainingStudio.sa3.preview.seconds', 'Preview seconds')} info={t('trainingStudio.sa3.preview.secondsInfo', 'Length of each rendered preview, from 1 to 384 seconds. Longer previews take longer at every checkpoint.')}
              meta={t('trainingStudio.sa3.preview.secondsMeta', 'default {{n}}', { n: PREVIEW_DEFAULTS.seconds })}
              onReset={preview.seconds !== PREVIEW_DEFAULTS.seconds && !busy ? () => setPreview({ seconds: PREVIEW_DEFAULTS.seconds }) : undefined}>
              <input className={PREVIEW_INPUT} type="number" min={1} max={384} step={1} value={preview.seconds} disabled={busy || !!resumeRun}
                onChange={e => setPreview({ seconds: e.target.valueAsNumber })} />
            </PreviewField>
            <PreviewField label={t('trainingStudio.sa3.preview.seeds', 'Preview seeds')} info={t('trainingStudio.sa3.preview.seedsInfo', 'Up to four whole-number seeds, separated by commas. Every checkpoint renders each prompt with each seed, so checkpoints compare like for like.')}
              meta={t('trainingStudio.sa3.preview.seedsMeta', 'default {{n}}', { n: PREVIEW_DEFAULTS.seedsText })}
              onReset={preview.seedsText !== PREVIEW_DEFAULTS.seedsText && !busy ? () => setPreview({ seedsText: PREVIEW_DEFAULTS.seedsText }) : undefined}>
              <input className={PREVIEW_INPUT} value={preview.seedsText} disabled={busy || !!resumeRun} onChange={e => setPreview({ seedsText: e.target.value })} />
            </PreviewField>
            <PreviewField wide label={t('trainingStudio.sa3.preview.prompts', 'Preview prompts')} info={t('trainingStudio.sa3.preview.promptsInfo', 'One prompt per line. Use instrumental descriptions that were not in the training captions, and keep them the same for the whole run.')}>
              <textarea className={`${PREVIEW_INPUT} min-h-16 resize-y`} value={preview.promptsText} disabled={busy || !!resumeRun} onChange={e => setPreview({ promptsText: e.target.value })} />
              {previewInvalid && !prompts.length && <span role="alert" className="text-[11px] text-amber-700 dark:text-amber-400">{t('trainingStudio.sa3.preview.needPrompt', 'Add at least one prompt, or turn previews off.')}</span>}
            </PreviewField>
          </>}
        </TrainingPreviewSettings>
        <button className={BUTTON} onClick={() => startMutation.mutate()} disabled={!status.available || busy || startMutation.isPending || invalidRecipe || previewInvalid
          || (resumeRun ? !run?.resumable || targetSteps < run.recipe.steps : !/^[a-z0-9][a-z0-9_-]{0,63}$/i.test(adapterName))}>
          {t(resumeRun ? 'trainingStudio.sa3.resume' : 'trainingStudio.sa3.start')}
        </button>
      </>}
    </section>
    {mine && job && <section className={`${CARD} space-y-3`} aria-live="polite">
      <p className="text-xs">{t(`trainingStudio.sa3.phase.${job.phase}`, { defaultValue: job.phase })} · {job.done}/{job.total} · {t(`trainingStudio.sa3.job.${job.status}`)}</p>
      <progress className="w-full accent-amber-500" max={Math.max(1, job.total)} value={job.done} />
      {job.error && <p role="alert" className="text-xs text-red-500">{job.error}</p>}
      {active(job) && <button className={BUTTON} disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate()}>{t('trainingStudio.sa3.cancel')}</button>}
      <TrainingChart epochs={[]} steps={steps} target={0} />
      <details><summary className="text-xs cursor-pointer">{t('trainingStudio.sa3.log')}</summary><pre className="max-h-48 overflow-auto whitespace-pre-wrap text-[11px]">{logs.map(line => `[${line.level}] ${line.message}`).join('\n')}</pre></details>
    </section>}
    {!!status?.runs.length && <section className={`${CARD} space-y-3`}>
      <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.sa3.preview.ladder', 'Checkpoint previews')}</h3>
      <StyledSelect accent="amber" value={viewRun} onChange={setPickedRun} options={status.runs.map(r => ({ value: r.run, label: r.run }))} />
      {(previewsQuery.error || renderMutation.error) && <p role="alert" className="text-xs text-red-500">{previewsQuery.error?.message || renderMutation.error?.message}</p>}
      {!previewSteps.length && <p className="text-xs text-zinc-500">{t(previewsQuery.isPending && !!viewRun ? 'trainingStudio.sa3.loading' : 'trainingStudio.sa3.preview.none', 'No previews for this run yet.')}</p>}
      {previewSteps.map(step => {
        const records = (previewsQuery.data?.previews ?? []).filter(p => p.step === step).sort((a, b) => a.promptIndex - b.promptIndex || a.seed - b.seed);
        const failed = records.some(p => p.status === 'failed' || p.status === 'skipped');
        return <TrainingPreviewRung key={step} title={t('trainingStudio.sa3.preview.step', 'Step {{step}}', { step })} takes={previewTakes(records, t)}
          badges={records.some(p => p.final) && <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-300 text-[10px] font-semibold">{t('trainingStudio.sa3.preview.final', 'final')}</span>}
          actions={<button type="button" className="px-2 py-1 rounded-lg text-[11px] border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40"
            disabled={busy || rendering !== null} onClick={() => renderMutation.mutate(step)}>
            {rendering === step ? t('trainingStudio.sa3.preview.rendering', 'Rendering…') : failed ? t('trainingStudio.sa3.preview.retry', 'Retry') : t('trainingStudio.sa3.preview.renderMore', 'Render more')}
          </button>} />;
      })}
    </section>}
    <section className={`${CARD} space-y-3`}>
      <h3 className="text-sm font-bold">{t('trainingStudio.sa3.audition')}</h3>
      <p className="text-xs text-zinc-500">{t(worker ? 'trainingStudio.sa3.localAudition' : 'trainingStudio.sa3.auditionInfo')}</p>
      {adaptersQuery.error && <p role="alert" className="text-xs text-red-500">{adaptersQuery.error.message}</p>}
      {status?.runs.filter(r => r.published).map(r => <p key={r.run} className="text-xs break-all">{t('trainingStudio.sa3.published', { name: r.published })}</p>)}
      <ParamLabel label={t('trainingStudio.sa3.adapter')} info={t('trainingStudio.sa3.adapterInfo')} />
      <StyledSelect accent="amber" value={selectedAdapter} onChange={setPickedAdapter} disabled={worker !== null || busy || auditionMutation.isPending}
        placeholder={t('trainingStudio.sa3.noAdapters')} options={adapters.map(a => ({ value: a.name, label: a.name, hint: `${a.sizeMb} MB` }))} />
      <Toggle checked={useAdapter} onChange={setUseAdapter} label={t('trainingStudio.sa3.useAdapter')} info={t('trainingStudio.sa3.useAdapterInfo')} accent="amber" />
      <label className="block space-y-1"><ParamLabel label={t('trainingStudio.sa3.caption')} info={t('trainingStudio.sa3.captionInfo')} />
        <textarea className={INPUT} rows={2} value={caption} onChange={e => setCaption(e.target.value)} />
      </label>
      <div className="grid grid-cols-3 gap-3">
        {([{ key: 'strength', value: strength, set: setStrength, min: 0, max: 2, step: 0.1 },
          { key: 'auditionSeed', value: auditionSeed, set: setAuditionSeed, min: 0, max: 0x7fffffff, step: 1 },
          { key: 'auditionSeconds', value: auditionSeconds, set: setAuditionSeconds, min: 1, max: 384, step: 1 }]).map(field => <label key={field.key} className="space-y-1">
          <ParamLabel label={t(`trainingStudio.sa3.${field.key}`)} info={t(`trainingStudio.sa3.${field.key}Info`)} />
          <input className={INPUT} type="number" value={field.value} min={field.min} max={field.max} step={field.step} onChange={e => field.set(e.target.valueAsNumber)} />
        </label>)}
      </div>
      {auditionMutation.error && <p role="alert" className="text-xs text-red-500">{auditionMutation.error.message}</p>}
      <button className={BUTTON} disabled={worker !== null || busy || startMutation.isPending || auditionMutation.isPending || !caption.trim()
        || (useAdapter && strength !== 0 && !adapters.some(a => a.name === selectedAdapter)) || !Number.isFinite(strength) || strength < 0 || strength > 2
        || !Number.isInteger(auditionSeed) || auditionSeed < 0 || auditionSeed > 0x7fffffff
        || !Number.isFinite(auditionSeconds) || auditionSeconds < 1 || auditionSeconds > 384} onClick={() => auditionMutation.mutate()}>{t('trainingStudio.sa3.renderAudition')}</button>
      {auditions.map(a => <AuditionResult key={a.jobId} {...a} />)}
    </section>
  </div>;
}
