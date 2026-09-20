import { Yue2OptimizerFields } from './Yue2OptimizerFields';
import type { Yue2OptimOptions } from '../../services/trainingApi';
// Yue2TrainCard.tsx — Training Studio phase 3, YuE2 branch: stages 1 and 4.
//
// Two of the five YuE2 stages Yue2TrainStages.tsx renders in order: stage 1,
// audio -> cached VAE latents (Yue2PreprocessCard), and stage 4, latents -> a
// LoRA on the frozen NAR half (Yue2NarTrainCard). Both read `status` as a PROP
// from the one useYue2Status() call Yue2TrainStages makes, which is also where
// the licence banner and the status-fetch error banner live now — a single
// fetch shared by every stage that reads it, rather than each stage repeating
// the same GET. Yue2RunsList (the NAR ladder) is exported for the same
// component, kept in this file rather than its own because it is read
// exclusively by Yue2NarTrainCard.
//
// STAGE 1 STANDING ALONE (rather than beside MM3's phase-2 codes card) is a
// server fact, not a style choice: `yue2-preprocess` does not read
// dataset.json, has no tensor-cache concept and does not touch the ACE
// preprocess variants, so there is nothing for it to sit beside in phase 2.
//
// THE DEFAULTS ARE NOT DUPLICATED HERE. Every number arrives in the `yue2`
// status payload from services/training/yue2Train.ts, which is where the
// measured recipe lives, and the VRAM model arrives as COEFFICIENTS so the
// estimate re-computes as rank moves without a second copy of the measurements.
// Same rule as Mm3TrainCard.
//
// THE LICENCE LINE IS THE SERVER'S STRING, rendered verbatim by
// Yue2TrainStages. YuE2 weights are CC BY-NC 4.0 and a trained adapter is a
// derivative that carries the same terms; the text is YUE2_LICENSE_NOTICE in
// services/backends/yue2, is shipped by the status route, and is never
// retyped, truncated or paraphrased on this side.

import React, { useState } from 'react';
import {
  AlertTriangle, ChevronDown, ChevronRight, Cpu, FileCode2, History, Loader2, PauseCircle,
  Play,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import {
  estimateYue2PeakMb, listYue2Runs,
  type Yue2CaptionMode, type Yue2NarTarget, type Yue2PresetName, type Yue2RunSummary,
  type Yue2Status, type Yue2TrainRequest,
} from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { formatDurationMs } from '../../utils/trainingEta';
import { JobProgress } from './JobProgress';
import { TrainingChart } from './TrainingChart';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 '
            + 'dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none '
            + 'focus:border-amber-500/50';
const BTN_SM = 'shrink-0 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border border-zinc-300 '
             + 'dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 '
             + 'dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors';
const BTN_GO = 'px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 '
             + 'disabled:opacity-40 transition-colors flex items-center gap-2';

const NumField: React.FC<{
  label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string;
  disabled?: boolean;
}> = ({ label, value, onChange, step = 1, hint, disabled }) => (
  <label className={`flex flex-col gap-1${disabled ? ' opacity-50' : ''}`}>
    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
    <input
      type="number" className={INPUT} value={value} step={step} disabled={disabled}
      onChange={e => onChange(Number(e.target.value))}
    />
    {hint && <span className="text-[10px] text-zinc-500 leading-snug">{hint}</span>}
  </label>
);

function gb(bytes: number): string {
  return bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(1)} GB`
                             : `${Math.round(bytes / 1048576)} MB`;
}

function when(ms?: number): string {
  if (!ms) return '';
  return new Date(ms).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

// ── Preprocess: audio -> cached VAE latents ─────────────────────────────────

interface PreprocessForm {
  clipSeconds: number;
  captionMode: Yue2CaptionMode;
  defaultCaption: string;
  decode: 'auto' | 'ffmpeg';
  tileFrames: number;
  haloFrames: number;
  only: string;
  limit: number;
  force: boolean;
  acknowledgeSidecarFormat: boolean;
}

export const Yue2PreprocessCard: React.FC<{ status: Yue2Status; onDone: () => void }> = ({ status, onDone }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startYue2Preprocess = useTrainingStore(s => s.startYue2Preprocess);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [edits, setEdits] = useState<Partial<PreprocessForm>>({});

  const jobRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const mine = activeJob?.kind === 'yue2-preprocess';

  const d = status.defaults;
  const form: PreprocessForm = {
    clipSeconds: d.clipSeconds,
    captionMode: d.captionMode,
    defaultCaption: d.defaultCaption,
    decode: d.decode,
    tileFrames: d.tileFrames,
    haloFrames: d.haloFrames,
    only: '',
    limit: 0,
    force: false,
    acknowledgeSidecarFormat: false,
    ...edits,
  };
  const set = <K extends keyof PreprocessForm>(k: K, v: PreprocessForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const blocked = status.missingForPreprocess.length > 0;
  const cache = status.cache;
  // The cache is keyed per SOURCE FILE and is clip-length independent, so
  // re-cutting at a different length is a manifest rewrite and no GPU time —
  // but only with --force, and the engine will otherwise leave the old cut in
  // place while the form says something else.
  const reCut = !!cache && cache.clipFrames > 0
    && Math.round(form.clipSeconds * 25) !== cache.clipFrames;

  const run = async () => {
    setBusy(true);
    try {
      await startYue2Preprocess({
        clipSeconds: form.clipSeconds,
        captionMode: form.captionMode,
        ...(form.captionMode === 'default' ? { defaultCaption: form.defaultCaption } : {}),
        ...(form.captionMode === 'txt' ? { acknowledgeSidecarFormat: form.acknowledgeSidecarFormat } : {}),
        decode: form.decode,
        tileFrames: form.tileFrames,
        haloFrames: form.haloFrames,
        ...(form.only.trim() ? { only: form.only.trim() } : {}),
        ...(form.limit > 0 ? { limit: form.limit } : {}),
        force: form.force || reCut,
      });
    } finally {
      setBusy(false);
      onDone();
    }
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <FileCode2 size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.yue2.ppTitle', 'Latent cache')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2.ppBlurb',
          'Encodes the dataset\'s audio through the YuE2 VAE once and caches the latents, which is what '
          + 'the NAR trainer reads — it never sees the audio. About 4 minutes for a 12 to 15 track album, '
          + 'and a 12-track album comes out at roughly 240 clips. Re-running is cheap: anything already '
          + 'encoded is reused.')}
      </p>

      {blocked ? (
        <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
          <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
          <span>
            {t('trainingStudio.yue2.missing', 'Missing model files')}: {status.missingForPreprocess.join(', ')}
          </span>
        </div>
      ) : (
        <>
          {/* What the FLAT scan sees, beside what the dataset holds. The engine
              takes --audio <folder> and no manifest, so subfolders and the
              dataset's excluded rows genuinely cannot reach it — and a user
              training on tracks they excluded, without being told, is the
              failure worth spending three lines on. */}
          <div className="text-[11px] text-zinc-500 leading-relaxed mb-3">
            {t('trainingStudio.yue2.ppScan',
              '{{files}} audio file(s) directly in the source folder; the dataset has {{samples}} sample(s)'
              + '{{excluded}}.',
              {
                files: status.scannableFiles,
                samples: status.datasetSamples,
                excluded: status.datasetExcluded > 0
                  ? t('trainingStudio.yue2.ppExcluded', ' and {{n}} excluded', { n: status.datasetExcluded })
                  : '',
              })}
            {cache && (
              <>
                {' '}
                <span className="text-emerald-500">
                  {t('trainingStudio.yue2.ppCache',
                    'Cached: {{clips}} clip(s) of {{frames}} frames from {{sources}} track(s).',
                    { clips: cache.clips, frames: cache.clipFrames, sources: cache.sources })}
                </span>
                {(cache.skipped > 0 || cache.failed > 0) && (
                  <span className="text-amber-500">
                    {' '}
                    {t('trainingStudio.yue2.ppCacheProblems', '{{skipped}} skipped, {{failed}} failed.',
                      { skipped: cache.skipped, failed: cache.failed })}
                  </span>
                )}
              </>
            )}
          </div>

          {(status.scannableFiles !== status.datasetSamples || status.recursive) && (
            <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mb-3">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('trainingStudio.yue2.ppFlatScan',
                  'This stage scans the source folder FLAT. Subfolders are not searched whatever the '
                  + 'dataset\'s recursive setting says, and rows excluded in the dataset are still '
                  + 'encoded — the engine takes a folder, not the dataset manifest.')}
              </span>
            </div>
          )}

          {!status.ffmpeg && status.filesNeedingFfmpeg > 0 && (
            <div className="flex items-start gap-2 text-[11px] text-rose-500 mb-3">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('trainingStudio.yue2.ppNoFfmpeg',
                  '{{n}} of the {{total}} files are FLAC/OGG/M4A and need ffmpeg to decode, which this '
                  + 'install does not have. They would all be skipped, so the run is refused.',
                  { n: status.filesNeedingFfmpeg, total: status.scannableFiles })}
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <NumField label={t('trainingStudio.yue2.clipSeconds', 'Clip length (s)')}
              value={form.clipSeconds} onChange={v => set('clipSeconds', v)}
              hint={t('trainingStudio.yue2.clipSecondsHint',
                '10 s = 250 frames, which is what every measured run used. The trainer takes the '
                + 'length from this cache, so changing it re-cuts the manifest.') as string} />
            <label className="flex flex-col gap-1">
              <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {t('trainingStudio.yue2.captionMode', 'Clip captions')}
              </span>
              <select className={INPUT} value={form.captionMode}
                onChange={e => set('captionMode', e.target.value as Yue2CaptionMode)}>
                <option value="ace">{t('trainingStudio.yue2.captionAce', 'Read the sidecars: caption and lyrics')}</option>
                <option value="yue2">{t('trainingStudio.yue2.captionYue2', 'Sidecar lyrics, with the one-sentence YuE2 caption (.yue2.txt) as the style')}</option>
                <option value="none">{t('trainingStudio.yue2.captionNone', 'Nothing: the trigger word is the whole style')}</option>
                <option value="default">{t('trainingStudio.yue2.captionDefault', 'One caption I type here, on every clip')}</option>
                <option value="txt">{t('trainingStudio.yue2.captionTxt', 'The raw sidecar file, field names and all')}</option>
              </select>
              {(status.sidecarsWithLyrics ?? 0) > 0 && (
                <span className={`text-[10px] leading-snug ${form.captionMode === 'ace'
                  ? 'text-emerald-600 dark:text-emerald-500' : 'text-amber-600 dark:text-amber-500'}`}>
                  {form.captionMode === 'ace'
                    ? t('trainingStudio.yue2.captionFound',
                        '{{n}} of the scanned tracks ship a sidecar with lyrics, and this mode reads them.',
                        { n: status.sidecarsWithLyrics })
                    : t('trainingStudio.yue2.captionFoundIgnored',
                        '{{n}} of the scanned tracks ship a sidecar with lyrics and this mode throws them '
                        + 'away. The cursor-span stage and the AR half both need them.',
                        { n: status.sidecarsWithLyrics })}
                </span>
              )}
              <span className="text-[10px] text-zinc-500 leading-snug">
                {form.captionMode === 'ace'
                  ? t('trainingStudio.yue2.captionAceHint',
                      'Reads the .txt beside each track as the fielded sidecar it is, and carries the '
                      + 'caption and the lyric sheet into the cache separately. This is the only mode the '
                      + 'later stages can use: cursor spans align against those lyrics, and the AR half '
                      + 'trains on that caption as its prefix. Any other mode and both skip every track. '
                      + 'Pick one of the others only for a NAR-only run.')
                  : form.captionMode === 'txt'
                  ? t('trainingStudio.yue2.captionTxtHint',
                      'There is no YuE2 caption sidecar. The .txt beside each track is the ACE one — '
                      + '"caption:", "genre:", then the lyrics — and it is fed in RAW AND WHOLE as the '
                      + 'style prompt, field syntax included.')
                  : form.captionMode === 'default'
                    ? t('trainingStudio.yue2.captionDefaultHint',
                        'The same style text on every clip, with the trigger word in front of it.')
                    : t('trainingStudio.yue2.captionNoneHint',
                        'The style is the trigger word alone, which is exactly how the adapter is '
                        + 'addressed at generation time. This is the default.')}
              </span>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {t('trainingStudio.yue2.decode', 'Decoder')}
              </span>
              <select className={INPUT} value={form.decode}
                onChange={e => set('decode', e.target.value as 'auto' | 'ffmpeg')}>
                <option value="auto">auto</option>
                <option value="ffmpeg" disabled={!status.ffmpeg}>ffmpeg</option>
              </select>
              <span className="text-[10px] text-zinc-500 leading-snug">
                {status.ffmpeg
                  ? t('trainingStudio.yue2.decodeHint',
                      'auto = the built-in WAV/MP3 decoder, ffmpeg for everything else. Forcing ffmpeg '
                      + 'puts one resampler across a mixed corpus.')
                  : t('trainingStudio.yue2.decodeNoFfmpeg',
                      'No ffmpeg in this install, so only WAV and MP3 can be decoded.')}
              </span>
            </label>
          </div>

          {form.captionMode === 'default' && (
            <label className="flex flex-col gap-1 mt-3">
              <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {t('trainingStudio.yue2.defaultCaption', 'Caption for every clip')}
              </span>
              <input className={INPUT} value={form.defaultCaption}
                onChange={e => set('defaultCaption', e.target.value)} />
            </label>
          )}

          {form.captionMode === 'txt' && (
            <label className="flex items-start gap-2 mt-3 text-[11px] text-zinc-600 dark:text-zinc-300">
              <input type="checkbox" className="mt-0.5" checked={form.acknowledgeSidecarFormat}
                onChange={e => set('acknowledgeSidecarFormat', e.target.checked)} />
              <span>
                {t('trainingStudio.yue2.ackSidecar', 'Train on the ACE sidecars as they are')}
                <span className="block text-[10px] text-zinc-500">
                  {t('trainingStudio.yue2.ackSidecarHint',
                    'The run is refused without this. Tick it only if training the style encoder on '
                    + '"caption:"/"genre:" field syntax and full lyrics is what you meant.')}
                </span>
              </span>
            </label>
          )}

          <button
            onClick={() => setAdvanced(v => !v)}
            className="flex items-center gap-1 mt-3 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          >
            {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {t('trainingStudio.yue2.advanced', 'Advanced')}
          </button>

          {advanced && (
            <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField label={t('trainingStudio.yue2.tileFrames', 'Encoder tile (frames)')}
                value={form.tileFrames} onChange={v => set('tileFrames', v)} step={50} />
              <NumField label={t('trainingStudio.yue2.haloFrames', 'Tile halo (frames)')}
                value={form.haloFrames} onChange={v => set('haloFrames', v)}
                hint={t('trainingStudio.yue2.haloHint',
                  'The floor is the encoder\'s own receptive field; the engine refuses less.') as string} />
              <label className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.yue2.only', 'Name filter')}
                </span>
                <input className={INPUT} value={form.only} onChange={e => set('only', e.target.value)} />
                <span className="text-[10px] text-zinc-500">
                  {t('trainingStudio.yue2.onlyHint', 'Case-insensitive. Blank = every file.')}
                </span>
              </label>
              <NumField label={t('trainingStudio.yue2.limit', 'File limit')}
                value={form.limit} onChange={v => set('limit', v)}
                hint={t('trainingStudio.yue2.limitHint', '0 = no limit') as string} />
              <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300 col-span-2">
                <input type="checkbox" className="mt-0.5" checked={form.force || reCut}
                  disabled={reCut}
                  onChange={e => set('force', e.target.checked)} />
                <span>
                  {t('trainingStudio.yue2.force', 'Rewrite the manifest')}
                  <span className="block text-[10px] text-zinc-500">
                    {t('trainingStudio.yue2.forceHint',
                      'Needed to re-cut an existing cache at a different clip length. The cached '
                      + 'latents themselves are clip-length independent, so this costs no GPU time.')}
                  </span>
                </span>
              </label>
            </div>
          )}

          {reCut && cache && (
            <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mt-3">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('trainingStudio.yue2.ppReCut',
                  'The cache was cut at {{have}} frames and this form says {{want}}. The manifest will be '
                  + 'rewritten — no audio is re-encoded.',
                  { have: cache.clipFrames, want: Math.round(form.clipSeconds * 25) })}
              </span>
            </div>
          )}

          <div className="flex items-center gap-3 flex-wrap mt-3">
            <button
              onClick={() => void run()}
              disabled={busy || jobRunning}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 disabled:opacity-40 transition-colors flex items-center gap-1.5"
            >
              {busy ? <Loader2 size={12} className="animate-spin" /> : null}
              {cache
                ? t('trainingStudio.yue2.ppReRun', 'Re-encode latents')
                : t('trainingStudio.yue2.ppRun', 'Encode latents')}
            </button>
            <span className="text-[11px] text-zinc-500">
              {t('trainingStudio.yue2.ppCost',
                'About {{gb}} GB of VRAM: the VAE encoder plus a 3.7 GB compute buffer.',
                { gb: (status.preprocessPeakMb / 1024).toFixed(1) })}
              {status.vaeFile ? ` · ${status.vaeFile}` : ''}
            </span>
          </div>
          <p className="text-[11px] text-zinc-500 flex items-center gap-1.5 mt-3">
            <PauseCircle size={12} className="flex-shrink-0" />
            {t('trainingStudio.yue2.enginePaused',
              'The engine is paused while this runs and restarted afterwards.')}
          </p>
        </>
      )}

      {mine && activeJob && (
        <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/10">
          <JobProgress />
        </div>
      )}
    </div>
  );
};

// ── Previous runs and their checkpoint ladders ──────────────────────────────

const OUTCOME: Record<Yue2RunSummary['outcome'], { label: string; tone: string }> = {
  completed: { label: 'Finished',      tone: 'text-emerald-500' },
  halted:    { label: 'Stopped early', tone: 'text-amber-500' },
  failed:    { label: 'Failed',        tone: 'text-rose-500' },
  unknown:   { label: 'Unknown',       tone: 'text-zinc-500' },
};

/** Compact, and read-only on purpose: there is no resume route for YuE2 yet
 *  (the engine refuses a resume whose rank/alpha/target/trigger/seed differ,
 *  and it is exact only within one machine and build), so a "continue" button
 *  here would be a promise the server cannot keep. What the ladder is FOR is
 *  picking which rung to load — the adapter picker takes the file path. */
export const Yue2RunsList: React.FC<{ datasetId: string; reloadKey: unknown }> = ({ datasetId, reloadKey }) => {
  const { t } = useTranslation();
  const [runs, setRuns] = useState<Yue2RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [copied, setCopied] = useState('');

  React.useEffect(() => {
    let cancelled = false;
    listYue2Runs(datasetId)
      .then(r => { if (!cancelled) { setRuns(r.runs); setError(null); } })
      .catch(err => {
        if (!cancelled) { setError(err instanceof Error ? err.message : String(err)); setRuns([]); }
      });
    return () => { cancelled = true; };
  }, [datasetId, reloadKey]);

  if (error) return <div className={`${CARD} text-xs text-rose-500`}>{error}</div>;
  if (!runs || runs.length === 0) return null;

  const copy = (p: string) => {
    void navigator.clipboard?.writeText(p).then(
      () => { setCopied(p); window.setTimeout(() => setCopied(''), 1500); },
      () => { /* clipboard blocked — the path is selectable on screen anyway */ },
    );
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <History size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.yue2.runsTitle', 'Previous runs')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2.runsBlurb',
          'Every NAR LoRA this dataset has produced. Each snapshot is a plain safetensors file, so the '
          + 'whole ladder is loadable — copy a path into the YuE2 adapter field to hear that rung.')}
      </p>
      <div className="flex flex-col gap-2">
        {runs.map(run => {
          const o = OUTCOME[run.outcome];
          return (
            <div key={run.runName} className="rounded-lg border border-zinc-200 dark:border-white/10 p-2.5">
              <div className="flex items-start gap-2 flex-wrap">
                <div className="min-w-0 flex-1">
                  <div className="text-xs font-medium text-zinc-800 dark:text-zinc-200 break-all">
                    {run.runName}
                  </div>
                  <div className="text-[10px] text-zinc-500 mt-0.5 flex flex-wrap gap-x-3 gap-y-0.5">
                    <span className={run.running ? 'text-amber-500' : o.tone}>
                      {run.running ? t('trainingStudio.yue2.runsRunning', 'Running now') : o.label}
                    </span>
                    <span>
                      {t('trainingStudio.yue2.runsSteps', 'step {{done}} of {{cap}}',
                        { done: run.lastStep, cap: run.configuredSteps || run.lastStep })}
                    </span>
                    <span>
                      {t('trainingStudio.yue2.runsCkpts', '{{n}} checkpoints',
                        { n: run.checkpoints.length })}
                    </span>
                    {run.trigger && (
                      <span>
                        {t('trainingStudio.yue2.runsTrigger', 'trigger "{{w}}"', { w: run.trigger })}
                      </span>
                    )}
                    {run.rank !== undefined && (
                      <span>
                        {t('trainingStudio.yue2.runsShape', 'rank {{rank}}, {{target}}',
                          { rank: run.rank, target: run.target || '—' })}
                      </span>
                    )}
                    {run.best && (
                      <span>
                        {t('trainingStudio.yue2.runsBest', 'best loss {{loss}} at {{step}}',
                          { loss: run.best.loss.toFixed(4), step: run.best.step })}
                      </span>
                    )}
                    <span>{gb(run.sizeBytes)}</span>
                    <span>{when(run.updatedAt)}</span>
                  </div>
                  {run.failure && (
                    <div className="text-[10px] text-rose-500 mt-1 break-words">{run.failure}</div>
                  )}
                  {run.resume && !run.running && (
                    <div className="text-[10px] text-amber-600/90 dark:text-amber-400/90 mt-1 leading-snug">
                      {t('trainingStudio.yue2.runsHalted',
                        'This run still holds its optimizer state, which means it stopped before its clean '
                        + 'finish — the engine deletes that file on export. Continuing a run is not wired '
                        + 'up yet; start a fresh one with the same recipe.')}
                    </div>
                  )}
                </div>
                {run.checkpoints.length > 0 && (
                  <button className={BTN_SM}
                    onClick={() => setOpen(open === run.runName ? null : run.runName)}>
                    {open === run.runName
                      ? t('trainingStudio.yue2.runsHide', 'Hide ladder')
                      : t('trainingStudio.yue2.runsShow', 'Ladder')}
                  </button>
                )}
              </div>

              {open === run.runName && (
                <div className="mt-2.5 pt-2.5 border-t border-zinc-200 dark:border-white/10 flex flex-col gap-1">
                  {run.checkpoints.map(c => (
                    <button key={c.name} onClick={() => copy(c.path)}
                      title={c.path}
                      className="flex items-center gap-2 text-left text-[10px] text-zinc-600 dark:text-zinc-300 hover:text-amber-600 dark:hover:text-amber-400 transition-colors">
                      <span className="font-mono break-all flex-1 min-w-0">{c.name}</span>
                      {c.final && (
                        <span className="text-emerald-500 shrink-0">
                          {t('trainingStudio.yue2.runsFinal', 'final')}
                        </span>
                      )}
                      {c.loss !== undefined && (
                        <span className="shrink-0 tabular-nums">{c.loss.toFixed(4)}</span>
                      )}
                      <span className="shrink-0 text-zinc-500">{gb(c.bytes)}</span>
                      <span className="shrink-0 text-zinc-500">
                        {copied === c.path
                          ? t('trainingStudio.yue2.runsCopied', 'copied')
                          : t('trainingStudio.yue2.runsCopy', 'copy path')}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

// ── Training ────────────────────────────────────────────────────────────────

interface TrainForm extends Yue2OptimOptions {
  lmType: string;
  trigger: string;
  rank: number;
  alpha: number;
  target: Yue2NarTarget;
  lr: number;
  lrScheduler: 'cosine' | 'constant';
  steps: number;
  warmup: number;
  saveEvery: number;
  logEvery: number;
  gradAccum: number;
  maxGradNorm: number;
  weightDecay: number;
  captionDropout: number;
  abcDropout: number;
  tSampling: 'logit-normal' | 'uniform';
  seed: number;
  kvCache: number;
  clipBlock: number;
}

const PRESET_ORDER: Yue2PresetName[] = ['balanced', 'thorough', 'fast'];

/** Stage 4: the NAR LoRA. `status` arrives as a prop from ONE useYue2Status()
 *  call in Yue2TrainStages, which also owns the loading gate, the licence
 *  banner and the status-fetch error banner — sharing the fetch across the
 *  five stages, rather than each stage re-requesting the same payload, is the
 *  whole reason this card no longer calls the hook itself. */
export const Yue2NarTrainCard: React.FC<{
  datasetId: string; trigger?: string; status: Yue2Status | null; reload: () => void;
}> = ({ datasetId, trigger, status, reload }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startYue2Train = useTrainingStore(s => s.startYue2Train);
  const yue2Live = useTrainingStore(s => s.yue2Live);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);
  const trainLmEpochs = useTrainingStore(s => s.trainLmEpochs);
  const trainMaxEpochs = useTrainingStore(s => s.trainMaxEpochs);

  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  // DERIVED, not seeded: server defaults underneath, the user's edits on top.
  // No effect copies one into the other, so there is no window where the form
  // holds stale numbers and the measured recipe still has one home.
  const [edits, setEdits] = useState<Partial<TrainForm>>({});

  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobRunning = jobStatus === 'queued' || jobStatus === 'running';
  const mine = jobKind === 'yue2-nar-train';

  const d = status?.defaults;
  const form: TrainForm | null = status && d ? {
    lmType: d.lmType,
    // The dataset's own trigger word first, then whatever the status route
    // reports, and only then blank. Without one the adapter has no handle at
    // generation time and the route refuses the run.
    trigger: trigger || status.trigger || '',
    optimizer: d.optimizer,
    prodigyD0: d.prodigyD0,
    muonLrScale: d.muonLrScale,
    muonNsSteps: d.muonNsSteps,
    rank: d.rank,
    alpha: d.alpha,
    target: d.target,
    lr: d.lr,
    lrScheduler: d.lrScheduler,
    steps: d.steps,
    warmup: d.warmup,
    saveEvery: d.saveEvery,
    logEvery: d.logEvery,
    gradAccum: d.gradAccum,
    maxGradNorm: d.maxGradNorm,
    weightDecay: d.weightDecay,
    captionDropout: d.captionDropout,
    abcDropout: d.abcDropout,
    tSampling: d.tSampling,
    seed: d.seed,
    kvCache: d.kvCache,
    clipBlock: d.clipBlock,
    ...edits,
  } : null;

  const set = <K extends keyof TrainForm>(k: K, v: TrainForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  // Presets are DERIVED from the fields, never stored, so a hand edit shows as
  // Custom rather than misreporting a preset. Step count is the only lever
  // measured here, so it is the only thing they move.
  const presets = status?.presets;
  const activePreset: Yue2PresetName | 'custom' = (() => {
    if (!presets || !form) return 'custom';
    const hit = PRESET_ORDER.find(p => {
      const q = presets[p];
      return q && form.steps === q.steps && form.saveEvery === q.saveEvery;
    });
    return hit ?? 'custom';
  })();
  const applyPreset = (p: Yue2PresetName) => {
    const q = presets?.[p];
    if (!q) return;
    setEdits(e => ({ ...e, steps: q.steps, saveEvery: q.saveEvery }));
  };

  const chosenBase = status?.bases.find(b => b.id === form?.lmType);
  const peak = (() => {
    if (!form || !status) return null;
    const mb = estimateYue2PeakMb(chosenBase?.bytes ?? 0, form.rank, status.vramModel, form.optimizer);
    const gbStr = (mb / 1024).toFixed(1);
    const total = status.gpuTotalMb || 0;
    // 0 means the engine could not be read, NOT a card with no memory. Show the
    // estimate without a verdict rather than inventing a scary one.
    if (total <= 0) {
      return { text: t('trainingStudio.yue2.peakUnknown', 'about {{gb}} GB of VRAM', { gb: gbStr }),
               tone: 'text-zinc-500' };
    }
    const totalGb = (total / 1024).toFixed(1);
    if (mb + 1536 <= total) {
      return { text: t('trainingStudio.yue2.peakFits', 'about {{gb}} GB of your {{totalGb}} GB',
                       { gb: gbStr, totalGb }), tone: 'text-emerald-500' };
    }
    if (mb <= total) {
      return { text: t('trainingStudio.yue2.peakTight',
                       'about {{gb}} GB of your {{totalGb}} GB — fits with nothing left for the desktop',
                       { gb: gbStr, totalGb }), tone: 'text-amber-500' };
    }
    return { text: t('trainingStudio.yue2.peakOver',
                     'about {{gb}} GB, more than your {{totalGb}} GB — lower the rank',
                     { gb: gbStr, totalGb }), tone: 'text-rose-500' };
  })();

  const startTrain = async () => {
    if (!form) return;
    setBusy(true);
    try {
      const body: Yue2TrainRequest = {
        // Informational: the fields below already carry the recipe, and the
        // route lays a named preset UNDER them. It only tells a log reader
        // which recipe the user started from.
        ...(activePreset === 'custom' ? {} : { preset: activePreset }),
        lmType: form.lmType,
        rank: form.rank, alpha: form.alpha, target: form.target,
        optimizer: form.optimizer, prodigyD0: form.prodigyD0,
        muonLrScale: form.muonLrScale, muonNsSteps: form.muonNsSteps,
        lr: form.lr, lrScheduler: form.lrScheduler,
        steps: form.steps, warmup: form.warmup,
        saveEvery: form.saveEvery, logEvery: form.logEvery,
        gradAccum: form.gradAccum, maxGradNorm: form.maxGradNorm,
        weightDecay: form.weightDecay, captionDropout: form.captionDropout,
        abcDropout: form.abcDropout,
        tSampling: form.tSampling, seed: form.seed, kvCache: form.kvCache,
        clipBlock: form.clipBlock,
        // Sent only when there is one: the route falls back to the dataset's
        // own trigger word, and sending '' would look like an answer.
        ...(form.trigger.trim() ? { trigger: form.trigger.trim() } : {}),
      };
      await startYue2Train(body);
    } finally {
      setBusy(false);
      reload();
    }
  };

  const trainBlocked = (status?.missingForTrain.length ?? 0) > 0;
  const clips = status?.cache?.clips ?? 0;
  const tensors = form && status ? status.targetTensors[form.target] : 0;

  return (
    <div className="flex flex-col gap-4">
      {/* ── NAR LoRA ── */}
      <div className={CARD}>
        <div className="flex items-center gap-2 mb-2">
          <Cpu size={15} className="text-amber-500" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.yue2.trainTitle', 'NAR LoRA training')}
          </h3>
        </div>
        <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
          {t('trainingStudio.yue2.trainBlurb',
            'Trains a LoRA on the frozen NAR half of the YuE2 LM from the cached latents. Snapshots are '
            + 'written as plain safetensors files you can load straight into the YuE2 adapter field — '
            + 'there is no install step. The engine is paused for the run.')}
        </p>

        {trainBlocked ? (
          <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t('trainingStudio.yue2.missing', 'Missing model files')}: {status?.missingForTrain.join(', ')}
            </span>
          </div>
        ) : clips <= 0 ? (
          <div className="flex items-start gap-2 text-xs text-zinc-500">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.yue2.needsLatents',
              'Encode the latents first — training reads the cache, not the audio.')}
          </div>
        ) : form && status && (
          <>
            {presets && (
              <div className="mb-3">
                <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-1">
                  {t('trainingStudio.yue2.preset', 'Recipe')}
                </div>
                <div className="flex flex-wrap gap-2">
                  {PRESET_ORDER.map(p => (
                    <button key={p} type="button" onClick={() => applyPreset(p)} disabled={busy}
                      className={BTN_SM + (activePreset === p
                        ? ' !border-amber-500 !text-amber-600 dark:!text-amber-400 !bg-amber-500/10' : '')}>
                      {t(`trainingStudio.yue2.preset.${p}`, p)}
                      <span className="ml-1 text-zinc-500">{presets[p].steps}</span>
                    </button>
                  ))}
                  {activePreset === 'custom' && (
                    <span className={BTN_SM + ' !border-amber-500 !text-amber-600 dark:!text-amber-400 !bg-amber-500/10 cursor-default'}>
                      {t('trainingStudio.yue2.preset.custom', 'Custom')}
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-zinc-500 mt-1">
                  {t('trainingStudio.yue2.presetInfo',
                    'Step count is the only lever measured here, so it is the only thing the presets '
                    + 'move. At the measured 0.065 s per step the wall clock is close to linear: 10,000 '
                    + 'steps is about 12 minutes and 20,000 about 23.')}
                </p>
              </div>
            )}

            <label className="flex flex-col gap-1 mb-3">
              <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {t('trainingStudio.yue2.trigger', 'Trigger word')}
              </span>
              <input className={INPUT} value={form.trigger}
                onChange={e => set('trigger', e.target.value)} />
              <span className="text-[10px] text-zinc-500 leading-snug">
                {form.trigger.trim()
                  ? t('trainingStudio.yue2.triggerHint',
                      'Put in front of every training caption, and the only handle the trained style '
                      + 'has at generation time.')
                  : t('trainingStudio.yue2.triggerMissing',
                      'Required. Without one the adapter has no word to address it by, and the run is '
                      + 'refused rather than warned about twelve minutes in.')}
              </span>
            </label>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField label={t('trainingStudio.yue2.steps', 'Steps')} value={form.steps}
                onChange={v => set('steps', v)} step={1000}
                hint={formatDurationMs(form.steps * status.vramModel.secondsPerStep * 1000)} />
              <NumField label={t('trainingStudio.yue2.rank', 'Rank')} value={form.rank}
                onChange={v => set('rank', v)} step={64}
                hint={t('trainingStudio.yue2.rankHint',
                  '256 is what was measured. 128 saves about 2.6 GB; the ladder between them, and '
                  + 'upstream\'s 16, have not been heard here.') as string} />
              <NumField label={t('trainingStudio.yue2.saveEvery', 'Snapshot every')}
                value={form.saveEvery} onChange={v => set('saveEvery', v)} step={500}
                hint={t('trainingStudio.yue2.saveEveryHint',
                  'Rungs on the ladder — and where a killed run comes back from.') as string} />
              <label className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.yue2.target', 'Trained sites')}
                </span>
                <select className={INPUT} value={form.target}
                  onChange={e => set('target', e.target.value as Yue2NarTarget)}>
                  <option value="nar_attn">nar_attn</option>
                  <option value="nar_attn_mlp">nar_attn_mlp</option>
                  <option value="nar_attn_mlp_proj">nar_attn_mlp_proj</option>
                </select>
                <span className="text-[10px] text-zinc-500 leading-snug">
                  {t('trainingStudio.yue2.targetHint',
                    '{{n}} exported tensors. nar_attn_mlp is attention plus the FFN on all 28 NAR '
                    + 'blocks — 196 sites — and is what the measured runs used.', { n: tensors })}
                </span>
              </label>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
              <label className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.yue2.base', 'Base model')}
                </span>
                <select className={INPUT} value={form.lmType}
                  onChange={e => set('lmType', e.target.value)}>
                  {status.bases.map(b => (
                    <option key={b.id} value={b.id}>
                      {b.id}{b.proven ? ` — ${t('trainingStudio.yue2.baseProven', 'measured')}` : ''}
                      {` · ${gb(b.bytes)}`}
                    </option>
                  ))}
                  {status.bases.length === 0 && <option value={form.lmType}>{form.lmType}</option>}
                </select>
                <span className="text-[10px] text-zinc-500 leading-snug">
                  {t('trainingStudio.yue2.baseHint',
                    'Only bf16 has been trained on here. Whether a quantized base trains usefully at '
                    + 'all is not established, so the others are offered without a ranking.')}
                </span>
              </label>
              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.yue2.cost', 'What this costs')}
                </span>
                {peak && <span className={`text-sm font-semibold ${peak.tone}`}>{peak.text}</span>}
                <span className="text-[10px] text-zinc-500 leading-snug">
                  {t('trainingStudio.yue2.costHint',
                    'Measured on one card: 17.0 GB at rank 128 and 19.6 GB at rank 256, the roughly 7 GB '
                    + 'bf16 base included in both. {{clips}} clips in the cache; 10,000 steps at rank 256 '
                    + 'took about 12 minutes.', { clips })}
                </span>
              </div>
            </div>

            <button
              onClick={() => setAdvanced(v => !v)}
              className="flex items-center gap-1 mt-3 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {t('trainingStudio.yue2.advanced', 'Advanced')}
            </button>

            {advanced && (
              <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 flex flex-col gap-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <NumField label={t('trainingStudio.yue2.alpha', 'Alpha')} value={form.alpha}
                    onChange={v => set('alpha', v)} step={64} />
                  <Yue2OptimizerFields value={form} onChange={patch => setEdits(p => ({ ...p, ...patch }))} />
                  {form.optimizer !== 'prodigy' && (<NumField label={t('trainingStudio.yue2.lr', 'Learning rate')} value={form.lr}
                    onChange={v => set('lr', v)} step={1e-5} />)}
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.yue2.lrScheduler', 'Schedule')}
                    </span>
                    <select className={INPUT} value={form.lrScheduler}
                      onChange={e => set('lrScheduler', e.target.value as 'cosine' | 'constant')}>
                      <option value="cosine">cosine</option>
                      <option value="constant">constant</option>
                    </select>
                  </label>
                  <NumField label={t('trainingStudio.yue2.warmup', 'Warmup steps')} value={form.warmup}
                    onChange={v => set('warmup', v)} step={10} />
                  <NumField label={t('trainingStudio.yue2.gradAccum', 'Grad accum')} value={form.gradAccum}
                    onChange={v => set('gradAccum', v)} />
                  <NumField label={t('trainingStudio.yue2.maxGradNorm', 'Clip grad norm')}
                    value={form.maxGradNorm} onChange={v => set('maxGradNorm', v)} step={0.1} />
                  <NumField label={t('trainingStudio.yue2.weightDecay', 'Weight decay')}
                    value={form.weightDecay} onChange={v => set('weightDecay', v)} step={0.01} />
                  <NumField label={t('trainingStudio.yue2.seed', 'Seed')} value={form.seed}
                    onChange={v => set('seed', v)} />
                  <NumField label={t('trainingStudio.yue2.captionDropout', 'Caption dropout')}
                    value={form.captionDropout} onChange={v => set('captionDropout', v)} step={0.05}
                    hint={t('trainingStudio.yue2.captionDropoutHint',
                      'Chance of swapping in the EMPTY style prefix. 0 is not "more likeness": it is '
                      + 'the trigger word ceasing to mean anything relative to no trigger word.') as string} />
                  <NumField label={t('trainingStudio.yue2.abcDropout', 'ABC dropout')}
                    value={form.abcDropout} onChange={v => set('abcDropout', v)} step={0.05}
                    hint={t('trainingStudio.yue2.abcDropoutHint',
                      'Chance a clip whose source has a lead sheet trains cot=off instead of cot=full '
                      + 'this draw. A source with no lead sheet always trains cot=off. 0.5 is upstream\'s '
                      + 'own split.') as string} />
                  <NumField label={t('trainingStudio.yue2.kvCache', 'K/V canvases')} value={form.kvCache}
                    onChange={v => set('kvCache', v)}
                    hint={t('trainingStudio.yue2.kvCacheHint',
                      'AR-prefix caches held at once, about 104 MB each at 10 s clips. A real VRAM '
                      + 'knob; 8 is what was measured.') as string} />
                  <NumField label={t('trainingStudio.yue2.clipBlock', 'Clip block')} value={form.clipBlock}
                    onChange={v => set('clipBlock', v)}
                    hint={t('trainingStudio.yue2.clipBlockHint',
                      'Steps one working set of clips is held for. With per-clip codec ids every clip '
                      + 'is its own conditioning, so drawing from the whole album misses the K/V cache '
                      + 'almost every step and re-runs the AR prefix. 0 draws from everything.') as string} />
                  <NumField label={t('trainingStudio.yue2.logEvery', 'Log every')} value={form.logEvery}
                    onChange={v => set('logEvery', v)} />
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.yue2.tSampling', 'Timestep sampling')}
                    </span>
                    <select className={INPUT} value={form.tSampling}
                      onChange={e => set('tSampling', e.target.value as 'logit-normal' | 'uniform')}>
                      <option value="logit-normal">logit-normal</option>
                      <option value="uniform">uniform</option>
                    </select>
                  </label>
                </div>
                <p className="text-[11px] text-zinc-500 leading-snug">
                  {status.defaults.deletesResumeStateOnFinish
                    ? t('trainingStudio.yue2.resumeStateNote',
                        'The optimizer state is deleted when a run exports cleanly — it is the run\'s '
                        + 'middle, not its result. A run that stops early keeps it, and the engine will '
                        + 'only continue from it if the rank, alpha, trained sites, trigger, seed and '
                        + 'grad-accum all match, on this same machine and build.')
                    : ''}
                </p>
              </div>
            )}

            <button
              onClick={() => void startTrain()}
              disabled={busy || jobRunning || !form.trigger.trim()}
              className={`mt-4 ${BTN_GO}`}
            >
              <Play size={14} />
              {t('trainingStudio.yue2.start', 'Start training')}
            </button>
          </>
        )}
      </div>

      {/* ── Previous runs and their ladders ── */}
      <Yue2RunsList datasetId={datasetId} reloadKey={`${activeJob?.id ?? ''}:${jobStatus ?? ''}`} />

      {/* ── Live run: the shared job machinery, unchanged ── */}
      {mine && activeJob && (
        <div className={CARD}>
          <JobProgress />
          {yue2Live && (
            <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2">
              {[
                { k: 'loss', v: yue2Live.loss ? yue2Live.loss.toFixed(4) : '—' },
                { k: 'runMean', v: yue2Live.runMean ? yue2Live.runMean.toFixed(4) : '—' },
                { k: 'gradNorm', v: yue2Live.gradNorm ? yue2Live.gradNorm.toFixed(3) : '—' },
                { k: 'stepTime', v: yue2Live.stepMs ? `${(yue2Live.stepMs / 1000).toFixed(2)}s` : '—' },
                {
                  k: 'vram',
                  v: yue2Live.totalMb
                    ? `${Math.round(yue2Live.usedMb / 1024)}/${Math.round(yue2Live.totalMb / 1024)} GB`
                    : '—',
                  warn: yue2Live.totalMb > 0 && yue2Live.usedMb > yue2Live.totalMb - 512,
                },
              ].map(tile => (
                <div key={tile.k}
                  className="rounded-lg border border-zinc-200 dark:border-white/5 px-2.5 py-1.5">
                  <div className="text-[10px] uppercase tracking-wider text-zinc-500">
                    {t(`trainingStudio.yue2.stat.${tile.k}`, tile.k)}
                  </div>
                  <div className={`text-sm font-semibold tabular-nums ${
                    (tile as { warn?: boolean }).warn ? 'text-amber-500' : 'text-zinc-800 dark:text-zinc-200'
                  }`}>{tile.v}</div>
                </div>
              ))}
            </div>
          )}
          {trainStepSeries.length > 1 && (
            <div className="mt-3">
              {/* target 0 hides the target line: the YuE2 trainer has no
                  stop-on-loss mode, so there is never a line to draw. */}
              <TrainingChart
                epochs={trainLmEpochs}
                steps={trainStepSeries}
                milestones={trainMilestones}
                target={0}
                maxEpochs={trainMaxEpochs}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
