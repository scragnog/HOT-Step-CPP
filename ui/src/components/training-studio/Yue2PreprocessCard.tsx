// Yue2PreprocessCard.tsx — YuE2 stage 1: audio -> cached VAE latents.
// Reads `status` as a prop from the single useYue2Status() call in
// Yue2TrainStages, so the five stages share one status fetch.

import { AlertTriangle, ChevronDown, ChevronRight, FileCode2, Loader2, PauseCircle } from 'lucide-react';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';
import type { Yue2CaptionMode, Yue2Status } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { JobProgress } from './JobProgress';
import { NumField } from './TrainingNumberField';
import { Yue2StageCard } from './Yue2StageCard';
import { INPUT } from './yue2CardUi';

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

export const Yue2PreprocessCard: React.FC<{ status: Yue2Status; done: boolean; onDone: () => void }> = ({ status, done, onDone }) => {
  const { t } = useTranslation();
  const activeJob = useActiveTrainingJob();
  const startYue2Preprocess = useTrainingStore(s => s.startYue2Preprocess);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [edits, setEdits] = useState<Partial<PreprocessForm>>({});

  const jobRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const mine = activeJob?.kind === 'yue2-preprocess';

  const d = status.defaults;
  // `base` is what the form shows with no edits — resolved fresh from the
  // server's current defaults on every render, same pattern as Mm3TrainCard.
  const base: PreprocessForm = {
    clipSeconds: d.clipSeconds,
    captionMode: d.captionMode,
    defaultCaption: d.defaultCaption,
    decode: d.decode,
    tileFrames: d.tileFrames,
    haloFrames: d.haloFrames,
    only: '',
    limit: 0,
    force: true,
    acknowledgeSidecarFormat: false,
  };
  const form: PreprocessForm = { ...base, ...edits };
  const set = <K extends keyof PreprocessForm>(k: K, v: PreprocessForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));
  const reset = <K extends keyof PreprocessForm>(k: K) =>
    setEdits(e => { const rest = { ...e }; delete rest[k]; return rest; });

  /** The two modes that carry the sidecar's lyrics into the cache. Everything
   *  downstream (cursor spans, the AR prefix) needs one of them. */
  const readsSidecar = form.captionMode === 'ace' || form.captionMode === 'yue2';

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

  const action = blocked ? (
    <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
      <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
      <span>
        {t('trainingStudio.yue2.missing', 'Missing model files')}: {status.missingForPreprocess.join(', ')}
      </span>
    </div>
  ) : (
    <div className="flex items-center gap-3 flex-wrap">
      <button
        onClick={() => void run()}
        disabled={busy || jobRunning}
        className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 disabled:opacity-40 transition-colors flex items-center gap-1.5"
      >
        {busy ? <Loader2 size={12} className="animate-spin" /> : null}
        {done
          ? t('trainingStudio.yue2.ppReRun', 'Encode latents again')
          : t('trainingStudio.yue2.ppRun', 'Encode latents')}
      </button>
      <span className="text-[11px] text-zinc-500">
        {t('trainingStudio.yue2.ppCost',
          'About {{gb}} GB of VRAM: the VAE encoder plus a 3.7 GB compute buffer.',
          { gb: (status.preprocessPeakMb / 1024).toFixed(1) })}
        {status.vaeFile ? ` · ${status.vaeFile}` : ''}
      </span>
    </div>
  );

  return (
    <Yue2StageCard
      icon={<FileCode2 size={15} className="text-amber-500" />}
      title={t('trainingStudio.yue2.ppTitle', 'Latent cache')}
      done={done}
      action={action}
      footer={mine && activeJob && (
        <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/10">
          <JobProgress />
        </div>
      )}
    >
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2.ppBlurb',
          'Encodes the dataset\'s audio through the YuE2 VAE once and caches the latents, which is what '
          + 'the NAR trainer reads — it never sees the audio. About 4 minutes for a 12 to 15 track album, '
          + 'and a 12-track album comes out at roughly 240 clips. Re-running is cheap: anything already '
          + 'encoded is reused.')}
      </p>

      {!blocked && (
        <>
          {cache && (
            <div className="text-[11px] text-zinc-500 leading-relaxed mb-3">
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
              defaultValue={base.clipSeconds} onReset={() => reset('clipSeconds')}
              meta={t('trainingStudio.yue2.clipSecondsMeta', 'default 10 s') as string}
              info={t('trainingStudio.yue2.clipSecondsInfo',
                'How many seconds of audio each training clip covers once cut from the cache. The trainer '
                + 'takes the clip length from this cache, so changing it here rewrites the manifest next '
                + 'time you encode. At 25 frames per second, 10 s is 250 frames, which is what every '
                + 'measured run used. Raising it gives the model more context per clip at the cost of '
                + 'slower encoding and training; lowering it is cheaper but each clip carries less.') as string} />
            <label className="flex flex-col gap-1">
              <ParamLabel
                label={t('trainingStudio.yue2.captionMode', 'Clip captions')}
                className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                info={t('trainingStudio.yue2.captionModeInfo',
                  'What text the trainer uses as the style prompt on each clip. The sidecar modes ("Read '
                  + 'the sidecars" and the YuE2-caption one) are the only ones that also carry the lyrics '
                  + 'through, which the cursor-span stage and the AR half both need — anything else and '
                  + 'those stages skip every track. "Nothing" leaves only the trigger word as the style, '
                  + '"One caption I type here" repeats the same sentence on every clip, and "The raw '
                  + 'sidecar file" feeds the whole ACE sidecar in verbatim, field names included.') as string}
                onReset={form.captionMode !== base.captionMode ? () => reset('captionMode') : undefined}
              />
              <StyledSelect
                accent="amber"
                value={form.captionMode}
                onChange={v => set('captionMode', v)}
                className="w-full"
                options={[
                  { value: 'ace', label: t('trainingStudio.yue2.captionAce', 'Read the sidecars: caption and lyrics') as string,
                    hint: t('trainingStudio.yue2.captionAceHintShort', 'Carries the caption and lyrics through; the only mode besides yue2 the later stages can use.') as string },
                  { value: 'yue2', label: t('trainingStudio.yue2.captionYue2', 'Sidecar lyrics, with the one-sentence YuE2 caption (.yue2.txt) as the style') as string,
                    hint: t('trainingStudio.yue2.captionYue2HintShort', 'Default. Same lyrics as the sidecar mode, but the style sentence is the planner caption.') as string },
                  { value: 'none', label: t('trainingStudio.yue2.captionNone', 'Nothing: the trigger word is the whole style') as string,
                    hint: t('trainingStudio.yue2.captionNoneHintShort', 'NAR-only: later stages skip every track without lyrics.') as string },
                  { value: 'default', label: t('trainingStudio.yue2.captionDefault', 'One caption I type here, on every clip') as string,
                    hint: t('trainingStudio.yue2.captionDefaultHintShort', 'The same style text on every clip, trigger word in front.') as string },
                  { value: 'txt', label: t('trainingStudio.yue2.captionTxt', 'The raw sidecar file, field names and all') as string,
                    hint: t('trainingStudio.yue2.captionTxtHintShort', 'The ACE sidecar fed in raw and whole, "caption:"/"genre:" syntax included.') as string },
                ]}
              />
              {/* `yue2` reads the same sidecar as `ace` — it only swaps the style
                  sentence for the planner one — so both are the healthy answer
                  here, and neither throws the lyrics away. */}
              {(status.sidecarsWithLyrics ?? 0) > 0 && (
                <span className={`text-[10px] leading-snug ${readsSidecar
                  ? 'text-emerald-600 dark:text-emerald-500' : 'text-amber-600 dark:text-amber-500'}`}>
                  {readsSidecar
                    ? t('trainingStudio.yue2.captionFound',
                        '{{n}} of the scanned tracks ship a sidecar with lyrics, and this mode reads them.',
                        { n: status.sidecarsWithLyrics })
                    : t('trainingStudio.yue2.captionFoundIgnored',
                        '{{n}} of the scanned tracks ship a sidecar with lyrics and this mode throws them '
                        + 'away. The cursor-span stage and the AR half both need them.',
                        { n: status.sidecarsWithLyrics })}
                </span>
              )}
              {/* The cache is cut once and then outlives several rounds of
                  captioning. A cache cut in another mode is silently ignoring
                  every planner caption written since, and nothing downstream
                  can tell — say so here, where the re-cut is one Force away. */}
              {!!cache && cache.loudnessLufs === null && (
                <span className="text-[10px] leading-snug text-amber-600 dark:text-amber-500">
                  {t('trainingStudio.yue2.loudnessStaleCache',
                    'This cache was cut by an older audio prep that clipped loud tracks instead of turning '
                    + 'them down, so loud albums render clipped. Run the cache again to re-encode every track.')}
                </span>
              )}
              {!!cache && (status.sidecarsWithYue2 ?? 0) > 0 && cache.captionMode !== 'yue2' && (
                <span className="text-[10px] leading-snug text-amber-600 dark:text-amber-500">
                  {t('trainingStudio.yue2.captionStaleCache',
                    '{{n}} tracks have a .yue2.txt planner caption, but this cache was cut with '
                    + '"{{mode}}" and trained on the ACE captions instead. Run the cache again with the caption mode above to use them (a manifest rewrite; nothing is re-encoded).',
                    { n: status.sidecarsWithYue2, mode: cache.captionMode || 'none' })}
                </span>
              )}
            </label>
            <label className="flex flex-col gap-1">
              <ParamLabel
                label={t('trainingStudio.yue2.decode', 'Decoder')}
                className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                info={status.ffmpeg
                  ? t('trainingStudio.yue2.decodeInfo',
                      'Which decoder reads the source audio before encoding. auto uses the built-in '
                      + 'WAV/MP3 decoder and falls back to ffmpeg for everything else; forcing ffmpeg puts '
                      + 'one resampler across the whole corpus instead of mixing two.') as string
                  : t('trainingStudio.yue2.decodeNoFfmpegInfo',
                      'Which decoder reads the source audio before encoding. No ffmpeg in this install, '
                      + 'so only WAV and MP3 can be decoded either way.') as string}
                onReset={form.decode !== base.decode ? () => reset('decode') : undefined}
              />
              <StyledSelect
                accent="amber"
                value={form.decode}
                onChange={v => set('decode', v)}
                className="w-full"
                options={[
                  { value: 'auto', label: 'auto' },
                  { value: 'ffmpeg', label: 'ffmpeg', disabled: !status.ffmpeg },
                ]}
              />
            </label>
          </div>

          {form.captionMode === 'default' && (
            <label className="flex flex-col gap-1 mt-3">
              <ParamLabel
                label={t('trainingStudio.yue2.defaultCaption', 'Caption for every clip')}
                className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                info={t('trainingStudio.yue2.defaultCaptionInfo',
                  'The exact text used as the style prompt on every training clip, with the trigger word '
                  + 'placed in front of it. There is no per-track variation in this mode.') as string}
                onReset={form.defaultCaption !== base.defaultCaption ? () => reset('defaultCaption') : undefined}
              />
              <input className={INPUT} value={form.defaultCaption}
                onChange={e => set('defaultCaption', e.target.value)} />
            </label>
          )}

          {form.captionMode === 'txt' && (
            <div className="mt-3">
              <Toggle
                accent="amber"
                checked={form.acknowledgeSidecarFormat}
                onChange={v => set('acknowledgeSidecarFormat', v)}
                defaultValue={base.acknowledgeSidecarFormat} onReset={() => reset('acknowledgeSidecarFormat')}
                label={t('trainingStudio.yue2.ackSidecar', 'Train on the ACE sidecars as they are')}
                info={t('trainingStudio.yue2.ackSidecarHint',
                  'The run is refused without this. Tick it only if training the style encoder on '
                  + '"caption:"/"genre:" field syntax and full lyrics is what you meant.') as string}
              />
            </div>
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
                value={form.tileFrames} onChange={v => set('tileFrames', v)} step={50}
                defaultValue={base.tileFrames} onReset={() => reset('tileFrames')}
                info={t('trainingStudio.yue2.tileFramesInfo',
                  'How many frames of audio the VAE encoder processes in one pass while cutting the cache.') as string} />
              <NumField label={t('trainingStudio.yue2.haloFrames', 'Tile halo (frames)')}
                value={form.haloFrames} onChange={v => set('haloFrames', v)}
                defaultValue={base.haloFrames} onReset={() => reset('haloFrames')}
                info={t('trainingStudio.yue2.haloHint',
                  'Extra frames of context added to each side of an encoder tile so neighbouring tiles '
                  + 'blend without a seam. The floor is the encoder\'s own receptive field; the engine '
                  + 'refuses anything less.') as string} />
              <label className="flex flex-col gap-1">
                <ParamLabel
                  label={t('trainingStudio.yue2.only', 'Name filter')}
                  className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                  info={t('trainingStudio.yue2.onlyHint',
                    'Only encodes files whose name contains this text, case-insensitive. Blank encodes '
                    + 'every file the scan finds.') as string}
                  onReset={form.only !== base.only ? () => reset('only') : undefined}
                />
                <input className={INPUT} value={form.only} onChange={e => set('only', e.target.value)} />
              </label>
              <NumField label={t('trainingStudio.yue2.limit', 'File limit')}
                value={form.limit} onChange={v => set('limit', v)}
                defaultValue={base.limit} onReset={() => reset('limit')}
                meta={t('trainingStudio.yue2.limitMeta', '0 = no limit') as string}
                info={t('trainingStudio.yue2.limitInfo',
                  'Caps how many files this run encodes, useful for a quick test before committing to the '
                  + 'whole dataset. 0 encodes every scanned file.') as string} />
              <div className="col-span-2">
                <Toggle
                  accent="amber"
                  checked={form.force || reCut}
                  disabled={reCut}
                  onChange={v => set('force', v)}
                  defaultValue={base.force} onReset={() => reset('force')}
                  label={t('trainingStudio.yue2.force', 'Rewrite the manifest')}
                  info={t('trainingStudio.yue2.forceHint',
                    'On by default: re-running rewrites the manifest with the caption mode chosen above '
                    + '(otherwise an old cut keeps its old captions) and re-cuts at the clip length. '
                    + 'Cached latents are clip-length independent, so this costs no GPU time.') as string}
                />
              </div>
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

          <p className="text-[11px] text-zinc-500 flex items-center gap-1.5 mt-3">
            <PauseCircle size={12} className="flex-shrink-0" />
            {t('trainingStudio.yue2.enginePaused',
              'The engine is paused while this runs and restarted afterwards.')}
          </p>
        </>
      )}
    </Yue2StageCard>
  );
};

