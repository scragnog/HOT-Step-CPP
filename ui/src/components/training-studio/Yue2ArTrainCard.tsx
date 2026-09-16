// Yue2ArTrainCard.tsx — Training Studio phase 3, the YuE2 AR half: stages 2,
// 3, 5 and 7.
//
// Four of the seven YuE2 stages Yue2TrainStages.tsx renders in order: stage 2
// (Yue2TokenizeCard, `codes/` — what the AR half predicts), stage 3
// (Yue2SheetCard, `abc`/`abc_error` — the SheetSage2 lead sheet both trainers'
// `--abc-dropout` reads), stage 5 (Yue2AlignCard, `cursor/` — what the
// lyric-timing loss is measured against), and stage 7 (Yue2ArTrainStageCard,
// the AR LoRA itself — the composer half, where artist likeness lives). All
// four read `status` as a PROP from one useYue2ArStatus() call in
// Yue2TrainStages, which also owns the loading gate and the status-fetch
// error banner. Stage 2, 3 and 5 all write into the SAME
// yue2_preprocess.json that stage 1 (Yue2TrainCard.tsx) starts.
//
// THE DEFAULTS ARE NOT DUPLICATED HERE. Every number in the form is a view of
// `status.defaults`, which is YUE2_AR_DEFAULTS from services/training/
// yue2ArTrain.ts — the recipe proven by ear, with one home. None of the NAR
// card's numbers apply: different model half, different recipe, no presets.
//
// NO LICENCE BANNER. Yue2TrainStages renders YUE2_LICENSE_NOTICE once, above
// every stage; repeating the same text a second time on one screen teaches
// people to skip it.

import abcjs from 'abcjs';
import 'abcjs/abcjs-audio.css';
import React, { useEffect, useRef, useState } from 'react';
import {
  AlertTriangle, Check, ChevronDown, ChevronRight, Download, FileText, History, Loader2, Mic2,
  Package, PauseCircle, Play, Scissors, Waves,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import {
  getYue2SheetSource, listYue2ArRuns, listYue2SheetSources, sampleAudioUrl,
  type Yue2AlignRequest, type Yue2ArAttn, type Yue2ArLrScheduler, type Yue2ArRunSummary,
  type Yue2ArStatus, type Yue2ArTarget, type Yue2ArTrainRequest, type Yue2SheetRequest,
  type Yue2SheetSourceDetail, type Yue2SheetSourceStatus, type Yue2StyleTemplate,
  type Yue2TokenizeRequest,
} from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { ModelManagerModal } from '../model-manager/ModelManagerModal';
import { JobProgress } from './JobProgress';
import { TrainingChart } from './TrainingChart';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 '
            + 'dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none '
            + 'focus:border-amber-500/50';
const BTN_SM = 'shrink-0 px-2.5 py-1.5 rounded-lg text-[11px] font-medium border border-zinc-300 '
             + 'dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-100 '
             + 'dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors';
const BTN_STAGE = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border '
                + 'border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 '
                + 'disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-1.5';
const BTN_GO = 'px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 '
             + 'disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2';
const LABEL = 'text-[11px] font-medium text-zinc-500 uppercase tracking-wider';
const HINT = 'text-[10px] text-zinc-500 leading-snug';

const NumField: React.FC<{
  label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string;
  disabled?: boolean;
}> = ({ label, value, onChange, step = 1, hint, disabled }) => (
  <label className={`flex flex-col gap-1${disabled ? ' opacity-50' : ''}`}>
    <span className={LABEL}>{label}</span>
    <input
      type="number" className={INPUT} value={value} step={step} disabled={disabled}
      onChange={e => onChange(Number(e.target.value))}
    />
    {hint && <span className={HINT}>{hint}</span>}
  </label>
);

const TextField: React.FC<{
  label: string; value: string; onChange: (v: string) => void; hint?: string; placeholder?: string;
}> = ({ label, value, onChange, hint, placeholder }) => (
  <label className="flex flex-col gap-1">
    <span className={LABEL}>{label}</span>
    <input className={INPUT} value={value} placeholder={placeholder}
      onChange={e => onChange(e.target.value)} />
    {hint && <span className={HINT}>{hint}</span>}
  </label>
);

const CheckField: React.FC<{
  label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string;
  className?: string;
}> = ({ label, checked, onChange, hint, className }) => (
  <label className={`flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300 ${className ?? ''}`}>
    <input type="checkbox" className="mt-0.5" checked={checked}
      onChange={e => onChange(e.target.checked)} />
    <span>
      {label}
      {hint && <span className={`block ${HINT}`}>{hint}</span>}
    </span>
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

/** A snapshot's step, read off its filename. The milestone events the AR relay
 *  emits carry a step and no epoch, and the store keeps only the epoch — so
 *  without this every ladder tick would land at x = 0. The path is written by
 *  the trainer as `<stem>_step<N>.safetensors`, so it is the step. */
function stepFromSnapshot(p: string): number {
  const m = /_step(\d+)\.safetensors$/i.exec(p);
  return m ? Number(m[1]) : 0;
}

// ── Stage 2: codes ──────────────────────────────────────────────────────────

interface TokenizeForm {
  decode: 'auto' | 'ffmpeg';
  only: string;
  limit: number;
  force: boolean;
}

export const Yue2TokenizeCard: React.FC<{ status: Yue2ArStatus; onDone: () => void }> = ({ status, onDone }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startYue2Tokenize = useTrainingStore(s => s.startYue2Tokenize);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [edits, setEdits] = useState<Partial<TokenizeForm>>({});

  const stage = status.stages.tokenize;
  const d = stage.defaults;
  const form: TokenizeForm = { decode: d.decode, only: d.only, limit: d.limit, force: d.force, ...edits };
  const set = <K extends keyof TokenizeForm>(k: K, v: TokenizeForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const jobRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const mine = activeJob?.kind === 'yue2-tokenize';
  const codes = stage.status;
  const needsLatents = !status.stages.preprocess.done;
  const blocked = stage.missing.length > 0;

  const run = async () => {
    setBusy(true);
    try {
      const body: Yue2TokenizeRequest = {
        decode: form.decode,
        ...(form.only.trim() ? { only: form.only.trim() } : {}),
        ...(form.limit > 0 ? { limit: form.limit } : {}),
        force: form.force,
      };
      await startYue2Tokenize(body);
    } finally {
      setBusy(false);
      onDone();
    }
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <Waves size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.yue2ar.tokTitle', 'Codes')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2ar.tokBlurb',
          'Runs the sources the latent cache already names back through the semantic tokenizer and caches '
          + 'the codec ids. These are the TARGETS the AR half learns to predict, so without them there is '
          + 'nothing for it to train on. Re-running is cheap: a source whose codes are already complete is '
          + 'skipped.')}
      </p>

      {blocked ? (
        <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
          <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
          <span>
            {t('trainingStudio.yue2ar.tokMissing',
              'Missing: {{files}}. Install it from the Model Manager (yue2-tok-f16) — nothing in generation '
              + 'needs it, so a fresh install will not have it.',
              { files: stage.missing.join(', ') })}
          </span>
        </div>
      ) : (
        <>
          <div className="text-[11px] text-zinc-500 leading-relaxed mb-3">
            {codes && codes.sourcesWithCodes > 0 ? (
              <span className={codes.sourcesWithCodes >= codes.sources ? 'text-emerald-500' : 'text-amber-500'}>
                {t('trainingStudio.yue2ar.tokHave',
                  '{{done}} of {{total}} source(s) have codes, {{clips}} of {{clipTotal}} clip(s).',
                  { done: codes.sourcesWithCodes, total: codes.sources,
                    clips: codes.clipsWithCodes, clipTotal: codes.clips })}
                {codes.tokenizer ? ` · ${codes.tokenizer}` : ''}
              </span>
            ) : (
              t('trainingStudio.yue2ar.tokNone', 'No codes cached yet.')
            )}
            {stage.tokenizerFile && !codes && ` · ${stage.tokenizerFile}`}
          </div>

          {codes && codes.tokenizer && stage.tokenizerFile && codes.tokenizer !== stage.tokenizerFile && (
            <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mb-3">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('trainingStudio.yue2ar.tokMixed',
                  'These codes came from {{had}} and the installed tokenizer is now {{have}}. Two tokenizers '
                  + 'across one cache is one corpus with two vocabularies — re-run with "Re-encode cached '
                  + 'sources" ticked.',
                  { had: codes.tokenizer, have: stage.tokenizerFile })}
              </span>
            </div>
          )}

          <button
            onClick={() => setAdvanced(v => !v)}
            className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          >
            {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {t('trainingStudio.yue2ar.advanced', 'Advanced')}
          </button>

          {advanced && (
            <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 grid grid-cols-2 md:grid-cols-4 gap-3">
              <label className="flex flex-col gap-1">
                <span className={LABEL}>{t('trainingStudio.yue2ar.decode', 'Decoder')}</span>
                <select className={INPUT} value={form.decode}
                  onChange={e => set('decode', e.target.value as 'auto' | 'ffmpeg')}>
                  <option value="auto">auto</option>
                  <option value="ffmpeg" disabled={!status.ffmpeg}>ffmpeg</option>
                </select>
                <span className={HINT}>
                  {t('trainingStudio.yue2ar.decodeHint',
                    'Must match what the latent cache used: the codes and the latents are only '
                    + 'frame-aligned because both came from identically decoded samples.')}
                </span>
              </label>
              <TextField label={t('trainingStudio.yue2ar.only', 'Name filter')}
                value={form.only} onChange={v => set('only', v)}
                hint={t('trainingStudio.yue2ar.onlyHint',
                  'Case-insensitive. Blank = every source.') as string} />
              <NumField label={t('trainingStudio.yue2ar.limit', 'Source limit')}
                value={form.limit} onChange={v => set('limit', v)}
                hint={t('trainingStudio.yue2ar.limitHint', '0 = no limit') as string} />
              <CheckField className="col-span-2 md:col-span-1 self-end pb-1.5"
                label={t('trainingStudio.yue2ar.tokForce', 'Re-encode cached sources')}
                checked={form.force} onChange={v => set('force', v)}
                hint={t('trainingStudio.yue2ar.tokForceHint',
                  'Off, an already complete source is skipped — which is what makes a resumed run '
                  + 'cheap.') as string} />
            </div>
          )}

          <div className="flex items-center gap-3 flex-wrap mt-3">
            <button onClick={() => void run()} disabled={busy || jobRunning || needsLatents}
              className={BTN_STAGE}>
              {busy ? <Loader2 size={12} className="animate-spin" /> : null}
              {codes && codes.sourcesWithCodes > 0
                ? t('trainingStudio.yue2ar.tokReRun', 'Tokenize again')
                : t('trainingStudio.yue2ar.tokRun', 'Tokenize')}
            </button>
            {needsLatents && (
              <span className="text-[11px] text-zinc-500">
                {t('trainingStudio.yue2ar.needsLatents',
                  'Encode the latents first — this stage reads the manifest preprocess writes, not the '
                  + 'source folder.')}
              </span>
            )}
          </div>
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

// ── Stage: lead sheets ────────────────────────────────────────────────────
//
// Independent of codes/stems/align: SheetSage2 reads a source's own audio,
// not its codes or cursor spans. What it produces feeds --abc-dropout on
// both trainers (Yue2NarTrainCard, Yue2ArTrainStageCard), not this card.

interface SheetForm {
  only: string;
  force: boolean;
  fast: boolean;
}

export const Yue2SheetCard: React.FC<{ datasetId: string; status: Yue2ArStatus; onDone: () => void }> =
    ({ datasetId, status, onDone }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startYue2Sheet = useTrainingStore(s => s.startYue2Sheet);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [edits, setEdits] = useState<Partial<SheetForm>>({});

  const stage = status.stages.sheet;
  const d = stage.defaults;
  const form: SheetForm = { only: d.only, force: d.force, fast: d.fast, ...edits };
  const set = <K extends keyof SheetForm>(k: K, v: SheetForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const jobRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const mine = activeJob?.kind === 'yue2-sheet';
  const abc = stage.status;
  const needsLatents = !status.stages.preprocess.done;
  const blocked = stage.missing.length > 0;

  const run = async () => {
    setBusy(true);
    try {
      const body: Yue2SheetRequest = {
        ...(form.only.trim() ? { only: form.only.trim() } : {}),
        force: form.force,
        fast: form.fast,
      };
      await startYue2Sheet(body);
    } finally {
      setBusy(false);
      onDone();
    }
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <FileText size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.yue2ar.sheetTitle', 'Lead sheets')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2ar.sheetBlurb',
          'Transcribes each source\'s own audio into a SheetSage2 lead sheet (chords + melody), which '
          + '--abc-dropout on both trainers reads to draw the cot=full conditioning instead of cot=off. '
          + 'Optional: a source with no lead sheet simply always trains cot=off, same as before this '
          + 'stage existed. Roughly 4% of real tracks decode fine but fail to render — that shows up as '
          + '"soft-failed", not an error, and those sources also always train cot=off.')}
      </p>

      {blocked ? (
        <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
          <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
          <span>
            {t('trainingStudio.yue2ar.sheetMissing',
              'Missing: {{files}}. Install it from the Model Manager (yue2-sheetsage2-f16) — nothing in '
              + 'generation needs it, so a fresh install will not have it.',
              { files: stage.missing.join(', ') })}
          </span>
        </div>
      ) : (
        <>
          <div className="text-[11px] text-zinc-500 leading-relaxed mb-3">
            {abc && (abc.sourcesWithAbc + abc.sourcesWithError) > 0 ? (
              <span className={stage.done ? 'text-emerald-500' : 'text-amber-500'}>
                {t('trainingStudio.yue2ar.sheetHave',
                  '{{done}} of {{total}} source(s) have a lead sheet, {{failed}} soft-failed (abc_error).',
                  { done: abc.sourcesWithAbc, total: abc.sources, failed: abc.sourcesWithError })}
              </span>
            ) : (
              t('trainingStudio.yue2ar.sheetNone', 'No lead sheets cached yet.')
            )}
            {stage.sheetModelFile && ` · ${stage.sheetModelFile}`}
          </div>

          <button
            onClick={() => setAdvanced(v => !v)}
            className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          >
            {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {t('trainingStudio.yue2ar.advanced', 'Advanced')}
          </button>

          {advanced && (
            <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 grid grid-cols-2 md:grid-cols-4 gap-3">
              <TextField label={t('trainingStudio.yue2ar.only', 'Name filter')}
                value={form.only} onChange={v => set('only', v)}
                hint={t('trainingStudio.yue2ar.onlyHint',
                  'Case-insensitive. Blank = every source.') as string} />
              <CheckField className="col-span-2 md:col-span-1 self-end pb-1.5"
                label={t('trainingStudio.yue2ar.sheetForce', 'Re-transcribe cached sources')}
                checked={form.force} onChange={v => set('force', v)}
                hint={t('trainingStudio.yue2ar.sheetForceHint',
                  'Off, a source that already has abc or abc_error is skipped — which is what makes a '
                  + 'resumed run cheap, since one transcription can run minutes.') as string} />
              <CheckField className="col-span-2 md:col-span-1 self-end pb-1.5"
                label={t('trainingStudio.yue2ar.sheetFast', 'Fast (less precise)')}
                checked={form.fast} onChange={v => set('fast', v)}
                hint={t('trainingStudio.yue2ar.sheetFastHint',
                  'Skips the exact-load precision fix. Leave off unless you have measured that the '
                  + 'default matters for your corpus.') as string} />
            </div>
          )}

          <div className="flex items-center gap-3 flex-wrap mt-3">
            <button onClick={() => void run()} disabled={busy || jobRunning || needsLatents}
              className={BTN_STAGE}>
              {busy ? <Loader2 size={12} className="animate-spin" /> : null}
              {abc && (abc.sourcesWithAbc + abc.sourcesWithError) > 0
                ? t('trainingStudio.yue2ar.sheetReRun', 'Transcribe again')
                : t('trainingStudio.yue2ar.sheetRun', 'Transcribe')}
            </button>
            {needsLatents && (
              <span className="text-[11px] text-zinc-500">
                {t('trainingStudio.yue2ar.needsLatents',
                  'Encode the latents first — this stage reads the manifest preprocess writes, not the '
                  + 'source folder.')}
              </span>
            )}
          </div>
        </>
      )}

      {mine && activeJob && (
        <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/10">
          <JobProgress />
        </div>
      )}

      {abc && abc.sourcesWithAbc + abc.sourcesWithError > 0 && (
        <Yue2SheetPreview datasetId={datasetId} reloadKey={abc.sourcesWithAbc + abc.sourcesWithError} />
      )}
    </div>
  );
};

// ── Lead sheets: score + audio preview ──────────────────────────────────────
//
// A collapsible section under Yue2SheetCard, not its own card: nothing here
// starts a job, it only reads what the stage already wrote. `reloadKey`
// changes whenever the manifest's abc/abc_error counts do (a run or re-run
// finished), which is the cue to refetch the picker list.

function Yue2SheetPreview({ datasetId, reloadKey }: { datasetId: string; reloadKey: number }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [sources, setSources] = useState<Yue2SheetSourceStatus[] | null>(null);
  const [listError, setListError] = useState('');
  const [selected, setSelected] = useState('');
  const [detail, setDetail] = useState<Yue2SheetSourceDetail | null>(null);
  const [detailError, setDetailError] = useState('');
  const [loadingDetail, setLoadingDetail] = useState(false);
  const scoreRef = useRef<HTMLDivElement | null>(null);
  const [renderNote, setRenderNote] = useState('');
  const audioControlRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    listYue2SheetSources(datasetId)
      .then(r => {
        if (cancelled) return;
        setSources(r.sources);
        setListError('');
        setSelected(prev => {
          if (prev && r.sources.some(s => s.name === prev && s.ok)) return prev;
          return r.sources.find(s => s.ok)?.name ?? '';
        });
      })
      .catch(err => { if (!cancelled) setListError(err?.message || String(err)); });
    return () => { cancelled = true; };
  }, [open, datasetId, reloadKey]);

  useEffect(() => {
    if (!open || !selected) return;
    let cancelled = false;
    setLoadingDetail(true);
    setDetailError('');
    getYue2SheetSource(datasetId, selected)
      .then(d => { if (!cancelled) setDetail(d); })
      .catch(err => { if (!cancelled) { setDetail(null); setDetailError(err?.message || String(err)); } })
      .finally(() => { if (!cancelled) setLoadingDetail(false); });
    return () => { cancelled = true; };
  }, [open, selected, datasetId]);

  // The fetch above never clears a stale `detail` synchronously on unmount of
  // interest (that would be a setState-in-effect lint violation) — this
  // guards the render instead, so switching tracks never shows the PREVIOUS
  // track's score while the new one is still loading.
  const showDetail = detail && detail.name === selected && !loadingDetail;

  // Render the score and wire the transport once a lead sheet comes back.
  // abcjs's SynthController builds its own play/pause/progress UI inside
  // audioControlRef; the first Play click is the user gesture the browser
  // needs before it lets an AudioContext make sound, and that click also
  // triggers abcjs's own soundfont fetch from its default CDN
  // (https://paulrosen.github.io/abcjs/soundfont/) — nothing bundled here.
  useEffect(() => {
    if (!showDetail || !detail?.abc || !scoreRef.current) return;
    scoreRef.current.innerHTML = '';
    let tunes: ReturnType<typeof abcjs.renderAbc> | undefined;
    try {
      tunes = abcjs.renderAbc(scoreRef.current, detail.abc, { responsive: 'resize', add_classes: true });
    } catch (e) {
      setRenderNote(`abcjs could not render this sheet: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    if (!tunes) return;
    const warnings = tunes[0]?.warnings;
    setRenderNote(warnings && warnings.length ? `abcjs warnings: ${warnings.slice(0, 3).join(' | ')}` : '');
    if (audioControlRef.current && abcjs.synth.supportsAudio() && tunes[0]) {
      audioControlRef.current.innerHTML = '';
      const synthControl = new abcjs.synth.SynthController();
      // Follow the playback: abcjs calls onEvent per note with the SVG
      // elements it drew for it (add_classes above), so we tint those and
      // keep them in view inside the scrolling paper.
      const box = scoreRef.current;
      let lit: Element[] = [];
      const cursorControl = {
        onStart() { lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = []; },
        onEvent(ev: { elements?: Element[][] }) {
          lit.forEach(el => el.classList.remove('abcjs-highlight'));
          lit = (ev.elements ?? []).flat();
          lit.forEach(el => el.classList.add('abcjs-highlight'));
          const first = lit[0] as (Element & { scrollIntoView?: (o: ScrollIntoViewOptions) => void }) | undefined;
          if (first && box) first.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
        },
        onFinished() { lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = []; },
      };
      synthControl.load(audioControlRef.current, cursorControl, {
        displayLoop: false, displayRestart: true, displayPlay: true,
        displayProgress: true, displayWarp: false,
      });
      synthControl.setTune(tunes[0], false).catch(() => { /* score still renders without audio */ });
    }
  }, [detail, showDetail]);

  return (
    <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/10">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {t('trainingStudio.yue2ar.sheetPreview', 'Preview a lead sheet')}
      </button>

      {open && (
        <div className="mt-3 space-y-3">
          {listError && (
            <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
              <span>{listError}</span>
            </div>
          )}

          {sources && (
            <label className="flex flex-col gap-1">
              <span className={LABEL}>{t('trainingStudio.yue2ar.sheetTrack', 'Track')}</span>
              <select className={INPUT} value={selected} onChange={e => setSelected(e.target.value)}>
                <option value="" disabled>
                  {t('trainingStudio.yue2ar.sheetTrackPick', 'Choose a source…')}
                </option>
                {sources.map(s => (
                  <option key={s.name} value={s.name} disabled={!s.ok}>
                    {s.ok ? s.name : `${s.name} — ${s.error || t('trainingStudio.yue2ar.sheetTrackNone', 'no lead sheet')}`}
                  </option>
                ))}
              </select>
            </label>
          )}

          {loadingDetail && (
            <div className="flex items-center gap-2 text-[11px] text-zinc-500">
              <Loader2 size={12} className="animate-spin" /> {t('trainingStudio.yue2ar.sheetLoading', 'Loading…')}
            </div>
          )}

          {detailError && (
            <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
              <span>{detailError}</span>
            </div>
          )}

          {showDetail && detail && (
            detail.abc_error ? (
              <div className="text-[11px] text-amber-600 dark:text-amber-400 leading-relaxed">
                {t('trainingStudio.yue2ar.sheetSoftFail',
                  'This source soft-failed to render a lead sheet: {{error}}',
                  { error: detail.abc_error })}
              </div>
            ) : detail.abc ? (
              <div className="space-y-2">
                <div ref={scoreRef} className="bg-white text-black rounded-lg p-2 max-h-[420px] overflow-auto [&_svg]:fill-current [&_.abcjs-highlight]:fill-amber-500 [&_.abcjs-highlight]:stroke-amber-500" />
                {renderNote && <div className="mt-1 text-[11px] text-amber-600 dark:text-amber-400 break-words">{renderNote}</div>}
                <div ref={audioControlRef} className="text-xs" />
                <div className="flex flex-col gap-1">
                  <span className={LABEL}>{t('trainingStudio.yue2ar.sheetOriginal', 'Original audio')}</span>
                  <audio controls className="w-full h-8" src={sampleAudioUrl(datasetId, detail.sampleId)} />
                </div>
              </div>
            ) : (
              <span className="text-[11px] text-zinc-500">
                {t('trainingStudio.yue2ar.sheetTrackNone', 'No lead sheet for this source.')}
              </span>
            )
          )}
        </div>
      )}
    </div>
  );
}

// ── Stage 2a: vocal stems ────────────────────────────────────────
//
// Its own stage rather than a step inside align, for the reason the server
// gives (routes/training.ts, POST .../yue2-stems): separation costs minutes a
// track and alignment costs seconds, so a failed alignment should cost one
// retry and not a whole corpus of separations.
//
// It had no UI at all until now — the endpoint and the job kind existed, the
// align card told people to run it, and there was nothing anywhere to press.

interface StemsForm {
  level: number;
  force: boolean;
}

export const Yue2StemsCard: React.FC<{ status: Yue2ArStatus; onDone: () => void }> = ({ status, onDone }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startYue2Stems = useTrainingStore(s => s.startYue2Stems);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [edits, setEdits] = useState<Partial<StemsForm>>({});

  const stage = status.stages.align;
  // 4 = SUPERSEP_VOCALS_ONLY. See the server's own note (yue2Stems.ts): level 0
  // is three model passes and five discarded stems for a stage that opens one
  // file.
  const form: StemsForm = { level: 4, force: false, ...edits };
  const set = <K extends keyof StemsForm>(k: K, v: StemsForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const jobRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const mine = activeJob?.kind === 'yue2-stems';
  const needsLatents = !status.stages.preprocess.done;
  const ready = stage.stemsReady;
  const needed = stage.stemsNeeded;
  const complete = ready > 0 && (needed === 0 || ready >= needed);

  const run = async () => {
    setBusy(true);
    try {
      await startYue2Stems({ level: form.level, force: form.force });
    } finally {
      setBusy(false);
      onDone();
    }
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <Scissors size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.yue2ar.stemsTitle', 'Vocal stems')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2ar.stemsBlurb',
          'Separates each song\'s vocal and writes it as <source>/vocals.wav, which is the only thing the '
          + 'aligner below reads. It is a separate stage because separation costs minutes a track while '
          + 'alignment costs seconds, and because it is worth doing once: a stem already on disk is skipped.')}
      </p>

      {/* The count comes from the status fetch, which does not re-read while a
          job runs — so mid-separation it says "2 of 12" under a live bar
          reading 7/12. Two numbers for one thing, one of them stale. The bar
          wins while it is on screen. */}
      <div className="text-[11px] text-zinc-600 dark:text-zinc-300 mb-3">
        {mine && activeJob ? (
          <span className="text-zinc-500">
            {t('trainingStudio.yue2ar.stemsLive', 'Separating now — the count below is live.')}
          </span>
        ) : ready > 0 ? (
          <span className={complete ? 'text-emerald-500' : 'text-amber-500'}>
            {t('trainingStudio.yue2ar.stemsHave', '{{ready}} of {{needed}} song(s) separated.',
              { ready, needed: needed || ready })}
          </span>
        ) : (
          t('trainingStudio.yue2ar.stemsNone', 'No stems yet.')
        )}
        <span className="block text-[10px] text-zinc-500 font-mono break-all mt-1">{stage.stemsDir}</span>
      </div>

      <button onClick={() => setAdvanced(a => !a)} className={`${BTN_SM} mb-3`}>
        {advanced
          ? t('trainingStudio.yue2ar.advancedHide', 'Hide advanced')
          : t('trainingStudio.yue2ar.advancedShow', 'Advanced')}
      </button>

      {advanced && (
        <div className="grid grid-cols-2 gap-3 mb-3">
          <label className="flex flex-col gap-1">
            <span className={LABEL}>{t('trainingStudio.yue2ar.stemsLevel', 'Separator')}</span>
            <select className={INPUT} value={form.level}
              onChange={e => set('level', Number(e.target.value))}>
              <option value={4}>
                {t('trainingStudio.yue2ar.stemsLevelVocals', 'Vocals only, one pass (recommended)')}
              </option>
              <option value={5}>
                {t('trainingStudio.yue2ar.stemsLevelLeap', 'Leap Xe pair, two passes')}
              </option>
              <option value={0}>
                {t('trainingStudio.yue2ar.stemsLevelFull', 'Full six-stem split (slowest)')}
              </option>
            </select>
            <span className={HINT}>
              {t('trainingStudio.yue2ar.stemsLevelHint',
                'The aligner opens one file, vocals.wav, so the default runs the separator with everything '
                + 'but the vocal stem masked off. The six-stem split produces a drum kit and a piano this '
                + 'stage then deletes, at roughly three model passes instead of one.')}
            </span>
          </label>
          <CheckField
            className="col-span-2 md:col-span-1 self-end pb-1.5"
            label={t('trainingStudio.yue2ar.stemsForce', 'Re-separate existing stems')}
            checked={form.force}
            onChange={v => set('force', v)}
            hint={t('trainingStudio.yue2ar.stemsForceHint',
              'Off, a song whose stem is already on disk is skipped, which is what makes a resumed run '
              + 'cheap. Turn it on only after changing the source audio or the level.') as string}
          />
        </div>
      )}

      <div className="flex items-center gap-3">
        <button onClick={run}
          disabled={busy || jobRunning || needsLatents}
          className={BTN_STAGE}>
          {busy ? <Loader2 size={12} className="animate-spin" /> : null}
          {ready > 0
            ? t('trainingStudio.yue2ar.stemsReRun', 'Separate again')
            : t('trainingStudio.yue2ar.stemsRun', 'Separate vocals')}
        </button>
        {needsLatents && (
          <span className="text-[11px] text-zinc-500">
            {t('trainingStudio.yue2ar.stemsNeedsLatents',
              'Encode the latents first \u2014 this stage separates the sources that manifest names.')}
          </span>
        )}
      </div>

      {mine && activeJob && (
        <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/10">
          <JobProgress />
        </div>
      )}
    </div>
  );
};

// ── Stage 3: cursor spans ───────────────────────────────────────────────────

interface AlignForm {
  stemsDir: string;
  only: string;
  limit: number;
  cpu: boolean;
}

export const Yue2AlignCard: React.FC<{ status: Yue2ArStatus; onDone: () => void }> = ({ status, onDone }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startYue2Align = useTrainingStore(s => s.startYue2Align);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [edits, setEdits] = useState<Partial<AlignForm>>({});

  const stage = status.stages.align;
  const d = stage.defaults;
  // stemsDir blank means "wherever the server looks by default", which is what
  // the placeholder shows. Sending '' would look like an answer.
  const form: AlignForm = { stemsDir: '', only: d.only, limit: d.limit, cpu: d.cpu, ...edits };
  const set = <K extends keyof AlignForm>(k: K, v: AlignForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const jobRunning = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const mine = activeJob?.kind === 'yue2-align';
  const cursor = stage.status;
  const needsLatents = !status.stages.preprocess.done;
  const blocked = stage.missing.length > 0;
  const noStems = stage.stemsReady === 0;
  // The other way this stage has nothing to do: the manifest carries no lyrics
  // at all because the latent cache was built with caption mode `none`. That
  // is the right build for a NAR-only run and unusable here, and the engine's
  // answer to it is twelve identical "no lyrics in the manifest" lines and a
  // failed job — true, and no help at all in working out that the fix is two
  // stages further up the page.
  const noLyrics = status.stages.preprocess.captionModeOk === false;

  const run = async () => {
    setBusy(true);
    try {
      const body: Yue2AlignRequest = {
        ...(form.stemsDir.trim() ? { stemsDir: form.stemsDir.trim() } : {}),
        ...(form.only.trim() ? { only: form.only.trim() } : {}),
        ...(form.limit > 0 ? { limit: form.limit } : {}),
        cpu: form.cpu,
      };
      await startYue2Align(body);
    } finally {
      setBusy(false);
      onDone();
    }
  };

  return (
    <div className={CARD}>
      <div className="flex items-center gap-2 mb-2">
        <Mic2 size={15} className="text-amber-500" />
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.yue2ar.alignTitle', 'Lyric cursor spans')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2ar.alignBlurb',
          'Force-aligns each song\'s lyrics against its vocal stem and caches where every word is sung. '
          + 'That is what the cursor loss is measured against, and it is measured to matter — with it the '
          + 'frame-to-lyric alignment loss fell where without it it rose. Optional only if you turn the '
          + 'cursor weight down to 0.')}
      </p>

      {blocked ? (
        <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
          <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
          <span>
            {t('trainingStudio.yue2ar.alignMissing',
              'Missing: {{files}}. The aligner has no Model Manager entry yet, so it has to be placed in '
              + '{{dir}} by hand.',
              // `minted.dir` is the YuE2 model folder, which is where the
              // aligner goes too — the status payload names it once.
              { files: stage.missing.join(', '), dir: status.minted.dir })}
          </span>
        </div>
      ) : (
        <>
          <div className="text-[11px] text-zinc-500 leading-relaxed mb-3">
            {cursor && cursor.sourcesWithCursor > 0 ? (
              <span className={cursor.sourcesWithCursor >= cursor.sources ? 'text-emerald-500' : 'text-amber-500'}>
                {t('trainingStudio.yue2ar.alignHave',
                  '{{done}} of {{total}} source(s) have cursor spans.',
                  { done: cursor.sourcesWithCursor, total: cursor.sources })}
                {cursor.model ? ` · ${cursor.model}` : ''}
              </span>
            ) : (
              t('trainingStudio.yue2ar.alignNone', 'No cursor spans cached yet.')
            )}
            {' '}
            {t('trainingStudio.yue2ar.alignStems',
              '{{n}} vocal stem(s) in {{dir}}.',
              { n: stage.stemsReady, dir: stage.stemsDir })}
          </div>

          {noLyrics && (
            <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mb-3">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('trainingStudio.yue2ar.alignNoLyrics',
                  'The latent cache was built with clip captions set to "{{mode}}", so it carries no lyrics '
                  + 'and this stage has nothing to align: it would skip every source and fail. Re-encode the '
                  + 'latents with clip captions set to "ace" — the sidecars beside the audio already hold the '
                  + 'lyrics, and the AR half needs the captions anyway, since the caption is its prefix.',
                  { mode: status.stages.preprocess.captionMode || 'none' })}
              </span>
            </div>
          )}

          {noStems && (
            <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mb-3">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('trainingStudio.yue2ar.alignNoStems',
                  'Separation happens outside this stage, so the stems are an input it cannot produce. It '
                  + 'wants <source stem>/vocals.wav per song and skips by name, so with none of them it '
                  + 'would align nothing and still report success — the run is refused instead. Run the '
                  + 'Vocal stems stage just above, or point Advanced\'s stems folder at stems you already '
                  + 'have.')}
              </span>
            </div>
          )}

          <button
            onClick={() => setAdvanced(v => !v)}
            className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          >
            {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            {t('trainingStudio.yue2ar.advanced', 'Advanced')}
          </button>

          {advanced && (
            <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="col-span-2">
                <TextField label={t('trainingStudio.yue2ar.stemsDir', 'Stems folder')}
                  value={form.stemsDir} onChange={v => set('stemsDir', v)}
                  placeholder={stage.stemsDir}
                  hint={t('trainingStudio.yue2ar.stemsDirHint',
                    'Blank uses the dataset\'s own. A path you type has to exist — the engine\'s answer to '
                    + 'a wrong one is to skip every source and exit cleanly.') as string} />
              </div>
              <TextField label={t('trainingStudio.yue2ar.only', 'Name filter')}
                value={form.only} onChange={v => set('only', v)}
                hint={t('trainingStudio.yue2ar.alignOnlyHint',
                  'Also the re-run mechanism: this stage has no force flag, so name the source whose '
                  + 'lyrics you edited.') as string} />
              <NumField label={t('trainingStudio.yue2ar.limit', 'Source limit')}
                value={form.limit} onChange={v => set('limit', v)}
                hint={t('trainingStudio.yue2ar.limitHint', '0 = no limit') as string} />
              <CheckField className="col-span-2"
                label={t('trainingStudio.yue2ar.alignCpu', 'Run the aligner on the CPU')}
                checked={form.cpu} onChange={v => set('cpu', v)}
                hint={t('trainingStudio.yue2ar.alignCpuHint',
                  'Several times slower, and both backends pass the port\'s gates. It cannot be changed '
                  + 'once the run starts.') as string} />
            </div>
          )}

          <div className="flex items-center gap-3 flex-wrap mt-3">
            <button onClick={() => void run()}
              disabled={busy || jobRunning || needsLatents || noLyrics || (noStems && !form.stemsDir.trim())}
              className={BTN_STAGE}>
              {busy ? <Loader2 size={12} className="animate-spin" /> : null}
              {cursor && cursor.sourcesWithCursor > 0
                ? t('trainingStudio.yue2ar.alignReRun', 'Align again')
                : t('trainingStudio.yue2ar.alignRun', 'Align lyrics')}
            </button>
            {needsLatents && (
              <span className="text-[11px] text-zinc-500">
                {t('trainingStudio.yue2ar.alignNeedsLatents',
                  'Encode the latents first — the lyrics the spans are measured against come out of that '
                  + 'manifest.')}
              </span>
            )}
          </div>
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

// ── Previous AR runs and their checkpoint ladders ───────────────────────────

const OUTCOME: Record<Yue2ArRunSummary['outcome'], { label: string; tone: string }> = {
  completed: { label: 'Finished',      tone: 'text-emerald-500' },
  halted:    { label: 'Stopped early', tone: 'text-amber-500' },
  failed:    { label: 'Failed',        tone: 'text-rose-500' },
  unknown:   { label: 'Unknown',       tone: 'text-zinc-500' },
};

/** Read-only, like the NAR list and for the same reason: there is no resume
 *  route, so a "continue" button would be a promise the server cannot keep.
 *  What the ladder is FOR is picking a rung by ear — copy a path into the YuE2
 *  adapter field. The suggested rung is marked, not chosen. */
export const Yue2ArRunsList: React.FC<{ datasetId: string; pickStep: number; reloadKey: unknown }> = (
  { datasetId, pickStep, reloadKey },
) => {
  const { t } = useTranslation();
  const [runs, setRuns] = useState<Yue2ArRunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [copied, setCopied] = useState('');

  React.useEffect(() => {
    let cancelled = false;
    listYue2ArRuns(datasetId)
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
          {t('trainingStudio.yue2ar.runsTitle', 'Previous AR runs')}
        </h3>
      </div>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        {t('trainingStudio.yue2ar.runsBlurb',
          'Every composer LoRA this dataset has produced. Each rung is a plain safetensors file — copy a '
          + 'path into the YuE2 adapter field to hear it. Step {{pick}} is where the ear landed on the '
          + 'reference run, which is a starting point and not a verdict on yours.', { pick: pickStep })}
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
                      {run.running ? t('trainingStudio.yue2ar.runsRunning', 'Running now') : o.label}
                    </span>
                    <span>
                      {t('trainingStudio.yue2ar.runsSteps', 'step {{done}} of {{cap}}',
                        { done: run.lastStep, cap: run.configuredSteps || run.lastStep })}
                    </span>
                    <span>
                      {t('trainingStudio.yue2ar.runsCkpts', '{{n}} checkpoints',
                        { n: run.checkpoints.length })}
                    </span>
                    {run.trigger && (
                      <span>
                        {t('trainingStudio.yue2ar.runsTrigger', 'trigger "{{w}}"', { w: run.trigger })}
                      </span>
                    )}
                    {run.rank !== undefined && (
                      <span>
                        {t('trainingStudio.yue2ar.runsShape', 'rank {{rank}}, {{target}}',
                          { rank: run.rank, target: run.target || '—' })}
                      </span>
                    )}
                    {/* How the prompt was built, because generation has to
                        compose the same string for the trigger to mean
                        anything. */}
                    {run.styleTemplate && (
                      <span>
                        {t('trainingStudio.yue2ar.runsTemplate', 'style {{tpl}}', { tpl: run.styleTemplate })}
                      </span>
                    )}
                    {run.best && (
                      <span>
                        {t('trainingStudio.yue2ar.runsBest', 'best loss {{loss}} at {{step}}',
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
                      {t('trainingStudio.yue2ar.runsHalted',
                        'This run still holds its optimizer state, which means it stopped before its clean '
                        + 'finish — the engine deletes that file on export. Continuing a run is not wired '
                        + 'up here; start a fresh one with the same recipe.')}
                    </div>
                  )}
                </div>
                {run.checkpoints.length > 0 && (
                  <button className={BTN_SM}
                    onClick={() => setOpen(open === run.runName ? null : run.runName)}>
                    {open === run.runName
                      ? t('trainingStudio.yue2ar.runsHide', 'Hide ladder')
                      : t('trainingStudio.yue2ar.runsShow', 'Ladder')}
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
                      {c.step === pickStep && (
                        <span className="text-amber-500 shrink-0">
                          {t('trainingStudio.yue2ar.runsSuggested', 'try first')}
                        </span>
                      )}
                      {c.final && (
                        <span className="text-emerald-500 shrink-0">
                          {t('trainingStudio.yue2ar.runsFinal', 'final')}
                        </span>
                      )}
                      {c.loss !== undefined && (
                        <span className="shrink-0 tabular-nums">{c.loss.toFixed(4)}</span>
                      )}
                      <span className="shrink-0 text-zinc-500">{gb(c.bytes)}</span>
                      <span className="shrink-0 text-zinc-500">
                        {copied === c.path
                          ? t('trainingStudio.yue2ar.runsCopied', 'copied')
                          : t('trainingStudio.yue2ar.runsCopy', 'copy path')}
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

// ── Stage 4: the AR LoRA ────────────────────────────────────────────────────

interface TrainForm {
  lmType: string;
  trigger: string;
  allowNoTrigger: boolean;
  rank: number;
  alpha: number;
  target: Yue2ArTarget;
  styleTemplate: Yue2StyleTemplate;
  lr: number;
  lrScheduler: Yue2ArLrScheduler;
  schedSteps: number;
  warmup: number;
  steps: number;
  allowOvertrain: boolean;
  gradAccum: number;
  maxGradNorm: number;
  weightDecay: number;
  adamBeta1: number;
  adamBeta2: number;
  artistFrac: number;
  captionDropout: number;
  cursorWeight: number;
  abcDropout: number;
  seed: number;
  maxLen: number;
  attn: Yue2ArAttn;
  chunk: number;
  saveEvery: number;
  ckptFrom: number;
  evalEvery: number;
  logEvery: number;
  sidecars: boolean;
  style: string;
  lyrics: string;
  allowNoMinted: boolean;
}

/** Stage 5: the AR LoRA — the composer half. `status` arrives as a prop from
 *  the one useYue2ArStatus() call Yue2TrainStages makes, which also owns the
 *  loading gate and the status-fetch error banner (shared with stages 2 and
 *  3, which read the same payload). The stage-readiness recap this card used
 *  to open with is gone: Yue2TrainStages now renders stages 1-4 directly
 *  above this one, which is the same information laid out as five separate
 *  sections instead of summarised in a fifth. */
export const Yue2ArTrainStageCard: React.FC<{
  datasetId: string; trigger?: string; status: Yue2ArStatus | null; reload: () => void;
}> = ({ datasetId, trigger, status, reload }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startYue2ArTrain = useTrainingStore(s => s.startYue2ArTrain);
  const yue2ArLive = useTrainingStore(s => s.yue2ArLive);
  const yue2ArEvalSeries = useTrainingStore(s => s.yue2ArEvalSeries);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);

  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [models, setModels] = useState(false);
  /** The server's advisory warnings from the last start. Advisory because the
   *  run has already begun by the time they arrive: a cache with partial codes
   *  or partial cursor spans trains, it just trains on less. */
  const [warnings, setWarnings] = useState<string[]>([]);
  // DERIVED, not seeded: server defaults underneath, the user's edits on top, so
  // there is no window where the form holds a stale recipe.
  const [edits, setEdits] = useState<Partial<TrainForm>>({});

  const jobStatus = activeJob?.status;
  const jobRunning = jobStatus === 'queued' || jobStatus === 'running';
  const mine = activeJob?.kind === 'yue2-ar-train';

  const d = status?.defaults;
  const form: TrainForm | null = status && d ? {
    lmType: d.lmType,
    // The dataset's own trigger word first, then whatever the status route
    // reports, and only then blank.
    trigger: trigger || status.trigger || '',
    allowNoTrigger: false,
    rank: d.rank,
    alpha: d.alpha,
    target: d.target,
    styleTemplate: d.styleTemplate,
    lr: d.lr,
    lrScheduler: d.lrScheduler,
    schedSteps: d.schedSteps,
    warmup: d.warmup,
    steps: d.steps,
    allowOvertrain: false,
    gradAccum: d.gradAccum,
    maxGradNorm: d.maxGradNorm,
    weightDecay: d.weightDecay,
    adamBeta1: d.adamBeta1,
    adamBeta2: d.adamBeta2,
    artistFrac: d.artistFrac,
    captionDropout: d.captionDropout,
    cursorWeight: d.cursorWeight,
    abcDropout: d.abcDropout,
    seed: d.seed,
    maxLen: d.maxLen,
    attn: d.attn,
    chunk: d.chunk,
    saveEvery: d.saveEvery,
    ckptFrom: d.ckptFrom,
    evalEvery: d.evalEvery,
    logEvery: d.logEvery,
    sidecars: d.sidecars,
    style: '',
    lyrics: '',
    allowNoMinted: false,
    ...edits,
  } : null;

  const set = <K extends keyof TrainForm>(k: K, v: TrainForm[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const pp = status?.stages.preprocess;
  const al = status?.stages.align;
  const trainMissing = status?.stages.train.missing ?? [];
  const mintedMissing = !!status && !status.minted.present;
  const hasCursor = (al?.status?.sourcesWithCursor ?? 0) > 0;
  const sh = status?.stages.sheet;
  const hasSheet = (sh?.status?.sourcesWithAbc ?? 0) > 0;
  const overtrain = !!form && !!status && form.steps > status.overtrainSteps;
  // Each of these is a server refusal reproduced, not guessed at: the route
  // sends back 400 and a paragraph explaining what the override costs, and
  // finding that out by pressing the button is a worse way to read it.
  const cursorWithoutSpans = !!form && form.cursorWeight > 0 && !hasCursor;
  const needsTrigger = !!form && !form.trigger.trim() && !form.allowNoTrigger;
  const needsMinted = mintedMissing && !!form && !form.allowNoMinted;
  const needsOvertrain = overtrain && !!form && !form.allowOvertrain;

  const startTrain = async () => {
    if (!form) return;
    setBusy(true);
    setWarnings([]);
    try {
      const body: Yue2ArTrainRequest = {
        lmType: form.lmType,
        ...(form.trigger.trim() ? { trigger: form.trigger.trim() } : { allowNoTrigger: true }),
        styleTemplate: form.styleTemplate,
        ...(form.style.trim() ? { style: form.style.trim() } : {}),
        ...(form.lyrics.trim() ? { lyrics: form.lyrics.trim() } : {}),
        sidecars: form.sidecars,
        target: form.target,
        rank: form.rank, alpha: form.alpha,
        lr: form.lr, lrScheduler: form.lrScheduler,
        schedSteps: form.schedSteps, warmup: form.warmup,
        steps: form.steps,
        ...(overtrain ? { allowOvertrain: true } : {}),
        gradAccum: form.gradAccum, maxGradNorm: form.maxGradNorm,
        weightDecay: form.weightDecay,
        artistFrac: form.artistFrac,
        adamBeta1: form.adamBeta1, adamBeta2: form.adamBeta2,
        captionDropout: form.captionDropout,
        attn: form.attn, maxLen: form.maxLen, chunk: form.chunk,
        cursorWeight: form.cursorWeight, abcDropout: form.abcDropout, seed: form.seed,
        ckptFrom: form.ckptFrom, saveEvery: form.saveEvery,
        evalEvery: form.evalEvery, logEvery: form.logEvery,
        ...(mintedMissing ? { allowNoMinted: true } : {}),
      };
      setWarnings(await startYue2ArTrain(body));
    } finally {
      setBusy(false);
      reload();
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* ── The AR LoRA ── */}
      {status && form && (
        <div className={CARD}>
          <div className="flex items-center gap-2 mb-2">
            <Package size={15} className="text-amber-500" />
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
              {t('trainingStudio.yue2ar.trainTitle', 'AR LoRA training')}
            </h3>
          </div>
          <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
            {t('trainingStudio.yue2ar.trainBlurb',
              'Trains a LoRA on the AR half from whole songs — no crops, so nothing teaches the model that '
              + 'a song may begin mid-flow. Snapshots are plain safetensors files you can load straight into '
              + 'the YuE2 adapter field. The engine is paused for the run.')}
          </p>

          {/* The regulariser pack, above the button that needs it. Absent is a
              legitimate state with an override, and the override is a bad
              trade, so the trade is stated rather than hidden behind a 400. */}
          {status.minted.present ? (
            <div className="flex items-start gap-2 text-[11px] text-emerald-600 dark:text-emerald-400 mb-3">
              <Check size={13} className="mt-0.5 flex-shrink-0" />
              <span className="min-w-0 break-all">
                {t('trainingStudio.yue2ar.mintedHave', 'Regulariser pack: {{path}}',
                  { path: status.minted.manifestPath })}
              </span>
            </div>
          ) : (
            <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 p-2.5 mb-3">
              <div className="flex items-start gap-2 text-[11px] text-amber-700 dark:text-amber-300">
                <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                <span className="min-w-0">
                  {t('trainingStudio.yue2ar.mintedMissing',
                    'The minted regulariser pack is not installed. Half of every batch is meant to be true '
                    + 'YuE2 tokens, and that mix is the counterweight to a real defect: our semantic encoder '
                    + 'repeats adjacent codes far more often than YuE2\'s own, which is out of distribution '
                    + 'in the direction that produces LOOPING.')}
                  <span className="block mt-1 text-amber-700/80 dark:text-amber-300/80 break-all">
                    {t('trainingStudio.yue2ar.mintedFiles', '{{files}} — both in {{dir}}',
                      { files: status.minted.files.join(', '), dir: status.minted.dir })}
                  </span>
                </span>
              </div>
              <div className="flex items-center gap-3 flex-wrap mt-2.5">
                <button onClick={() => setModels(true)} className={BTN_SM}>
                  <span className="flex items-center gap-1.5">
                    <Download size={11} />
                    {t('trainingStudio.yue2ar.mintedInstall', 'Open the Model Manager')}
                  </span>
                </button>
                <span className="text-[10px] text-zinc-500">
                  {t('trainingStudio.yue2ar.mintedEntry',
                    'The entry is "YuE2 Minted Regulariser Pack" on the YuE2 tab.')}
                </span>
              </div>
              <CheckField className="mt-2.5"
                label={t('trainingStudio.yue2ar.allowNoMinted', 'Train artist-only anyway')}
                checked={form.allowNoMinted} onChange={v => set('allowNoMinted', v)}
                hint={t('trainingStudio.yue2ar.allowNoMintedHint',
                  'The run is refused without this, and the held-out minted loss — the one number that '
                  + 'separates learning the style from memorising the songs — is not produced at all.') as string} />
            </div>
          )}

          {trainMissing.length > 0 ? (
            <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
              <span>
                {t('trainingStudio.yue2ar.trainMissing', 'Missing model files')}: {trainMissing.join(', ')}
              </span>
            </div>
          ) : !pp?.done ? (
            <div className="flex items-start gap-2 text-xs text-zinc-500">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.yue2ar.trainNeedsLatents',
                'Encode the latents first, in the Latent cache card above — every stage here reads the '
                + 'manifest that writes.')}
            </div>
          ) : (
            <>
              <label className="flex flex-col gap-1 mb-3">
                <span className={LABEL}>{t('trainingStudio.yue2ar.trigger', 'Trigger word')}</span>
                <input className={INPUT} value={form.trigger}
                  onChange={e => set('trigger', e.target.value)} />
                <span className={HINT}>
                  {form.trigger.trim()
                    ? t('trainingStudio.yue2ar.triggerHint',
                        'Goes in front of every training caption and is stamped into the adapter, so half a '
                        + 'run under a different one is a different adapter.')
                    : t('trainingStudio.yue2ar.triggerMissing',
                        'Required. Without one the trained style has no handle at generation time, and the '
                        + 'engine only warns — a silent warning inside a long run is not a warning.')}
                </span>
              </label>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <NumField label={t('trainingStudio.yue2ar.steps', 'Steps')} value={form.steps}
                  onChange={v => set('steps', v)} step={50}
                  hint={t('trainingStudio.yue2ar.stepsHint',
                    '{{n}} is the settled recipe, with the ear picking a rung around {{pick}}. Nothing has '
                    + 'been timed here, so there is no clock estimate to give you.',
                    { n: d?.steps, pick: d?.ckptPickStep }) as string} />
                <NumField label={t('trainingStudio.yue2ar.rank', 'Rank')} value={form.rank}
                  onChange={v => set('rank', v)} step={16}
                  hint={t('trainingStudio.yue2ar.rankHint',
                    'Upstream\'s rank, and the gate\'s floor: below 64 the probes come back '
                    + 'inconclusive.') as string} />
                <NumField label={t('trainingStudio.yue2ar.saveEvery', 'Snapshot every')}
                  value={form.saveEvery} onChange={v => set('saveEvery', v)} step={10}
                  hint={t('trainingStudio.yue2ar.saveEveryHint',
                    'Rungs on the ladder. The rung is picked by ear afterwards, so a coarse ladder is a '
                    + 'smaller choice.') as string} />
                <label className="flex flex-col gap-1">
                  <span className={LABEL}>{t('trainingStudio.yue2ar.target', 'Trained sites')}</span>
                  <select className={INPUT} value={form.target}
                    onChange={e => set('target', e.target.value as Yue2ArTarget)}>
                    <option value="attn">attn</option>
                    <option value="attn_mlp">attn_mlp</option>
                  </select>
                  <span className={HINT}>
                    {t('trainingStudio.yue2ar.targetHint',
                      'attn_mlp is qkvo plus gate/up/down on all 28 AR layers — upstream\'s own group, and '
                      + 'what the recipe was proven with.')}
                  </span>
                </label>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                <NumField label={t('trainingStudio.yue2ar.captionDropout', 'Caption dropout')}
                  value={form.captionDropout} onChange={v => set('captionDropout', v)} step={0.05}
                  hint={t('trainingStudio.yue2ar.captionDropoutHint',
                    'Chance of dropping the caption and training on the trigger alone. 0 is not "more '
                    + 'likeness": it is the trigger ceasing to mean anything on its own.') as string} />
                <NumField label={t('trainingStudio.yue2ar.artistFrac', 'Artist fraction')}
                  value={form.artistFrac} onChange={v => set('artistFrac', v)} step={0.05}
                  disabled={mintedMissing}
                  hint={mintedMissing
                    ? t('trainingStudio.yue2ar.artistFracNoMinted',
                        'Without the regulariser pack every draw is an artist song whatever this says.') as string
                    : t('trainingStudio.yue2ar.artistFracHint',
                        'One draw per micro-step: artist song or minted song. 0.5 is the measured '
                        + 'mix.') as string} />
                <NumField label={t('trainingStudio.yue2ar.cursorWeight', 'Cursor weight')}
                  value={form.cursorWeight} onChange={v => set('cursorWeight', v)} step={0.01}
                  hint={hasCursor
                    ? t('trainingStudio.yue2ar.cursorWeightHint',
                        'The lyric-timing loss. 0 turns it off.') as string
                    : t('trainingStudio.yue2ar.cursorWeightNone',
                        'No source has cursor spans yet — run the align stage, or set this to 0.') as string} />
                <NumField label={t('trainingStudio.yue2ar.abcDropout', 'ABC dropout')}
                  value={form.abcDropout} onChange={v => set('abcDropout', v)} step={0.05}
                  hint={hasSheet
                    ? t('trainingStudio.yue2ar.abcDropoutHint',
                        'Chance a source with a lead sheet trains cot=off instead of cot=full this draw. '
                        + '0.5 is upstream\'s own split.') as string
                    : t('trainingStudio.yue2ar.abcDropoutNone',
                        'No source has a lead sheet yet — run the lead-sheet stage, or leave this at any '
                        + 'value; every source trains cot=off either way.') as string} />
                <label className="flex flex-col gap-1">
                  <span className={LABEL}>{t('trainingStudio.yue2ar.styleTemplate', 'Style template')}</span>
                  <select className={INPUT} value={form.styleTemplate}
                    onChange={e => set('styleTemplate', e.target.value as Yue2StyleTemplate)}>
                    <option value="upstream">upstream</option>
                    <option value="bare">bare</option>
                  </select>
                  <span className={HINT}>
                    {t('trainingStudio.yue2ar.styleTemplateHint',
                      'How the prompt is built from the trigger and the caption. Generation has to compose '
                      + 'the same string, which is why it is stamped into the adapter.')}
                  </span>
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
                <label className="flex flex-col gap-1">
                  <span className={LABEL}>{t('trainingStudio.yue2ar.base', 'Base model')}</span>
                  <select className={INPUT} value={form.lmType}
                    onChange={e => set('lmType', e.target.value)}>
                    {status.bases.map(b => (
                      <option key={b.id} value={b.id}>
                        {b.id}{b.proven ? ` — ${t('trainingStudio.yue2ar.baseProven', 'measured')}` : ''}
                        {` · ${gb(b.bytes)}`}
                      </option>
                    ))}
                    {status.bases.length === 0 && <option value={form.lmType}>{form.lmType}</option>}
                  </select>
                  <span className={HINT}>
                    {t('trainingStudio.yue2ar.baseHint',
                      'Only bf16 has been trained on here. Whether a quantized base trains usefully at all '
                      + 'is not established, so the others are offered without a ranking.')}
                  </span>
                </label>
                <NumField label={t('trainingStudio.yue2ar.lr', 'Learning rate')} value={form.lr}
                  onChange={v => set('lr', v)} step={1e-5}
                  hint={t('trainingStudio.yue2ar.lrHint',
                    'Cosine over a {{h}}-step horizon that a {{n}}-step run never reaches the end of — the '
                    + 'run is the head of that curve, not a complete decay.',
                    { h: d?.schedSteps, n: d?.steps }) as string} />
              </div>

              <button
                onClick={() => setAdvanced(v => !v)}
                className="flex items-center gap-1 mt-3 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
              >
                {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                {t('trainingStudio.yue2ar.advanced', 'Advanced')}
              </button>

              {advanced && (
                <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 flex flex-col gap-3">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <NumField label={t('trainingStudio.yue2ar.alpha', 'Alpha')} value={form.alpha}
                      onChange={v => set('alpha', v)} step={16} />
                    <label className="flex flex-col gap-1">
                      <span className={LABEL}>{t('trainingStudio.yue2ar.lrScheduler', 'Schedule')}</span>
                      <select className={INPUT} value={form.lrScheduler}
                        onChange={e => set('lrScheduler', e.target.value as Yue2ArLrScheduler)}>
                        <option value="cosine">cosine</option>
                        <option value="constant">constant</option>
                      </select>
                    </label>
                    <NumField label={t('trainingStudio.yue2ar.schedSteps', 'Cosine horizon')}
                      value={form.schedSteps} onChange={v => set('schedSteps', v)} step={100}
                      hint={t('trainingStudio.yue2ar.schedStepsHint',
                        'Not the run length. Moving the steps without moving this changes where on the '
                        + 'curve the run stops.') as string} />
                    <NumField label={t('trainingStudio.yue2ar.warmup', 'Warmup steps')}
                      value={form.warmup} onChange={v => set('warmup', v)} step={10} />
                    <NumField label={t('trainingStudio.yue2ar.gradAccum', 'Grad accum')}
                      value={form.gradAccum} onChange={v => set('gradAccum', v)} />
                    <NumField label={t('trainingStudio.yue2ar.maxGradNorm', 'Clip grad norm')}
                      value={form.maxGradNorm} onChange={v => set('maxGradNorm', v)} step={0.1} />
                    <NumField label={t('trainingStudio.yue2ar.weightDecay', 'Weight decay')}
                      value={form.weightDecay} onChange={v => set('weightDecay', v)} step={0.01} />
                    <NumField label={t('trainingStudio.yue2ar.seed', 'Seed')} value={form.seed}
                      onChange={v => set('seed', v)} />
                    <NumField label={t('trainingStudio.yue2ar.adamBeta1', 'Adam β1')}
                      value={form.adamBeta1} onChange={v => set('adamBeta1', v)} step={0.01} />
                    <NumField label={t('trainingStudio.yue2ar.adamBeta2', 'Adam β2')}
                      value={form.adamBeta2} onChange={v => set('adamBeta2', v)} step={0.005}
                      hint={t('trainingStudio.yue2ar.adamBeta2Hint',
                        'Upstream\'s, not the LM optimizer\'s 0.999. It is how long the second moment '
                        + 'remembers, which sets how hard a small corpus is memorised.') as string} />
                    <NumField label={t('trainingStudio.yue2ar.maxLen', 'Max song length')}
                      value={form.maxLen} onChange={v => set('maxLen', v)} step={1024}
                      hint={t('trainingStudio.yue2ar.maxLenHint',
                        'Tokens. A longer song truncates without its end marker, so it never teaches a fake '
                        + 'ending. This also sizes the per-layer buffers — it is the VRAM lever.') as string} />
                    <NumField label={t('trainingStudio.yue2ar.chunk', 'CE chunk')} value={form.chunk}
                      onChange={v => set('chunk', v)} step={64} />
                    <label className="flex flex-col gap-1">
                      <span className={LABEL}>{t('trainingStudio.yue2ar.attn', 'Attention')}</span>
                      <select className={INPUT} value={form.attn}
                        onChange={e => set('attn', e.target.value as Yue2ArAttn)}>
                        <option value="exact">exact</option>
                        <option value="flash">flash</option>
                        <option value="flash-f32">flash-f32</option>
                      </select>
                      <span className={HINT}>
                        {t('trainingStudio.yue2ar.attnHint',
                          'Flash drops the retained softmax — several GB at full song length — and errors '
                          + 'out on a backend that cannot do it. The recipe was proven on exact.')}
                      </span>
                    </label>
                    <NumField label={t('trainingStudio.yue2ar.ckptFrom', 'First snapshot at')}
                      value={form.ckptFrom} onChange={v => set('ckptFrom', v)} step={10} />
                    <NumField label={t('trainingStudio.yue2ar.evalEvery', 'Eval every')}
                      value={form.evalEvery} onChange={v => set('evalEvery', v)} step={10}
                      disabled={mintedMissing}
                      hint={t('trainingStudio.yue2ar.evalEveryHint',
                        'Held-out loss on the regulariser pack. 0 turns it off.') as string} />
                    <NumField label={t('trainingStudio.yue2ar.logEvery', 'Log every')}
                      value={form.logEvery} onChange={v => set('logEvery', v)} />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <TextField label={t('trainingStudio.yue2ar.style', 'Fallback style')}
                      value={form.style} onChange={v => set('style', v)}
                      hint={t('trainingStudio.yue2ar.styleHint',
                        'Used only for a source with no sidecar caption.') as string} />
                    <TextField label={t('trainingStudio.yue2ar.lyrics', 'Fallback lyrics')}
                      value={form.lyrics} onChange={v => set('lyrics', v)}
                      hint={t('trainingStudio.yue2ar.lyricsHint',
                        'Same, for the lyrics half of the prefix.') as string} />
                  </div>
                  <CheckField
                    label={t('trainingStudio.yue2ar.sidecars', 'Read the .txt beside each track')}
                    checked={form.sidecars} onChange={v => set('sidecars', v)}
                    hint={t('trainingStudio.yue2ar.sidecarsHint',
                      'The manifest carries no lyrics, so with this off the prefix falls back to the two '
                      + 'fields above — and a style-less, lyric-less prefix trains happily and is '
                      + 'wrong.') as string} />
                </div>
              )}

              {cursorWithoutSpans && (
                <div className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400 mt-3">
                  <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                  <span>
                    {t('trainingStudio.yue2ar.cursorBlocked',
                      'The cursor loss is on but no source has cursor spans, and the engine refuses that '
                      + 'rather than print "cursor nan" for the whole run. Run the align stage above, or set '
                      + 'the cursor weight to 0.')}
                  </span>
                </div>
              )}

              {overtrain && (
                <CheckField className="mt-3"
                  label={t('trainingStudio.yue2ar.allowOvertrain', 'Train past {{n}} steps anyway',
                    { n: status.overtrainSteps })}
                  checked={form.allowOvertrain} onChange={v => set('allowOvertrain', v)}
                  hint={t('trainingStudio.yue2ar.allowOvertrainHint',
                    'Past there upstream says the model stops learning the style and starts memorising the '
                    + 'songs. The settled recipe is {{n}} steps.', { n: d?.steps }) as string} />
              )}

              {/* On the empty trigger, not on `needsTrigger`: that goes false
                  the moment the box is ticked, and a checkbox that disappears
                  when checked cannot be unticked. */}
              {!form.trigger.trim() && (
                <CheckField className="mt-3"
                  label={t('trainingStudio.yue2ar.allowNoTrigger', 'Train without a trigger word')}
                  checked={form.allowNoTrigger} onChange={v => set('allowNoTrigger', v)}
                  hint={t('trainingStudio.yue2ar.allowNoTriggerHint',
                    'The adapter then has no word that addresses it, and it is the caption dropout\'s '
                    + 'counterpart — there is nothing left for the empty-caption draws to teach.') as string} />
              )}

              <button
                onClick={() => void startTrain()}
                disabled={busy || jobRunning || cursorWithoutSpans || needsTrigger || needsMinted
                  || needsOvertrain}
                className={`mt-4 ${BTN_GO}`}
              >
                <Play size={14} />
                {t('trainingStudio.yue2ar.start', 'Start AR training')}
              </button>
              <p className="text-[11px] text-zinc-500 flex items-center gap-1.5 mt-3">
                <PauseCircle size={12} className="flex-shrink-0" />
                {t('trainingStudio.yue2ar.enginePaused',
                  'The engine is paused while this runs and restarted afterwards.')}
              </p>
            </>
          )}

          {warnings.length > 0 && (
            <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/10 flex flex-col gap-1.5">
              {warnings.map(w => (
                <div key={w} className="flex items-start gap-2 text-[11px] text-amber-600 dark:text-amber-400">
                  <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                  <span className="min-w-0">{w}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Previous runs and their ladders ── */}
      <Yue2ArRunsList datasetId={datasetId} pickStep={status?.defaults.ckptPickStep ?? 0}
        reloadKey={`${activeJob?.id ?? ''}:${jobStatus ?? ''}`} />

      {/* ── Live run ── */}
      {mine && activeJob && (
        <div className={CARD}>
          <JobProgress />
          {yue2ArLive && (
            <div className="mt-3 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
              {[
                { k: 'loss', v: yue2ArLive.loss ? yue2ArLive.loss.toFixed(4) : '—' },
                { k: 'runMean', v: yue2ArLive.runMean ? yue2ArLive.runMean.toFixed(4) : '—' },
                {
                  k: 'mintedVal',
                  v: yue2ArLive.mintedVal !== null ? yue2ArLive.mintedVal.toFixed(4) : '—',
                },
                { k: 'gradNorm', v: yue2ArLive.gradNorm ? yue2ArLive.gradNorm.toFixed(3) : '—' },
                {
                  k: 'seqLen',
                  v: yue2ArLive.seqLen ? yue2ArLive.seqLen.toLocaleString() : '—',
                },
                {
                  // Seconds, not milliseconds: AR steps run ~5 s, and "5.0 s"
                  // is the number to compare against the recipe's expectation.
                  k: 'stepTime',
                  v: yue2ArLive.stepMs ? `${(yue2ArLive.stepMs / 1000).toFixed(2)} s` : '—',
                },
                {
                  k: 'vram',
                  v: yue2ArLive.totalMb
                    ? `${Math.round(yue2ArLive.usedMb / 1024)}/${Math.round(yue2ArLive.totalMb / 1024)} GB`
                    : '—',
                  warn: yue2ArLive.totalMb > 0 && yue2ArLive.usedMb > yue2ArLive.totalMb - 512,
                },
              ].map(tile => (
                <div key={tile.k}
                  className="rounded-lg border border-zinc-200 dark:border-white/5 px-2.5 py-1.5">
                  <div className="text-[10px] uppercase tracking-wider text-zinc-500">
                    {t(`trainingStudio.yue2ar.stat.${tile.k}`, tile.k)}
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
              {/* Everything is mapped into a STEP x-domain: an AR run counts
                  songs, so the relay emits no epoch and the fractional-epoch
                  positions the store computes are all 0. target 0 hides the
                  target line — this trainer has no stop-on-loss mode. */}
              <TrainingChart
                epochs={[]}
                steps={trainStepSeries.map(s => ({ ...s, ep: s.step }))}
                milestones={trainMilestones
                  .map(m => ({ ...m, epoch: stepFromSnapshot(m.path) }))
                  .filter(m => m.epoch > 0)}
                evals={yue2ArEvalSeries.map(e => ({ ep: e.step, loss: e.loss }))}
                target={0}
              />
              <p className="text-[10px] text-zinc-500 mt-1.5 leading-snug">
                {t('trainingStudio.yue2ar.chartNote',
                  'Faint line: the training loss, which falls whether the model is learning the style or '
                  + 'memorising the songs. The green points are the held-out loss on the regulariser pack, '
                  + 'which is the one that does not.')}
              </p>
            </div>
          )}
        </div>
      )}

      {models && <ModelManagerModal onClose={() => { setModels(false); reload(); }} />}
    </div>
  );
};
