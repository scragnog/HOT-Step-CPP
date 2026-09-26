// AuditionCard.tsx — the pure-LM A/B codes audition
//
// Same caption, same lyrics, same lm_seed, two `/lm` runs — one base, one with
// the planner adapter — decoded straight through the FSQ detokenizer and the
// VAE. No DiT, no sound adapter, no sampler. This is the acceptance instrument
// for "did this artist-style planner adapter actually learn the artist".
//
// Everything the LM handed back is rendered next to each player, not just the
// audio: the adapter changes caption/bpm/key/sig/lyrics too (C13), and judging
// on audio alone throws away half the evidence.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, ChevronDown, ChevronRight, Dices, Headphones, History, Loader2, PauseCircle, PlugZap,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { AuditionOptions, AuditionPreview, AuditionSideSpec, LsGenerationsResponse } from '../../services/trainingApi';
import { getLsGenerations } from '../../services/trainingApi';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useGlobalParamsStore } from '../../stores/globalParamsStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { AuditionPlayer } from './AuditionPlayer';
import { JobProgress } from './JobProgress';
import { LmAdapterPicker, type LmAdapterOption } from './LmAdapterPicker';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const FIELD = 'w-full rounded-lg px-3 py-2 text-xs bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500 disabled:opacity-50';

type PromptSource = 'sample' | 'text' | 'lm' | 'lyricstudio';

/** "B Major" → "B major" — the same normalization Lyric Studio's own
 *  send-to-Create applies (useAudioGeneration.ts). */
function normalizeKey(k: string): string {
  const parts = k.trim().split(/\s+/);
  return parts.length === 2 ? `${parts[0]} ${parts[1].toLowerCase()}` : k.trim();
}

/** A milestone the badge row asked to audition. `nonce` makes repeat clicks on
 *  the same milestone distinguishable — the effect keys on it, not on the path. */
export interface MilestoneAuditionRequest {
  path: string;
  label: string;
  nonce: number;
}

interface AuditionCardProps {
  milestoneRequest?: MilestoneAuditionRequest | null;
}

/** Base always renders left, adapter right, regardless of completion order. */
function orderSides(preview: AuditionPreview) {
  return [...preview.sides].sort((a, b) => (a.slot === 'base' ? -1 : 1) - (b.slot === 'base' ? -1 : 1));
}

/** True when every side of the preview reported the same codes hash — the
 *  adapter did nothing (§6.4). Only meaningful with 2+ successful sides. */
function hasIdenticalCodes(preview: AuditionPreview): boolean {
  const hashes = preview.sides.filter(s => s.ok && s.codesSha1).map(s => s.codesSha1);
  return hashes.length >= 2 && hashes.every(h => h === hashes[0]);
}

export const AuditionCard: React.FC<AuditionCardProps> = ({ milestoneRequest }) => {
  const { t } = useTranslation();
  const capabilities = useTrainingStore(s => s.capabilities);
  const detail = useTrainingStore(s => s.detail);
  const selectedDatasetId = useTrainingStore(s => s.selectedDatasetId);
  const activeJob = useTrainingStore(s => s.activeJob);
  const trainLmStatus = useTrainingStore(s => s.trainLmStatus);
  const trainDitStatus = useTrainingStore(s => s.trainDitStatus);
  const auditions = useTrainingStore(s => s.auditions);
  const auditionRunning = useTrainingStore(s => s.auditionRunning);
  const auditionError = useTrainingStore(s => s.auditionError);
  const loadAuditions = useTrainingStore(s => s.loadAuditions);
  const startAudition = useTrainingStore(s => s.startAudition);

  const [expandedByUser, setExpandedByUser] = useState(false);
  const [source, setSource] = useState<PromptSource>('text');
  const [caption, setCaption] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [sampleId, setSampleId] = useState('');
  const [sampleLocked, setSampleLocked] = useState(true);
  const [seed, setSeed] = useState('');
  const [durationSec, setDurationSec] = useState(180);
  const [twoSided, setTwoSided] = useState(true);
  const [baseLabel, setBaseLabel] = useState('');
  const [baseAdapter, setBaseAdapter] = useState('');
  const [adapterLabel, setAdapterLabel] = useState('');
  const [adapterPath, setAdapterPath] = useState('');
  const [adapterScale, setAdapterScale] = useState(1);
  const [baseScale, setBaseScale] = useState(1);
  // Persisted (Rob, 2026-07-29): the "hear it as music" choice tends to be a
  // standing preference, not a per-run one.
  const [renderDit, setRenderDit] = usePersistedState('hs-auditionRenderDit', false);
  // Sub-option of the render: a second render per side through the dataset's
  // latest trained DiT adapter — the 2×2 {base LM, LM adapter} × {bare DiT,
  // DiT adapter}. Only offered when TrainPanel's status says one exists; the
  // server re-resolves by name (never a client path) and pins every render to
  // the adapter's training base.
  const [renderDitAdapter, setRenderDitAdapter] = usePersistedState('hs-auditionRenderDitAdapter', false);
  const ditAdapterAvailable = !!trainDitStatus?.adapterExists;
  // Render step count (Rob, 2026-08-13): server default is 8; more steps =
  // slower but closer to a real generation's render quality. Clamped 2..60
  // server-side either way.
  const [renderSteps, setRenderSteps] = usePersistedState('hs-auditionRenderSteps', 8);
  // Lyric Studio prompt source: the linked album's generated lyrics.
  const [lsData, setLsData] = useState<LsGenerationsResponse | null>(null);
  const [lsLoading, setLsLoading] = useState(false);
  const [lsGenId, setLsGenId] = useState(0);
  // The render runs on whatever DiT the user has selected in the models tab
  // (Rob, 2026-07-29) — the same hs-ditModel every generation uses. '' falls
  // back server-side (newest xl-turbo, else the detok DiT).
  const selectedDitModel: string = useGlobalParamsStore(s => s.ditModel) || '';
  const [temperature, setTemperature] = useState(0.85);
  const [topP, setTopP] = useState(0.9);
  const [cfgScale, setCfgScale] = useState(2);
  // 1.1 default (Rob, 2026-08-12): adapters sharpen the code distribution; a
  // mild presence penalty (engine window 64 codes ≈ 13 s) is the baseline.
  const [repPenalty, setRepPenalty] = useState(1.1);
  const [openHistory, setOpenHistory] = useState<Record<string, boolean>>({});
  const [milestoneNotice, setMilestoneNotice] = useState('');
  const [milestoneOptions, setMilestoneOptions] = useState<LmAdapterOption[]>([]);
  const cardRef = useRef<HTMLDivElement>(null);
  const lastMilestoneNonce = useRef(0);

  const adapterExists = !!trainLmStatus?.adapterExists;
  const expanded = adapterExists || expandedByUser;

  // Preview history is per-dataset and outlives the page — read it whenever the
  // card first becomes visible for a dataset.
  useEffect(() => {
    if (!selectedDatasetId || !expanded) return;
    void loadAuditions();
  }, [selectedDatasetId, expanded, loadAuditions]);

  // Seed side B from the adapter this panel just trained, and FOLLOW it when a
  // newer run lands. The old seed-once (`prev ? prev : trainedDir`) captured
  // whatever run dir the status held at first render — after a failed run that
  // still wrote weights, the audition stayed pinned to the FAILED run even once
  // a later run completed (observed live, albumA4 2026-07-29). Only an
  // EXPLICIT user pick (picker or milestone badge) stops the auto-follow, and a
  // dataset switch re-arms it.
  const trainedDir = trainLmStatus?.adapterDir ?? '';
  const [adapterPickedByUser, setAdapterPickedByUser] = useState(false);
  // Dataset switch re-arms the auto-follow — adjust-during-render, the React-
  // sanctioned form of "reset some state when a prop changes".
  const [seenDatasetId, setSeenDatasetId] = useState(selectedDatasetId);
  const [seenTrainedDir, setSeenTrainedDir] = useState('');
  if (selectedDatasetId !== seenDatasetId) {
    setSeenDatasetId(selectedDatasetId);
    setSeenTrainedDir('');
    setAdapterPickedByUser(false);
    setAdapterPath('');
    setAdapterLabel('');
    setLsData(null);
    setLsGenId(0);
  } else if (adapterExists && trainedDir && trainedDir !== seenTrainedDir) {
    setSeenTrainedDir(trainedDir);
    if (!adapterPickedByUser) {
      setAdapterPath(trainedDir);
      // artist/run-stamp — the stamp alone is unreadable across 190+ adapters.
      setAdapterLabel(trainedDir.replace(/[\\/]+$/, '').split(/[\\/]/).slice(-2).join('/'));
    }
  }

  const jobActive = activeJob?.status === 'queued' || activeJob?.status === 'running';
  const auditionJobActive = activeJob?.kind === 'audition' && jobActive;

  const engineUp = capabilities?.engine.up !== false;
  const engineSuspended = !!capabilities?.preprocess?.engineSuspended;

  // Pin the base LM to the size the adapter was TRAINED against.
  //
  // Leaving lmModel unset means the engine's resolve_name falls back to
  // registry[0], which is acestep-5Hz-lm-0.6B-BF16.gguf — 28 layers. A 4B
  // planner adapter has 36, so the store refuses it
  // ("[Store] LM adapter has 36 layers but model has 28 — mismatch, refusing")
  // and the whole audition fails. That is the headline use case of this feature
  // failing 100% of the time, and it was observed live in ace_engine.log before
  // this line existed. The plan (§5.3) accepted the registry[0] default on the
  // grounds that a wrong base "would fail loudly in ace_lm_load anyway" — true,
  // but loud-and-always-broken is not a usable default when the right answer is
  // already sitting in capabilities.
  const auditionLmModel =
    (trainLmStatus?.lmSize && capabilities?.trainLm?.defaultLmBySize?.[trainLmStatus.lmSize]) || '';

  // Declared HERE, above the milestone hand-off effect, so that effect can
  // consult it before dispatching. Every blocker in one place, one named reason
  // each — §6.2.5 asks for a named reason per disabled state, never a bare
  // failure.
  const disabledReason =
    !engineUp ? t('trainingStudio.audition.engineDown')
      : engineSuspended ? t('trainingStudio.audition.engineSuspended')
        : jobActive ? t('trainingStudio.audition.jobRunning')
          : !caption.trim() ? t('trainingStudio.audition.needsCaption')
            : '';

  // Samples that can supply a prompt. `hasAudioCodes` is the LEGACY /understand
  // payload only (datasetScan.ts) — it says nothing about the lm_codes.jsonl row
  // this audition actually reads, so filtering on it would hide every sample of
  // a normally-preprocessed dataset. Filter on what is knowable instead.
  const promptSamples = useMemo(
    () => (detail?.samples ?? []).filter(s => !s.excluded && !s.fileMissing && !!s.caption.trim()),
    [detail],
  );

  const pickSample = (id: string) => {
    setSampleId(id);
    setSampleLocked(true);
    const s = promptSamples.find(x => x.sampleId === id);
    if (!s) return;
    setCaption(s.caption);
    setLyrics(s.lyrics);
  };

  // Lyric Studio source: fetch the linked album's generations once per dataset,
  // lazily on first selection of the source. The first fetch for a never-linked
  // dataset runs the export preview's artist/album detection server-side and
  // persists the resolved link, so later fetches are one indexed query.
  useEffect(() => {
    if (source !== 'lyricstudio' || !selectedDatasetId || lsData || lsLoading) return;
    setLsLoading(true);
    getLsGenerations(selectedDatasetId)
      .then(d => setLsData(d))
      .catch(() => setLsData({ lyricsSetId: 0, artist: '', album: '', generations: [] }))
      .finally(() => setLsLoading(false));
  }, [source, selectedDatasetId, lsData, lsLoading]);

  const lsGen = lsData?.generations.find(g => g.id === lsGenId);

  const pickLsGeneration = (id: number) => {
    setLsGenId(id);
    const g = lsData?.generations.find(x => x.id === id);
    if (!g) return;
    setCaption(g.caption);
    setLyrics(g.lyrics);
    if (g.duration > 0) setDurationSec(Math.min(300, Math.max(10, g.duration)));
  };

  const buildOptions = (kind: 'ab' | 'milestone'): AuditionOptions => {
    const sides: AuditionSideSpec[] = [{
      slot: 'base',
      label: baseLabel.trim() || t('trainingStudio.audition.sideBase'),
      lmAdapter: baseAdapter,
      lmAdapterScale: baseAdapter ? baseScale : 1,
    }];
    if (twoSided) {
      sides.push({
        slot: 'adapter',
        label: adapterLabel.trim() || t('trainingStudio.audition.sideAdapter'),
        lmAdapter: adapterPath,
        lmAdapterScale: adapterScale,
      });
    }
    const seedNum = seed.trim() === '' ? -1 : Number(seed);
    // The caption shown for a dataset sample is the RAW sidecar caption. The
    // trainer did not condition on that string — lm-extract composes
    // lm_apply_tag(caption, custom_tag, position) and stores the result in
    // lm_codes.jsonl, so the trainer's caption carries the dataset's trigger
    // word ("abba, This track is …") and this one does not. Sending it would
    // audition the adapter's prompt WITHOUT its trigger: both sides emit
    // near-identical plans and the card then shows the red "identical codes —
    // the adapter had no effect" banner, blaming the adapter for a prompt bug.
    // Sending empty strings + sampleId hands the choice to the server, which
    // reads the literal trainer strings out of the row (C12a). Only while the
    // fields are still locked — once the user unlocks and edits, what they typed
    // wins.
    const useTrainerStrings = source === 'sample' && !!sampleId && sampleLocked;
    return {
      sides,
      caption: useTrainerStrings ? '' : caption.trim(),
      lyrics: useTrainerStrings ? '' : (source === 'lm' ? '' : lyrics),
      ...(Number.isFinite(seedNum) && seedNum >= 0 ? { seed: seedNum } : {}),
      durationSec,
      ...(trainLmStatus?.variantKey ? { variantKey: trainLmStatus.variantKey } : {}),
      ...(auditionLmModel ? { lmModel: auditionLmModel } : {}),
      ...(source === 'sample' && sampleId ? { sampleId } : {}),
      // Lyric Studio source: pin the plan to the generation's bpm/key so the
      // audition anchors where a real generation of this song would (the same
      // force_fields path as the sample-row pins).
      ...(source === 'lyricstudio' && lsGen
        ? {
          ...(lsGen.bpm > 0 ? { bpm: lsGen.bpm } : {}),
          ...(lsGen.key.trim() ? { keyscale: normalizeKey(lsGen.key) } : {}),
        }
        : {}),
      temperature,
      topP,
      cfgScale,
      repPenalty,
      format: 'wav16',
      coResident: true,
      kind,
      // Opt-in DiT render (Rob, 2026-07-29): the user's selected DiT at the
      // server-default 8 steps. With the DiT-adapter sub-toggle (2026-08-12)
      // the server adds a second render per side through the dataset's latest
      // trained DiT adapter and pins all renders to its training base.
      ...(renderDit
        ? {
          renderDit: true,
          renderSteps,
          ...(selectedDitModel ? { renderDitModel: selectedDitModel } : {}),
          ...(renderDitAdapter && ditAdapterAvailable
            ? {
              renderDitAdapter: true,
              ...(trainDitStatus?.adapterName ? { renderDitAdapterName: trainDitStatus.adapterName } : {}),
            }
            : {}),
        }
        : {}),
    };
  };

  const run = (kind: 'ab' | 'milestone' = 'ab') => {
    if (!caption.trim()) return;
    void startAudition(buildOptions(kind));
  };

  // ── Milestone hand-off ─────────────────────────────────────────────────
  // The badge row cannot start a job on its own: an audition needs a prompt,
  // and the prompt lives here. A click loads the milestone into side B and
  // starts immediately IF a prompt is already set; otherwise it expands the
  // card and says what is missing rather than failing a 400 silently.
  useEffect(() => {
    if (!milestoneRequest || milestoneRequest.nonce === lastMilestoneNonce.current) return;
    lastMilestoneNonce.current = milestoneRequest.nonce;
    setAdapterPickedByUser(true);
    setExpandedByUser(true);
    setTwoSided(true);
    setAdapterPath(milestoneRequest.path);
    setAdapterLabel(milestoneRequest.label);
    setMilestoneOptions(prev => (
      prev.some(o => o.value === milestoneRequest.path)
        ? prev
        : [...prev, { label: milestoneRequest.label, value: milestoneRequest.path }]
    ));
    cardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (!caption.trim()) {
      setMilestoneNotice(t('trainingStudio.audition.milestoneNeedsPrompt'));
      return;
    }
    // The milestone path must clear the SAME bar as the Run button. Dispatching
    // while the engine is down, while a preprocess/train run owns the GPU, or
    // while another job holds this dataset just trades the card's named reason
    // for a raw 503/409 in the red error box.
    if (disabledReason) {
      setMilestoneNotice(disabledReason);
      return;
    }
    setMilestoneNotice('');
    void startAudition({
      ...buildOptions('milestone'),
      sides: [
        {
          slot: 'base',
          label: baseLabel.trim() || t('trainingStudio.audition.sideBase'),
          lmAdapter: baseAdapter,
          lmAdapterScale: baseAdapter ? baseScale : 1,
        },
        {
          slot: 'adapter',
          label: milestoneRequest.label,
          lmAdapter: milestoneRequest.path,
          lmAdapterScale: adapterScale,
        },
      ],
    });
    // buildOptions closes over every form field; re-running this effect on any
    // of them would re-fire the job on each keystroke. The nonce is the trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [milestoneRequest]);

  // Only A/B and milestone previews belong to this card. decodeStoredCodes
  // writes a kind:'sample' preview under the SAME datasetId (the SampleDrawer
  // audition), and listPreviews sorts across all kinds — so without this filter
  // a single-sided "Stored codes" preview with a seed-0 receipt can land under
  // the "Latest" heading and read as the A/B that was just run.
  const cardAuditions = useMemo(
    () => auditions.filter(p => p.kind !== 'sample'),
    [auditions],
  );
  const latest = cardAuditions[0];
  const history = cardAuditions.slice(1, 6);

  // ── Collapsed ──────────────────────────────────────────────────────────
  if (!expanded) {
    return (
      <button
        onClick={() => setExpandedByUser(true)}
        className={`${CARD} flex items-center gap-2 text-left hover:border-amber-500/40 transition-colors`}
      >
        <Headphones size={15} className="text-amber-500 flex-shrink-0" />
        <span className="flex-1 min-w-0">
          <span className="block text-sm font-bold text-zinc-900 dark:text-white">
            {t('trainingStudio.audition.title')}
          </span>
          <span className="block text-[11px] text-zinc-500">
            {t('trainingStudio.audition.collapsedHint')}
          </span>
        </span>
        <ChevronRight size={15} className="text-zinc-500 flex-shrink-0" />
      </button>
    );
  }

  return (
    <div ref={cardRef} className={`${CARD} flex flex-col gap-4`}>
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <div className="flex items-start gap-2">
        <Headphones size={15} className="text-amber-500 mt-0.5 flex-shrink-0" />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.audition.title')}</h3>
          <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">{t('trainingStudio.audition.subtitle')}</p>
          <p className="text-[11px] text-zinc-500 dark:text-zinc-400 italic mt-1">{t('trainingStudio.audition.hint')}</p>
        </div>
      </div>

      {/* ── Prompt source ────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-2">
        <ParamLabel
          label={t('trainingStudio.audition.promptSource')}
          className="text-[10px] uppercase tracking-wide text-zinc-500"
          info={t('trainingStudio.audition.promptSourceInfo')}
        />
        <div className="flex items-center gap-1 p-0.5 rounded-lg bg-zinc-100 dark:bg-black/20 border border-zinc-200 dark:border-white/10 w-fit">
          {(['sample', 'text', 'lm', 'lyricstudio'] as PromptSource[]).map(s => (
            <button
              key={s}
              onClick={() => setSource(s)}
              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-colors ${
                source === s
                  ? 'bg-amber-500/20 text-amber-600 dark:text-amber-400'
                  : 'text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
            >
              {t(
                s === 'sample' ? 'trainingStudio.audition.sourceSample'
                  : s === 'text' ? 'trainingStudio.audition.sourceText'
                    : s === 'lm' ? 'trainingStudio.audition.sourceLm'
                      : 'trainingStudio.audition.sourceLyricStudio',
              )}
            </button>
          ))}
        </div>

        {source === 'sample' && (
          <div className="flex flex-col gap-2">
            <StyledSelect
              accent="amber"
              value={sampleId}
              onChange={(v) => pickSample(v)}
              placeholder={t('trainingStudio.audition.pickSample')}
              options={[
                { value: '', label: t('trainingStudio.audition.pickSample') },
                ...promptSamples.map(s => ({ value: s.sampleId, label: s.filename })),
              ]}
              searchPlaceholder="Filter songs…"
            />
            {sampleId && (
              <Toggle
                size="sm"
                accent="amber"
                checked={!sampleLocked}
                onChange={(v) => setSampleLocked(!v)}
                label={t('trainingStudio.audition.editSamplePrompt')}
                info={sampleLocked ? t('trainingStudio.audition.trainerCaptionNote') : undefined}
              />
            )}
          </div>
        )}

        {source === 'lm' && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-600 dark:text-amber-400">
            <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.audition.sourceLmWarning')}
          </div>
        )}

        {source === 'lyricstudio' && (
          <div className="flex flex-col gap-2">
            {lsLoading && (
              <div className="flex items-center gap-2 text-[11px] text-zinc-500">
                <Loader2 size={12} className="animate-spin" /> {t('trainingStudio.audition.lsLoading')}
              </div>
            )}
            {!lsLoading && lsData && lsData.lyricsSetId === 0 && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-600 dark:text-amber-400">
                <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                {t('trainingStudio.audition.lsNoLink')}
              </div>
            )}
            {!lsLoading && lsData && lsData.lyricsSetId > 0 && lsData.generations.length === 0 && (
              <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-600 dark:text-amber-400">
                <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
                {t('trainingStudio.audition.lsNoGens', { artist: lsData.artist, album: lsData.album })}
              </div>
            )}
            {!lsLoading && lsData && lsData.generations.length > 0 && (
              <>
                <StyledSelect
                  accent="amber"
                  value={lsGenId ? String(lsGenId) : ''}
                  onChange={(v) => pickLsGeneration(Number(v) || 0)}
                  placeholder={t('trainingStudio.audition.lsPick')}
                  options={[
                    { value: '', label: t('trainingStudio.audition.lsPick') },
                    ...lsData.generations.map(g => ({
                      value: String(g.id),
                      label: `${g.title || '(untitled)'} · ${g.createdAt.slice(0, 10)}`,
                    })),
                  ]}
                  searchPlaceholder="Filter songs…"
                />
                <p className="text-[11px] text-zinc-500 dark:text-zinc-500">
                  {t('trainingStudio.audition.lsLinkedNote', { artist: lsData.artist, album: lsData.album })}
                  {lsGen && (lsGen.bpm > 0 || lsGen.key.trim()) && (
                    <span className="ml-1 tabular-nums">
                      {lsGen.bpm > 0 ? `· ${lsGen.bpm} BPM ` : ''}{lsGen.key.trim() ? `· ${normalizeKey(lsGen.key)}` : ''}
                    </span>
                  )}
                </p>
              </>
            )}
          </div>
        )}

        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.audition.caption')}
            className="text-[10px] uppercase tracking-wide text-zinc-500"
            info={t('trainingStudio.audition.captionInfo')}
          />
          <textarea
            rows={2}
            value={caption}
            maxLength={4000}
            disabled={source === 'sample' && sampleLocked}
            placeholder={t('trainingStudio.audition.captionPlaceholder')}
            onChange={(e) => setCaption(e.target.value)}
            className={FIELD}
          />
        </label>

        {source !== 'lm' && (
          <label className="flex flex-col gap-1">
            <ParamLabel
              label={t('trainingStudio.audition.lyrics')}
              className="text-[10px] uppercase tracking-wide text-zinc-500"
              info={t('trainingStudio.audition.lyricsInfo')}
            />
            <textarea
              rows={5}
              value={lyrics}
              disabled={source === 'sample' && sampleLocked}
              placeholder={t('trainingStudio.audition.lyricsPlaceholder')}
              onChange={(e) => setLyrics(e.target.value)}
              className={`${FIELD} font-mono text-[11px] leading-relaxed`}
            />
          </label>
        )}
      </div>

      {/* ── Sides ────────────────────────────────────────────────────────── */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <ParamLabel
            label={t('trainingStudio.audition.sides')}
            className="text-[10px] uppercase tracking-wide text-zinc-500"
            rootClassName="flex-1"
            info={t('trainingStudio.audition.sidesInfo')}
          />
          <Toggle
            size="sm"
            accent="amber"
            checked={twoSided}
            onChange={setTwoSided}
            label={t('trainingStudio.audition.twoSided')}
            info={t('trainingStudio.audition.twoSidedInfo')}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Side A — base by default, but adapter-vs-adapter is legitimate. */}
          <div className="flex flex-col gap-1.5 rounded-lg border border-zinc-200 dark:border-white/10 p-2.5">
            <input
              type="text"
              value={baseLabel}
              maxLength={64}
              placeholder={t('trainingStudio.audition.sideBase')}
              onChange={(e) => setBaseLabel(e.target.value)}
              className={FIELD}
            />
            <LmAdapterPicker
              value={baseAdapter}
              onChange={setBaseAdapter}
              extraOptions={milestoneOptions}
              placeholder={t('trainingStudio.audition.baseLm')}
            />
          </div>

          {/* Side B — the adapter under test. */}
          <div className={`flex flex-col gap-1.5 rounded-lg border p-2.5 ${
            twoSided ? 'border-amber-500/25' : 'border-zinc-200 dark:border-white/10 opacity-50'
          }`}>
            <input
              type="text"
              value={adapterLabel}
              maxLength={64}
              disabled={!twoSided}
              placeholder={t('trainingStudio.audition.sideAdapter')}
              onChange={(e) => setAdapterLabel(e.target.value)}
              className={FIELD}
            />
            <LmAdapterPicker
              value={adapterPath}
              onChange={(v) => { setAdapterPickedByUser(true); setAdapterPath(v); }}
              disabled={!twoSided}
              extraOptions={milestoneOptions}
            />
          </div>
        </div>
      </div>

      {/* ── Controls ─────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.audition.seed')}
            className="text-[10px] uppercase tracking-wide text-zinc-500"
            meta={t('trainingStudio.audition.seedMeta')}
            info={t('trainingStudio.audition.seedInfo')}
          />
          <div className="flex items-center gap-1">
            <input
              type="number"
              min={0}
              value={seed}
              placeholder={t('trainingStudio.audition.seedRandom')}
              onChange={(e) => setSeed(e.target.value)}
              className={`${FIELD} tabular-nums`}
            />
            <button
              onClick={() => setSeed(String(Math.floor(Math.random() * 2 ** 31)))}
              title={t('trainingStudio.audition.seedDice')}
              className="p-2 rounded-lg text-zinc-500 hover:text-amber-500 hover:bg-amber-500/10 transition-colors flex-shrink-0"
            >
              <Dices size={14} />
            </button>
          </div>
        </label>

        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.audition.duration')}
            className="text-[10px] uppercase tracking-wide text-zinc-500"
            meta={t('trainingStudio.audition.durationMeta')}
            info={t('trainingStudio.audition.durationInfo')}
          />
          <input
            type="number"
            min={10}
            max={300}
            value={durationSec}
            onChange={(e) => setDurationSec(Number(e.target.value) || 180)}
            className={`${FIELD} tabular-nums`}
          />
        </label>
      </div>

      <details className="rounded-lg border border-zinc-200 dark:border-white/10 px-3 py-2">
        <summary className="cursor-pointer text-[11px] font-semibold text-zinc-600 dark:text-zinc-400">
          {t('trainingStudio.audition.advanced')}
        </summary>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-2">
          {([
            ['temperature', temperature, setTemperature, 0.1, 2, 0.05,
              t('trainingStudio.audition.temperatureInfo')],
            ['topP', topP, setTopP, 0.05, 1, 0.05,
              t('trainingStudio.audition.topPInfo')],
            ['cfgScale', cfgScale, setCfgScale, 0, 10, 0.1,
              t('trainingStudio.audition.cfgScaleInfo')],
            ['repPenalty', repPenalty, setRepPenalty, 1, 1.5, 0.01,
              t('trainingStudio.audition.repPenaltyInfo')],
            ['adapterScale', adapterScale, setAdapterScale, 0, 2, 0.05,
              t('trainingStudio.audition.adapterScaleInfo')],
            ['baseScale', baseScale, setBaseScale, 0, 2, 0.05,
              t('trainingStudio.audition.baseScaleInfo')],
            // DiT render step count — only used when the render toggle is on.
            ['renderSteps', renderSteps, setRenderSteps, 2, 60, 1,
              t('trainingStudio.audition.renderStepsInfo')],
          ] as Array<[string, number, (v: number) => void, number, number, number, string]>).map(
            ([key, val, setter, min, max, step, info]) => (
              <label key={key} className="flex flex-col gap-1">
                <ParamLabel
                  label={t(`trainingStudio.audition.${key}`)}
                  className="text-[10px] uppercase tracking-wide text-zinc-500"
                  meta={t(`trainingStudio.audition.${key}Meta`)}
                  info={info}
                />
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={val}
                  onChange={(e) => setter(Number(e.target.value))}
                  className={`${FIELD} tabular-nums`}
                />
              </label>
            ),
          )}
        </div>
        <p className="text-[10px] text-zinc-500 mt-2">{t('trainingStudio.audition.coResidentHint')}</p>
      </details>

      {/* ── Render-through-DiT opt-in ────────────────────────────────────── */}
      <div className="flex flex-col gap-2">
        <Toggle
          accent="amber"
          checked={renderDit}
          onChange={setRenderDit}
          label={t('trainingStudio.audition.renderDit')}
          info={t('trainingStudio.audition.renderDitHint')}
        />
        {/* Hidden entirely (not disabled) when no DiT adapter is trained for
            this dataset — an option that can never work is noise. */}
        {renderDit && ditAdapterAvailable && (
          <div className="ml-6 flex flex-col gap-0.5">
            <Toggle
              accent="amber"
              checked={renderDitAdapter}
              onChange={setRenderDitAdapter}
              label={t('trainingStudio.audition.renderDitAdapter')}
              info={t('trainingStudio.audition.renderDitAdapterHint')}
            />
            {trainDitStatus?.adapterDir && (
              <span
                className="block text-[10px] font-mono text-zinc-500 ml-[52px] truncate max-w-[420px]"
                title={trainDitStatus.adapterDir}
              >
                {trainDitStatus.adapterDir.replace(/[\\/]+$/, '').split(/[\\/]/).slice(-3).join('/')}
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── Run ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={() => run('ab')}
          disabled={!!disabledReason || auditionRunning}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {auditionRunning || auditionJobActive
            ? <><Loader2 size={13} className="animate-spin" /> {t('trainingStudio.audition.running')}</>
            : <><Headphones size={13} /> {t('trainingStudio.audition.run')}</>}
        </button>
        {disabledReason && (
          <span className="flex items-center gap-1.5 text-[11px] text-amber-600 dark:text-amber-400">
            {!engineUp ? <PlugZap size={12} /> : engineSuspended ? <PauseCircle size={12} /> : <AlertTriangle size={12} />}
            {disabledReason}
          </span>
        )}
      </div>

      {/* The notice is only true while the prompt is still empty — once a
          caption exists the Run button says everything it needs to. */}
      {milestoneNotice && !caption.trim() && (
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-600 dark:text-amber-400">
          <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
          {milestoneNotice}
        </div>
      )}

      {auditionError && (
        <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">
          {auditionError}
        </div>
      )}

      {activeJob?.kind === 'audition' && <JobProgress />}

      {/* ── Results — base LEFT, adapter RIGHT, always ───────────────────── */}
      {latest && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-zinc-500">
            <span>{t('trainingStudio.audition.latest')}</span>
            <span className="text-zinc-400 dark:text-zinc-600">·</span>
            <span className="tabular-nums normal-case">
              {t('trainingStudio.audition.seedUsed', { seed: latest.seed })}
            </span>
            {latest.ditModel && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span className="font-mono normal-case truncate max-w-[220px]" title={latest.ditModel}>
                  {latest.ditModel}
                </span>
              </>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {orderSides(latest).map(side => (
              <AuditionPlayer
                key={`${latest.previewId}-${side.slot}`}
                side={side}
                identicalCodes={hasIdenticalCodes(latest)}
                preview={latest}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── History ──────────────────────────────────────────────────────── */}
      {history.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-zinc-500">
            <History size={11} /> {t('trainingStudio.audition.history')}
          </span>
          {history.map(p => {
            const open = !!openHistory[p.previewId];
            return (
              <div key={p.previewId} className="rounded-lg border border-zinc-200 dark:border-white/10">
                <button
                  onClick={() => setOpenHistory(h => ({ ...h, [p.previewId]: !open }))}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-left"
                >
                  {open ? <ChevronDown size={12} className="text-zinc-500 flex-shrink-0" />
                    : <ChevronRight size={12} className="text-zinc-500 flex-shrink-0" />}
                  <span className="flex-1 min-w-0 truncate text-[11px] text-zinc-700 dark:text-zinc-300">
                    {p.caption || t('trainingStudio.audition.noCaption')}
                  </span>
                  <span className="text-[10px] text-zinc-500 tabular-nums flex-shrink-0">
                    {p.createdAt.slice(0, 16).replace('T', ' ')}
                  </span>
                </button>
                {open && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 px-3 pb-3">
                    {orderSides(p).map(side => (
                      <AuditionPlayer
                        key={`${p.previewId}-${side.slot}`}
                        side={side}
                        identicalCodes={hasIdenticalCodes(p)}
                        preview={p}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default AuditionCard;
