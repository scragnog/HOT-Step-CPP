import React, { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Trash2, Pencil, Music2, Wand2, Play, Loader2, ChevronDown, ChevronRight, Send, FileText, Headphones, Sparkles, Zap, Download } from 'lucide-react';
import { lireekApi, streamRefine, skipThinking } from '../../services/lireekApi';
import type { Generation, Profile } from '../../services/lireekApi';
import { StreamingPanel } from './StreamingPanel';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { useStreamingStore, startStreamGenerate } from '../../stores/streamingStore';
import { useBackendStore } from '../../stores/backendStore';
import { MM3_BACKEND_ID } from '../../utils/captionForBackend';
import {
  pickNearestBpmTrack, readMm3CaptionSelection, resolveMm3Caption, writeMm3CaptionSelection,
  type Mm3CaptionSelection, type Mm3SourceTrack,
} from '../../utils/mm3CaptionSource';
import {
  ensureYue2CaptionSource, hasStoredYue2CaptionSelection, hasStoredYue2SongSelection,
  readYue2SongSelection, resolveYue2Caption,
  writeYue2SongSelection, yue2PresetAdapterPath, yue2TrackBpm, YUE2_BACKEND_ID,
  pickNearestBpmTrack as pickNearestYue2Track,
  type Yue2CaptionSelection, type Yue2SourceTrack,
} from '../../utils/yue2CaptionSource';

/**
 * Whether these lyrics have already produced a track, and whether one was kept.
 *
 * "Kept" is the one that matters: a version was downloaded, so generating again
 * only adds another good take to choose between. "Generated" means a track was
 * made but nothing was kept — worth another roll. Both markers survive the audio
 * being deleted, which is what normally happens once a track has been kept.
 */
const GenerationStatusBadge: React.FC<{ gen: Generation }> = ({ gen }) => {
  const { t } = useTranslation();
  const kept = gen.download_count ?? 0;
  const made = gen.audio_generated_count ?? 0;

  if (kept > 0) {
    const first = gen.first_downloaded_at
      ? ` ${t('lyric.badgeFirstOn', { date: new Date(gen.first_downloaded_at).toLocaleDateString() })}`
      : '';
    return (
      <span
        title={t('lyric.badgeKeptHint', { count: kept }) + first}
        className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/25 text-emerald-300 ring-1 ring-emerald-400/30 flex items-center gap-1"
      >
        <Download className="w-2.5 h-2.5" />
        {t('lyric.badgeKept')}{kept > 1 ? ` ×${kept}` : ''}
      </span>
    );
  }

  if (made > 0) {
    return (
      <span
        title={t('lyric.badgeGeneratedHint', { count: made })}
        className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-300/90 flex items-center gap-1"
      >
        <Music2 className="w-2.5 h-2.5" />
        {t('lyric.badgeGenerated')}{made > 1 ? ` ×${made}` : ''}
      </span>
    );
  }

  return null;
};

/**
 * The MM3 caption for one written song, plus WHERE it comes from.
 *
 * On MiniMax-Music3 the caption that renders is a choice. Rendering new lyrics
 * under one of the artist's own training-track captions — verbatim, not a fresh
 * caption in the same style — is what reliably produces a song in the band's
 * style and, especially, one that ends naturally. So the default is a source
 * track picked by tempo, and this song's own caption becomes the opt-in.
 *
 * On ACE-Step none of this appears and the box behaves exactly as it always has.
 */
/**
 * The ACE-Step caption box, plus a source picker when YuE2 is the active
 * backend.
 *
 * It sits on THIS box and not on the MM3 one below because YuE2 renders from
 * this caption — MM3's three-heading Structured Caption is a different text for
 * a different model. Same control, different home.
 *
 * The choice is stored per song AND per dataset (see writeYue2SongSelection):
 * the caption list belongs to one training dataset, so a track title chosen
 * under one album means nothing under another's.
 *
 * The default is Automatic when the album's preset names a YuE2 adapter — a
 * caption the adapter was actually trained on is what lands in the album's
 * character — and Custom on a base-model render, which has no adapter to be
 * in distribution for. This song's own caption is always the opt-in, and it
 * is never altered by the choice — the picker only decides which text
 * conditions the render.
 *
 * On any other backend this renders exactly the box that was here before.
 */
const Yue2CaptionField: React.FC<{
  gen: Generation;
  yue2Mode: boolean;
  datasetId: string;
  datasetName: string;
  /** Whether the album's preset names a YuE2 adapter — decides the default
   *  mode when nothing has been stored for this song yet. */
  hasAdapter: boolean;
  tracks: Yue2SourceTrack[];
  onSave: (value: string) => void;
}> = ({ gen, yue2Mode, datasetId, datasetName, hasAdapter, tracks, onSave }) => {
  const { t } = useTranslation();
  // Nothing stored anywhere for this song or this dataset: fall back to a
  // default that depends on whether the album has an adapter, rather than
  // readYue2SongSelection's own default (Automatic whenever a dataset is
  // linked, adapter or not — right for Create, wrong for a base-model album).
  const resolveSel = useCallback((): Yue2CaptionSelection => {
    if (!datasetId) return { mode: 'custom' };
    if (hasStoredYue2SongSelection(datasetId, gen.id) || hasStoredYue2CaptionSelection(datasetId)) {
      return readYue2SongSelection(datasetId, gen.id);
    }
    return hasAdapter ? { mode: 'auto' } : { mode: 'custom' };
  }, [datasetId, gen.id, hasAdapter]);
  const [sel, setSel] = useState<Yue2CaptionSelection>(resolveSel);
  useEffect(() => { setSel(resolveSel()); }, [resolveSel]);

  const hasTracks = yue2Mode && !!datasetId && tracks.length > 0;
  const resolved = hasTracks
    ? resolveYue2Caption(gen.caption || '', gen.bpm, tracks, sel)
    : { caption: gen.caption || '', mode: 'custom' as const, fromName: undefined };
  const readOnly = resolved.mode !== 'custom';
  const autoTrack = pickNearestYue2Track(tracks, gen.bpm);

  const selectValue = resolved.mode === 'track' && resolved.fromName
    ? `track:${resolved.fromName}`
    : resolved.mode;

  const onSelect = (value: string) => {
    const next: Yue2CaptionSelection = (value === 'auto' || value === 'custom')
      ? { mode: value }
      : { mode: 'track', selectedName: value.slice('track:'.length) };
    writeYue2SongSelection(datasetId, gen.id, next);
    setSel(next);
  };

  return (
    <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
      <ParamLabel
        label={t('lyric.yue2CaptionLabel', 'Caption — ACE-Step')}
        info={t('lyric.yue2CaptionLabelInfo', 'The caption box that ACE-Step and YuE2 both render from. Edit it directly, or on YuE2, with a source picker below, let a training-track caption take over instead.')}
        className="text-[10px] text-zinc-500 uppercase tracking-wider"
        rootClassName="block mb-1"
      />

      {hasTracks && (
        <div className="flex items-center gap-2 mb-2">
          <ParamLabel
            label={t('lyric.yue2CaptionSource', 'Caption source') + (datasetName ? ` · ${datasetName}` : '')}
            info={t('lyric.yue2CaptionSourceInfo', "Which caption text conditions a YuE2 render for this song. Automatic uses the training dataset track nearest this song's BPM; picking a named track pins that track's caption instead; Custom uses this song's own caption below, editable directly.")}
            className="text-[10px] text-zinc-500 uppercase tracking-wider"
            rootClassName="flex-shrink-0"
          />
          <StyledSelect
            accent="pink"
            value={selectValue}
            onChange={onSelect}
            options={[
              {
                value: 'auto',
                label: t('lyric.yue2CaptionAuto', 'Automatic from dataset')
                  + (autoTrack ? ` (${t('lyric.yue2CaptionNearestTempo', 'nearest tempo')}: ${autoTrack.name})` : ''),
              },
              ...tracks.map(track => ({
                value: `track:${track.name}`,
                label: `${t('lyric.yue2CaptionTrack', 'Track')}: ${track.name}`
                  + (yue2TrackBpm(track) ? ` · ${yue2TrackBpm(track)} BPM` : ''),
              })),
              { value: 'custom', label: t('lyric.yue2CaptionCustom', "Custom (this song's own caption)") },
            ]}
            className="flex-1 min-w-0"
          />
        </div>
      )}

      {readOnly ? (
        <>
          <textarea
            key={`yue2-resolved-${gen.id}`}
            readOnly
            rows={4}
            className="w-full bg-transparent text-sm font-mono text-zinc-500 dark:text-zinc-500 focus:outline-none border-b border-transparent resize-y cursor-default"
            value={resolved.caption}
          />
          <p className="text-[10px] text-cyan-400/70 mt-1">
            {t('lyric.yue2CaptionFromTrack', 'From dataset track')}: {resolved.fromName}
          </p>
        </>
      ) : (
        <textarea
          key={`yue2-custom-${gen.id}`}
          className="w-full bg-transparent text-sm text-zinc-700 dark:text-zinc-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-pink-500/50 transition-colors resize-none"
          rows={2}
          defaultValue={gen.caption || ''}
          onBlur={(e) => { if (e.target.value !== (gen.caption || '')) onSave(e.target.value); }}
        />
      )}
    </div>
  );
};

const Mm3CaptionField: React.FC<{
  gen: Generation;
  mm3Mode: boolean;
  tracks: Mm3SourceTrack[];
  selection: Mm3CaptionSelection;
  onSelectionChange: (sel: Mm3CaptionSelection) => void;
  onSaveCustom: (value: string) => void;
}> = ({ gen, mm3Mode, tracks, selection, onSelectionChange, onSaveCustom }) => {
  const { t } = useTranslation();

  // An album with no captioned source tracks has only one thing to offer, so
  // the control collapses to a hint and the box stays editable.
  const hasTracks = mm3Mode && tracks.length > 0;
  const resolved = hasTracks
    ? resolveMm3Caption(gen, tracks, selection)
    : { caption: gen.caption_mm3 || '', mode: 'custom' as const, fromTitle: undefined };
  const readOnly = resolved.mode !== 'custom';
  const autoTrack = pickNearestBpmTrack(tracks, gen.bpm);

  const selectValue = resolved.mode === 'track' && resolved.fromTitle
    ? `track:${resolved.fromTitle}`
    : resolved.mode;

  const onSelect = (value: string) => {
    if (value === 'auto' || value === 'custom') onSelectionChange({ mode: value });
    else onSelectionChange({ mode: 'track', selectedTitle: value.slice('track:'.length) });
  };

  return (
    <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
      <ParamLabel
        label={t('lyric.mm3CaptionLabel', 'MM3 Caption — MiniMax-Music3 Structured Caption')}
        info={t('lyric.mm3CaptionLabelInfo', "MiniMax-Music3's three-heading Structured Caption, a different text from the ACE-Step/YuE2 caption above and generated separately. MM3 lands off-genre when handed the other caption instead, so this box is what an MM3 render actually reads.")}
        className="text-[10px] text-zinc-500 uppercase tracking-wider"
        rootClassName="block mb-1"
      />

      {mm3Mode && (
        hasTracks ? (
          <div className="flex items-center gap-2 mb-2">
            <ParamLabel
              label={t('lyric.mm3CaptionSource', 'Caption source')}
              info={t('lyric.mm3CaptionSourceInfo', "Which MM3 caption conditions this song's render. Automatic uses the training dataset track nearest this song's BPM; picking a named track pins that track's caption instead; Custom uses this song's own MM3 caption below, editable directly.")}
              className="text-[10px] text-zinc-500 uppercase tracking-wider"
              rootClassName="flex-shrink-0"
            />
            <StyledSelect
              accent="pink"
              value={selectValue}
              onChange={onSelect}
              options={[
                {
                  value: 'auto',
                  label: t('lyric.mm3CaptionAuto', 'Automatic from dataset')
                    + (autoTrack ? ` (${t('lyric.mm3CaptionNearestTempo', 'nearest tempo')}: ${autoTrack.title})` : ''),
                },
                ...tracks.map(track => ({
                  value: `track:${track.title}`,
                  label: `${t('lyric.mm3CaptionTrack', 'Track')}: ${track.title}` + (track.bpm ? ` · ${track.bpm} BPM` : ''),
                })),
                { value: 'custom', label: t('lyric.mm3CaptionCustom', "Custom (this song's own caption)") },
              ]}
              className="flex-1 min-w-0"
            />
          </div>
        ) : (
          <p className="text-[10px] text-amber-400/70 mb-2">
            {t('lyric.mm3CaptionNoTracks', 'No source track on this album has an MM3 caption — using this song’s own caption.')}
          </p>
        )
      )}

      {readOnly ? (
        <>
          <textarea
            key={`mm3-resolved-${gen.id}`}
            readOnly
            className="w-full bg-transparent text-xs font-mono text-zinc-500 dark:text-zinc-500 focus:outline-none border-b border-transparent resize-y cursor-default"
            rows={6}
            value={resolved.caption}
          />
          <p className="text-[10px] text-cyan-400/70 mt-1">
            {t('lyric.mm3CaptionFromTrack', 'From dataset track')}: {resolved.fromTitle}
          </p>
        </>
      ) : (
        <textarea
          key={`mm3-custom-${gen.id}`}
          className="w-full bg-transparent text-xs font-mono text-zinc-700 dark:text-zinc-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-cyan-500/50 transition-colors resize-y"
          rows={gen.caption_mm3 ? 6 : 2}
          placeholder="None — written before this field existed, or the MM3 caption call failed. The MM3 backend will fall back to the caption above."
          defaultValue={gen.caption_mm3 || ''}
          onBlur={e => { if (e.target.value !== (gen.caption_mm3 || '')) onSaveCustom(e.target.value); }}
        />
      )}
    </div>
  );
};

/** The YuE2 PLANNER caption: one sentence, fixed order. Plain editable text —
 *  no source control here, because a song that HAS one no longer borrows a
 *  dataset track's caption (captionForBackend prefers it over the source pick
 *  in Yue2CaptionField above). */
const Yue2PlannerCaptionField: React.FC<{ gen: Generation; onSave: (value: string) => void }> = ({ gen, onSave }) => (
  <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
    <ParamLabel
      label="YuE2 Caption — one sentence: language → genre → vocal → instruments → mood → production → BPM"
      info="YuE2's own one-sentence planner caption, in that fixed field order. When present it is what a YuE2 render is prompted with, taking priority over the ACE-Step caption's dataset-track source pick above. Blank falls back to a dataset-track caption instead."
      className="text-[10px] text-zinc-500 uppercase tracking-wider"
      rootClassName="block mb-1"
    />
    <textarea
      key={`yue2-${gen.id}`}
      className="w-full bg-transparent text-xs font-mono text-zinc-700 dark:text-zinc-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-cyan-500/50 transition-colors resize-y"
      rows={gen.caption_yue2 ? 3 : 2}
      placeholder="None — written before this field existed, or the YuE2 caption call failed. A YuE2 render falls back to a dataset-track caption."
      defaultValue={gen.caption_yue2 || ''}
      onBlur={e => { if (e.target.value !== (gen.caption_yue2 || '')) onSave(e.target.value); }}
    />
  </div>
);

interface WrittenSongsTabProps {
  generations: Generation[];
  profiles: Profile[];
  /** The album these songs belong to. Its preset owns the YuE2 adapter, and
   *  therefore which dataset's captions this tab may offer. */
  lyricsSetId: number;
  /** Album source tracks that carry an MM3 caption, in album order. */
  mm3SourceTracks?: Mm3SourceTrack[];
  onRefresh: () => void;
  onGenerateAudio: (gen: Generation) => void;
  onSendToCreate?: (gen: Generation) => void;
  onViewRecordings?: (genId: number) => void;
  showToast: (msg: string) => void;
  generationModel: { provider: string; model?: string };
  refinementModel: { provider: string; model?: string };
}

export const WrittenSongsTab: React.FC<WrittenSongsTabProps> = ({
  generations, profiles, lyricsSetId, mm3SourceTracks = [], onRefresh, onGenerateAudio,
  onSendToCreate, onViewRecordings, showToast, generationModel, refinementModel,
}) => {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const { t } = useTranslation();
  const [generating, setGenerating] = useState(false);
  const [refiningId, setRefiningId] = useState<number | null>(null);
  const [genCount, setGenCount] = useState(1);
  const [userSubject, setUserSubject] = useState('');

  // ── MM3 caption source ──
  // The choice has no column to live in, so it is per-generation localStorage.
  // This map is only a render mirror of that: absent means "not touched this
  // session", and the stored value (default Automatic) is read on demand.
  const mm3Mode = useBackendStore(s => s.activeBackendId) === MM3_BACKEND_ID;
  const [captionSelections, setCaptionSelections] = useState<Record<number, Mm3CaptionSelection>>({});

  // ── YuE2 caption source ──
  // THIS ALBUM's training dataset, resolved by its lyrics-set id
  // (training_datasets.lyrics_set_id server-side) — not the adapter the engine
  // happens to be holding, and not the adapter's own run manifest either: both
  // break when a run folder moves or its prepared cache is swept, and neither
  // has anything to do with the dataset the album is actually built on.
  //
  // Still fetches the preset, but only to know whether the album has an
  // adapter — that decides the default mode (Automatic vs Custom) down in
  // Yue2CaptionField, since a base-model render has no adapter to be
  // in-distribution for even when the album's dataset itself has captions.
  //
  // One lookup serves every card; the per-song CHOICE is stored per
  // (dataset, song) instead.
  const yue2Mode = useBackendStore(s => s.activeBackendId) === YUE2_BACKEND_ID;
  const yue2Catalogue = useBackendStore(s => s.models[YUE2_BACKEND_ID] ?? null);
  const fetchBackendModels = useBackendStore(s => s.fetchModels);
  const [yue2Tracks, setYue2Tracks] = useState<Yue2SourceTrack[]>([]);
  const [yue2Dataset, setYue2Dataset] = useState<{ id: string; name: string } | null>(null);
  const [yue2HasAdapter, setYue2HasAdapter] = useState(false);
  useEffect(() => {
    if (yue2Mode && !yue2Catalogue) void fetchBackendModels(YUE2_BACKEND_ID);
  }, [yue2Mode, yue2Catalogue, fetchBackendModels]);
  useEffect(() => {
    let live = true;
    const load = async () => {
      if (!yue2Mode || !lyricsSetId) return;
      try {
        const preset = (await lireekApi.getPreset(lyricsSetId)).preset;
        if (live) setYue2HasAdapter(!!yue2PresetAdapterPath(preset));
      } catch { if (live) setYue2HasAdapter(false); }
      const src = await ensureYue2CaptionSource({ lyricsSet: lyricsSetId });
      if (live) {
        setYue2Dataset(src.datasetId ? { id: src.datasetId, name: src.datasetName } : null);
        setYue2Tracks(src.tracks);
      }
    };
    if (!yue2Mode) { setYue2Dataset(null); setYue2Tracks([]); setYue2HasAdapter(false); return; }
    void load();
    return () => { live = false; };
  }, [yue2Mode, lyricsSetId]);
  const captionSelectionFor = useCallback(
    (genId: number): Mm3CaptionSelection => captionSelections[genId] ?? readMm3CaptionSelection(genId),
    [captionSelections],
  );
  const setCaptionSelectionFor = useCallback((genId: number, sel: Mm3CaptionSelection) => {
    writeMm3CaptionSelection(genId, sel);
    setCaptionSelections(prev => ({ ...prev, [genId]: sel }));
  }, []);

  // Persistent streaming state — survives tab navigation
  const streaming = useStreamingStore();

  // Local streaming state for refine (one-off, doesn't need persistence)
  const [refineStreamVisible, setRefineStreamVisible] = useState(false);
  const [refineStreamText, setRefineStreamText] = useState('');
  const [refineStreamPhase, setRefineStreamPhase] = useState('');
  const [refineStreamDone, setRefineStreamDone] = useState(false);

  const handleQuickGenerate = useCallback(async (noThink = false) => {
    if (profiles.length === 0) {
      showToast('Build a profile first');
      return;
    }
    setGenerating(true);

    const profile = profiles[0];
    try {
      for (let i = 0; i < genCount; i++) {
        await startStreamGenerate(
          profile.id,
          {
            profile_id: profile.id,
            provider: generationModel.provider,
            model: generationModel.model,
            user_subject: userSubject.trim() || undefined,
            no_think: noThink || undefined,
          },
          () => onRefresh(),
        );
      }
      showToast(`Generated ${genCount} new song${genCount > 1 ? 's' : ''}`);
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    } finally {
      setGenerating(false);
    }
  }, [profiles, genCount, generationModel, onRefresh, showToast, userSubject]);

  const handleRefine = async (gen: Generation) => {
    const { provider, model } = refinementModel;
    if (!provider) {
      showToast('Select a refinement model first');
      return;
    }
    setRefiningId(gen.id);
    setRefineStreamVisible(true);
    setRefineStreamText('');
    setRefineStreamPhase('');
    setRefineStreamDone(false);
    try {
      await streamRefine(
        gen.id,
        { provider, model },
        {
          onChunk: (text) => setRefineStreamText(prev => {
            const next = prev + text;
            // Cap at 200KB to prevent OOM — matches streaming store limit
            return next.length > 200_000 ? '\u2026(earlier output trimmed)\u2026\n' + next.slice(-200_000) : next;
          }),
          onPhase: (phase) => setRefineStreamPhase(phase),
          onResult: () => {
            showToast(`Refined: ${gen.title || 'Untitled'}`);
            onRefresh();
          },
          onError: (err) => showToast(`Refinement failed: ${err}`),
        },
      );
      setRefineStreamDone(true);
    } catch (err: any) {
      showToast(`Refinement failed: ${err.message}`);
      setRefineStreamDone(true);
    } finally {
      setRefiningId(null);
    }
  };

  const handleDelete = async (gen: Generation) => {
    if (!confirm(`Delete "${gen.title || 'Untitled'}"?`)) return;
    try {
      await lireekApi.deleteGeneration(gen.id);
      showToast('Deleted');
      onRefresh();
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    }
  };

  const handleSaveField = async (genId: number, field: string, value: any) => {
    try {
      await lireekApi.updateMetadata(genId, { [field]: value });
      onRefresh();
    } catch (err: any) {
      showToast(`Failed to save: ${err.message}`);
    }
  };

  // Show streaming panel if either generation or refinement is active
  const showGenerationStream = streaming.visible || generating;

  return (
    <div className="p-4 space-y-4">
      {/* Generate controls */}
      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={() => handleQuickGenerate(false)}
          disabled={generating || profiles.length === 0}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-200 dark:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-semibold transition-all"
        >
          {generating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              {t('lyric.generating')}
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4" />
              {t('lyric.generateLyrics')}
            </>
          )}
        </button>
        <button
          onClick={() => handleQuickGenerate(true)}
          disabled={generating || profiles.length === 0}
          title={t('lyric.generateNoThinkHint')}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-500 disabled:bg-zinc-200 dark:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-semibold transition-all"
        >
          <Zap className="w-4 h-4" />
          {t('lyric.generateNoThink')}
        </button>
        <div className="flex items-center gap-2">
          <ParamLabel
            label={t('lyric.count')}
            info={t('lyric.countInfo', 'How many new songs Generate Lyrics writes in a row from the same profile. Each one still uses past generations for this artist to avoid repeating subjects, keys, titles, BPMs and durations, so a higher count gives more takes to pick between rather than more repetition.')}
            className="text-xs text-zinc-500"
          />
          <StyledSelect
            accent="pink"
            value={genCount}
            onChange={setGenCount}
            options={[1, 2, 3, 4, 5, 8, 10].map(n => ({ value: n, label: String(n) }))}
            className="w-16"
          />
        </div>
        {profiles.length === 0 && (
          <span className="text-xs text-amber-400/60">{t('lyric.buildProfileFirst')}</span>
        )}
      </div>
      {/* Optional subject input */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={userSubject}
          onChange={(e) => setUserSubject(e.target.value)}
          placeholder={t('lyric.subjectPlaceholder')}
          className="flex-1 px-3 py-2 rounded-lg bg-white/5 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-amber-500/50 transition-colors"
        />
        {userSubject && (
          <button
            onClick={() => setUserSubject('')}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors px-1.5"
            title={t('lyric.clearSubject')}
          >
            ✕
          </button>
        )}
      </div>

      {/* Generation streaming panel — persists across tab navigation */}
      {showGenerationStream && (
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 overflow-hidden">
          <StreamingPanel
            visible={true}
            streamText={streaming.text}
            phase={streaming.phase}
            done={streaming.done}
            onSkipThinking={() => skipThinking()}
          />
        </div>
      )}

      {/* Refine streaming panel — local, only shown during active refinement */}
      {refineStreamVisible && (
        <div className="rounded-xl border border-purple-500/20 bg-purple-500/5 overflow-hidden">
          <StreamingPanel
            visible={refineStreamVisible}
            streamText={refineStreamText}
            phase={refineStreamPhase}
            done={refineStreamDone}
            onSkipThinking={() => skipThinking()}
          />
        </div>
      )}

      {/* Generations list */}
      {generations.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Music2 className="w-7 h-7 text-zinc-600" />
          </div>
          <h3 className="text-base font-semibold text-zinc-600 dark:text-zinc-400 mb-2">{t('lyric.noGeneratedLyricsYet')}</h3>
          <p className="text-sm text-zinc-500 max-w-xs">
            {t('lyric.generateFromProfile')}
          </p>
        </div>
      ) : (
        <div className="space-y-1">
          {generations.map((gen, idx) => {
            const isExpanded = expandedId === gen.id;
            const kept = (gen.download_count ?? 0) > 0;

            return (
              <div
                key={gen.id}
                className={`rounded-xl border border-zinc-200 dark:border-white/5 hover:border-zinc-300 dark:border-white/10 overflow-hidden transition-colors ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)}`}
              >
                {/* Header */}
                <button
                  className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : gen.id)}
                >
                  {isExpanded
                    ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                    : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                  }
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate">
                      {gen.title || 'Untitled'}
                    </p>
                    <p className="text-xs text-zinc-500 mt-0.5">
                      {gen.subject || 'No subject'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <GenerationStatusBadge gen={gen} />
                    {gen.parent_generation_id && (
                      <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 flex items-center gap-0.5">
                        <Sparkles className="w-2.5 h-2.5" /> {t('lyric.refined')}
                      </span>
                    )}
                    {/* Still offered on a kept song — just played down, so the eye
                        goes to the ones with nothing kept yet. */}
                    <button
                      onClick={(e) => { e.stopPropagation(); onGenerateAudio(gen); }}
                      className={`flex items-center gap-1 px-2 py-1 rounded-md text-[11px] transition-colors ${
                        kept
                          ? 'text-zinc-500 hover:text-emerald-400 hover:bg-emerald-500/10'
                          : 'text-emerald-400 hover:bg-emerald-500/10'
                      }`}
                      title={kept ? t('lyric.audioAgainHint') : t('lyric.audioHint')}
                    >
                      <Play className="w-3 h-3" />
                      {t('lyric.audio')}
                    </button>
                    {onViewRecordings && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onViewRecordings(gen.id); }}
                        className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-pink-400 hover:bg-pink-500/10 transition-colors"
                        title="View generated songs from these lyrics"
                      >
                        <Headphones className="w-3 h-3" />
                        {t('lyric.songs')}
                      </button>
                    )}
                    {gen.bpm ? (
                      <span className="text-[11px] text-zinc-500 font-mono">{gen.bpm} BPM</span>
                    ) : null}
                    {gen.key ? (
                      <span className="text-[11px] text-zinc-500 font-mono">{gen.key}</span>
                    ) : null}
                  </div>
                </button>

                {/* Expanded content */}
                {isExpanded && (
                  <div className="border-t border-zinc-200 dark:border-white/5">
                    <div className="p-4 space-y-4">
                      {/* Editable title */}
                      <div className="flex items-center gap-3">
                        <FileText className="w-5 h-5 text-green-400 flex-shrink-0" />
                        <input
                          className="flex-1 text-lg font-bold text-white bg-transparent border-b border-transparent hover:border-white/20 focus:border-pink-500/50 focus:outline-none transition-colors"
                          defaultValue={gen.title || 'Untitled'}
                          onBlur={(e) => { if (e.target.value !== gen.title) handleSaveField(gen.id, 'title', e.target.value); }}
                        />
                        <Pencil className="w-3.5 h-3.5 text-zinc-600" />
                      </div>

                      {/* Metadata grid */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <ParamLabel
                            label="Subject"
                            info="What the song is about. Generate Lyrics fills this in from the profile and the optional subject field above; edit it here to correct or retitle the topic after the fact."
                            className="text-[10px] text-zinc-500 uppercase tracking-wider"
                            rootClassName="block mb-1"
                          />
                          <input
                            className="w-full bg-transparent text-sm text-amber-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-amber-500/50 transition-colors"
                            defaultValue={gen.subject || ''}
                            onBlur={(e) => { if (e.target.value !== (gen.subject || '')) handleSaveField(gen.id, 'subject', e.target.value); }}
                          />
                        </div>
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <ParamLabel
                            label="BPM"
                            info="The song's tempo in beats per minute. Generation and refinement avoid repeating past BPMs for this artist; on YuE2 and MM3 it also picks which dataset track a caption source-picker offers as 'nearest tempo'. Edit it to correct a wrong guess or to steer that nearest-tempo pick."
                            className="text-[10px] text-zinc-500 uppercase tracking-wider"
                            rootClassName="block mb-1"
                          />
                          <input
                            type="number"
                            className="w-full bg-transparent text-sm text-pink-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-pink-500/50 transition-colors"
                            defaultValue={gen.bpm || 0}
                            onBlur={(e) => { const v = parseInt(e.target.value) || 0; if (v !== gen.bpm) handleSaveField(gen.id, 'bpm', v); }}
                          />
                        </div>
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <ParamLabel
                            label="Key"
                            info="The song's musical key. Generation and refinement avoid repeating past keys for this artist. It is free text (edit it if the model's guess is wrong) and carries through to Send to Custom-Gen."
                            className="text-[10px] text-zinc-500 uppercase tracking-wider"
                            rootClassName="block mb-1"
                          />
                          <input
                            className="w-full bg-transparent text-sm text-blue-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-blue-500/50 transition-colors"
                            defaultValue={gen.key || ''}
                            onBlur={(e) => { if (e.target.value !== (gen.key || '')) handleSaveField(gen.id, 'key', e.target.value); }}
                          />
                        </div>
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <ParamLabel
                            label="Duration (seconds)"
                            info="How long the rendered audio should run. Generation and refinement avoid repeating past durations for this artist; a wrong or estimated value here is what Generate Audio and Send to Custom-Gen use as the target length, so edit it before rendering if it looks off."
                            className="text-[10px] text-zinc-500 uppercase tracking-wider"
                            rootClassName="block mb-1"
                          />
                          <input
                            type="number"
                            className="w-full bg-transparent text-sm text-purple-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-purple-500/50 transition-colors"
                            defaultValue={gen.duration || 0}
                            onBlur={(e) => { const v = parseInt(e.target.value) || 0; if (v !== gen.duration) handleSaveField(gen.id, 'duration', v); }}
                          />
                        </div>
                      </div>

                      {/* Editable caption — and on YuE2 the SOURCE control lives
                          here rather than on the MM3 box below, because YuE2
                          renders from this caption. Its adapter was trained on
                          whole songs under their own captions with half of them
                          dropped, so every dataset caption is a prompt it has
                          really seen and picking one steers toward that track. */}
                      <Yue2CaptionField
                        gen={gen}
                        yue2Mode={yue2Mode}
                        datasetId={yue2Dataset?.id || ''}
                        datasetName={yue2Dataset?.name || ''}
                        hasAdapter={yue2HasAdapter}
                        tracks={yue2Tracks}
                        onSave={(value) => handleSaveField(gen.id, 'caption', value)}
                      />

                      {/* MM3 caption — a genuinely different caption, not a reformatting
                          of the one above. MiniMax-Music3 was trained on a three-heading
                          Structured Caption and lands off-genre when handed an ACE-Step
                          caption instead. On MM3 it also carries a SOURCE control; see
                          Mm3CaptionField. */}
                      <Mm3CaptionField
                        gen={gen}
                        mm3Mode={mm3Mode}
                        tracks={mm3SourceTracks}
                        selection={captionSelectionFor(gen.id)}
                        onSelectionChange={sel => setCaptionSelectionFor(gen.id, sel)}
                        onSaveCustom={value => handleSaveField(gen.id, 'caption_mm3', value)}
                      />

                      {/* YuE2 caption — the third format: one sentence in the
                          planner's own field order. */}
                      <Yue2PlannerCaptionField gen={gen} onSave={value => handleSaveField(gen.id, 'caption_yue2', value)} />

                      {/* Action buttons */}
                      <div className="flex items-center gap-2 flex-wrap">
                        <button
                          onClick={() => onGenerateAudio(gen)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-pink-500/30 to-purple-500/30 text-white hover:from-pink-500/40 hover:to-purple-500/40 text-sm font-semibold transition-all border border-pink-500/20"
                        >
                          <Play className="w-3.5 h-3.5" />
                          {t('lyric.generateAudio')}
                        </button>
                        <button
                          onClick={() => handleRefine(gen)}
                          disabled={refiningId === gen.id}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                          title="Refine these lyrics using the refinement LLM"
                        >
                          {refiningId === gen.id ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                          {t('lyric.refine')}
                        </button>
                        {onSendToCreate && (
                          <button
                            onClick={() => onSendToCreate(gen)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 text-sm font-medium transition-colors border border-amber-500/10"
                          >
                            <Send className="w-3.5 h-3.5" />
                            {t('lyric.sendToCreate')}
                          </button>
                        )}
                        <div className="flex-1" />
                        <button
                          onClick={() => handleDelete(gen)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                        >
                          <Trash2 className="w-3 h-3" />
                          {t('common.delete')}
                        </button>
                      </div>

                      {/* Editable lyrics */}
                      <div>
                        <h3 className="text-sm font-semibold text-zinc-600 dark:text-zinc-400 uppercase tracking-wider mb-2">{t('lyric.lyrics')}</h3>
                        <textarea
                          className="w-full p-4 rounded-xl bg-black/20 dark:bg-black/40 border border-zinc-200 dark:border-white/5 text-sm text-zinc-800 dark:text-zinc-200 font-mono leading-relaxed focus:outline-none focus:border-pink-500/30 resize-y transition-colors"
                          style={{ minHeight: '300px' }}
                          defaultValue={gen.lyrics || ''}
                          onBlur={(e) => { if (e.target.value !== (gen.lyrics || '')) handleSaveField(gen.id, 'lyrics', e.target.value); }}
                        />
                      </div>
                    </div>
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
