// SongActionsMenu.tsx — the one 3-dot menu for a generated track.
//
// Every surface that lists songs (Library, Recent Songs, the Lyric Studio
// queue, the playlist, Song Details, Cover Studio) mounts this same component,
// so an action added here appears everywhere at once. Before it existed the
// menu was copy-pasted twice inside SongList alone, and most surfaces had no
// menu at all.
//
// Two kinds of action live here:
//   - self-contained ones the component implements itself (Export Params,
//     Retranscribe, Cover Art, A/B pinning, Post-Processing, Rename) — always present,
//     because they need nothing from the host
//   - host-supplied ones passed as callbacks (Edit, Download, Send to Cover,
//     Edit Metadata, Add to Playlist, Delete) — each hidden when its callback
//     is absent, since "Add to Playlist" makes no sense inside the playlist
//
// The dropdown portals to document.body to escape overflow clipping, and tags
// itself data-portal-layer so a click inside it does not read as a click
// outside to BarSection's dismiss handler (#129/#126).

import React, { useEffect, useRef, useState, useLayoutEffect, useSyncExternalStore } from 'react';
import ReactDOM from 'react-dom';
import {
  MoreHorizontal, RotateCcw, Download, Disc3, Tags, Trash2, Upload,
  Mic2, Image, ArrowLeftRight, ListPlus, Sparkles, Loader2, AlertCircle, Undo2, FileMusic, FileText,
  Pencil, AudioLines,
} from 'lucide-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import type { Song, UnifiedRecentSong } from '@/types';
import { cue, songToTrack } from '@/stores/playbackStore';
import { songApi } from '@/services/api';
import { coerceDuration } from '@/stores/playbackConverters';
import { saveJson, saveText } from '@/utils/storage/saveFile';
import {
  getSnapshot as getEngineSnapshot, subscribe as subscribeEngine,
} from '@/audio/engine';
import { LyricEditorModal } from '@/components/player/LyricEditorModal';
import { seedLinesFromLrc } from '@/components/player/lyric-editor/lrcSeed';
import type { EditLine } from '@/components/player/lyric-editor/timingModel';
import { useABCompareSelector, setTrackA, setTrackB } from '@/stores/abCompareStore';
import {
  startPostProcessing, canPostProcess, revertPostProcessing, dismissRun, usePostProcessRun,
} from '@/stores/postProcessStore';
import { openCoverArtPrompt } from '@/components/library/CoverArtPromptModal';
import { scoreAbcOptions, lrcOptions, lyricsJsonOptions } from '@/queries/shell-queue/lyrics';
import { downloadLyricsLrc } from '@/utils/audio/downloadTrack';
import { useRetranscribe } from '@/stores/retranscribeStore';
import { BottomSheet, useIsMobile } from '@/components/shared/overlays/BottomSheet';
import { YUE2_BACKEND_ID } from '@/utils/lyrics/yue2CaptionSource';
import { errorMessage } from '@/utils/format/errorMessage';
import { emit } from '@/utils/appEvents';

// ── Portal ───────────────────────────────────────────────────────────────────

interface PortalMenuProps {
  anchorRef: React.RefObject<HTMLElement | null>;
  onClose: () => void;
  children: React.ReactNode;
}

const MENU_WIDTH = 220;
const MENU_MAX_HEIGHT = 420;

const PortalMenu: React.FC<PortalMenuProps & { sheet?: boolean; title?: React.ReactNode }> = ({ anchorRef, onClose, children, sheet, title }) => {
  if (sheet) {
    return <BottomSheet open onClose={onClose} title={title} ariaLabel="Song actions">{children}</BottomSheet>;
  }
  return <DropdownMenu anchorRef={anchorRef} onClose={onClose}>{children}</DropdownMenu>;
};

const DropdownMenu: React.FC<PortalMenuProps> = ({ anchorRef, onClose, children }) => {
  const [pos, setPos] = useState({ top: 0, left: 0, flipped: false });

  useLayoutEffect(() => {
    if (!anchorRef.current) return;
    const rect = anchorRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const flipped = spaceBelow < MENU_MAX_HEIGHT && rect.top > spaceBelow;
    setPos({
      top: flipped ? rect.top : rect.bottom + 4,
      // Right-align to the button, then keep the whole menu on screen.
      left: Math.min(
        Math.max(8, rect.right - MENU_WIDTH),
        Math.max(8, window.innerWidth - MENU_WIDTH - 8),
      ),
      flipped,
    });
  }, [anchorRef]);

  return ReactDOM.createPortal(
    <>
      <div data-portal-layer className="fixed inset-0 z-[9998]" onClick={onClose} />
      <div
        data-portal-layer
        className="fixed z-[9999] bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-xl shadow-xl py-1 overflow-y-auto"
        style={{
          top: pos.flipped ? undefined : pos.top,
          bottom: pos.flipped ? (window.innerHeight - pos.top + 4) : undefined,
          left: pos.left,
          width: MENU_WIDTH,
          maxHeight: MENU_MAX_HEIGHT,
        }}
      >
        {children}
      </div>
    </>,
    document.body,
  );
};

// ── Menu items ───────────────────────────────────────────────────────────────

const ITEM_BASE = 'w-full flex items-center gap-2 px-3 py-2 max-[1023px]:py-3 text-sm text-left transition-colors';

interface ItemProps {
  icon: React.ReactNode;
  label: React.ReactNode;
  onClick: () => void;
  /** Tailwind colour classes for the enabled state. */
  tone?: string;
  disabled?: boolean;
  /** Shown on hover — the place to explain a disabled item. */
  title?: string;
}

const Item: React.FC<ItemProps> = ({ icon, label, onClick, tone, disabled, title }) => (
  <button
    type="button"
    title={title}
    disabled={disabled}
    onClick={(e) => { e.stopPropagation(); if (!disabled) onClick(); }}
    className={`${ITEM_BASE} ${
      disabled
        ? 'text-zinc-400 dark:text-zinc-600 cursor-not-allowed'
        : (tone || 'text-zinc-700 dark:text-zinc-300 hover:bg-white/5 hover:text-white')
    }`}
  >
    <span className="flex-shrink-0">{icon}</span>
    <span className="truncate">{label}</span>
  </button>
);

const Divider: React.FC = () => (
  <div className="border-t border-zinc-200 dark:border-white/5 my-1" />
);

/**
 * The lyric editor, mounted only while it is open.
 *
 * The editor's transport is the app's single audio element, so it has to be
 * handed the engine's live playhead — a subscription that would otherwise sit
 * on every row of the library. One row, while the modal is up, is the cost.
 */
const WordPlacementEditor: React.FC<{
  song: Song; audioUrl: string; seed: EditLine[] | null; onClose: () => void;
}> = ({ song, audioUrl, seed, onClose }) => {
  const engine = useSyncExternalStore(subscribeEngine, getEngineSnapshot);
  // A song row's duration is seconds or a formatted "m:ss"; the editor works in
  // seconds. The engine's length is the fallback for a row that has none yet.
  const duration = coerceDuration(song.duration) || engine.duration;
  return (
    <LyricEditorModal
      audioUrl={audioUrl}
      title={song.title}
      duration={duration}
      currentTime={engine.currentTime}
      seed={seed}
      onClose={onClose}
    />
  );
};

// ── Shape adapters ───────────────────────────────────────────────────────────

/** Words in a `.lyrics.json`, for deciding whether there is anything to place. */
function countSidecarWords(data: { lines?: Array<{ words?: unknown[] }> } | null): number {
  return data?.lines?.reduce((n, line) => n + (line.words?.length ?? 0), 0) ?? 0;
}

/**
 * Widen a /songs/recent row into the Song shape this menu (and downloadTrack,
 * and the cover-art modal) expects.
 *
 * mastered_audio_url is the field that matters most here: it is what decides
 * whether the post-processing item is offered, so dropping it would offer a
 * second pass on an already-mastered track.
 */
export function songFromRecent(rs: UnifiedRecentSong): Song {
  return {
    id: rs.id,
    title: rs.title || 'Untitled',
    style: rs.style || rs.caption || '',
    caption: rs.caption || '',
    lyrics: rs.lyrics || '',
    audioUrl: rs.audio_url || '',
    audio_url: rs.audio_url || '',
    coverUrl: rs.cover_url || rs.artist_image || '',
    cover_url: rs.cover_url || '',
    duration: rs.duration || 0,
    bpm: rs.bpm,
    key_scale: rs.key_scale,
    tags: [],
    masteredAudioUrl: rs.mastered_audio_url || '',
    mastered_audio_url: rs.mastered_audio_url || '',
    noAdapterAudioUrl: rs.noadapter_audio_url || '',
    latentUrl: rs.latent_url || '',
    latent_url: rs.latent_url || '',
  };
}

/**
 * Widen a playlist entry into the Song shape.
 *
 * A playlist item is a thin projection stored in localStorage, so it carries no
 * lyrics and its `style` doubles as the caption. That is enough for every menu
 * action: the post-processing gate reads masteredAudioUrl and audioUrl, and the
 * server re-reads the real row before it runs anything.
 */
export function songFromPlaylistItem(item: {
  id: string;
  title: string;
  audioUrl: string;
  masteredAudioUrl?: string;
  noAdapterAudioUrl?: string;
  coverUrl?: string;
  duration?: number;
  style?: string;
  generationParams?: any;
}): Song {
  return {
    id: item.id,
    title: item.title || 'Untitled',
    style: item.style || '',
    caption: item.style || '',
    lyrics: '',
    audioUrl: item.audioUrl || '',
    audio_url: item.audioUrl || '',
    coverUrl: item.coverUrl || '',
    duration: item.duration || 0,
    tags: [],
    masteredAudioUrl: item.masteredAudioUrl || '',
    mastered_audio_url: item.masteredAudioUrl || '',
    noAdapterAudioUrl: item.noAdapterAudioUrl || '',
    generationParams: item.generationParams,
  };
}

/**
 * Widen a finished audio-queue item into the Song shape.
 *
 * Returns null without `songId`: that field is the DB row id, and every action
 * here that reaches the server needs one. A queue item can be playable before
 * its row exists, and offering post-processing on it would 404.
 */
export function songFromQueueItem(item: {
  id: string;
  songId?: string;
  audioUrl?: string;
  masteredAudioUrl?: string;
  noAdapterAudioUrl?: string;
  coverUrl?: string;
  artistImageUrl?: string;
  audioDuration?: number;
  artistName?: string;
  generation?: { title?: string; caption?: string; lyrics?: string };
}): Song | null {
  if (!item.songId) return null;
  return {
    id: item.songId,
    title: item.generation?.title || 'Untitled',
    lyrics: item.generation?.lyrics || '',
    style: item.generation?.caption || '',
    caption: item.generation?.caption || '',
    audioUrl: item.audioUrl || '',
    audio_url: item.audioUrl || '',
    masteredAudioUrl: item.masteredAudioUrl || '',
    mastered_audio_url: item.masteredAudioUrl || '',
    noAdapterAudioUrl: item.noAdapterAudioUrl || '',
    coverUrl: item.coverUrl || item.artistImageUrl || '',
    duration: item.audioDuration || 0,
    artistName: item.artistName || '',
    tags: [],
  };
}

// ── Actions the component owns ───────────────────────────────────────────────

/** Download this song's params as a hot-step preset JSON. */
function exportParams(song: Song): void {
  const params = (song.generationParams || (song as any).generation_params || {}) as Record<string, any>;
  const exportData = {
    _format: 'hot-step-preset',
    _version: 1,
    ...params,
    title: song.title || '',
    caption: params.caption || song.style || '',
    lyrics: params.lyrics || song.lyrics || '',
  };
  saveJson(exportData, `${(song.title || 'song').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}_params.json`);
}

/** Download YuE2's ABC lead sheet for this song. */
function downloadScoreAbc(song: Song, abc: string): void {
  saveText(abc, `${(song.title || 'song').slice(0, 40).replace(/[^a-zA-Z0-9]/g, '_').toLowerCase()}_lead_sheet.abc`, 'text/vnd.abc');
}

/**
 * Save a new title, then announce it. Every list that caches the title (the
 * library and recent-songs queries, the playlist, the audio queue, the open
 * Song Details) listens for `song-renamed`, so the menu needs no host wiring.
 */
async function renameSong(song: Song, title: string): Promise<void> {
  try {
    await songApi.update(song.id, { title });
    emit('song-renamed', { songId: song.id, title });
  } catch (err) {
    toast.error(`Rename failed: ${errorMessage(err)}`);
  }
}

// ── Component ────────────────────────────────────────────────────────────────

export interface SongActionsMenuProps {
  song: Song;

  // Host-supplied actions. Each item is omitted when its handler is absent.
  onReuse?: () => void;
  onDownload?: () => void;
  onSendToCover?: () => void;
  onEditMetadata?: () => void;
  onAddToPlaylist?: () => void;
  onDelete?: () => void;

  /** Extra items appended before the destructive section — for surface-specific
   *  actions like the playlist's "Remove from playlist". */
  extraItems?: React.ReactNode;

  /** Hide the self-contained block (Export Params, Retranscribe, Cover Art,
   *  A/B) on surfaces where it is noise — a compact queue row, say. */
  compact?: boolean;

  /** Size of the trigger glyph. */
  size?: number;
  className?: string;
  /** Rendered instead of the default 3-dot button. */
  trigger?: React.ReactNode;
}

export const SongActionsMenu: React.FC<SongActionsMenuProps> = ({
  song,
  onReuse,
  onDownload,
  onSendToCover,
  onEditMetadata,
  onAddToPlaylist,
  onDelete,
  extraItems,
  compact = false,
  size = 16,
  className = '',
  trigger,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  /** The word-placement editor is a modal of its own, opened from this menu.
   *  The row's song and the lines it opened with are held from the click until
   *  the modal closes, so a list that renders a different track into this
   *  position cannot rebind an open editor to another file mid-edit. */
  const [editing, setEditing] = useState<{ song: Song; seed: EditLine[] | null } | null>(null);
  /** A retranscription started from this item, whose result opens the editor. */
  const [waitingForTimings, setWaitingForTimings] = useState(false);
  const queryClient = useQueryClient();
  const isMobile = useIsMobile();
  const btnRef = useRef<HTMLButtonElement>(null);
  const abTrackAId = useABCompareSelector(s => s.trackA?.id);
  const abTrackBId = useABCompareSelector(s => s.trackB?.id);
  const run = usePostProcessRun(song.id);
  // Whisper runs for a while; the store keeps the spinner alive after this menu
  // closes, and the outcome arrives as a toast (retranscribeStore).
  const { data: { transcribing }, start: startTranscribe } = useRetranscribe(song);

  const pp = canPostProcess(song);
  const ppBusy = !!run && (run.status === 'starting' || run.status === 'pending' || run.status === 'running');
  const ppFailed = run?.status === 'failed';
  const ppDone = run?.status === 'succeeded';

  // Rename edits in place inside the open menu, so no modal is needed.
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState('');
  // Enter closes the menu, and the input can then also blur: commit once.
  // Named apart from the word-placement `editing` state below.
  const renameCommitted = useRef(false);
  const close = () => { setOpen(false); setRenaming(false); };
  const commitRename = () => {
    if (!renameCommitted.current) return;
    renameCommitted.current = false;
    const title = draft.trim();
    close();
    if (title && title !== song.title) void renameSong(song, title);
  };
  const act = (fn: () => void) => { fn(); close(); };

  const gp = song.generationParams as any;
  const hasStoredScore = (song.backend ?? gp?.backend) === YUE2_BACKEND_ID && !!(gp?.yue2Abc ?? gp?.yue2Request?.abc);

  const hasMastered = !!(song.masteredAudioUrl || (song as any).mastered_audio_url);
  // Only YuE2 renders have a lead sheet; probe for it once the menu opens.
  const audioUrl = song.audioUrl || song.audio_url || '';
  const { data: scoreAbc } = useQuery({ ...scoreAbcOptions(audioUrl), enabled: open && !!audioUrl });
  // Only offer the .lrc export when the track actually has synced lyrics —
  // either the engine's LRC or the word-level JSON a retranscription writes.
  const { data: lrc } = useQuery({ ...lrcOptions(audioUrl), enabled: open && !!audioUrl });
  const { data: lyricsJson } = useQuery({ ...lyricsJsonOptions(audioUrl), enabled: open && !!audioUrl });
  const hasSyncedLyrics = !!lrc || !!lyricsJson;

  /** Cue the track and open the editor, seeded when there is no sidecar. */
  const openEditor = (seed: EditLine[] | null) => {
    // The editor's transport and waveform are the player's own, so the track
    // has to be in the element — loaded, not played. Cueing is what lets the
    // item work on a row nobody has listened to yet, and on the restored track
    // that a reload put in the bar without ever handing its file to the engine.
    cue(songToTrack(song));
    setEditing({ song, seed });
  };

  const handleEditWordPlacement = async () => {
    close();
    if (!audioUrl) {
      toast.error(t('songActions.editWordsNoAudio', 'This track has no audio file to place words against.'));
      return;
    }
    // Read the sidecar rather than trusting whatever the menu happened to have
    // fetched: it is the difference between opening an empty editor and saying
    // why there is nothing to edit.
    const data = await queryClient
      .ensureQueryData(lyricsJsonOptions(audioUrl))
      .catch(() => null);
    if (countSidecarWords(data) > 0) {
      openEditor(null);
      return;
    }

    // No word timings, but maybe line timings: the `.lrc` gives every line a
    // start, and that is enough to open with each line's words spread across it.
    const lrc = await queryClient.ensureQueryData(lrcOptions(audioUrl)).catch(() => null);
    const seed = seedLinesFromLrc(lrc ?? '', coerceDuration(song.duration) || 0);
    if (seed.length > 0) {
      openEditor(seed);
      return;
    }

    // Nothing on this track is timed at all. The lyrics exist, so what is
    // missing is *when*, and that is what a retranscription supplies: Whisper
    // hears the audio and the reconciliation keeps this track's lyrics as the
    // text, taking only the times from what it heard. So run it and open the
    // editor when it lands — stacking every line at 0:00 would be a worse lie
    // than waiting a minute, and sending the person to press another menu item
    // was a step for nobody.
    if (transcribing) {
      toast.info(
        t('songActions.editWordsAligning',
          'Whisper is already fitting this track’s lyrics to the audio. The editor opens as soon as the timings are written.'),
        { autoClose: 6000 },
      );
      return;
    }
    setWaitingForTimings(true);
    void startTranscribe();
  };

  /**
   * Open the editor once the transcription this item started lands.
   *
   * The store's flag goes false on failure too, so the sidecar read is what
   * separates "there are timings now" from "that did not work"; a failure has
   * already said why in its own toast, and opening an empty editor would just
   * repeat it.
   */
  useEffect(() => {
    if (!waitingForTimings || transcribing) return;
    void (async () => {
      // A session refresh can land between the run finishing and this read, and
      // one refused request is not "there are no timings": the sidecar is on
      // disk either way. A run that took a minute must not come out looking like
      // nothing happened because of an expired token, so try again before
      // deciding to keep the editor closed. Three tries, 1.2 s apart.
      for (let attempt = 0; attempt < 3; attempt++) {
        const data = await queryClient
          .ensureQueryData({ ...lyricsJsonOptions(audioUrl), staleTime: 0 })
          .catch(() => null);
        if (countSidecarWords(data) > 0) {
          // After the read, not before it: a state write in an effect's own body
          // makes React render twice for nothing.
          setWaitingForTimings(false);
          openEditor(null);
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 1200));
      }
      setWaitingForTimings(false);
    })();
    // The read is fresh because the store invalidates the sidecar queries before
    // it clears its flag, and `openEditor` acts on the row's own song.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [transcribing, waitingForTimings]);

  const handleRevert = async () => {
    close();
    try {
      await revertPostProcessing(song);
    } catch (err: any) {
      console.error('[PostProcess] Revert failed:', err.message);
    }
  };

  const handlePostProcess = async () => {
    close();
    try {
      await startPostProcessing(song);
    } catch (err: any) {
      // startPostProcessing has already parked the message on the run, which
      // the trigger badge surfaces. Nothing to do but keep it out of the void.
      console.error('[PostProcess]', err.message);
    }
  };

  // The trigger doubles as the progress indicator: this menu is the only place
  // a re-run can be started from, so it is the natural place to watch one.
  const triggerIcon = ppBusy
    ? <Loader2 size={size} className="animate-spin text-violet-400" />
    : ppFailed
      ? <AlertCircle size={size} className="text-red-400" />
      : <MoreHorizontal size={size} />;

  const triggerTitle = ppBusy
    ? `Post-processing: ${run?.stage || 'running'}`
    : ppFailed
      ? `Post-processing failed: ${run?.error || 'unknown error'}`
      : ppDone
        ? 'Post-processing complete'
        : 'Song actions';

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        title={triggerTitle}
        onClick={(e) => { e.stopPropagation(); setOpen(o => !o); }}
        className={`p-1.5 rounded-lg text-zinc-600 dark:text-zinc-400 hover:text-white hover:bg-white/10 transition-colors ${className}`}
      >
        {trigger ?? triggerIcon}
      </button>

      {open && (
        <PortalMenu anchorRef={btnRef} onClose={close} sheet={isMobile} title={song.title}>
          {/* Post-processing — the reason this menu exists. Deliberately first
              and deliberately shown (disabled) rather than hidden when it can't
              run, so "why can't I master this?" has an answer on hover. */}
          <Item
            icon={ppBusy ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
            label={ppBusy ? (run?.stage || 'Post-processing...') : t('songActions.postProcess', 'Run Post-Processing')}
            onClick={handlePostProcess}
            disabled={!pp.enabled}
            title={pp.enabled
              ? t('songActions.postProcessHint',
                  'Runs the full PP chain with your current settings, even if the global toggle is off')
              : pp.reason}
            tone="text-violet-400 hover:bg-violet-500/10"
          />
          {/* The way back out. Without it a single bad pass — wrong settings,
              or a chain that was half-configured — locks the track out of
              post-processing for good. */}
          {hasMastered && (
            <Item
              icon={<Undo2 size={14} />}
              label={t('songActions.removePostProcess', 'Remove Post-Processed Version')}
              onClick={handleRevert}
              title="Deletes the processed file so the chain can be run again. The raw render is untouched."
              tone="text-amber-400 hover:bg-amber-500/10"
            />
          )}
          {scoreAbc && (
            <Item
              icon={<FileMusic size={14} />}
              label={t('songActions.downloadLeadSheet', 'Download ABC Lead Sheet')}
              onClick={() => act(() => downloadScoreAbc(song, scoreAbc))}
              title={t('songActions.downloadLeadSheetHint', 'The ABC score YuE2 planned this track from')}
            />
          )}
          {ppFailed && (
            <Item
              icon={<AlertCircle size={14} />}
              label="Dismiss PP error"
              onClick={() => act(() => dismissRun(song.id))}
              title={run?.error}
              tone="text-red-400 hover:bg-red-500/10"
            />
          )}

          <Divider />

          {onReuse && (
            <Item icon={<RotateCcw size={14} />} label={t('library.edit')} onClick={() => act(onReuse)} />
          )}
          {onReuse && hasStoredScore && (
            <Item
              icon={<FileMusic size={14} />}
              label={t('songActions.reuseScore', 'Reuse score')}
              onClick={() => act(() => window.dispatchEvent(new CustomEvent('song-reuse-score', { detail: { song } })))}
              title={t('songActions.reuseScoreHint', 'Open Create with the ABC score of this track, so a new prompt, genre or singer keeps the same score')}
            />
          )}
          {onAddToPlaylist && (
            <Item
              icon={<ListPlus size={14} />}
              label={t('library.addToPlaylist')}
              onClick={() => act(onAddToPlaylist)}
              tone="text-pink-400 hover:bg-pink-500/10"
            />
          )}
          {onDownload && (
            <Item icon={<Download size={14} />} label={t('library.download')} onClick={() => act(onDownload)} />
          )}
          {onSendToCover && (
            <Item
              icon={<Disc3 size={14} />}
              label={t('library.sendToCover', 'Send to Cover Studio')}
              onClick={() => act(onSendToCover)}
              tone="text-cyan-400 hover:bg-cyan-500/10"
            />
          )}
          {renaming ? (
            <div className={`${ITEM_BASE} text-zinc-700 dark:text-zinc-300`}>
              <Pencil size={14} className="flex-shrink-0" />
              <input
                autoFocus
                value={draft}
                aria-label={t('library.rename', 'Rename')}
                onChange={(e) => setDraft(e.target.value)}
                onFocus={(e) => e.target.select()}
                // The menu portals out of the row, but React still bubbles
                // portal events to the row, which would start playback.
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => {
                  e.stopPropagation();
                  if (e.key === 'Enter') commitRename();
                  if (e.key === 'Escape') { renameCommitted.current = false; setRenaming(false); }
                }}
                onBlur={commitRename}
                className="flex-1 min-w-0 bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 rounded px-1.5 py-0.5 text-sm outline-none focus:border-violet-400"
              />
            </div>
          ) : (
            <Item
              icon={<Pencil size={14} />}
              label={t('library.rename', 'Rename')}
              onClick={() => { setDraft(song.title || ''); renameCommitted.current = true; setRenaming(true); }}
            />
          )}
          {onEditMetadata && (
            <Item
              icon={<Tags size={14} />}
              label={t('metadata.editTitle', 'Edit Metadata')}
              onClick={() => act(onEditMetadata)}
              tone="text-amber-400 hover:bg-amber-500/10"
            />
          )}

          {extraItems}

          {!compact && (
            <>
              <Divider />
              <Item
                icon={<Upload size={14} />}
                label={t('library.exportParams')}
                onClick={() => act(() => exportParams(song))}
              />
              <Item
                icon={<AudioLines size={14} />}
                label={t('songActions.editWordPlacement', 'Edit Word Placement')}
                onClick={() => { void handleEditWordPlacement(); }}
                title={t('songActions.editWordPlacementHint',
                  'Opens the lyric editor for this track: one chip per word, draggable, retypeable and checkable by ear. It starts on the track\u2019s word timings, on its .lrc line timings split into words when it has none, or — when nothing is timed yet — on a retranscription, which keeps your lyrics and takes only the timing from the audio. Opening it also puts the track in the player, paused.',)}
                tone="text-indigo-400 hover:bg-indigo-500/10"
              />
              <Item
                icon={transcribing ? <Loader2 size={14} className="animate-spin" /> : <Mic2 size={14} />}
                label={transcribing
                  ? t('songActions.retranscribing', 'Transcribing lyrics...')
                  : t('songActions.retranscribe', 'Retranscribe Lyrics')}
                onClick={() => act(() => { void startTranscribe(); })}
                disabled={transcribing}
                title={transcribing
                  ? 'Whisper is running on this track. The toast reports when it finishes.'
                  : 'Creates the word-level timings the lyric editor needs. Enable Whisper Lyrics before a render to get them automatically, or run this on a track that already exists. Once the timings are written, Edit appears in the Lyrics strip so they can be placed by hand.'}
                tone="text-sky-400 hover:bg-sky-500/10"
              />
              {hasSyncedLyrics && (
                <Item
                  icon={<FileText size={14} />}
                  label={t('songActions.downloadLrc', 'Download .LRC')}
                  onClick={() => act(() => { void downloadLyricsLrc(song); })}
                  title={t('songActions.downloadLrcHint',
                    'Synced lyrics with timestamps — for external players, DAWs and datasets')}
                  tone="text-emerald-400 hover:bg-emerald-500/10"
                />
              )}
              <Item
                icon={<Image size={14} />}
                label={song.coverUrl
                  ? t('songActions.regenerateCoverArt', 'Regenerate Cover Art')
                  : t('songActions.generateCoverArt', 'Generate Cover Art')}
                onClick={() => act(() => openCoverArtPrompt(song))}
                tone="text-violet-400 hover:bg-violet-500/10"
              />

              <Divider />
              <Item
                icon={<ArrowLeftRight size={14} />}
                label={t('songActions.setTrackA', 'Set as Track A')}
                onClick={() => act(() => setTrackA(songToTrack(song)))}
                tone={song.id === abTrackAId
                  ? 'text-blue-400 bg-blue-500/10'
                  : 'text-blue-400/70 hover:bg-blue-500/10 hover:text-blue-400'}
              />
              <Item
                icon={<ArrowLeftRight size={14} />}
                label={t('songActions.setTrackB', 'Set as Track B')}
                onClick={() => act(() => setTrackB(songToTrack(song)))}
                tone={song.id === abTrackBId
                  ? 'text-orange-400 bg-orange-500/10'
                  : 'text-orange-400/70 hover:bg-orange-500/10 hover:text-orange-400'}
              />
            </>
          )}

          {onDelete && (
            <>
              <Divider />
              <Item
                icon={<Trash2 size={14} />}
                label={t('library.delete')}
                onClick={() => act(onDelete)}
                tone="text-red-400 hover:bg-red-500/10"
              />
            </>
          )}
        </PortalMenu>
      )}

      {editing && (
        <WordPlacementEditor
          song={editing.song}
          audioUrl={editing.song.audioUrl || editing.song.audio_url || ''}
          seed={editing.seed}
          onClose={() => setEditing(null)}
        />
      )}
    </>
  );
};

export default SongActionsMenu;
