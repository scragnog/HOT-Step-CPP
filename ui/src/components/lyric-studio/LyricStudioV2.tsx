/**
 * LyricStudioV2.tsx — Main container for the Lyric Studio interface.
 *
 * Three navigation levels:
 *   1. Artist Grid with settings sidebar
 *   2. Album Grid for selected artist
 *   3. Album Detail with tabbed content (Source Lyrics, Profiles, Written Songs)
 *
 * State management: Query owns server snapshots; URL restore consumes the
 * artists observer once, with popstate support for browser back/forward.
 *
 * Ported from hot-step-9000 with import path + API adaptations for the C++ engine.
 */

import { useMutation, useQuery } from '@tanstack/react-query';
import { snapshotOptions } from '../../queries/cover-lyrics/reads';
import type { AlbumPreset } from '../../services/lireekApi';
import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { ChevronLeft } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { lireekApi } from '../../services/lireekApi';
import type { Artist, LyricsSet, Profile, Generation, SongLyric } from '../../services/lireekApi';
import { ArtistGrid } from './ArtistGrid';
import { ArtistSidebar } from './ArtistSidebar';
import { ArtistPageSidebar } from './ArtistPageSidebar';
import { AlbumGrid } from './AlbumGrid';
import { AlbumHeader } from './AlbumHeader';
import { FetchLyricsModal } from './FetchLyricsModal';
import { AddArtistModal } from './AddArtistModal';
import { AddAlbumModal } from './AddAlbumModal';
import { AddSongModal } from './AddSongModal';
import { CuratedProfileModal } from './CuratedProfileModal';
import { PresetSettingsModal } from './PresetSettingsModal';
import { ContentTabs } from './ContentTabs';
import type { TabId } from './ContentTabs';
import { SourceLyricsTab } from './SourceLyricsTab';
import { ProfilesTab } from './ProfilesTab';
import { WrittenSongsTab } from './WrittenSongsTab';
import { WriteLyricsTab } from './WriteLyricsTab';
import { RecordingsTab } from './RecordingsTab';
import { RenderAcrossModal } from './RenderAcrossModal';
import type { RenderTarget } from './RenderAcrossModal';
import { useAudioGeneration } from './useAudioGeneration';
import { enqueueAudioGen, useResumeQueue, useAudioGenQueueSelector } from '../../stores/audioGenQueueStore';
import { usePlaybackSelector } from '../../stores/playbackStore';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { QueuePanel } from './QueuePanel';
import { PromptEditor } from './PromptEditor';
import { restoreLyricQueue } from '../../stores/streamingStore';
import { loadSelections, saveSelections, type ModelSelections } from './ProviderSelector';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';
import { cacheMm3SourceTracks, collectMm3SourceTracks } from '../../utils/mm3CaptionSource';

// ── URL helpers ──────────────────────────────────────────────────────────────

const attemptedArtwork = new Set<string>();
async function fillMissingArtwork(items: Array<{ id: number; image_url?: string }>, kind: 'artist' | 'album') {
  for (const item of items) {
    const key = `${kind}:${item.id}`;
    if (item.image_url || attemptedArtwork.has(key)) continue;
    attemptedArtwork.add(key);
    try {
      if (kind === 'artist') await lireekApi.refreshArtistImage(item.id);
      else await lireekApi.refreshAlbumImage(item.id);
    } catch { /* Artwork is advisory; explicit refresh remains available. */ }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}

const LS_BASE = '/lyric-studio';

function buildUrl(artistId?: number, albumId?: number, tab?: TabId): string {
  if (artistId && albumId && tab) return `${LS_BASE}/artist/${artistId}/album/${albumId}/${tab}`;
  if (artistId && albumId) return `${LS_BASE}/artist/${artistId}/album/${albumId}`;
  if (artistId) return `${LS_BASE}/artist/${artistId}`;
  return LS_BASE;
}

function parseUrl(path: string): { artistId?: number; albumId?: number; tab?: TabId } {
  const m = path.match(/\/lyric-studio\/artist\/(\d+)(?:\/album\/(\d+)(?:\/(source-lyrics|profiles|write-lyrics|written-songs|recordings))?)?/);
  if (!m) return {};
  return {
    artistId: Number(m[1]),
    albumId: m[2] ? Number(m[2]) : undefined,
    tab: (m[3] as TabId) || undefined,
  };
}

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

// ── Navigation state ─────────────────────────────────────────────────────────

type NavLevel = 'artists' | 'albums' | 'album-detail';

interface NavState {
  level: NavLevel;
  selectedArtist: Artist | null;
  selectedAlbum: LyricsSet | null;
}

// ── Main Component ──────────────────────────────────────────────────────────

export const LyricStudioV2: React.FC = () => {
  const { token } = useAuth();
  const { t } = useTranslation();
  const { disguiseArtist } = useDisguiseMode();

  // ── Navigation ──
  const [nav, setNav] = useState<NavState>({ level: 'artists', selectedArtist: null, selectedAlbum: null });

  // ── Data ──
  const artistsQuery = useQuery(snapshotOptions<Artist[]>('lireek', '/lireek/artists'));
  const albumListPath = `/lireek/lyrics-sets?artist_id=${nav.selectedArtist?.id ?? ''}`;
  const albumsQuery = useQuery({ ...snapshotOptions<LyricsSet[]>('lireek', albumListPath), enabled: !!nav.selectedArtist });
  const detailPath = `/lireek/lyrics-sets/${nav.selectedAlbum?.id ?? ''}/full-detail`;
  const detailQuery = useQuery({ ...snapshotOptions<{ lyrics_set: LyricsSet; profiles: Profile[]; generations: Generation[] }>('lireek', detailPath), enabled: !!nav.selectedAlbum });
  const artists = artistsQuery.data ?? [];
  const albums = nav.selectedArtist ? albumsQuery.data ?? [] : [];
  const profiles = nav.selectedAlbum ? detailQuery.data?.profiles ?? [] : [];
  const generations = nav.selectedAlbum ? detailQuery.data?.generations ?? [] : [];
  const artistsLoading = artistsQuery.isPending;
  const albumsLoading = albumsQuery.isPending && !!nav.selectedArtist;
  useEffect(() => {
    const album = detailQuery.data?.lyrics_set;
    if (album) setNav(prev => prev.selectedAlbum?.id === album.id ? { ...prev, selectedAlbum: album } : prev);
  }, [detailQuery.data]);

  // ── Tabs ──
  const [activeTab, setActiveTab] = useState<TabId>('source-lyrics');
  const [recordingsFilter, setRecordingsFilter] = useState<number | null>(null);
  const [songCount, setSongCount] = useState(0);
  const [recordingsRefreshKey, setRecordingsRefreshKey] = useState(0);

  const isRestoringUrl = useRef(false);

  // ── Modals ──
  const [fetchModalOpen, setFetchModalOpen] = useState(false);
  const [fetchModalPrefill, setFetchModalPrefill] = useState<string | undefined>();
  const [presetModalOpen, setPresetModalOpen] = useState(false);
  const [queueOpen, setQueueOpen] = useState(false);
  // Single source of truth for the provider/model picker: AlbumHeader and
  // ArtistPageSidebar used to each keep their own copy loaded from
  // localStorage, so picking a provider there never re-rendered this
  // component — the tabs/QueuePanel below kept submitting whatever
  // loadSelections() had returned on LyricStudioV2's last unrelated render.
  const [modelSelections, setModelSelections] = useState<ModelSelections>(loadSelections);
  const handleSelectionsChange = useCallback((sel: ModelSelections) => {
    setModelSelections(sel);
    saveSelections(sel);
  }, []);
  const [promptEditorOpen, setPromptEditorOpen] = useState(false);
  const [addArtistModalOpen, setAddArtistModalOpen] = useState(false);
  const [addAlbumModalOpen, setAddAlbumModalOpen] = useState(false);
  const [addSongModalOpen, setAddSongModalOpen] = useState(false);
  const [curatedModalOpen, setCuratedModalOpen] = useState(false);

  // ── Fetch lyrics progress ──
  // Derived from the mutation: the label comes from the variables that were
  // actually submitted, so a second request cannot show the first one's name.

  // ── Bulk queue data ──
  const allSetsQuery = useQuery({ ...snapshotOptions<LyricsSet[]>('lireek', '/lireek/lyrics-sets'), enabled: queueOpen });
  const allProfilesQuery = useQuery({ ...snapshotOptions<Profile[]>('lireek', '/lireek/profiles'), enabled: queueOpen });
  const allLyricsSets = allSetsQuery.data ?? [];
  const allProfiles = allProfilesQuery.data ?? [];

  const presetsQuery = useQuery(snapshotOptions<{ presets: Array<AlbumPreset & { artist_id?: number }> }>('lireek', '/lireek/presets'));
  const artistIdsWithAdapters = new Set((presetsQuery.data?.presets ?? []).filter(p => p.adapter_path && p.artist_id).map(p => p.artist_id!));

  // ── Playback (for backdrop effect) ──
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  const currentPlaybackTrack = usePlaybackSelector(s => s.currentTrack);

  // ── Audio generation ──
  useResumeQueue(token || undefined);
  useEffect(() => { if (token) void restoreLyricQueue(token).catch(console.error); }, [token]);
  const completionCounter = useAudioGenQueueSelector(s => s.completionCounter);

  // ── Toast ──
  const [toast, setToast] = useState<string | null>(null);
  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  }, []);

  // ── Initial artwork + URL restore from the artists observer ──
  const initialLoad = useRef({ path: window.location.pathname, artworkStarted: false, restored: false });
  useEffect(() => {
    const artistsList = artistsQuery.data;
    if (!artistsList || initialLoad.current.restored) return;
    if (!initialLoad.current.artworkStarted) {
      initialLoad.current.artworkStarted = true;
      void fillMissingArtwork(artistsList, 'artist');
    }
    let live = true;
    const initialPath = initialLoad.current.path;
    const init = async () => {
      if (!live || initialPath !== window.location.pathname) return;
      const parsed = parseUrl(initialPath);
      if (parsed.artistId) {
        isRestoringUrl.current = true;
        try {
          const artist = artistsList.find(a => a.id === parsed.artistId);
          if (!artist) {
            initialLoad.current.restored = true;
            return;
          }
          const albumRes = await lireekApi.listLyricsSets(artist.id);
          if (!live || initialPath !== window.location.pathname) return;
          initialLoad.current.restored = true;

          if (parsed.albumId) {
            const album = albumRes.lyrics_sets.find(a => a.id === parsed.albumId);
            if (!album) {
              setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
              return;
            }
            setNav({ level: 'album-detail', selectedArtist: artist, selectedAlbum: album });
            if (parsed.tab) setActiveTab(parsed.tab);
          } else {
            setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
          }
        } finally {
          isRestoringUrl.current = false;
        }
      } else {
        initialLoad.current.restored = true;
      }
    };
    void init().catch(console.error);
    return () => { live = false; };
  }, [artistsQuery.data]);

  // ── Load albums ──
  const loadAlbums = useCallback(async (artistId: number) => {
    const { lyrics_sets } = await lireekApi.listLyricsSets(artistId);
    void fillMissingArtwork(lyrics_sets.filter(album => !!album.album), 'album');
  }, []);

  // ── Load album detail data ──
  const loadAlbumData = useCallback(async (albumId: number) => {
    try { await lireekApi.getAlbumFullDetail(albumId); }
    catch (err) { console.warn('[LyricStudioV2] Failed to load album:', err); }
  }, []);

  // ── Navigation handlers ──
  const pushUrl = useCallback((artistId?: number, albumId?: number, tab?: TabId) => {
    if (isRestoringUrl.current) return;
    const url = buildUrl(artistId, albumId, tab);
    if (window.location.pathname !== url) window.history.pushState({}, '', url);
  }, []);

  const handleSelectArtist = useCallback((artist: Artist) => {
    setSongCount(0);
    setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
    loadAlbums(artist.id);
    pushUrl(artist.id);
  }, [loadAlbums, pushUrl]);

  const handleSelectAlbum = useCallback((album: LyricsSet) => {
    setNav(prev => ({ ...prev, level: 'album-detail', selectedAlbum: album }));
    setActiveTab('source-lyrics');
    pushUrl(nav.selectedArtist?.id, album.id, 'source-lyrics');
  }, [pushUrl, nav.selectedArtist]);

  // Reactive album data loading
  const albumIdRef = useRef<number | null>(null);
  useEffect(() => {
    const albumId = nav.selectedAlbum?.id ?? null;
    if (albumId === albumIdRef.current) return;
    albumIdRef.current = albumId;
    if (albumId == null) return;

    setSongCount(0);
    loadAlbumData(albumId);
  }, [nav.selectedAlbum?.id, loadAlbumData]);

  const handleBackToArtists = useCallback(() => {
    setNav({ level: 'artists', selectedArtist: null, selectedAlbum: null });
    setSongCount(0);
    pushUrl();
  }, [pushUrl]);

  const handleBackToAlbums = useCallback(() => {
    setNav(prev => ({ ...prev, level: 'albums', selectedAlbum: null }));
    setSongCount(0);
    pushUrl(nav.selectedArtist?.id);
  }, [pushUrl, nav.selectedArtist]);

  const handleTabChange = useCallback((tab: TabId) => {
    setActiveTab(tab);
    pushUrl(nav.selectedArtist?.id, nav.selectedAlbum?.id, tab);
  }, [pushUrl, nav.selectedArtist, nav.selectedAlbum]);

  // ── popstate ──
  useEffect(() => {
    const handlePopState = async () => {
      const parsed = parseUrl(window.location.pathname);
      if (!parsed.artistId) {
        setNav({ level: 'artists', selectedArtist: null, selectedAlbum: null });
         setSongCount(0);
        return;
      }
      const artist = artists.find(a => a.id === parsed.artistId);
      if (!artist) return;
      if (!parsed.albumId) {
        setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
        setSongCount(0);
        loadAlbums(artist.id);
        return;
      }
      const album = albums.find(a => a.id === parsed.albumId);
      if (album) {
        setNav({ level: 'album-detail', selectedArtist: artist, selectedAlbum: album });
        if (parsed.tab) setActiveTab(parsed.tab);
      }
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [artists, albums, loadAlbums]);

  // ── Actions ──
  // Every command below is a `useMutation`: pending state and double-click
  // guards come from `isPending`, and failures keep the transport as their one
  // notifier (lireekApi invalidates the affected reads on write).
  const deleteArtistMut = useMutation({
    mutationFn: (artist: Artist) => lireekApi.deleteArtist(artist.id),
    onSuccess: (_res, artist) => {
      showToast(`Deleted ${artist.name}`);
    },
  });

  const refreshArtistImageMut = useMutation({
    mutationFn: (artist: Artist) => lireekApi.refreshArtistImage(artist.id),
    onSuccess: (_res, artist) => showToast(`Updated image for ${artist.name}`),
  });

  const setArtistImageMut = useMutation({
    mutationFn: ({ artist, url }: { artist: Artist; url: string }) => lireekApi.setArtistImage(artist.id, url),
    onSuccess: (_res, { artist }) => showToast(`Image updated for ${artist.name}`),
  });

  const deleteAlbumMut = useMutation({
    mutationFn: (album: LyricsSet) => lireekApi.deleteLyricsSet(album.id),
    onSuccess: () => {
      showToast('Album deleted');
      if (nav.selectedArtist) void loadAlbums(nav.selectedArtist.id);
    },
  });

  const refreshAlbumImageMut = useMutation({
    mutationFn: (album: LyricsSet) => lireekApi.refreshAlbumImage(album.id),
    onSuccess: () => showToast('Album image updated'),
  });

  const setAlbumImageMut = useMutation({
    mutationFn: ({ album, url }: { album: LyricsSet; url: string }) => lireekApi.setAlbumImage(album.id, url),
    onSuccess: () => showToast('Album image set'),
  });

  const deleteSongMut = useMutation({
    mutationFn: ({ albumId, index }: { albumId: number; index: number }) => lireekApi.removeSong(albumId, index),
    onSuccess: async (_res, { albumId }) => {
      showToast('Song removed');
      const updated = await lireekApi.getLyricsSet(albumId);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
    },
  });

  const editSongMut = useMutation({
    mutationFn: ({ albumId, index, lyrics }: { albumId: number; index: number; lyrics: string }) =>
      lireekApi.editSong(albumId, index, lyrics),
    onSuccess: (updated) => {
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
      showToast('Lyrics updated');
    },
  });

  const fetchLyricsMut = useMutation({
    mutationFn: ({ artist, album, maxSongs }: { artist: string; album: string; maxSongs: number }) =>
      lireekApi.fetchLyrics({ artist, album: album || undefined, max_songs: maxSongs }),
    onSuccess: async (res) => {
      showToast(`Fetched ${res.songs_fetched} songs`);
      if (nav.selectedArtist && res.artist.id === nav.selectedArtist.id) {
        await loadAlbums(nav.selectedArtist.id);
      }
      if (nav.level === 'artists') handleSelectArtist(res.artist);
    },
  });

  const fetchingLyrics = fetchLyricsMut.isPending;
  const fetchingLabel = fetchLyricsMut.variables
    ? `${fetchLyricsMut.variables.artist}${fetchLyricsMut.variables.album ? ` — ${fetchLyricsMut.variables.album}` : ''}`
    : '';

  const handleDeleteArtist = useCallback((artist: Artist) => {
    if (!confirm(`Delete ${artist.name} and ALL their albums, profiles, and generations?`)) return;
    deleteArtistMut.mutate(artist);
  }, [deleteArtistMut]);

  const handleRefreshImage = useCallback((artist: Artist) => {
    showToast(`Refreshing image for ${artist.name}...`);
    refreshArtistImageMut.mutate(artist);
  }, [refreshArtistImageMut, showToast]);

  const handleSetImage = useCallback((artist: Artist, url: string) => {
    setArtistImageMut.mutate({ artist, url });
  }, [setArtistImageMut]);

  const handleDeleteAlbum = useCallback((album: LyricsSet) => {
    if (!confirm(`Delete album "${album.album || 'Top Songs'}" and all associated data?`)) return;
    deleteAlbumMut.mutate(album);
  }, [deleteAlbumMut]);

  const handleRefreshAlbumImage = useCallback((album: LyricsSet) => {
    refreshAlbumImageMut.mutate(album);
  }, [refreshAlbumImageMut]);

  const handleSetAlbumImage = useCallback((album: LyricsSet, url: string) => {
    setAlbumImageMut.mutate({ album, url });
  }, [setAlbumImageMut]);

  const handleDeleteSong = useCallback((index: number) => {
    if (!nav.selectedAlbum) return;
    deleteSongMut.mutate({ albumId: nav.selectedAlbum.id, index });
  }, [nav.selectedAlbum, deleteSongMut]);

  const handleEditSong = useCallback((index: number, lyrics: string) => {
    if (!nav.selectedAlbum) return;
    editSongMut.mutate({ albumId: nav.selectedAlbum.id, index, lyrics });
  }, [nav.selectedAlbum, editSongMut]);

  const handleFetchLyrics = useCallback(async (artist: string, album: string, maxSongs: number) => {
    showToast(`Fetching lyrics for ${artist}${album ? ` — ${album}` : ''}…`);
    try { await fetchLyricsMut.mutateAsync({ artist, album, maxSongs }); } catch { /* interceptor notified */ }
  }, [fetchLyricsMut, showToast]);

  const refreshAlbumData = useCallback(() => {
    if (nav.selectedAlbum) loadAlbumData(nav.selectedAlbum.id);
  }, [nav.selectedAlbum, loadAlbumData]);

  // ── Audio generation ──
  const { sendToCreate } = useAudioGeneration({ profiles, showToast });
  const globalParams = useGlobalParamsStore();

  /** `renderAs`: render with another album's preset and caption source, and
   *  list the recording there ("Render as" / "Render from"). */
  const handleGenerateAudio = useCallback(async (gen: Generation, renderAs?: RenderTarget) => {
    if (!token) { showToast('Not authenticated'); return; }
    const profile = profiles.find(p => p.id === gen.profile_id);
    const sourceLyricsSetId = profile?.lyrics_set_id ?? gen.lyrics_set_id;
    if (!renderAs && !profile) { showToast('Profile not found'); return; }
    // Capture globalParams snapshot NOW — same as Create page's getGlobalParams().
    // This ensures every engine param (solver, guidance, DCW, latent, LM, etc.)
    // flows through identically to the Create page path.
    const paramsSnapshot = globalParams.getGlobalParams();
    await enqueueAudioGen(gen, renderAs ? {
      ...renderAs,
      profileId: gen.profile_id,
      sourceLyricsSetId,
    } : {
      artistId: nav.selectedArtist?.id || 0,
      artistName: nav.selectedArtist?.name || 'Unknown',
      artistImageUrl: nav.selectedArtist?.image_url || '',
      profileId: profile!.id,
      lyricsSetId: profile!.lyrics_set_id,
    }, paramsSnapshot, token);
    showToast(renderAs
      ? `Queued: ${gen.title || 'Untitled'} as ${disguiseArtist(renderAs.artistName)}`
      : `Queued: ${gen.title || 'Untitled'}`);
  }, [token, profiles, nav.selectedArtist, globalParams, showToast, disguiseArtist]);

  // ── Render as / Render from ──
  const [renderAcross, setRenderAcross] = useState<{ mode: 'as' | 'from'; gen?: Generation } | null>(null);
  const currentTarget = useMemo<RenderTarget | null>(() => (nav.selectedArtist && nav.selectedAlbum ? {
    lyricsSetId: nav.selectedAlbum.id,
    artistId: nav.selectedArtist.id,
    artistName: nav.selectedArtist.name,
    artistImageUrl: nav.selectedArtist.image_url || '',
  } : null), [nav.selectedArtist, nav.selectedAlbum]);

  // Refresh album data on audio queue completions
  useEffect(() => {
    if (completionCounter > 0) {
      refreshAlbumData();
      setRecordingsRefreshKey(k => k + 1);
    }
  }, [completionCounter]);

  const handleSendToCreate = useCallback(async (gen: Generation, renderAs?: RenderTarget) => {
    // Inject artist name — gen from getAlbumFullDetail doesn't include it
    const enriched = { ...gen, artist_name: gen.artist_name || nav.selectedArtist?.name || '' };
    await sendToCreate(enriched, renderAs);
  }, [sendToCreate, nav.selectedArtist]);

  const openFetchForArtist = useCallback(() => {
    setFetchModalPrefill(nav.selectedArtist?.name);
    setFetchModalOpen(true);
  }, [nav.selectedArtist]);

  const openFetchNew = useCallback(() => {
    setFetchModalPrefill(undefined);
    setFetchModalOpen(true);
  }, []);

  // ── Manual add handlers ──
  // The modals await these, so each handler returns a Promise; `mutateAsync`
  // keeps that contract while the mutation owns pending/error state.
  const addArtistMut = useMutation({
    mutationFn: ({ name, imageUrl }: { name: string; imageUrl?: string }) =>
      lireekApi.createArtist({ name, image_url: imageUrl }),
    onSuccess: (res) => {
      showToast(`Added ${res.artist.name}`);
      handleSelectArtist(res.artist);
    },
  });

  const addAlbumMut = useMutation({
    mutationFn: ({ albumName, imageUrl }: { albumName?: string; imageUrl?: string }) =>
      lireekApi.createLyricsSet({ artist_id: nav.selectedArtist!.id, album: albumName, image_url: imageUrl }),
    onSuccess: async (res, { albumName }) => {
      showToast(`Created ${albumName || 'lyrics collection'}`);
      if (nav.selectedArtist) await loadAlbums(nav.selectedArtist.id);
      handleSelectAlbum(res.lyrics_set);
    },
  });

  const addSongMut = useMutation({
    mutationFn: ({ albumId, title, lyrics }: { albumId: number; title: string; lyrics: string }) =>
      lireekApi.addSongToSet(albumId, { title, lyrics }),
    onSuccess: (updated, { title }) => {
      showToast(`Added "${title}"`);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
    },
  });

  const handleAddArtistManual = useCallback(async (name: string, imageUrl?: string) => {
    try { await addArtistMut.mutateAsync({ name, imageUrl }); } catch { /* interceptor notified */ }
  }, [addArtistMut]);

  const handleAddAlbumManual = useCallback(async (albumName: string | undefined, imageUrl?: string) => {
    if (!nav.selectedArtist) return;
    try { await addAlbumMut.mutateAsync({ albumName, imageUrl }); } catch { /* interceptor notified */ }
  }, [nav.selectedArtist, addAlbumMut]);

  const handleAddSong = useCallback(async (title: string, lyrics: string) => {
    if (!nav.selectedAlbum) return;
    try { await addSongMut.mutateAsync({ albumId: nav.selectedAlbum.id, title, lyrics }); } catch { /* interceptor notified */ }
  }, [nav.selectedAlbum, addSongMut]);

  const handleCuratedComplete = useCallback(async (lyricsSet: any) => {
    if (nav.selectedArtist) await loadAlbums(nav.selectedArtist.id);
    handleSelectAlbum(lyricsSet);
  }, [nav.selectedArtist, loadAlbums, handleSelectAlbum]);

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (fetchModalOpen) setFetchModalOpen(false);
        else if (nav.level === 'album-detail') handleBackToAlbums();
        else if (nav.level === 'albums') handleBackToArtists();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [nav.level, fetchModalOpen, handleBackToAlbums, handleBackToArtists]);

  const sourceLyricsCount = nav.selectedAlbum ? parseSongs(nav.selectedAlbum.songs).length : 0;

  // ── MM3 caption sources ──
  // The album's own source tracks that carry an MM3 Structured Caption. A
  // written song renders under ONE of these by default (nearest tempo), which
  // is what reliably lands in the band's style and reaches a natural ending.
  // Cached to localStorage because the two render paths — the audio queue and
  // Send-to-Create — are plain modules with no access to this component's data.
  const mm3SourceTracks = useMemo(
    () => (nav.selectedAlbum ? collectMm3SourceTracks(parseSongs(nav.selectedAlbum.songs)) : []),
    [nav.selectedAlbum],
  );
  useEffect(() => {
    if (nav.selectedAlbum) cacheMm3SourceTracks(nav.selectedAlbum.id, mm3SourceTracks);
  }, [nav.selectedAlbum?.id, mm3SourceTracks]);

  // Queue open helper
  const openQueuePanel = useCallback(async () => {
    try {
      await Promise.all([lireekApi.listLyricsSets(), lireekApi.listProfiles()]);
    } catch (err) { console.error('[LyricStudioV2] Failed to load queue data:', err); }
    setQueueOpen(true);
  }, []);

  // ── Render ──
  return (
    <div className="h-full w-full flex flex-col relative bg-white dark:bg-zinc-950">
      {/* Toast */}
      {toast && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 px-5 py-2.5 rounded-xl bg-zinc-100/90 dark:bg-zinc-800/90 backdrop-blur-sm border border-zinc-300 dark:border-white/10 text-sm text-white shadow-2xl ls2-slide-up">
          {toast}
        </div>
      )}

      {/* Fetch-lyrics indicator */}
      {fetchingLyrics && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-5 py-2.5 rounded-xl bg-pink-950/80 backdrop-blur-sm border border-pink-500/20 text-sm text-pink-200 shadow-2xl ls2-slide-up">
          <svg className="w-4 h-4 animate-spin flex-shrink-0" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <span>{t('lyric.fetchingLyrics')} <strong className="text-white">{fetchingLabel}</strong>…</span>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        {nav.level === 'artists' && (
          <div className="h-full flex ls2-fade-in">
            <div className="w-48 flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-hidden max-[1023px]:hidden portrait-wide:hidden">
              <ArtistSidebar artists={artists} selectedArtistId={-1}
                onSelectArtist={handleSelectArtist} onBack={handleBackToArtists}
                artistIdsWithAdapters={artistIdsWithAdapters} />
            </div>
            <div className="flex-1 flex flex-col overflow-hidden min-w-0">
              {/* Header bar — spans above ArtistPageSidebar + Grid, matches ContentTabs height */}
              <div className="flex-shrink-0 flex items-center px-5 py-3 border-b border-zinc-200 dark:border-white/5 bg-zinc-50/30 dark:bg-zinc-950/30">
                <span className="text-sm font-medium text-zinc-600 dark:text-zinc-400">{t('lyric.allArtists')}</span>
                {artists.length > 0 && (
                  <span className="ml-2 min-w-[20px] h-5 px-1.5 rounded-full text-[11px] font-bold flex items-center justify-center bg-white/10 text-zinc-600 dark:text-zinc-400">
                    {artists.length}
                  </span>
                )}
                {/* Narrow/portrait: queue + prompt editor buttons since sidebar is hidden */}
                <div className="max-[1023px]:flex portrait-wide:flex hidden items-center gap-2 ml-auto">
                  <button onClick={openQueuePanel} className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors">
                    {t('lyric.queueButton')}
                  </button>
                  <button onClick={() => setPromptEditorOpen(true)} className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors">
                    Prompts
                  </button>
                </div>
              </div>
              <div className="flex-1 flex min-h-0">
                <div className="w-64 flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-hidden max-[1023px]:hidden portrait-wide:hidden">
                  <ArtistPageSidebar onOpenQueue={openQueuePanel} onOpenPromptEditor={() => setPromptEditorOpen(true)}
                    modelSelections={modelSelections} onModelSelectionsChange={handleSelectionsChange} />
                </div>
                <div className="flex-1 overflow-y-auto">
                  <ArtistGrid
                    artists={artists} loading={artistsLoading}
                    onSelectArtist={handleSelectArtist} onAddNew={openFetchNew}
                    onAddManual={() => setAddArtistModalOpen(true)}
                    onDelete={handleDeleteArtist} onRefreshImage={handleRefreshImage} onSetImage={handleSetImage}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {nav.level === 'albums' && nav.selectedArtist && (
          <div className="h-full flex ls2-fade-in">
            <div className="w-48 flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-hidden max-[1023px]:hidden portrait-wide:hidden">
              <ArtistSidebar artists={artists} selectedArtistId={nav.selectedArtist.id}
                onSelectArtist={handleSelectArtist} onBack={handleBackToArtists}
                artistIdsWithAdapters={artistIdsWithAdapters} />
            </div>
            <div className="flex-1 flex flex-col overflow-hidden min-w-0">
              {/* Header bar — spans above ArtistPageSidebar + Grid, matches ContentTabs height */}
              <div className="flex-shrink-0 flex items-center px-5 py-3 border-b border-zinc-200 dark:border-white/5 bg-zinc-50/30 dark:bg-zinc-950/30">
                {/* Narrow/portrait: back button to artists */}
                <button onClick={handleBackToArtists} className="max-[1023px]:flex portrait-wide:flex hidden items-center gap-1 mr-3 text-xs text-zinc-500 hover:text-white transition-colors flex-shrink-0">
                  <ChevronLeft className="w-3.5 h-3.5" />
                  All Artists
                </button>
                <span className="text-sm font-medium text-zinc-600 dark:text-zinc-400 truncate min-w-0">{disguiseArtist(nav.selectedArtist.name)}</span>
                {albums.length > 0 && (
                  <span className="ml-2 min-w-[20px] h-5 px-1.5 rounded-full text-[11px] font-bold flex items-center justify-center bg-white/10 text-zinc-600 dark:text-zinc-400 flex-shrink-0">
                    {albums.length}
                  </span>
                )}
              </div>
              <div className="flex-1 flex min-h-0">
                <div className="w-64 flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-hidden max-[1023px]:hidden portrait-wide:hidden">
                  <ArtistPageSidebar artist={nav.selectedArtist} albumCount={albums.length}
                    onOpenQueue={openQueuePanel} onOpenPromptEditor={() => setPromptEditorOpen(true)}
                    modelSelections={modelSelections} onModelSelectionsChange={handleSelectionsChange} />
                </div>
                <div className="flex-1 overflow-y-auto">
                  <AlbumGrid
                    albums={albums} loading={albumsLoading} artistName={nav.selectedArtist.name}
                    onSelectAlbum={handleSelectAlbum} onAddAlbum={openFetchForArtist}
                    onAddManual={() => setAddAlbumModalOpen(true)}
                    onDeleteAlbum={handleDeleteAlbum} onRefreshImage={handleRefreshAlbumImage}
                    onSetImage={handleSetAlbumImage} onCuratedProfile={() => setCuratedModalOpen(true)}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {nav.level === 'album-detail' && nav.selectedArtist && nav.selectedAlbum && (
          <div className="h-full flex flex-col ls2-fade-in">
            {/* Narrow/portrait nav bar: back to albums + album name */}
            <div className="max-[1023px]:flex portrait-wide:flex hidden flex-shrink-0 items-center gap-2 px-3 py-2 border-b border-zinc-200 dark:border-white/5 bg-zinc-50/30 dark:bg-zinc-950/30">
              <button onClick={handleBackToAlbums} className="flex items-center gap-1 text-xs text-zinc-500 hover:text-white transition-colors flex-shrink-0">
                <ChevronLeft className="w-3.5 h-3.5" />
                Albums
              </button>
              <span className="text-zinc-600 dark:text-zinc-600 text-xs flex-shrink-0">/</span>
              <span className="text-xs text-zinc-600 dark:text-zinc-400 truncate min-w-0">{nav.selectedAlbum.album || 'Top Songs'}</span>
            </div>
            <div className="flex-1 flex min-h-0">
              {/* Left: artist sidebar + album header */}
              <div className="w-48 flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-hidden max-[1023px]:hidden portrait-wide:hidden">
                <ArtistSidebar artists={artists} selectedArtistId={nav.selectedArtist.id}
                  onSelectArtist={handleSelectArtist} onBack={handleBackToArtists}
                  artistIdsWithAdapters={artistIdsWithAdapters} />
              </div>
              <div className="w-64 flex-shrink-0 border-r border-zinc-200 dark:border-white/5 overflow-hidden relative max-[1023px]:hidden portrait-wide:hidden">
                <div className="relative z-[1] h-full">
                  <AlbumHeader
                    artist={nav.selectedArtist} album={nav.selectedAlbum}
                    onBack={handleBackToAlbums} onOpenPreset={() => setPresetModalOpen(true)}
                    profileCount={profiles.length} generationCount={generations.length} songCount={songCount}
                    modelSelections={modelSelections} onModelSelectionsChange={handleSelectionsChange}
                  />
                </div>
              </div>

              {/* Middle: tabbed content */}
              <div className="flex-1 overflow-hidden relative min-w-0">
                {/* Cover art backdrop when playing */}
                {isPlaying && currentPlaybackTrack?.coverUrl && (
                  <>
                    <style>{`
                      @keyframes ls-random-zoom { 0%, 100% { scale: 1.4; } 50% { scale: 1.6; } }
                      @keyframes ls-random-rotate { 0%, 100% { rotate: -5deg; } 25% { rotate: 15deg; } 50% { rotate: 2deg; } 75% { rotate: -15deg; } }
                      @keyframes ls-random-pan { 0%, 100% { translate: 0% 0%; } 20% { translate: -5% 4%; } 40% { translate: 6% -5%; } 60% { translate: -4% -6%; } 80% { translate: 5% 5%; } }
                      .ls-dynamic-backdrop { animation: ls-random-zoom 47s ease-in-out infinite, ls-random-rotate 61s ease-in-out infinite, ls-random-pan 53s ease-in-out infinite; }
                    `}</style>
                    <div className="absolute inset-0 z-0 pointer-events-none transition-[background-image] duration-700 ls-dynamic-backdrop"
                      style={{ backgroundImage: `url(${currentPlaybackTrack!.coverUrl})`, backgroundSize: 'cover', backgroundPosition: 'center', filter: 'brightness(0.15) blur(2px) saturate(1.4)' }} />
                  </>
                )}
                <div className="relative z-[1] h-full">
                  <ContentTabs activeTab={activeTab} onTabChange={handleTabChange}
                    sourceLyricsCount={sourceLyricsCount} profilesCount={profiles.length}
                    writtenSongsCount={generations.length} recordingsCount={songCount}>
                    {activeTab === 'source-lyrics' && (
                      <SourceLyricsTab album={nav.selectedAlbum} onDeleteSong={handleDeleteSong}
                        onEditSong={handleEditSong} onAddSong={() => setAddSongModalOpen(true)} />
                    )}
                    {activeTab === 'profiles' && (
                      <ProfilesTab lyricsSetId={nav.selectedAlbum.id} profiles={profiles}
                        onRefresh={refreshAlbumData} showToast={showToast}
                        profilingModel={modelSelections.profiling} />
                    )}
                    {activeTab === 'write-lyrics' && (
                      <WriteLyricsTab key={nav.selectedAlbum.id} lyricsSetId={nav.selectedAlbum.id}
                        onRefresh={refreshAlbumData} onViewLyrics={() => handleTabChange('written-songs')} />
                    )}
                    {activeTab === 'written-songs' && (
                      <WrittenSongsTab generations={generations} profiles={profiles}
                        lyricsSetId={nav.selectedAlbum.id}
                        mm3SourceTracks={mm3SourceTracks}
                        onRefresh={refreshAlbumData} onGenerateAudio={handleGenerateAudio}
                        onSendToCreate={handleSendToCreate}
                        onRenderAs={(gen) => setRenderAcross({ mode: 'as', gen })}
                        onRenderFrom={() => setRenderAcross({ mode: 'from' })}
                        onViewRecordings={(genId) => {
                          setRecordingsFilter(genId);
                          setActiveTab('recordings');
                          pushUrl(nav.selectedArtist?.id, nav.selectedAlbum?.id, 'recordings');
                        }}
                        showToast={showToast} generationModel={modelSelections.generation}
                        refinementModel={modelSelections.refinement} />
                    )}
                    {activeTab === 'recordings' && (
                      <RecordingsTab
                        generations={generations}
                        showToast={showToast}
                        filterGenerationId={recordingsFilter}
                        onClearFilter={() => setRecordingsFilter(null)}
                        onSongCountChange={setSongCount}
                        refreshKey={recordingsRefreshKey}
                        lyricsSetId={nav.selectedAlbum.id}
                        artistName={nav.selectedArtist?.name}
                        onDeleteComplete={() => setRecordingsRefreshKey(k => k + 1)}
                      />
                    )}
                  </ContentTabs>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      {/* Fetch modal */}
      <FetchLyricsModal isOpen={fetchModalOpen} onClose={() => setFetchModalOpen(false)}
        onFetch={handleFetchLyrics} prefillArtist={fetchModalPrefill} />

      {/* Preset modal */}
      {nav.selectedAlbum && (
        <PresetSettingsModal isOpen={presetModalOpen} lyricsSetId={nav.selectedAlbum.id}
          albumName={nav.selectedAlbum.album || 'Top Songs'} onClose={() => setPresetModalOpen(false)}
          showToast={showToast} />
      )}

      {/* Render as / Render from */}
      {nav.selectedAlbum && (
        <RenderAcrossModal
          mode={renderAcross?.mode ?? null}
          gen={renderAcross?.gen}
          target={currentTarget}
          currentLyricsSetId={nav.selectedAlbum.id}
          onClose={() => setRenderAcross(null)}
          onRender={(gen, target) => { handleGenerateAudio(gen, target); }}
          onSendToCreate={(gen, target) => { handleSendToCreate(gen, target); }}
        />
      )}

      {/* Queue modal */}
      <QueuePanel open={queueOpen} onClose={() => setQueueOpen(false)}
        artists={artists} lyricsSets={allLyricsSets} profiles={allProfiles}
        profilingModel={modelSelections.profiling} generationModel={modelSelections.generation}
        refinementModel={modelSelections.refinement} showToast={showToast}
        onFetchComplete={async () => {
          if (nav.selectedArtist) loadAlbums(nav.selectedArtist.id);
          try {
            await Promise.all([lireekApi.listLyricsSets(), lireekApi.listProfiles()]);
          } catch {}
        }} />

      {/* Prompt Editor modal */}
      <PromptEditor open={promptEditorOpen} onClose={() => setPromptEditorOpen(false)} />

      {/* Manual add modals */}
      <AddArtistModal isOpen={addArtistModalOpen} onClose={() => setAddArtistModalOpen(false)} onSubmit={handleAddArtistManual} />
      {nav.selectedArtist && (
        <AddAlbumModal isOpen={addAlbumModalOpen} onClose={() => setAddAlbumModalOpen(false)}
          onSubmit={handleAddAlbumManual} artistName={nav.selectedArtist.name} />
      )}
      {nav.selectedAlbum && (
        <AddSongModal isOpen={addSongModalOpen} onClose={() => setAddSongModalOpen(false)}
          onSubmit={handleAddSong} albumName={nav.selectedAlbum.album || 'Lyrics Collection'} />
      )}
      {nav.selectedArtist && (
        <CuratedProfileModal isOpen={curatedModalOpen} onClose={() => setCuratedModalOpen(false)}
          artistId={nav.selectedArtist.id} artistName={nav.selectedArtist.name}
          albums={albums} showToast={showToast} onComplete={handleCuratedComplete} />
      )}
    </div>
  );
};
