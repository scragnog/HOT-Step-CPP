/**
 * LyricStudioV2.tsx — Main container for the Lyric Studio interface.
 *
 * Three navigation levels:
 *   1. Artist Grid with settings sidebar
 *   2. Album Grid for selected artist
 *   3. Album Detail with tabbed content (Source Lyrics, Profiles, Written Songs)
 *
 * State management: All data is loaded via lireekApi, with URL-based routing
 * and popstate support for browser back/forward.
 *
 * Ported from hot-step-9000 with import path + API adaptations for the C++ engine.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { usePersistedState } from '../../hooks/usePersistedState';
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
import { RecordingsTab } from './RecordingsTab';
import { ActivitySidebar } from '../shared/ActivitySidebar';
import { useAudioGeneration } from './useAudioGeneration';
import { enqueueAudioGen, useAudioGenQueue } from '../../stores/audioGenQueueStore';
import { usePlayback } from '../../stores/playbackStore';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { QueuePanel } from './QueuePanel';
import { PromptEditor } from './PromptEditor';
// streamingStore used via queue panel
import { loadSelections } from './ProviderSelector';


// ── URL helpers ──────────────────────────────────────────────────────────────

const LS_BASE = '/lyric-studio';

function buildUrl(artistId?: number, albumId?: number, tab?: TabId): string {
  if (artistId && albumId && tab) return `${LS_BASE}/artist/${artistId}/album/${albumId}/${tab}`;
  if (artistId && albumId) return `${LS_BASE}/artist/${artistId}/album/${albumId}`;
  if (artistId) return `${LS_BASE}/artist/${artistId}`;
  return LS_BASE;
}

function parseUrl(path: string): { artistId?: number; albumId?: number; tab?: TabId } {
  const m = path.match(/\/lyric-studio\/artist\/(\d+)(?:\/album\/(\d+)(?:\/(source-lyrics|profiles|written-songs|recordings))?)?/);
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

  // ── Navigation ──
  const [nav, setNav] = useState<NavState>({ level: 'artists', selectedArtist: null, selectedAlbum: null });

  // ── Right panel width (persisted, pixel-based) ──
  const [lsRightPanelWidth, setLsRightPanelWidth] = usePersistedState('ls-rightPanelWidth', 380);
  const compactRight = lsRightPanelWidth < 380;

  const handleRightPanelResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = lsRightPanelWidth;
    const onMove = (ev: MouseEvent) => {
      const newW = Math.min(700, Math.max(240, startW + startX - ev.clientX));
      setLsRightPanelWidth(newW);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [lsRightPanelWidth, setLsRightPanelWidth]);

  // ── Data ──
  const [artists, setArtists] = useState<Artist[]>([]);
  const [albums, setAlbums] = useState<LyricsSet[]>([]);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [artistsLoading, setArtistsLoading] = useState(true);
  const [albumsLoading, setAlbumsLoading] = useState(false);

  const albumLoadIdRef = useRef(0);
  const albumDataLoadIdRef = useRef(0);

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
  const [promptEditorOpen, setPromptEditorOpen] = useState(false);
  const [addArtistModalOpen, setAddArtistModalOpen] = useState(false);
  const [addAlbumModalOpen, setAddAlbumModalOpen] = useState(false);
  const [addSongModalOpen, setAddSongModalOpen] = useState(false);
  const [curatedModalOpen, setCuratedModalOpen] = useState(false);

  // ── Fetch lyrics progress ──
  const [fetchingLyrics, setFetchingLyrics] = useState(false);
  const [fetchingLabel, setFetchingLabel] = useState('');

  // ── Bulk queue data ──
  const [allLyricsSets, setAllLyricsSets] = useState<LyricsSet[]>([]);
  const [allProfiles, setAllProfiles] = useState<Profile[]>([]);

  const [artistIdsWithAdapters, setArtistIdsWithAdapters] = useState<Set<number>>(new Set());

  // ── Playback (for backdrop effect) ──
  const { isPlaying, currentTrack: currentPlaybackTrack } = usePlayback();

  // ── Audio generation ──
  const audioQueue = useAudioGenQueue(token || undefined);

  // ── Toast ──
  const [toast, setToast] = useState<string | null>(null);
  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3500);
  }, []);

  // ── Load artists ──
  const loadArtists = useCallback(async (retries = 5): Promise<Artist[]> => {
    setArtistsLoading(true);
    let artistsList: Artist[] = [];
    try {
      const res = await lireekApi.listArtists();
      artistsList = res.artists;
      setArtists(res.artists);
    } catch (err) {
      console.warn(`[LyricStudioV2] Failed to load artists (retries left: ${retries}):`, err);
      if (retries > 0) {
        await new Promise(r => setTimeout(r, 2000));
        return loadArtists(retries - 1);
      }
    } finally {
      setArtistsLoading(false);
    }

    // Background: fetch missing artist images
    const missing = artistsList.filter(a => !a.image_url);
    if (missing.length > 0) {
      const fetchNext = (idx: number) => {
        if (idx >= missing.length) return;
        lireekApi.refreshArtistImage(missing[idx].id)
          .then(result => {
            if (result.image_url) {
              setArtists(prev => prev.map(a => a.id === missing[idx].id ? { ...a, image_url: result.image_url } : a));
            }
          })
          .catch(() => {})
          .finally(() => setTimeout(() => fetchNext(idx + 1), 500));
      };
      fetchNext(0);
    }
    return artistsList;
  }, []);

  // ── Initial load: artists + URL restore ──
  useEffect(() => {
    const init = async () => {
      const artistsList = await loadArtists();
      // Background: load adapter mapping
      lireekApi.listAllPresets().then(({ presets }) => {
        const ids = new Set<number>();
        for (const p of presets as any[]) {
          if (p.adapter_path && p.artist_id) ids.add(p.artist_id);
        }
        setArtistIdsWithAdapters(ids);
      }).catch(() => {});
      const parsed = parseUrl(window.location.pathname);
      if (parsed.artistId) {
        isRestoringUrl.current = true;
        try {
          const artist = artistsList.find(a => a.id === parsed.artistId);
          if (!artist) return;
          const albumRes = await lireekApi.listLyricsSets(artist.id);
          setAlbums(albumRes.lyrics_sets);

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
      }
    };
    init();
  }, [loadArtists]);

  // ── Load albums ──
  const loadAlbums = useCallback(async (artistId: number) => {
    const loadId = ++albumLoadIdRef.current;
    setAlbumsLoading(true);
    let albumsList: LyricsSet[] = [];
    try {
      const res = await lireekApi.listLyricsSets(artistId);
      if (loadId !== albumLoadIdRef.current) return;
      albumsList = res.lyrics_sets;
      setAlbums(res.lyrics_sets);
    } catch (err) {
      console.error('[LyricStudioV2] Failed to load albums:', err);
    } finally {
      if (loadId === albumLoadIdRef.current) setAlbumsLoading(false);
    }

    // Background: fetch missing album images
    const missing = albumsList.filter(a => !a.image_url && a.album);
    if (missing.length > 0) {
      const fetchNext = (idx: number) => {
        if (idx >= missing.length || loadId !== albumLoadIdRef.current) return;
        lireekApi.refreshAlbumImage(missing[idx].id)
          .then(result => {
            if (loadId !== albumLoadIdRef.current) return;
            if (result.image_url) {
              setAlbums(prev => prev.map(a => a.id === missing[idx].id ? { ...a, image_url: result.image_url } : a));
            }
          })
          .catch(() => {})
          .finally(() => setTimeout(() => fetchNext(idx + 1), 500));
      };
      fetchNext(0);
    }
  }, []);

  // ── Load album detail data ──
  const loadAlbumData = useCallback(async (albumId: number, retries = 2) => {
    const loadId = ++albumDataLoadIdRef.current;
    try {
      const { lyrics_set, profiles: p, generations: g } = await lireekApi.getAlbumFullDetail(albumId);
      if (loadId !== albumDataLoadIdRef.current) return;
      setNav(prev => ({ ...prev, selectedAlbum: lyrics_set }));
      setProfiles(p);
      setGenerations(g);
    } catch (err) {
      if (loadId !== albumDataLoadIdRef.current) return;
      if (retries > 0) {
        await new Promise(r => setTimeout(r, 2000));
        return loadAlbumData(albumId, retries - 1);
      }
    }
  }, []);

  // ── Navigation handlers ──
  const pushUrl = useCallback((artistId?: number, albumId?: number, tab?: TabId) => {
    if (isRestoringUrl.current) return;
    const url = buildUrl(artistId, albumId, tab);
    if (window.location.pathname !== url) window.history.pushState({}, '', url);
  }, []);

  const handleSelectArtist = useCallback((artist: Artist) => {
    setAlbums([]);
    setProfiles([]);
    setGenerations([]);
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
    setProfiles([]);
    setGenerations([]);
    setSongCount(0);
    loadAlbumData(albumId);
  }, [nav.selectedAlbum?.id, loadAlbumData]);

  const handleBackToArtists = useCallback(() => {
    setNav({ level: 'artists', selectedArtist: null, selectedAlbum: null });
    setAlbums([]); setProfiles([]); setGenerations([]); setSongCount(0);
    loadArtists();
    pushUrl();
  }, [loadArtists, pushUrl]);

  const handleBackToAlbums = useCallback(() => {
    setNav(prev => ({ ...prev, level: 'albums', selectedAlbum: null }));
    setProfiles([]); setGenerations([]); setSongCount(0);
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
        setAlbums([]); setProfiles([]); setGenerations([]); setSongCount(0);
        return;
      }
      const artist = artists.find(a => a.id === parsed.artistId);
      if (!artist) return;
      if (!parsed.albumId) {
        setNav({ level: 'albums', selectedArtist: artist, selectedAlbum: null });
        setProfiles([]); setGenerations([]); setSongCount(0);
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
  const handleDeleteArtist = useCallback(async (artist: Artist) => {
    if (!confirm(`Delete ${artist.name} and ALL their albums, profiles, and generations?`)) return;
    try {
      await lireekApi.deleteArtist(artist.id);
      showToast(`Deleted ${artist.name}`);
      loadArtists();
    } catch (err: any) { showToast(`Failed to delete: ${err.message}`); }
  }, [loadArtists, showToast]);

  const handleRefreshImage = useCallback(async (artist: Artist) => {
    try {
      showToast(`Refreshing image for ${artist.name}...`);
      const res = await lireekApi.refreshArtistImage(artist.id);
      setArtists(prev => prev.map(a => a.id === artist.id ? { ...a, image_url: res.image_url } : a));
      showToast(`Updated image for ${artist.name}`);
    } catch (err: any) { showToast(`Couldn't find image: ${err.message}`); }
  }, [showToast]);

  const handleSetImage = useCallback(async (artist: Artist, url: string) => {
    try {
      const res = await lireekApi.setArtistImage(artist.id, url);
      setArtists(prev => prev.map(a => a.id === artist.id ? { ...a, image_url: res.image_url } : a));
      showToast(`Image updated for ${artist.name}`);
    } catch (err: any) { showToast(`Failed: ${err.message}`); }
  }, [showToast]);

  const handleDeleteAlbum = useCallback(async (album: LyricsSet) => {
    if (!confirm(`Delete album "${album.album || 'Top Songs'}" and all associated data?`)) return;
    try {
      await lireekApi.deleteLyricsSet(album.id);
      showToast('Album deleted');
      if (nav.selectedArtist) loadAlbums(nav.selectedArtist.id);
    } catch (err: any) { showToast(`Failed to delete: ${err.message}`); }
  }, [nav.selectedArtist, loadAlbums, showToast]);

  const handleRefreshAlbumImage = useCallback(async (album: LyricsSet) => {
    try {
      const res = await lireekApi.refreshAlbumImage(album.id);
      setAlbums(prev => prev.map(a => a.id === album.id ? { ...a, image_url: res.image_url } : a));
      showToast('Album image updated');
    } catch { showToast('Could not find album image on Genius'); }
  }, [showToast]);

  const handleSetAlbumImage = useCallback(async (album: LyricsSet, url: string) => {
    try {
      const res = await lireekApi.setAlbumImage(album.id, url);
      setAlbums(prev => prev.map(a => a.id === album.id ? { ...a, image_url: res.image_url } : a));
      showToast('Album image set');
    } catch (err: any) { showToast(`Failed: ${err.message}`); }
  }, [showToast]);

  const handleDeleteSong = useCallback(async (index: number) => {
    if (!nav.selectedAlbum) return;
    try {
      await lireekApi.removeSong(nav.selectedAlbum.id, index);
      showToast('Song removed');
      const updated = await lireekApi.getLyricsSet(nav.selectedAlbum.id);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
    } catch (err: any) { showToast(`Failed: ${err.message}`); }
  }, [nav.selectedAlbum, showToast]);

  const handleEditSong = useCallback(async (index: number, lyrics: string) => {
    if (!nav.selectedAlbum) return;
    try {
      const updated = await lireekApi.editSong(nav.selectedAlbum.id, index, lyrics);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
      showToast('Lyrics updated');
    } catch (err: any) { showToast(`Failed: ${err.message}`); }
  }, [nav.selectedAlbum, showToast]);

  const handleFetchLyrics = useCallback(async (artist: string, album: string, maxSongs: number) => {
    const label = `${artist}${album ? ` — ${album}` : ''}`;
    setFetchingLyrics(true);
    setFetchingLabel(label);
    showToast(`Fetching lyrics for ${label}…`);
    try {
      const res = await lireekApi.fetchLyrics({ artist, album: album || undefined, max_songs: maxSongs });
      showToast(`Fetched ${res.songs_fetched} songs`);
      await loadArtists();
      if (nav.selectedArtist && res.artist.id === nav.selectedArtist.id) {
        await loadAlbums(nav.selectedArtist.id);
      }
      if (nav.level === 'artists') handleSelectArtist(res.artist);
    } catch (err: any) { showToast(`Fetch failed: ${err.message}`); }
    finally { setFetchingLyrics(false); setFetchingLabel(''); }
  }, [loadArtists, loadAlbums, nav.selectedArtist, nav.level, handleSelectArtist, showToast]);

  const refreshAlbumData = useCallback(() => {
    if (nav.selectedAlbum) loadAlbumData(nav.selectedAlbum.id);
  }, [nav.selectedAlbum, loadAlbumData]);

  // ── Audio generation ──
  const { sendToCreate } = useAudioGeneration({ profiles, showToast });
  const globalParams = useGlobalParams();

  const handleGenerateAudio = useCallback(async (gen: Generation) => {
    if (!token) { showToast('Not authenticated'); return; }
    const profile = profiles.find(p => p.id === gen.profile_id);
    if (!profile) { showToast('Profile not found'); return; }
    // Capture globalParams snapshot NOW — same as Create page's getGlobalParams().
    // This ensures every engine param (solver, guidance, DCW, latent, LM, etc.)
    // flows through identically to the Create page path.
    const paramsSnapshot = globalParams.getGlobalParams();
    await enqueueAudioGen(gen, {
      artistId: nav.selectedArtist?.id || 0,
      artistName: nav.selectedArtist?.name || 'Unknown',
      artistImageUrl: nav.selectedArtist?.image_url || '',
      profileId: profile.id,
      lyricsSetId: profile.lyrics_set_id,
    }, paramsSnapshot, token);
    showToast(`Queued: ${gen.title || 'Untitled'}`);
  }, [token, profiles, nav.selectedArtist, globalParams, showToast]);

  // Refresh album data on audio queue completions
  useEffect(() => {
    if (audioQueue.completionCounter > 0) {
      refreshAlbumData();
      setRecordingsRefreshKey(k => k + 1);
    }
  }, [audioQueue.completionCounter]);

  const handleSendToCreate = useCallback(async (gen: Generation) => {
    // Inject artist name — gen from getAlbumFullDetail doesn't include it
    const enriched = { ...gen, artist_name: gen.artist_name || nav.selectedArtist?.name || '' };
    await sendToCreate(enriched);
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
  const handleAddArtistManual = useCallback(async (name: string, imageUrl?: string) => {
    try {
      const res = await lireekApi.createArtist({ name, image_url: imageUrl });
      showToast(`Added ${res.artist.name}`);
      await loadArtists();
      handleSelectArtist(res.artist);
    } catch (err: any) { showToast(`Failed to add artist: ${err.message}`); }
  }, [loadArtists, handleSelectArtist, showToast]);

  const handleAddAlbumManual = useCallback(async (albumName: string | undefined, imageUrl?: string) => {
    if (!nav.selectedArtist) return;
    try {
      const res = await lireekApi.createLyricsSet({ artist_id: nav.selectedArtist.id, album: albumName, image_url: imageUrl });
      showToast(`Created ${albumName || 'lyrics collection'}`);
      await loadAlbums(nav.selectedArtist.id);
      handleSelectAlbum(res.lyrics_set);
    } catch (err: any) { showToast(`Failed to create album: ${err.message}`); }
  }, [nav.selectedArtist, loadAlbums, handleSelectAlbum, showToast]);

  const handleAddSong = useCallback(async (title: string, lyrics: string) => {
    if (!nav.selectedAlbum) return;
    try {
      const updated = await lireekApi.addSongToSet(nav.selectedAlbum.id, { title, lyrics });
      showToast(`Added "${title}"`);
      setNav(prev => ({ ...prev, selectedAlbum: updated }));
    } catch (err: any) { showToast(`Failed to add song: ${err.message}`); }
  }, [nav.selectedAlbum, showToast]);

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

  // Queue open helper
  const openQueuePanel = useCallback(async () => {
    try {
      const [lsRes, pRes] = await Promise.all([lireekApi.listLyricsSets(), lireekApi.listProfiles()]);
      setAllLyricsSets(lsRes.lyrics_sets);
      setAllProfiles(pRes.profiles);
    } catch (err) { console.error('[LyricStudioV2] Failed to load queue data:', err); }
    setQueueOpen(true);
  }, []);

  // ── Render ──
  return (
    <div className="h-full w-full flex flex-col relative bg-zinc-950">
      {/* Toast */}
      {toast && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 px-5 py-2.5 rounded-xl bg-zinc-800/90 backdrop-blur-sm border border-white/10 text-sm text-white shadow-2xl ls2-slide-up">
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
          <span>Fetching lyrics for <strong className="text-white">{fetchingLabel}</strong>…</span>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        {nav.level === 'artists' && (
          <div className="h-full flex ls2-fade-in">
            <div className="w-48 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <ArtistSidebar artists={artists} selectedArtistId={-1}
                onSelectArtist={handleSelectArtist} onBack={handleBackToArtists}
                artistIdsWithAdapters={artistIdsWithAdapters} />
            </div>
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Header bar — spans above ArtistPageSidebar + Grid, matches ContentTabs height */}
              <div className="flex-shrink-0 flex items-center px-5 py-3 border-b border-white/5 bg-zinc-950/30">
                <span className="text-sm font-medium text-zinc-400">All Artists</span>
                {artists.length > 0 && (
                  <span className="ml-2 min-w-[20px] h-5 px-1.5 rounded-full text-[11px] font-bold flex items-center justify-center bg-white/10 text-zinc-400">
                    {artists.length}
                  </span>
                )}
              </div>
              <div className="flex-1 flex min-h-0">
                <div className="w-64 flex-shrink-0 border-r border-white/5 overflow-hidden">
                  <ArtistPageSidebar onOpenQueue={openQueuePanel} onOpenPromptEditor={() => setPromptEditorOpen(true)} />
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
            {/* Resize handle */}
            <div
              className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
              onMouseDown={handleRightPanelResize}
            >
              <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
            </div>
            <div className="h-full flex-shrink-0 border-l border-white/5 overflow-hidden" style={{ width: lsRightPanelWidth }}>
              <ActivitySidebar source="lyric-studio" showToast={showToast}
                refreshKey={recordingsRefreshKey} compact={compactRight} />
            </div>
          </div>
        )}

        {nav.level === 'albums' && nav.selectedArtist && (
          <div className="h-full flex ls2-fade-in">
            <div className="w-48 flex-shrink-0 border-r border-white/5 overflow-hidden">
              <ArtistSidebar artists={artists} selectedArtistId={nav.selectedArtist.id}
                onSelectArtist={handleSelectArtist} onBack={handleBackToArtists}
                artistIdsWithAdapters={artistIdsWithAdapters} />
            </div>
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Header bar — spans above ArtistPageSidebar + Grid, matches ContentTabs height */}
              <div className="flex-shrink-0 flex items-center px-5 py-3 border-b border-white/5 bg-zinc-950/30">
                <span className="text-sm font-medium text-zinc-400">{nav.selectedArtist.name}</span>
                {albums.length > 0 && (
                  <span className="ml-2 min-w-[20px] h-5 px-1.5 rounded-full text-[11px] font-bold flex items-center justify-center bg-white/10 text-zinc-400">
                    {albums.length}
                  </span>
                )}
              </div>
              <div className="flex-1 flex min-h-0">
                <div className="w-64 flex-shrink-0 border-r border-white/5 overflow-hidden">
                  <ArtistPageSidebar artist={nav.selectedArtist} albumCount={albums.length}
                    onOpenQueue={openQueuePanel} onOpenPromptEditor={() => setPromptEditorOpen(true)} />
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
            {/* Resize handle */}
            <div
              className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
              onMouseDown={handleRightPanelResize}
            >
              <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
            </div>
            <div className="h-full flex-shrink-0 border-l border-white/5 overflow-hidden" style={{ width: lsRightPanelWidth }}>
              <ActivitySidebar source="lyric-studio" showToast={showToast}
                refreshKey={recordingsRefreshKey} compact={compactRight} />
            </div>
          </div>
        )}

        {nav.level === 'album-detail' && nav.selectedArtist && nav.selectedAlbum && (
          <div className="h-full flex flex-col ls2-fade-in">
            <div className="flex-1 flex min-h-0">
              {/* Left: artist sidebar + album header */}
              <div className="w-48 flex-shrink-0 border-r border-white/5 overflow-hidden">
                <ArtistSidebar artists={artists} selectedArtistId={nav.selectedArtist.id}
                  onSelectArtist={handleSelectArtist} onBack={handleBackToArtists}
                  artistIdsWithAdapters={artistIdsWithAdapters} />
              </div>
              <div className="w-64 flex-shrink-0 border-r border-white/5 overflow-hidden relative">
                <div className="relative z-[1] h-full">
                  <AlbumHeader
                    artist={nav.selectedArtist} album={nav.selectedAlbum}
                    onBack={handleBackToAlbums} onOpenPreset={() => setPresetModalOpen(true)}
                    profileCount={profiles.length} generationCount={generations.length} songCount={songCount}
                  />
                </div>
              </div>

              {/* Middle: tabbed content */}
              <div className="flex-1 overflow-hidden relative">
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
                        profilingModel={loadSelections().profiling} />
                    )}
                    {activeTab === 'written-songs' && (
                      <WrittenSongsTab generations={generations} profiles={profiles}
                        onRefresh={refreshAlbumData} onGenerateAudio={handleGenerateAudio}
                        onSendToCreate={handleSendToCreate}
                        onViewRecordings={(genId) => {
                          setRecordingsFilter(genId);
                          setActiveTab('recordings');
                          pushUrl(nav.selectedArtist?.id, nav.selectedAlbum?.id, 'recordings');
                        }}
                        showToast={showToast} generationModel={loadSelections().generation}
                        refinementModel={loadSelections().refinement} />
                    )}
                    {activeTab === 'recordings' && (
                      <RecordingsTab
                        generations={generations}
                        showToast={showToast}
                        filterGenerationId={recordingsFilter}
                        onClearFilter={() => setRecordingsFilter(null)}
                        onSongCountChange={setSongCount}
                        refreshKey={recordingsRefreshKey}
                        artistName={nav.selectedArtist?.name}
                        onDeleteComplete={() => setRecordingsRefreshKey(k => k + 1)}
                      />
                    )}
                  </ContentTabs>
                </div>
              </div>

              {/* Resize handle */}
              <div
                className="flex-shrink-0 w-1.5 h-full cursor-col-resize group z-20 flex items-center hover:bg-pink-500/20 active:bg-pink-500/30 transition-colors"
                onMouseDown={handleRightPanelResize}
              >
                <div className="w-0.5 h-8 rounded-full bg-zinc-600 group-hover:bg-pink-400 transition-colors" />
              </div>

              {/* Right: sidebar panel */}
              <div className="flex-shrink-0 border-l border-white/5 overflow-hidden flex flex-col relative" style={{ width: lsRightPanelWidth }}>
                <div className="relative z-[1] flex-1 min-h-0 overflow-hidden">
                  <ActivitySidebar source="lyric-studio"
                    showToast={showToast}
                    refreshKey={recordingsRefreshKey} compact={compactRight} />
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

      {/* Queue modal */}
      <QueuePanel open={queueOpen} onClose={() => setQueueOpen(false)}
        artists={artists} lyricsSets={allLyricsSets} profiles={allProfiles}
        profilingModel={loadSelections().profiling} generationModel={loadSelections().generation}
        refinementModel={loadSelections().refinement} showToast={showToast}
        onFetchComplete={async () => {
          await loadArtists();
          if (nav.selectedArtist) loadAlbums(nav.selectedArtist.id);
          try {
            const [lsRes, pRes] = await Promise.all([lireekApi.listLyricsSets(), lireekApi.listProfiles()]);
            setAllLyricsSets(lsRes.lyrics_sets);
            setAllProfiles(pRes.profiles);
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
