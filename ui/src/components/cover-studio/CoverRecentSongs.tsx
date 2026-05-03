/**
 * CoverRecentSongs.tsx — Shows recently generated covers in Cover Studio.
 *
 * Fetches songs with source=cover-studio, provides play/download/delete.
 * Download uses cover-specific filename: Target Artist - Track Name (Source Artist Cover)
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Loader2, Music, Download, Trash2, ListPlus, Check } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { songApi } from '../../services/api';
import type { Song } from '../../types';
import { DownloadModal } from '../shared/DownloadModal';
import { usePlaylist } from '../lyric-studio/playlistStore';
import { playFromList, songToTrack, usePlayback } from '../../stores/playbackStore';

interface CoverRecentSongsProps {
  showToast: (msg: string, type?: 'success' | 'error') => void;
  refreshKey?: number;
}

// ── Module-level cache ───────────────────────────────────────────────────────

let _cachedCovers: Song[] = [];
let _cachedRefreshKey = -1;
let _fetchInFlight = false;

// ── Component ────────────────────────────────────────────────────────────────

export const CoverRecentSongs: React.FC<CoverRecentSongsProps> = ({ showToast, refreshKey = 0 }) => {
  const { token } = useAuth();
  const pb = usePlayback();
  const [songs, setSongs] = useState<Song[]>(_cachedCovers);
  const [loading, setLoading] = useState(_cachedCovers.length === 0);
  const mountedRef = useRef(true);
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);
  const [downloadArtist, setDownloadArtist] = useState('');

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      _cachedRefreshKey = -1;
    };
  }, []);

  useEffect(() => {
    if (!token) { setLoading(false); return; }
    if (_cachedRefreshKey === refreshKey && _cachedCovers.length > 0) return;
    if (_fetchInFlight) return;
    if (_cachedCovers.length === 0) setLoading(true);

    _fetchInFlight = true;
    fetch('/api/songs?source=cover-studio', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => res.json())
      .then(data => {
        const allSongs: Song[] = (data.songs || [])
          .filter((s: any) => !!s.audio_url)
          .map((s: any): Song => ({
            id: s.id,
            title: s.title || 'Untitled Cover',
            lyrics: s.lyrics || '',
            style: s.style || '',
            caption: s.caption || '',
            audioUrl: s.audio_url || '',
            masteredAudioUrl: s.mastered_audio_url || '',
            mastered_audio_url: s.mastered_audio_url || '',
            duration: s.duration || 0,
            tags: s.tags || [],
            createdAt: new Date(s.created_at),
            artistName: s.artist || '',
            generationParams: typeof s.generation_params === 'string'
              ? JSON.parse(s.generation_params || '{}')
              : (s.generation_params || {}),
          }))
          .sort((a: Song, b: Song) => {
            const aTime = a.createdAt instanceof Date ? a.createdAt.getTime() : 0;
            const bTime = b.createdAt instanceof Date ? b.createdAt.getTime() : 0;
            return bTime - aTime;
          })
          .slice(0, 50);
        _cachedCovers = allSongs;
        _cachedRefreshKey = refreshKey;
        _fetchInFlight = false;
        if (mountedRef.current) { setSongs(allSongs); setLoading(false); }
      })
      .catch(() => {
        _fetchInFlight = false;
        if (mountedRef.current) setLoading(false);
      });
  }, [refreshKey, token]);

  const handlePlay = useCallback((song: Song) => {
    playFromList(songToTrack(song), songs.map(songToTrack), 'cover-studio');
  }, [songs]);

  const handleDelete = useCallback(async (e: React.MouseEvent, song: Song) => {
    e.stopPropagation();
    if (!token) return;
    try {
      await songApi.delete(song.id, token);
      setSongs(prev => {
        const updated = prev.filter(s => s.id !== song.id);
        _cachedCovers = updated;
        return updated;
      });
      showToast('Cover deleted');
    } catch {
      showToast('Failed to delete', 'error');
    }
  }, [token, showToast]);

  const handleDownloadClick = useCallback((e: React.MouseEvent, song: Song) => {
    e.stopPropagation();
    // Extract target artist from generation params
    const gp = song.generationParams as any;
    const targetArtist = gp?.artistName || song.artistName || '';
    setDownloadSong(song);
    setDownloadArtist(targetArtist);
  }, []);

  if (loading && songs.length === 0) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
      </div>
    );
  }

  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-center px-4">
        <Music className="w-5 h-5 text-zinc-600 mb-2" />
        <p className="text-xs text-zinc-500">No covers yet</p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 auto-rows-[4.5rem] gap-1 px-2 py-1.5 overflow-y-auto scrollbar-hide" style={{ maxHeight: '100%' }}>
        {songs.map((song) => {
          const dur = typeof song.duration === 'number' ? song.duration : 0;
          const mins = Math.floor(dur / 60);
          const secs = String(Math.floor(dur % 60)).padStart(2, '0');
          const isCurrent = pb.currentTrack?.id === song.id;
          const gp = song.generationParams as any;
          const targetArtist = gp?.artistName || song.artistName || '';

          return (
            <div key={song.id}
              className={`flex items-center gap-2.5 rounded-lg hover:bg-white/[0.06] transition-colors text-left group px-2 overflow-hidden relative cursor-pointer ${isCurrent ? 'bg-cyan-500/10 ring-1 ring-cyan-500/30' : ''}`}
              onClick={() => handlePlay(song)}>
              <div className="w-14 h-14 rounded-md flex-shrink-0 overflow-hidden bg-zinc-800 relative">
                <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-cyan-900/40 to-purple-900/40">
                  <Music className="w-5 h-5 text-cyan-500/60" />
                </div>
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <Play className="w-4 h-4 text-white ml-0.5" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-zinc-200 truncate leading-snug">
                  {song.title || 'Untitled Cover'}
                </p>
                {targetArtist && (
                  <p className="text-[10px] text-zinc-500 truncate leading-snug">{targetArtist}</p>
                )}
                {dur > 0 && (
                  <p className="text-[10px] text-zinc-600 font-mono mt-0.5">{mins}:{secs}</p>
                )}
              </div>
              <div className="absolute right-1 top-1/2 -translate-y-1/2 flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <CoverAddToPlaylistBtn song={song} />
                <button onClick={(e) => handleDownloadClick(e, song)}
                  className="p-1.5 rounded-md bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors"
                  title="Download">
                  <Download className="w-3 h-3" />
                </button>
                <button onClick={(e) => handleDelete(e, song)}
                  className="p-1.5 rounded-md bg-zinc-800/80 hover:bg-red-900/60 text-zinc-400 hover:text-red-400 transition-colors"
                  title="Delete">
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Download Modal */}
      {downloadSong && (
        <DownloadModal
          song={downloadSong}
          isOpen={!!downloadSong}
          onClose={() => setDownloadSong(null)}
          artistName={downloadArtist}
        />
      )}
    </>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const CoverAddToPlaylistBtn: React.FC<{ song: Song }> = ({ song }) => {
  const playlist = usePlaylist();
  const inPlaylist = playlist.isIn(song.id);
  const gp = song.generationParams as any;
  const targetArtist = gp?.artistName || song.artistName || '';

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inPlaylist) {
      playlist.remove(song.id);
    } else {
      playlist.add({
        id: song.id,
        title: song.title || 'Untitled Cover',
        audioUrl: song.audioUrl || '',
        masteredAudioUrl: song.masteredAudioUrl || '',
        artistName: targetArtist,
        coverUrl: '',
        duration: typeof song.duration === 'number' ? song.duration : 0,
      });
    }
  };

  return (
    <button onClick={toggle}
      className={`p-1.5 rounded-md transition-colors ${
        inPlaylist ? 'bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30'
          : 'bg-zinc-800/80 hover:bg-zinc-700 text-zinc-400 hover:text-cyan-400'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}>
      {inPlaylist ? <Check className="w-3 h-3" /> : <ListPlus className="w-3 h-3" />}
    </button>
  );
};
