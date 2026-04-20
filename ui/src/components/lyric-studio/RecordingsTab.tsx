/**
 * RecordingsTab.tsx — Shows generated audio recordings grouped by lyric generation.
 *
 * Data flow:
 *   1. For each Generation, fetch audio_generations from Lireek DB
 *   2. For each audio gen, check job status via generateApi
 *   3. Render playable songs grouped by generation
 *
 * Adapted for cpp engine: uses generateApi.status() per-job (no bulk history endpoint).
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Play, Trash2, Headphones, ChevronDown, ChevronRight, Loader2, Clock, X, Filter, Download, ListPlus, Check } from 'lucide-react';
import { lireekApi } from '../../services/lireekApi';
import type { Generation, AudioGeneration } from '../../services/lireekApi';
import { generateApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import type { Song } from '../../types';
import { DownloadModal } from '../shared/DownloadModal';
import { usePlaylist } from './playlistStore';
import { playFromList, songToTrack } from '../../stores/playbackStore';

interface SongGroup {
  generation: Generation;
  audioGens: AudioGeneration[];
  songs: Song[];
}

interface RecordingsTabProps {
  generations: Generation[];
  showToast: (msg: string) => void;
  filterGenerationId?: number | null;
  onClearFilter?: () => void;
  onSongCountChange?: (count: number) => void;
  refreshKey?: number;
  artistName?: string;
}

export const RecordingsTab: React.FC<RecordingsTabProps> = ({
  generations, showToast, filterGenerationId, onClearFilter, onSongCountChange, refreshKey = 0, artistName,
}) => {
  const { token } = useAuth();
  const [groups, setGroups] = useState<SongGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedGenId, setExpandedGenId] = useState<number | null>(null);
  const [localRefreshKey, setLocalRefreshKey] = useState(0);
  const [downloadSong, setDownloadSong] = useState<Song | null>(null);

  const generationsRef = useRef(generations);
  generationsRef.current = generations;

  const genKey = useMemo(() => {
    const ids = generations.map(g => g.id).sort().join(',');
    return `${ids}|${filterGenerationId ?? 'all'}|${refreshKey}|${localRefreshKey}`;
  }, [generations, filterGenerationId, refreshKey, localRefreshKey]);

  const filteredGenerations = useMemo(() =>
    filterGenerationId
      ? generations.filter(g => g.id === filterGenerationId)
      : generations,
    [generations, filterGenerationId]
  );

  useEffect(() => {
    if (!token || genKey === '|all') {
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);

    const load = async () => {
      try {
        const gens = filterGenerationId
          ? generationsRef.current.filter(g => g.id === filterGenerationId)
          : generationsRef.current;

        const results: SongGroup[] = [];
        for (const gen of gens) {
          try {
            const res = await lireekApi.getAudioGenerations(gen.id);
            if (res.audio_generations.length > 0) {
              const songs: Song[] = [];
              for (const ag of res.audio_generations) {
                // Use pre-resolved audio URL from Lireek DB first
                if (ag.audio_url) {
                  songs.push({
                    id: ag.hotstep_job_id || `ag-${ag.id}`,
                    title: gen.title || 'Untitled',
                    style: gen.caption || '',
                    caption: gen.caption || '',
                    lyrics: gen.lyrics || '',
                    coverUrl: ag.cover_url || '',
                    duration: gen.duration || 0,
                    tags: [],
                    audioUrl: ag.audio_url,
                    masteredAudioUrl: ag.mastered_audio_url || '',
                    created_at: ag.created_at,
                  });
                } else {
                  // Fallback: check job status
                  try {
                    const status = await generateApi.status(ag.hotstep_job_id);
                    if (status?.status === 'succeeded' && status.result?.audioUrls) {
                      for (const audioUrl of status.result.audioUrls) {
                        songs.push({
                          id: ag.hotstep_job_id,
                          title: gen.title || 'Untitled',
                          style: gen.caption || '',
                          caption: gen.caption || '',
                          lyrics: gen.lyrics || '',
                          coverUrl: '',
                          duration: status.result.duration || 0,
                          tags: [],
                          audioUrl,
                          masteredAudioUrl: status.result.masteredAudioUrl || '',
                          created_at: ag.created_at,
                        });
                      }
                      // Resolve in Lireek DB for next time
                      const firstUrl = status.result.audioUrls[0];
                      if (firstUrl) {
                        lireekApi.resolveAudioGeneration(ag.hotstep_job_id, firstUrl).catch(() => {});
                      }
                    }
                  } catch {
                    console.warn(`[RecordingsTab] Could not resolve job ${ag.hotstep_job_id}`);
                  }
                }
              }
              if (songs.length > 0) {
                results.push({ generation: gen, audioGens: res.audio_generations, songs });
              }
            }
          } catch (err) {
            console.error(`[RecordingsTab] Failed to get audio gens for gen ${gen.id}:`, err);
          }
        }

        if (!cancelled) {
          setGroups(results);
          const totalSongs = results.reduce((n, g) => n + g.songs.length, 0);
          onSongCountChange?.(totalSongs);
        }
      } catch (err) {
        console.error('[RecordingsTab] Failed to load:', err);
      } finally {
        setLoading(false);
      }
    };

    load();
    return () => { cancelled = true; };
  }, [genKey, token]);

  useEffect(() => {
    if (filterGenerationId && groups.length === 1) {
      setExpandedGenId(groups[0].generation.id);
    }
  }, [filterGenerationId, groups.length]);

  const handleDeleteAudioGen = useCallback(async (ag: AudioGeneration) => {
    if (!confirm('Delete this audio generation?')) return;
    try {
      await lireekApi.deleteAudioGeneration(ag.id);
      showToast('Audio generation deleted');
      setLocalRefreshKey(k => k + 1);
    } catch (err: any) {
      showToast(`Failed to delete: ${err.message}`);
    }
  }, [showToast]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
      </div>
    );
  }

  return (
    <>
      <div className="p-4 space-y-2">
        {/* Filter indicator */}
        {filterGenerationId && onClearFilter && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-pink-500/10 border border-pink-500/20 mb-2">
            <Filter className="w-3.5 h-3.5 text-pink-400" />
            <span className="text-xs text-pink-300 flex-1">
              Showing songs from: <strong>{filteredGenerations[0]?.title || 'Untitled'}</strong>
            </span>
            <button onClick={onClearFilter}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-xs text-zinc-400 hover:text-white hover:bg-white/10 transition-colors">
              <X className="w-3 h-3" /> Clear
            </button>
          </div>
        )}

        {groups.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-center px-8">
            <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
              <Headphones className="w-7 h-7 text-zinc-600" />
            </div>
            <h3 className="text-base font-semibold text-zinc-400 mb-2">
              {filterGenerationId ? 'No songs generated from these lyrics yet' : 'No generated songs yet'}
            </h3>
            <p className="text-sm text-zinc-500 max-w-xs">
              Go to the Generated Lyrics tab and generate audio to see songs here.
            </p>
          </div>
        ) : (
          groups.map((group, idx) => {
            const isExpanded = expandedGenId === group.generation.id;
            return (
              <div key={group.generation.id}
                className={`rounded-xl border border-white/5 hover:border-white/10 overflow-hidden transition-colors ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)}`}>
                <button className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors"
                  onClick={() => setExpandedGenId(isExpanded ? null : group.generation.id)}>
                  {isExpanded
                    ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                    : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-200 truncate">{group.generation.title || 'Untitled'}</p>
                    <p className="text-xs text-zinc-500 mt-0.5">
                      {group.generation.subject || group.generation.caption?.slice(0, 60) || 'No caption'}
                    </p>
                  </div>
                  <span className="text-xs text-zinc-500 flex items-center gap-1">
                    <Headphones className="w-3 h-3" />
                    {group.songs.length} song{group.songs.length !== 1 ? 's' : ''}
                  </span>
                </button>

                {isExpanded && (
                  <div className="border-t border-white/5">
                    {group.songs.length === 0 ? (
                      <p className="px-4 py-6 text-sm text-zinc-500 text-center">
                        Audio generation is pending or failed. Check the queue.
                      </p>
                    ) : (
                      <div className="divide-y divide-white/5">
                        {group.songs.map((song, idx) => {
                          const ag = group.audioGens[idx];
                          return (
                            <div key={idx} className="flex items-center gap-3 px-4 py-2.5 hover:bg-white/[0.02] transition-colors">
                              <button onClick={() => {
                                playFromList(songToTrack(song), group.songs.map(songToTrack), 'lireek-recordings');
                              }}
                                className="w-8 h-8 rounded-full bg-pink-600/20 hover:bg-pink-600/30 flex items-center justify-center flex-shrink-0 transition-colors">
                                <Play className="w-3.5 h-3.5 text-pink-400 ml-0.5" />
                              </button>
                              <AddToPlaylistButton song={song} artistName={artistName} />
                              <div className="flex-1 min-w-0">
                                <p className="text-sm text-zinc-300 truncate">{song.title || `Song ${idx + 1}`}</p>
                                {song.duration && (
                                  <p className="text-[11px] text-zinc-500 flex items-center gap-1">
                                    <Clock className="w-3 h-3" />
                                    {Math.floor(Number(song.duration) / 60)}:{String(Math.floor(Number(song.duration) % 60)).padStart(2, '0')}
                                  </p>
                                )}
                              </div>
                              <button onClick={() => setDownloadSong(song)}
                                className="p-1.5 rounded-lg text-zinc-500 hover:text-blue-400 hover:bg-blue-500/10 transition-colors"
                                title="Download">
                                <Download className="w-3.5 h-3.5" />
                              </button>
                              {ag && (
                                <button onClick={() => handleDeleteAudioGen(ag)}
                                  className="p-1.5 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                                  title="Delete">
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Download Modal */}
      {downloadSong && (
        <DownloadModal
          song={downloadSong}
          isOpen={!!downloadSong}
          onClose={() => setDownloadSong(null)}
        />
      )}
    </>
  );
};

// ── Add-to-playlist helper ───────────────────────────────────────────────────

const AddToPlaylistButton: React.FC<{ song: Song; artistName?: string }> = ({ song, artistName }) => {
  const playlist = usePlaylist();
  const inPlaylist = playlist.isIn(song.id);

  const toggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inPlaylist) {
      playlist.remove(song.id);
    } else {
      const dur = song.duration;
      let seconds = 0;
      if (typeof dur === 'string' && dur.includes(':')) {
        const [m, s] = dur.split(':').map(Number);
        seconds = (m || 0) * 60 + (s || 0);
      } else if (typeof dur === 'number') {
        seconds = dur;
      }
      playlist.add({
        id: song.id,
        title: song.title || 'Untitled',
        audioUrl: song.audioUrl || '',
        masteredAudioUrl: song.masteredAudioUrl || '',
        artistName: artistName || '',
        coverUrl: song.coverUrl || '',
        duration: seconds,
        style: song.style || '',
        generationParams: song.generationParams,
      });
    }
  };

  return (
    <button onClick={toggle}
      className={`p-1 rounded-md transition-colors flex-shrink-0 ${
        inPlaylist ? 'text-pink-400 bg-pink-500/10 hover:bg-pink-500/20'
          : 'text-zinc-600 hover:text-pink-400 hover:bg-pink-500/10'
      }`}
      title={inPlaylist ? 'Remove from playlist' : 'Add to playlist'}>
      {inPlaylist ? <Check className="w-3.5 h-3.5" /> : <ListPlus className="w-3.5 h-3.5" />}
    </button>
  );
};
