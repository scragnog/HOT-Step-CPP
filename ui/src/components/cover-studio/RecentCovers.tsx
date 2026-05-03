// RecentCovers.tsx — Recent covers sub-component for Cover Studio
import React, { useState, useEffect } from 'react';
import { Music, Play, Pause, Loader2, Trash2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { songApi } from '../../services/api';
import type { Song } from '../../types';
import { playFromList, songToTrack, usePlayback } from '../../stores/playbackStore';

interface RecentCoversProps {
  refreshTrigger: number;
}

export const RecentCovers: React.FC<RecentCoversProps> = ({ refreshTrigger }) => {
  const { token } = useAuth();
  const pb = usePlayback();
  const [covers, setCovers] = useState<Song[]>([]);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    fetch('/api/songs?source=cover-studio', {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(res => res.json())
      .then(data => {
        const allSongs = data.songs || [];
        setCovers(
          allSongs
            .map((s: any): Song => ({
              id: s.id,
              title: s.title || 'Untitled Cover',
              lyrics: s.lyrics || '',
              style: s.style || '',
              caption: s.caption || '',
              audioUrl: s.audio_url || '',
              masteredAudioUrl: s.mastered_audio_url || '',
              mastered_audio_url: s.mastered_audio_url || '',
              duration: s.duration && s.duration > 0
                ? `${Math.floor(s.duration / 60)}:${String(Math.floor(s.duration % 60)).padStart(2, '0')}`
                : '0:00',
              tags: s.tags || [],
              createdAt: new Date(s.created_at),
              generationParams: typeof s.generation_params === 'string'
                ? JSON.parse(s.generation_params || '{}')
                : (s.generation_params || {}),
            }))
            .sort((a: Song, b: Song) => {
              const aTime = a.createdAt instanceof Date ? a.createdAt.getTime() : 0;
              const bTime = b.createdAt instanceof Date ? b.createdAt.getTime() : 0;
              return bTime - aTime;
            })
            .slice(0, 20)
        );
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [token, refreshTrigger]);

  const handleClearAll = async () => {
    if (!token || covers.length === 0) return;
    if (!window.confirm(`Delete all ${covers.length} covers?`)) return;
    setClearing(true);
    try {
      await songApi.bulkDelete(covers.map(c => c.id), token);
      setCovers([]);
    } finally {
      setClearing(false);
    }
  };

  const handlePlay = (cover: Song) => {
    playFromList(songToTrack(cover), covers.map(songToTrack), 'cover-studio');
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Music className="w-4 h-4 text-cyan-400" />
          Recent Covers
          {covers.length > 0 && <span className="text-[10px] text-zinc-500 font-normal">({covers.length})</span>}
        </div>
        {covers.length > 0 && (
          <button
            onClick={handleClearAll}
            disabled={clearing}
            className="p-1 rounded-md hover:bg-red-900/30 text-zinc-500 hover:text-red-400 transition-colors"
            title="Clear all covers"
          >
            {clearing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="w-4 h-4 text-zinc-500 animate-spin" />
        </div>
      ) : covers.length === 0 ? (
        <p className="text-[10px] text-zinc-500 text-center py-4">No covers yet. Generate your first one!</p>
      ) : (
        <div className="space-y-1">
          {covers.map(cover => {
            const isCurrent = pb.currentTrack?.id === cover.id;
            return (
              <div
                key={cover.id}
                onClick={() => handlePlay(cover)}
                className={`
                  w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200 text-left cursor-pointer group relative
                  ${isCurrent ? 'bg-cyan-500/20 ring-1 ring-cyan-400/50' : 'hover:bg-white/5'}
                `}
              >
                <div className="w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center flex-shrink-0">
                  {isCurrent && pb.isPlaying ? (
                    <Pause className="w-3 h-3 text-cyan-400" />
                  ) : (
                    <Play className="w-3 h-3 text-cyan-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300 truncate">{cover.title}</div>
                  <div className="text-[10px] text-zinc-500">{cover.duration}</div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
