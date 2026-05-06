// RecentBuilds.tsx — Recent stem builds sidebar section
//
// Fetches recent songs generated via Stem Builder (source = 'stem-builder')
// and displays them in a compact list. Each item can be played or used as
// a new source for iterative composition.

import React, { useState, useEffect, useCallback } from 'react';
import { Play, ArrowUp } from 'lucide-react';
import { songApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';

interface RecentBuild {
  id: string;
  title: string;
  audioUrl: string;
  masteredAudioUrl?: string;
  trackName?: string;
  createdAt?: string;
}

interface RecentBuildsProps {
  refreshTrigger: number;
  onPlay: (build: RecentBuild) => void;
  onUseAsSource: (build: RecentBuild) => void;
}

/** Track emoji lookup */
const TRACK_EMOJI: Record<string, string> = {
  vocals: '🎤', backing_vocals: '🎙️', drums: '🥁', bass: '🎵',
  guitar: '🎸', keyboard: '🎹', percussion: '🪘', strings: '🎻',
  synth: '🎛️', fx: '🔊', brass: '🎺', woodwinds: '🪕',
};

export const RecentBuilds: React.FC<RecentBuildsProps> = ({
  refreshTrigger,
  onPlay,
  onUseAsSource,
}) => {
  const { token } = useAuth();
  const [builds, setBuilds] = useState<RecentBuild[]>([]);
  const [loading, setLoading] = useState(false);

  const loadBuilds = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const { songs } = await songApi.list(token);
      // Filter to stem-builder generations by checking generation_params
      const stemBuilds = songs
        .filter((s: any) => {
          try {
            const params = typeof s.generation_params === 'string'
              ? JSON.parse(s.generation_params)
              : s.generation_params;
            return params?.source === 'stem-builder' || params?.taskType === 'lego';
          } catch { return false; }
        })
        .slice(0, 20)  // limit to 20 most recent
        .map((s: any) => {
          let trackName = '';
          try {
            const params = typeof s.generation_params === 'string'
              ? JSON.parse(s.generation_params)
              : s.generation_params;
            trackName = params?.trackName || '';
          } catch { /* ignore */ }
          return {
            id: s.id,
            title: s.title || 'Untitled',
            audioUrl: s.audio_url || s.audioUrl || '',
            masteredAudioUrl: s.mastered_audio_url || s.masteredAudioUrl,
            trackName,
            createdAt: s.created_at,
          };
        });
      setBuilds(stemBuilds);
    } catch (err) {
      console.error('[RecentBuilds] Failed to load:', err);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => { loadBuilds(); }, [loadBuilds, refreshTrigger]);

  if (loading && builds.length === 0) {
    return (
      <div className="text-xs text-zinc-600 p-3 text-center">Loading...</div>
    );
  }

  if (builds.length === 0) {
    return (
      <div className="text-xs text-zinc-600 p-3 text-center">
        No stem builds yet. Generate your first layer!
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 p-2">
      {builds.map(build => (
        <div
          key={build.id}
          className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg hover:bg-white/[0.04] group transition-colors"
        >
          <span className="text-sm flex-shrink-0">
            {TRACK_EMOJI[build.trackName || ''] || '🎵'}
          </span>
          <div className="flex-1 min-w-0">
            <div className="text-xs text-zinc-300 truncate">{build.title}</div>
            {build.trackName && (
              <div className="text-[10px] text-zinc-600 capitalize">
                {build.trackName.replace('_', ' ')}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => onPlay(build)}
            className="flex-shrink-0 w-6 h-6 rounded-full bg-white/[0.06] flex items-center justify-center hover:bg-amber-500/20 transition-colors opacity-0 group-hover:opacity-100"
            title="Play"
          >
            <Play size={10} className="text-zinc-400" />
          </button>
          <button
            type="button"
            onClick={() => onUseAsSource(build)}
            className="flex-shrink-0 w-6 h-6 rounded-full bg-white/[0.06] flex items-center justify-center hover:bg-amber-500/20 transition-colors opacity-0 group-hover:opacity-100"
            title="Use as source"
          >
            <ArrowUp size={10} className="text-zinc-400" />
          </button>
        </div>
      ))}
    </div>
  );
};
