import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Pencil, Trash2, Save, X, FileText, Plus } from 'lucide-react';
import type { LyricsSet, SongLyric } from '../../services/lireekApi';

function parseSongs(songs: SongLyric[] | string): SongLyric[] {
  if (typeof songs === 'string') {
    try { return JSON.parse(songs); } catch { return []; }
  }
  return songs || [];
}

interface SourceLyricsTabProps {
  album: LyricsSet;
  onDeleteSong: (index: number) => void;
  onEditSong?: (index: number, lyrics: string) => void;
  onAddSong?: () => void;
}

export const SourceLyricsTab: React.FC<SourceLyricsTabProps> = ({ album, onDeleteSong, onEditSong, onAddSong }) => {
  const songs = parseSongs(album.songs);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [editText, setEditText] = useState('');

  if (songs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center px-8">
        <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
          <FileText className="w-7 h-7 text-zinc-600" />
        </div>
        <h3 className="text-base font-semibold text-zinc-600 dark:text-zinc-400 mb-2">No source lyrics</h3>
        <p className="text-sm text-zinc-500 max-w-xs mb-4">
          Add songs manually or fetch lyrics from Genius.
        </p>
        {onAddSong && (
          <button
            onClick={onAddSong}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white text-sm font-semibold transition-all"
          >
            <Plus className="w-4 h-4" />
            Add Song Manually
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="p-4 space-y-1">
      {/* Add Song button */}
      {onAddSong && (
        <div className="flex justify-end mb-2">
          <button
            onClick={onAddSong}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-amber-400 hover:bg-amber-500/10 transition-colors font-medium"
          >
            <Plus className="w-3 h-3" />
            Add Song
          </button>
        </div>
      )}
      {songs.map((song, idx) => {
        const isExpanded = expandedIdx === idx;
        const isEditing = editingIdx === idx;
        const lyrics = song.lyrics || '';
        return (
          <div
            key={idx}
            className={`rounded-xl border border-zinc-200 dark:border-white/5 overflow-hidden transition-colors hover:border-zinc-300 dark:border-white/10 ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)}`}
          >
            {/* Song header */}
            <button
              className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/[0.02] transition-colors"
              onClick={() => { setExpandedIdx(isExpanded ? null : idx); setEditingIdx(null); }}
            >
              {isExpanded
                ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />
              }
              <span className="flex-1 text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate">
                {song.title}
              </span>
              {song.genre ? (
                <span className="text-[11px] text-amber-500/80 truncate max-w-[180px]" title={song.genre}>
                  {song.genre.split(',')[0].trim()}
                </span>
              ) : null}
              {song.bpm ? (
                <span className="text-[11px] text-zinc-500 font-mono">{song.bpm} BPM</span>
              ) : null}
              {song.key ? (
                <span className="text-[11px] text-zinc-500 font-mono">{song.key}</span>
              ) : null}
              <span className="text-xs text-zinc-500">
                {lyrics.split('\n').length} lines
              </span>
            </button>

            {/* Expanded content */}
            {isExpanded && (
              <div className="border-t border-zinc-200 dark:border-white/5">
                {/* Enriched metadata from a Training Studio export */}
                {(song.caption || song.mm3Caption || song.genre || song.bpm || song.key || song.signature) && (
                  <div className="px-4 pt-3 space-y-3">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {song.genre && (
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5 col-span-2 sm:col-span-4">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Genre</label>
                          <p className="text-sm text-amber-300">{song.genre}</p>
                        </div>
                      )}
                      {song.bpm ? (
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">BPM</label>
                          <p className="text-sm text-pink-300">{song.bpm}</p>
                        </div>
                      ) : null}
                      {song.key && (
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Key</label>
                          <p className="text-sm text-blue-300">{song.key}</p>
                        </div>
                      )}
                      {song.signature && (
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Time Sig</label>
                          <p className="text-sm text-purple-300">{song.signature}</p>
                        </div>
                      )}
                      {song.language && (
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Language</label>
                          <p className="text-sm text-emerald-300">{song.language}</p>
                        </div>
                      )}
                    </div>
                    {song.caption && (
                      <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                        <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Caption</label>
                        <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed">{song.caption}</p>
                      </div>
                    )}
                    {song.mm3Caption && (
                      <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                        <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">MM3 Caption</label>
                        <pre className="text-xs text-zinc-700 dark:text-zinc-300 leading-relaxed whitespace-pre-wrap font-sans m-0">{song.mm3Caption}</pre>
                      </div>
                    )}
                    {song.yue2Caption && (
                      <div className="px-3 py-2 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                        <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">YuE2 Caption</label>
                        <p className="text-xs text-zinc-700 dark:text-zinc-300 leading-relaxed">{song.yue2Caption}</p>
                      </div>
                    )}
                  </div>
                )}
                <div className="px-4 py-3">
                  {isEditing ? (
                    <textarea
                      value={editText}
                      onChange={e => setEditText(e.target.value)}
                      className="w-full h-80 text-sm text-zinc-800 dark:text-zinc-200 bg-black/30 border border-zinc-300 dark:border-white/10 rounded-lg p-3 font-sans leading-relaxed resize-y focus:outline-none focus:border-indigo-500/50"
                    />
                  ) : (
                    <pre className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap font-sans leading-relaxed max-h-96 overflow-y-auto">
                      {lyrics || '(No lyrics available)'}
                    </pre>
                  )}
                </div>
                <div className="flex items-center gap-2 px-4 py-2 border-t border-zinc-200 dark:border-white/5 bg-white/[0.01]">
                  {isEditing ? (
                    <>
                      <button
                        onClick={() => {
                          if (onEditSong) onEditSong(idx, editText);
                          setEditingIdx(null);
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-emerald-400 hover:bg-emerald-500/10 transition-colors"
                      >
                        <Save className="w-3 h-3" />
                        Save
                      </button>
                      <button
                        onClick={() => setEditingIdx(null)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-zinc-600 dark:text-zinc-400 hover:bg-white/5 transition-colors"
                      >
                        <X className="w-3 h-3" />
                        Cancel
                      </button>
                    </>
                  ) : (
                    <>
                      {onEditSong && (
                        <button
                          onClick={() => { setEditingIdx(idx); setEditText(lyrics); }}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-indigo-400 hover:bg-indigo-500/10 transition-colors"
                        >
                          <Pencil className="w-3 h-3" />
                          Edit
                        </button>
                      )}
                      <button
                        onClick={() => {
                          if (confirm(`Delete "${song.title}" from this album?`)) {
                            onDeleteSong(idx);
                          }
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <Trash2 className="w-3 h-3" />
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
