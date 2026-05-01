import React, { useState, useCallback } from 'react';
import { Trash2, Pencil, Music2, Wand2, Play, Loader2, ChevronDown, ChevronRight, Send, FileText, Headphones, Sparkles } from 'lucide-react';
import { lireekApi, streamRefine, skipThinking } from '../../services/lireekApi';
import type { Generation, Profile } from '../../services/lireekApi';
import { StreamingPanel } from './StreamingPanel';
import { useStreamingStore, startStreamGenerate } from '../../stores/streamingStore';

interface WrittenSongsTabProps {
  generations: Generation[];
  profiles: Profile[];
  onRefresh: () => void;
  onGenerateAudio: (gen: Generation) => void;
  onSendToCreate?: (gen: Generation) => void;
  onViewRecordings?: (genId: number) => void;
  showToast: (msg: string) => void;
  generationModel: { provider: string; model?: string };
  refinementModel: { provider: string; model?: string };
}

export const WrittenSongsTab: React.FC<WrittenSongsTabProps> = ({
  generations, profiles, onRefresh, onGenerateAudio, onSendToCreate, onViewRecordings, showToast,
  generationModel, refinementModel,
}) => {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [generating, setGenerating] = useState(false);
  const [refiningId, setRefiningId] = useState<number | null>(null);
  const [genCount, setGenCount] = useState(1);

  // Persistent streaming state — survives tab navigation
  const streaming = useStreamingStore();

  // Local streaming state for refine (one-off, doesn't need persistence)
  const [refineStreamVisible, setRefineStreamVisible] = useState(false);
  const [refineStreamText, setRefineStreamText] = useState('');
  const [refineStreamPhase, setRefineStreamPhase] = useState('');
  const [refineStreamDone, setRefineStreamDone] = useState(false);

  const handleQuickGenerate = useCallback(async () => {
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
  }, [profiles, genCount, generationModel, onRefresh, showToast]);

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
          onChunk: (text) => setRefineStreamText(prev => prev + text),
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
          onClick={handleQuickGenerate}
          disabled={generating || profiles.length === 0}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-semibold transition-all"
        >
          {generating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4" />
              Generate Lyrics
            </>
          )}
        </button>
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-500">Count:</label>
          <select
            value={genCount}
            onChange={(e) => setGenCount(parseInt(e.target.value))}
            className="px-2 py-1.5 rounded-lg bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-emerald-500/50"
          >
            {[1, 2, 3, 4, 5, 8, 10].map(n => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
        {profiles.length === 0 && (
          <span className="text-xs text-amber-400/60">Build a profile first</span>
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
          <h3 className="text-base font-semibold text-zinc-400 mb-2">No generated lyrics yet</h3>
          <p className="text-sm text-zinc-500 max-w-xs">
            Generate lyrics from a profile to see them here.
          </p>
        </div>
      ) : (
        <div className="space-y-1">
          {generations.map((gen, idx) => {
            const isExpanded = expandedId === gen.id;

            return (
              <div
                key={gen.id}
                className={`rounded-xl border border-white/5 hover:border-white/10 overflow-hidden transition-colors ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)}`}
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
                    <p className="text-sm font-medium text-zinc-200 truncate">
                      {gen.title || 'Untitled'}
                    </p>
                    <p className="text-xs text-zinc-500 mt-0.5">
                      {gen.subject || 'No subject'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {gen.parent_generation_id && (
                      <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 flex items-center gap-0.5">
                        <Sparkles className="w-2.5 h-2.5" /> Refined
                      </span>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); onGenerateAudio(gen); }}
                      className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-emerald-400 hover:bg-emerald-500/10 transition-colors"
                      title="Generate audio from these lyrics"
                    >
                      <Play className="w-3 h-3" />
                      Audio
                    </button>
                    {onViewRecordings && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onViewRecordings(gen.id); }}
                        className="flex items-center gap-1 px-2 py-1 rounded-md text-[11px] text-pink-400 hover:bg-pink-500/10 transition-colors"
                        title="View generated songs from these lyrics"
                      >
                        <Headphones className="w-3 h-3" />
                        Songs
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
                  <div className="border-t border-white/5">
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
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Subject</label>
                          <input
                            className="w-full bg-transparent text-sm text-amber-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-amber-500/50 transition-colors"
                            defaultValue={gen.subject || ''}
                            onBlur={(e) => { if (e.target.value !== (gen.subject || '')) handleSaveField(gen.id, 'subject', e.target.value); }}
                          />
                        </div>
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">BPM</label>
                          <input
                            type="number"
                            className="w-full bg-transparent text-sm text-pink-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-pink-500/50 transition-colors"
                            defaultValue={gen.bpm || 0}
                            onBlur={(e) => { const v = parseInt(e.target.value) || 0; if (v !== gen.bpm) handleSaveField(gen.id, 'bpm', v); }}
                          />
                        </div>
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Key</label>
                          <input
                            className="w-full bg-transparent text-sm text-blue-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-blue-500/50 transition-colors"
                            defaultValue={gen.key || ''}
                            onBlur={(e) => { if (e.target.value !== (gen.key || '')) handleSaveField(gen.id, 'key', e.target.value); }}
                          />
                        </div>
                        <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                          <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Duration (seconds)</label>
                          <input
                            type="number"
                            className="w-full bg-transparent text-sm text-purple-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-purple-500/50 transition-colors"
                            defaultValue={gen.duration || 0}
                            onBlur={(e) => { const v = parseInt(e.target.value) || 0; if (v !== gen.duration) handleSaveField(gen.id, 'duration', v); }}
                          />
                        </div>
                      </div>

                      {/* Editable caption */}
                      <div className="px-3 py-2 rounded-lg bg-white/5 border border-white/5">
                        <label className="text-[10px] text-zinc-500 uppercase tracking-wider block mb-1">Caption</label>
                        <textarea
                          className="w-full bg-transparent text-sm text-zinc-300 focus:outline-none border-b border-transparent hover:border-white/20 focus:border-pink-500/50 transition-colors resize-none"
                          rows={2}
                          defaultValue={gen.caption || ''}
                          onBlur={(e) => { if (e.target.value !== (gen.caption || '')) handleSaveField(gen.id, 'caption', e.target.value); }}
                        />
                      </div>

                      {/* Action buttons */}
                      <div className="flex items-center gap-2 flex-wrap">
                        <button
                          onClick={() => onGenerateAudio(gen)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-pink-500/30 to-purple-500/30 text-white hover:from-pink-500/40 hover:to-purple-500/40 text-sm font-semibold transition-all border border-pink-500/20"
                        >
                          <Play className="w-3.5 h-3.5" />
                          Generate Audio
                        </button>
                        <button
                          onClick={() => handleRefine(gen)}
                          disabled={refiningId === gen.id}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-purple-500/20 text-purple-300 hover:bg-purple-500/30 text-sm font-medium transition-colors disabled:opacity-50"
                          title="Refine these lyrics using the refinement LLM"
                        >
                          {refiningId === gen.id ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                          Refine
                        </button>
                        {onSendToCreate && (
                          <button
                            onClick={() => onSendToCreate(gen)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 text-sm font-medium transition-colors border border-amber-500/10"
                          >
                            <Send className="w-3.5 h-3.5" />
                            Send to Create
                          </button>
                        )}
                        <div className="flex-1" />
                        <button
                          onClick={() => handleDelete(gen)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-red-400 hover:bg-red-500/10 transition-colors"
                        >
                          <Trash2 className="w-3 h-3" />
                          Delete
                        </button>
                      </div>

                      {/* Editable lyrics */}
                      <div>
                        <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-2">Lyrics</h3>
                        <textarea
                          className="w-full p-4 rounded-xl bg-black/40 border border-white/5 text-sm text-zinc-200 font-mono leading-relaxed focus:outline-none focus:border-pink-500/30 resize-y transition-colors"
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
