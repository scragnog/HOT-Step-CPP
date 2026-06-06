import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Trash2, Loader2, Users, Sparkles, ChevronDown, ChevronRight } from 'lucide-react';
import { lireekApi, streamBuildProfile, skipThinking } from '../../services/lireekApi';
import type { Profile } from '../../services/lireekApi';
import { StreamingPanel } from './StreamingPanel';

interface ProfilesTabProps {
  lyricsSetId: number;
  profiles: Profile[];
  onRefresh: () => void;
  showToast: (msg: string) => void;
  profilingModel: { provider: string; model?: string };
}

export const ProfilesTab: React.FC<ProfilesTabProps> = ({
  lyricsSetId, profiles, onRefresh, showToast, profilingModel,
}) => {
  const [building, setBuilding] = useState(false);
  const { t } = useTranslation();
  const [selectedProfile, setSelectedProfile] = useState<Profile | null>(null);

  // Inline streaming state (replaces zustand streamingStore)
  const [streamVisible, setStreamVisible] = useState(false);
  const [streamText, setStreamText] = useState('');
  const [streamPhase, setStreamPhase] = useState('');
  const [streamDone, setStreamDone] = useState(false);

  const provider = profilingModel.provider;
  const model = profilingModel.model || '';

  const handleBuild = useCallback(async () => {
    setBuilding(true);
    setStreamVisible(true);
    setStreamText('');
    setStreamPhase('');
    setStreamDone(false);

    try {
      await streamBuildProfile(lyricsSetId, { provider, model: model || undefined }, {
        onChunk: (text) => setStreamText(prev => {
            const next = prev + text;
            return next.length > 200_000 ? '\u2026(earlier output trimmed)\u2026\n' + next.slice(-200_000) : next;
          }),
        onPhase: (phase) => setStreamPhase(phase),
        onResult: () => {
          setStreamDone(true);
          setBuilding(false);
          onRefresh();
          showToast('Profile built successfully');
        },
        onError: (err) => {
          setStreamDone(true);
          setBuilding(false);
          showToast(`Build failed: ${err}`);
        },
      });
    } catch (err: any) {
      showToast(`Build failed: ${err.message}`);
      setBuilding(false);
      setStreamDone(true);
    }
  }, [lyricsSetId, provider, model, onRefresh, showToast]);

  const handleDelete = async (profile: Profile) => {
    if (!confirm('Delete this profile?')) return;
    try {
      await lireekApi.deleteProfile(profile.id);
      showToast('Profile deleted');
      onRefresh();
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    }
  };

  const MetaBadge: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => {
    const colors: Record<string, string> = {
      pink: 'bg-pink-500/20 text-pink-300 border-pink-500/20',
      blue: 'bg-blue-500/20 text-blue-300 border-blue-500/20',
      purple: 'bg-purple-500/20 text-purple-300 border-purple-500/20',
      green: 'bg-green-500/20 text-green-300 border-green-500/20',
      amber: 'bg-amber-500/20 text-amber-300 border-amber-500/20',
    };
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${colors[color] || colors.pink}`}>
        {label}: {value}
      </span>
    );
  };

  return (
    <div className="p-4 space-y-4">
      {/* Build button */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleBuild}
          disabled={building}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-200 dark:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-semibold transition-all"
        >
          {building ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              {t('lyric.building')}
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              {t('lyric.buildNewProfile')}
            </>
          )}
        </button>
      </div>

      {/* Streaming panel */}
      {streamVisible && (
        <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 overflow-hidden">
          <StreamingPanel
            visible={streamVisible}
            streamText={streamText}
            phase={streamPhase}
            done={streamDone}
            onSkipThinking={() => skipThinking()}
          />
        </div>
      )}

      {/* Profile list */}
      {profiles.length === 0 && !building ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Users className="w-7 h-7 text-zinc-600" />
          </div>
          <h3 className="text-base font-semibold text-zinc-600 dark:text-zinc-400 mb-2">{t('lyric.noProfilesYet')}</h3>
          <p className="text-sm text-zinc-500 max-w-xs">
            {t('lyric.buildProfileDesc')}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {profiles.map((profile, idx) => {
            const data = profile.profile_data;
            const themes = data?.themes as string[] | undefined;
            return (
              <div
                key={profile.id}
                className={`group rounded-xl border border-zinc-200 dark:border-white/5 hover:border-zinc-300 dark:border-white/10 bg-white/[0.01] overflow-hidden transition-colors ls2-card-in ls2-stagger-${Math.min(idx + 1, 11)}`}
              >
              <div
                className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-white/[0.02] transition-colors"
                onClick={() => setSelectedProfile(selectedProfile?.id === profile.id ? null : profile)}
              >
                {selectedProfile?.id === profile.id
                  ? <ChevronDown className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                  : <ChevronRight className="w-4 h-4 text-zinc-500 flex-shrink-0" />
                }
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs px-2 py-0.5 rounded-md bg-indigo-500/10 text-indigo-400 font-medium">
                        {profile.provider}
                      </span>
                      {profile.model && (
                        <span className="text-xs text-zinc-500 truncate">
                          {profile.model}
                        </span>
                      )}
                    </div>
                    {themes && themes.length > 0 && (
                      <p className="text-xs text-zinc-600 dark:text-zinc-400 truncate">
                        {themes.slice(0, 4).join(', ')}
                      </p>
                    )}
                    <p className="text-[11px] text-zinc-600 mt-1">
                      {new Date(profile.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-1" onClick={e => e.stopPropagation()}>
                    <button
                      onClick={() => handleDelete(profile)}
                      className="p-2 rounded-lg hover:bg-red-500/10 text-zinc-600 dark:text-zinc-400 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100"
                      title={t('lyric.deleteProfile')}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Expanded profile detail */}
                {selectedProfile?.id === profile.id && data && (
                  <div className="border-t border-zinc-200 dark:border-white/5 p-4 space-y-4">
                    {/* Themes */}
                    {data.themes?.length > 0 && (
                      <div>
                        <span className="text-[10px] text-amber-400 uppercase tracking-wider font-semibold">{t('lyric.themes')}</span>
                        <div className="flex flex-wrap gap-1.5 mt-1">
                          {(Array.isArray(data.themes) ? data.themes : [data.themes]).map((t: string, i: number) => (
                            <span key={i} className="px-2 py-0.5 rounded-md text-xs bg-amber-500/15 text-amber-300 border border-amber-500/20">{t}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Common Subjects */}
                    {data.common_subjects?.length > 0 && (
                      <div>
                        <span className="text-[10px] text-green-400 uppercase tracking-wider font-semibold">{t('lyric.commonSubjects')}</span>
                        <div className="flex flex-wrap gap-1.5 mt-1">
                          {(Array.isArray(data.common_subjects) ? data.common_subjects : [data.common_subjects]).map((s: string, i: number) => (
                            <span key={i} className="px-2 py-0.5 rounded-md text-xs bg-green-500/15 text-green-300 border border-green-500/20">{s}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Subject Categories */}
                    {data.subject_categories?.length > 0 && (
                      <div>
                        <span className="text-[10px] text-blue-400 uppercase tracking-wider font-semibold">{t('lyric.subjectCategories')}</span>
                        <div className="flex flex-wrap gap-1.5 mt-1">
                          {(Array.isArray(data.subject_categories) ? data.subject_categories : [data.subject_categories]).map((c: string, i: number) => (
                            <span key={i} className="px-2 py-0.5 rounded-md text-xs bg-blue-500/15 text-blue-300 border border-blue-500/20">{c}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Stat badges */}
                    <div className="flex flex-wrap gap-2">
                      {data.avg_verse_lines > 0 && <MetaBadge label="Avg Verse" value={`${data.avg_verse_lines} lines`} color="blue" />}
                      {data.avg_chorus_lines > 0 && <MetaBadge label="Avg Chorus" value={`${data.avg_chorus_lines} lines`} color="pink" />}
                      {data.rhyme_schemes?.length > 0 && <MetaBadge label="Rhyme" value={data.rhyme_schemes.slice(0, 3).join(', ')} color="purple" />}
                      {data.perspective && <MetaBadge label="Voice" value={typeof data.perspective === 'string' ? data.perspective.split('—')[0].trim() : ''} color="amber" />}
                    </div>

                    {/* Text sections */}
                    {[
                      { label: 'Tone & Mood', value: data.tone_and_mood, color: 'text-pink-400' },
                      { label: 'Vocabulary', value: data.vocabulary_notes, color: 'text-blue-400' },
                      { label: 'Structural Patterns', value: data.structural_patterns, color: 'text-purple-400' },
                      { label: 'Narrative Techniques', value: data.narrative_techniques, color: 'text-green-400' },
                      { label: 'Imagery Patterns', value: data.imagery_patterns, color: 'text-amber-400' },
                      { label: 'Signature Devices', value: data.signature_devices, color: 'text-cyan-400' },
                      { label: 'Emotional Arc', value: data.emotional_arc, color: 'text-rose-400' },
                    ].filter(s => s.value).map((section, i) => (
                      <div key={i}>
                        <span className={`text-[10px] uppercase tracking-wider font-semibold ${section.color}`}>{section.label}</span>
                        <p className="text-sm text-zinc-700 dark:text-zinc-300 mt-1 leading-relaxed">{section.value}</p>
                      </div>
                    ))}

                    {/* Song subjects */}
                    {data.song_subjects && Object.keys(data.song_subjects).length > 0 && (
                      <details>
                        <summary className="text-[10px] text-amber-400 uppercase tracking-wider font-semibold cursor-pointer hover:text-amber-300">
                          Song Subjects ({Object.keys(data.song_subjects).length} songs)
                        </summary>
                        <div className="mt-2 space-y-1">
                          {Object.entries(data.song_subjects).map(([title, subject]: [string, any]) => (
                            <div key={title} className="flex gap-2 text-xs">
                              <span className="text-zinc-600 dark:text-zinc-400 font-medium shrink-0 w-32 truncate" title={title}>{title}</span>
                              <span className="text-zinc-500">{subject}</span>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {/* Detailed stats */}
                    {(data.meter_stats || data.vocabulary_stats || data.repetition_stats || data.rhyme_quality) && (
                      <details>
                        <summary className="text-[10px] text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-700 dark:text-zinc-300">{t('lyric.detailedStats')}</summary>
                        <div className="mt-2 grid grid-cols-2 gap-3">
                          {data.meter_stats && (
                            <div className="p-3 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                              <span className="text-[10px] text-blue-400 uppercase tracking-wider font-semibold">Meter</span>
                              <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1 space-y-0.5">
                                <div>Avg syllables: {data.meter_stats.avg_syllables_per_line}/line</div>
                                <div>σ = {data.meter_stats.syllable_std_dev}</div>
                                <div>Words: {data.meter_stats.avg_words_per_line}/line</div>
                              </div>
                            </div>
                          )}
                          {data.vocabulary_stats && (
                            <div className="p-3 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                              <span className="text-[10px] text-green-400 uppercase tracking-wider font-semibold">Vocabulary</span>
                              <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1 space-y-0.5">
                                <div>TTR: {data.vocabulary_stats.type_token_ratio}</div>
                                <div>{data.vocabulary_stats.total_words} words ({data.vocabulary_stats.unique_words} unique)</div>
                                <div>Contractions: {data.vocabulary_stats.contraction_pct}%</div>
                              </div>
                            </div>
                          )}
                          {data.repetition_stats && (
                            <div className="p-3 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                              <span className="text-[10px] text-pink-400 uppercase tracking-wider font-semibold">Repetition</span>
                              <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1 space-y-0.5">
                                <div>Chorus: {data.repetition_stats.chorus_repetition_pct}% repeated</div>
                                <div>Pattern: {data.repetition_stats.pattern}</div>
                              </div>
                            </div>
                          )}
                          {data.rhyme_quality && (
                            <div className="p-3 rounded-lg bg-white/5 border border-zinc-200 dark:border-white/5">
                              <span className="text-[10px] text-purple-400 uppercase tracking-wider font-semibold">Rhyme Quality</span>
                              <div className="text-xs text-zinc-600 dark:text-zinc-400 mt-1 space-y-0.5">
                                <div>Perfect: {data.rhyme_quality.perfect}</div>
                                <div>Slant: {data.rhyme_quality.slant}</div>
                                <div>Assonance: {data.rhyme_quality.assonance}</div>
                              </div>
                            </div>
                          )}
                        </div>
                      </details>
                    )}

                    {/* Raw summary */}
                    {data.raw_summary && (
                      <details>
                        <summary className="text-[10px] text-zinc-500 uppercase tracking-wider cursor-pointer hover:text-zinc-700 dark:text-zinc-300">{t('lyric.fullSummary')}</summary>
                        <div className="mt-2 p-3 rounded-lg bg-black/20 dark:bg-black/40 border border-zinc-200 dark:border-white/5 text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap leading-relaxed max-h-[40vh] overflow-y-auto">
                          {data.raw_summary}
                        </div>
                      </details>
                    )}
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
