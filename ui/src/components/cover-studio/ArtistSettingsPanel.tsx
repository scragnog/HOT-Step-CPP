// ArtistSettingsPanel.tsx — Right panel: artist selector + cover settings + generate
import React from 'react';
import { Guitar, Disc3, Zap, Music, ChevronDown, Loader2 } from 'lucide-react';
import { EditableSlider } from './EditableSlider';
import { transposeKey, type AudioAnalysis } from './coverStudioUtils';
import type { Artist, AlbumPreset } from '../../services/lireekApi';

interface ArtistSettingsPanelProps {
  artists: Artist[];
  isLoadingArtists: boolean;
  selectedArtistId: number | null;
  onSelectArtist: (a: Artist) => void;
  artistPresets: { lsId: number; album: string; preset: AlbumPreset | null }[];
  selectedPreset: AlbumPreset | null;
  onSelectPreset: (p: AlbumPreset | null) => void;
  audioCoverStrength: number;
  onAudioCoverStrength: (v: number) => void;
  coverNoiseStrength: number;
  onCoverNoiseStrength: (v: number) => void;
  tempoScale: number;
  onTempoScale: (v: number) => void;
  pitchShift: number;
  onPitchShift: (v: number) => void;
  analysis: AudioAnalysis | null;
  canGenerate: boolean;
  isGenerating: boolean;
  genProgress: number;
  genStage: string;
  onGenerate: () => void;
  onCancel: () => void;
}

export const ArtistSettingsPanel: React.FC<ArtistSettingsPanelProps> = (props) => {
  const {
    artists, isLoadingArtists, selectedArtistId, onSelectArtist,
    artistPresets, selectedPreset, onSelectPreset,
    audioCoverStrength, onAudioCoverStrength, coverNoiseStrength, onCoverNoiseStrength,
    tempoScale, onTempoScale, pitchShift, onPitchShift, analysis,
    canGenerate, isGenerating, genProgress, genStage, onGenerate, onCancel,
  } = props;

  const presetsWithAdapters = artistPresets.filter(p => p.preset?.adapter_path);

  return (
    <div className="w-[540px] flex-shrink-0 overflow-y-auto scrollbar-hide p-4 space-y-4">
      {/* Target Artist */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Guitar className="w-4 h-4 text-cyan-400" />
          Cover As Artist
        </div>
        {isLoadingArtists ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 text-zinc-500 animate-spin" />
          </div>
        ) : artists.length === 0 ? (
          <div className="text-center py-6">
            <p className="text-xs text-zinc-500">No artists in Lyric Studio yet.</p>
            <p className="text-[10px] text-zinc-400 mt-1">Add artists in Lyric Studio first.</p>
          </div>
        ) : (
          <div className="grid grid-cols-5 gap-2 max-h-[240px] overflow-y-auto scrollbar-hide">
            {artists.map(artist => (
              <button
                key={artist.id}
                onClick={() => onSelectArtist(artist)}
                className={`
                  flex flex-col items-center gap-1.5 p-2 rounded-xl transition-all duration-200
                  ${selectedArtistId === artist.id
                    ? 'bg-cyan-500/20 ring-2 ring-cyan-400 shadow-lg shadow-cyan-500/10'
                    : 'bg-black/5 dark:bg-white/5 hover:bg-cyan-500/10 hover:ring-1 hover:ring-cyan-400/50'}
                `}
              >
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center overflow-hidden flex-shrink-0">
                  {artist.image_url ? (
                    <img src={artist.image_url} alt={artist.name} className="w-full h-full object-cover" />
                  ) : (
                    <span className="text-white text-sm font-bold">{artist.name.charAt(0)}</span>
                  )}
                </div>
                <span className="text-[10px] font-medium text-zinc-700 dark:text-zinc-300 truncate w-full text-center">
                  {artist.name}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* Album selector */}
        {presetsWithAdapters.length > 1 && (
          <div className="flex items-center gap-2">
            <label className="text-[10px] font-medium text-zinc-500 uppercase whitespace-nowrap">Album</label>
            <div className="relative flex-1">
              <select
                value={artistPresets.findIndex(p => p.preset === selectedPreset)}
                onChange={e => {
                  const chosen = artistPresets[parseInt(e.target.value)];
                  if (chosen?.preset) {
                    onSelectPreset(chosen.preset);
                    if (chosen.preset.audio_cover_strength != null) onAudioCoverStrength(chosen.preset.audio_cover_strength);
                  }
                }}
                className="w-full appearance-none rounded-lg bg-black/5 dark:bg-white/5 border border-zinc-200 dark:border-zinc-700 px-3 py-1.5 pr-8 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer focus:ring-2 focus:ring-cyan-500/50 focus:outline-none"
              >
                {presetsWithAdapters.map(p => {
                  const idx = artistPresets.indexOf(p);
                  return <option key={p.lsId} value={idx}>{p.album}</option>;
                })}
              </select>
              <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-400 pointer-events-none" />
            </div>
          </div>
        )}

        {/* Selected preset info */}
        {selectedPreset && (
          <div className="rounded-lg bg-cyan-500/5 border border-cyan-500/20 px-3 py-2 space-y-1">
            {selectedPreset.adapter_path && (
              <div className="flex items-center gap-2">
                <Zap className="w-3 h-3 text-pink-400" />
                <span className="text-[10px] text-zinc-500 truncate flex-1">
                  {selectedPreset.adapter_path.split(/[\\/]/).pop()}
                </span>
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-pink-900/30 text-pink-400">ADAPTER</span>
              </div>
            )}
            {selectedPreset.reference_track_path && (
              <div className="flex items-center gap-2">
                <Music className="w-3 h-3 text-amber-400" />
                <span className="text-[10px] text-zinc-500 truncate flex-1">
                  {selectedPreset.reference_track_path.split(/[\\/]/).pop()}
                </span>
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-900/30 text-amber-400">REF</span>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="border-t border-zinc-200 dark:border-white/5" />

      {/* Cover Settings */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Disc3 className="w-4 h-4 text-teal-400" />
          Cover Settings
        </div>
        <EditableSlider label="Structure Fidelity" value={audioCoverStrength} min={0} max={1} step={0.05}
          onChange={onAudioCoverStrength} formatDisplay={v => v.toFixed(2)}
          helpText="How closely the output follows the source's arrangement" />
        <EditableSlider label="Source Timbre" value={coverNoiseStrength} min={0} max={1} step={0.05}
          onChange={onCoverNoiseStrength} formatDisplay={v => v.toFixed(2)}
          helpText="How much of the original artist's sound is preserved" />
        <EditableSlider label="Tempo Scale" value={tempoScale} min={0.5} max={2.0} step={0.05}
          onChange={onTempoScale}
          formatDisplay={v => {
            const bpm = analysis?.bpm;
            return bpm ? `${v.toFixed(2)}x (${Math.round(bpm * v)} BPM)` : `${v.toFixed(2)}x`;
          }}
          helpText={`1.0 = original tempo${analysis?.bpm ? ` (${Math.round(analysis.bpm)} BPM)` : ''}`} />
        <EditableSlider label="Pitch Shift" value={pitchShift} min={-12} max={12} step={1}
          onChange={onPitchShift}
          formatDisplay={v => {
            const shifted = analysis?.key ? transposeKey(analysis.key, v) : null;
            const sign = v > 0 ? '+' : '';
            return shifted && v !== 0 ? `${sign}${v} st → ${shifted}` : `${sign}${v} st`;
          }}
          helpText={`Semitones (-12 to +12)${analysis?.key ? `. Source: ${analysis.key}` : ''}`} />
      </div>

      <div className="border-t border-zinc-200 dark:border-white/5" />

      {/* Generate */}
      {isGenerating ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-cyan-400 font-medium">{genStage || 'Generating...'}</span>
            <span className="text-zinc-500 font-mono">{genProgress}%</span>
          </div>
          <div className="w-full h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-teal-500 transition-all duration-500 rounded-full"
              style={{ width: `${genProgress}%` }} />
          </div>
          <button onClick={onCancel}
            className="w-full py-2 rounded-xl text-xs font-medium text-red-400 hover:bg-red-500/10 transition-colors">
            Cancel
          </button>
        </div>
      ) : (
        <button onClick={onGenerate} disabled={!canGenerate}
          className={`w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl text-sm font-bold transition-all duration-300 shadow-lg
            ${canGenerate
              ? 'bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white shadow-cyan-500/20 hover:shadow-cyan-400/30 hover:scale-[1.02]'
              : 'bg-zinc-200 dark:bg-white/5 text-zinc-400 cursor-not-allowed shadow-none'}`}>
          <Disc3 className="w-4 h-4" />
          Generate Cover
        </button>
      )}
    </div>
  );
};
