// ArtistSettingsPanel.tsx — Right panel: artist selector + cover settings + generate
import React from 'react';
import { Guitar, Disc3, Zap, Music, ChevronDown, Loader2, Type, X, Mic } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { EditableSlider } from './EditableSlider';
import { transposeKey, type AudioAnalysis } from './coverStudioUtils';
import type { Artist, AlbumPreset } from '../../services/lireekApi';

interface ArtistSettingsPanelProps {
  artists: Artist[];
  isLoadingArtists: boolean;
  selectedArtistId: number | null;
  onSelectArtist: (a: Artist) => void;
  onClearArtist: () => void;
  artistPresets: { lsId: number; album: string; preset: AlbumPreset | null }[];
  selectedPreset: AlbumPreset | null;
  onSelectPreset: (p: AlbumPreset | null) => void;
  audioCoverStrength: number;
  onAudioCoverStrength: (v: number) => void;
  coverNoiseStrength: number;
  onCoverNoiseStrength: (v: number) => void;
  coverNoiseMethod: string;
  onCoverNoiseMethodChange: (v: string) => void;
  noFsq: boolean;
  onNoFsqChange: (v: boolean) => void;
  instrumental: boolean;
  onInstrumentalChange: (v: boolean) => void;
  tempoScale: number;
  onTempoScale: (v: number) => void;
  pitchShift: number;
  onPitchShift: (v: number) => void;
  analysis: AudioAnalysis | null;
  bpmCorrection: number;
  keyOverride: string | null;
  artistCaption: string;
  onArtistCaptionChange: (v: string) => void;
  canGenerate: boolean;
  isGenerating: boolean;
  genProgress: number;
  genStage: string;
  onGenerate: () => void;
  onCancel: () => void;
}

export const ArtistSettingsPanel: React.FC<ArtistSettingsPanelProps> = (props) => {
  const {
    artists, isLoadingArtists, selectedArtistId, onSelectArtist, onClearArtist,
    artistPresets, selectedPreset, onSelectPreset,
    audioCoverStrength, onAudioCoverStrength, coverNoiseStrength, onCoverNoiseStrength,
    coverNoiseMethod, onCoverNoiseMethodChange,
    noFsq, onNoFsqChange, instrumental, onInstrumentalChange,
    tempoScale, onTempoScale, pitchShift, onPitchShift, analysis, bpmCorrection, keyOverride,
    artistCaption, onArtistCaptionChange,
    canGenerate, isGenerating, genProgress, genStage, onGenerate, onCancel,
  } = props;

  const { t } = useTranslation();
  const presetsWithAdapters = artistPresets.filter(p => p.preset?.adapter_path);
  const effectiveKey = keyOverride || analysis?.key || null;

  return (
    <div className="w-[540px] flex-shrink-0 overflow-y-auto scrollbar-hide p-4 space-y-4">
      {/* Target Artist */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Guitar className="w-4 h-4 text-cyan-400" />
          {t('cover.targetArtist')}
          <span className="text-[10px] font-normal text-zinc-500">(optional)</span>
          {selectedArtistId && (
            <button
              onClick={onClearArtist}
              className="ml-auto flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-medium text-zinc-400 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            >
              <X className="w-3 h-3" />
              {t('cover.clearArtist')}
            </button>
          )}
        </div>
        {isLoadingArtists ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 text-zinc-500 animate-spin" />
          </div>
        ) : artists.length === 0 ? (
          <div className="text-center py-4">
            <p className="text-xs text-zinc-500">{t('cover.noArtistsAvailable')}</p>
            <p className="text-[10px] text-zinc-600 dark:text-zinc-400 mt-1">Describe the target style below, or add artists in Lyric Studio for adapter-powered covers.</p>
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
                  }
                }}
                className="w-full appearance-none rounded-lg bg-black/5 dark:bg-white/5 border border-zinc-200 dark:border-zinc-700 px-3 py-1.5 pr-8 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer focus:ring-2 focus:ring-cyan-500/50 focus:outline-none"
              >
                {presetsWithAdapters.map(p => {
                  const idx = artistPresets.indexOf(p);
                  return <option key={p.lsId} value={idx}>{p.album}</option>;
                })}
              </select>
              <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-600 dark:text-zinc-400 pointer-events-none" />
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

      {/* Style Description */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Type className="w-4 h-4 text-purple-400" />
          Style Description
        </div>
        <textarea
          value={artistCaption}
          onChange={e => onArtistCaptionChange(e.target.value)}
          placeholder="Describe the target style, e.g. 'indie rock, breathy female vocal, lo-fi production, dreamy reverb, 2010s alternative'"
          className="w-full h-24 resize-none rounded-xl bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 px-3 py-2 text-xs text-zinc-900 dark:text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-purple-500 transition-colors leading-relaxed"
        />
        <p className="text-[10px] text-zinc-500 leading-tight">
          {selectedArtistId
            ? 'Auto-filled from artist. Edit freely to refine the style.'
            : 'Describe genre, instruments, vocal style, production, mood — used as the generation caption.'}
        </p>
      </div>

      <div className="border-t border-zinc-200 dark:border-white/5" />

      {/* Instrumental Toggle */}
      <div className="flex items-center justify-between px-1 py-1">
        <div className="flex items-center gap-2">
          <Mic className="w-4 h-4 text-amber-400" />
          <div>
            <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">{t('cover.instrumental')}</span>
            <p className="text-[10px] text-zinc-500 leading-tight mt-0.5">{t('cover.instrumentalHelp')}</p>
          </div>
        </div>
        <button
          onClick={() => onInstrumentalChange(!instrumental)}
          className={`relative w-9 h-5 rounded-full transition-colors duration-200 flex-shrink-0 ${
            instrumental ? 'bg-amber-500' : 'bg-zinc-300 dark:bg-zinc-700'
          }`}
        >
          <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${
            instrumental ? 'translate-x-[18px]' : 'translate-x-0.5'
          }`} />
        </button>
      </div>

      <div className="border-t border-zinc-200 dark:border-white/5" />

      {/* Cover Settings */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Disc3 className="w-4 h-4 text-teal-400" />
          {t('cover.coverSettings')}
        </div>
        <EditableSlider label="Structure Fidelity" value={audioCoverStrength} min={0} max={1} step={0.05}
          onChange={onAudioCoverStrength} formatDisplay={v => v.toFixed(2)}
          helpText="How closely the output follows the source's arrangement" />
        <EditableSlider label={t('cover.sourcePreservation')} value={coverNoiseStrength} min={0} max={1} step={0.05}
          onChange={onCoverNoiseStrength} formatDisplay={v => v.toFixed(2)}
          helpText={t('cover.sourcePreservationHelp')} />
        {coverNoiseStrength > 0 && (
          <div className="flex items-center justify-between px-1 py-1">
            <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">Noise Method</span>
            <select
              className="appearance-none rounded-lg bg-black/5 dark:bg-white/5 border border-zinc-200 dark:border-zinc-700 px-3 py-1.5 pr-8 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer focus:ring-2 focus:ring-cyan-500/50 focus:outline-none"
              value={coverNoiseMethod}
              onChange={(e) => onCoverNoiseMethodChange(e.target.value)}
            >
              <option value="">Classic (Truncate)</option>
              <option value="rescale">Full Denoise (Rescale)</option>
            </select>
          </div>
        )}
        {/* NoFSQ toggle */}
        <div className="flex items-center justify-between px-1 py-1">
          <div>
            <span className="text-xs font-medium text-zinc-700 dark:text-zinc-300">NoFSQ Mode</span>
            <p className="text-[10px] text-zinc-500 leading-tight mt-0.5">Skip quantization — more faithful to source</p>
          </div>
          <button
            onClick={() => onNoFsqChange(!noFsq)}
            className={`relative w-9 h-5 rounded-full transition-colors duration-200 flex-shrink-0 ${
              noFsq ? 'bg-cyan-500' : 'bg-zinc-300 dark:bg-zinc-700'
            }`}
          >
            <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${
              noFsq ? 'translate-x-[18px]' : 'translate-x-0.5'
            }`} />
          </button>
        </div>
        <EditableSlider label="Tempo Scale" value={tempoScale} min={0.5} max={2.0} step={0.05}
          onChange={onTempoScale}
          formatDisplay={v => {
            const bpm = analysis?.bpm ? Math.round(analysis.bpm * bpmCorrection) : null;
            return bpm ? `${v.toFixed(2)}x (${Math.round(bpm * v)} BPM)` : `${v.toFixed(2)}x`;
          }}
          helpText={`1.0 = original tempo${analysis?.bpm ? ` (${Math.round(analysis.bpm * bpmCorrection)} BPM)` : ''}`} />
        <EditableSlider label="Pitch Shift" value={pitchShift} min={-12} max={12} step={1}
          onChange={onPitchShift}
          formatDisplay={v => {
            const shifted = effectiveKey ? transposeKey(effectiveKey, v) : null;
            const sign = v > 0 ? '+' : '';
            return shifted && v !== 0 ? `${sign}${v} st → ${shifted}` : `${sign}${v} st`;
          }}
          helpText={`Semitones (-12 to +12)${effectiveKey ? `. Source: ${effectiveKey}` : ''}`} />
      </div>

      <div className="border-t border-zinc-200 dark:border-white/5" />

      {/* Generate */}
      {isGenerating ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-cyan-400 font-medium">{genStage || 'Generating...'}</span>
            <span className="text-zinc-500 font-mono">{genProgress}%</span>
          </div>
          <div className="w-full h-2 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-teal-500 transition-all duration-500 rounded-full"
              style={{ width: `${genProgress}%` }} />
          </div>
          <button onClick={onCancel}
            className="w-full py-2 rounded-xl text-xs font-medium text-red-400 hover:bg-red-500/10 transition-colors">
            {t('common.cancel')}
          </button>
        </div>
      ) : (
        <button onClick={onGenerate} disabled={!canGenerate}
          className={`w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl text-sm font-bold transition-all duration-300 shadow-lg
            ${canGenerate
              ? 'bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white shadow-cyan-500/20 hover:shadow-cyan-400/30 hover:scale-[1.02]'
              : 'bg-zinc-200 dark:bg-white/5 text-zinc-600 dark:text-zinc-400 cursor-not-allowed shadow-none'}`}>
          <Disc3 className="w-4 h-4" />
          {t('cover.generateCover')}
        </button>
      )}
    </div>
  );
};
