// PostProcessingDropdown.tsx — Unified post-processing panel for the global bar
//
// Four collapsible accordion sections:
//   0. PP-VAE          — Neural VAE re-encode for spectral cleanup (auto-detected)
//   1. Spectral Lifter — AI artifact removal with tunable parameters
//   2. VST Chain       — Plugin chain management (existing VstChainDropdown content)
//   3. Mastering       — Reference-based mastering (existing MasteringDropdown content)

import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Sparkles, AudioWaveform, Zap, Image, Mic2, BarChart3,
  Upload, Trash2, Music2,
  ChevronDown, RotateCcw,
} from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { masteringApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { useVstChainStore } from '../../stores/vstChainStore';
import { ToggleSwitch } from './BarSection';
import { formatReferenceName } from './modelLabels';
import { VstChainDropdown } from './VstChainDropdown';
import { CoverArtContent, CoverArtBadge } from './CoverArtDropdown';
import { EditableSlider } from '../shared/EditableSlider';

// ── Accordion Section ───────────────────────────────────────────

interface AccordionProps {
  icon: React.ReactNode;
  label: string;
  badge?: React.ReactNode;
  accentColor: string;
  toggle?: { checked: boolean; onChange: (v: boolean) => void };
  persistKey: string;  // localStorage key for remembering open/closed state
  children: React.ReactNode;
}

const Accordion: React.FC<AccordionProps> = ({
  icon, label, badge, accentColor, toggle, persistKey, children,
}) => {
  const [open, setOpen] = usePersistedState(persistKey, false);

  const accentMap: Record<string, string> = {
    cyan: 'border-cyan-500/20 bg-cyan-500/5',
    violet: 'border-violet-500/20 bg-violet-500/5',
    amber: 'border-amber-500/20 bg-amber-500/5',
    emerald: 'border-emerald-500/20 bg-emerald-500/5',
    pink: 'border-pink-500/20 bg-pink-500/5',
  };
  const activeStyle = toggle?.checked ? (accentMap[accentColor] || '') : '';

  return (
    <div className={`rounded-xl border transition-all ${activeStyle || 'border-zinc-200 dark:border-white/5 bg-zinc-100/30 dark:bg-zinc-800/30'}`}>
      {/* Header */}
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
      >
        <span className={`flex-shrink-0 ${toggle?.checked ? `text-${accentColor}-400` : 'text-zinc-500'}`}>
          {icon}
        </span>
        <span className="text-sm text-zinc-700 dark:text-zinc-300 font-medium flex-1">{label}</span>
        {badge && <span className="flex-shrink-0">{badge}</span>}
        {toggle && (
          <span onClick={e => e.stopPropagation()}>
            <ToggleSwitch checked={toggle.checked} onChange={toggle.onChange} accentColor={accentColor as 'pink' | 'emerald' | 'sky' | 'purple' | 'amber'} />
          </span>
        )}
        <ChevronDown
          size={14}
          className={`text-zinc-500 transition-transform flex-shrink-0 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Content */}
      {open && (
        <div className="px-3 pb-3 pt-0 border-t border-zinc-200 dark:border-white/5">
          {children}
        </div>
      )}
    </div>
  );
};

// ── Mastering Content ───────────────────────────────────────────
// (Extracted from MasteringDropdown — reference track management)

interface ReferenceTrack {
  name: string;
  size: number;
  url: string;
}

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const MasteringContent: React.FC = () => {
  const gp = useGlobalParams();
  const { t } = useTranslation();
  const { token } = useAuth();
  const [references, setReferences] = useState<ReferenceTrack[]>([]);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    masteringApi.listReferences()
      .then(data => setReferences(data.references))
      .catch(() => {});
  }, []);

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !token) return;
    try {
      setUploading(true);
      const result = await masteringApi.uploadReference(file, token);
      gp.setMasteringReference(result.name);
      const data = await masteringApi.listReferences();
      setReferences(data.references);
    } catch (err) {
      console.error('[Mastering] Upload failed:', err);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  }, [token, gp]);

  const handleDelete = useCallback(async (name: string) => {
    if (!token) return;
    try {
      await masteringApi.deleteReference(name, token);
      if (gp.masteringReference === name) gp.setMasteringReference('');
      const data = await masteringApi.listReferences();
      setReferences(data.references);
    } catch (err) {
      console.error('[Mastering] Delete failed:', err);
    }
  }, [token, gp]);

  if (!gp.masteringEnabled) {
    return (
      <div className="text-xs text-zinc-500 italic text-center py-2 mt-2">
        {t('mastering.enableToggle')}
      </div>
    );
  }

  return (
    <div className="space-y-3 mt-2">
      {/* Reference selector */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          {t('mastering.referenceTrack')}
        </label>
        {references.length > 0 ? (
          <select
            className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 outline-none transition-colors cursor-pointer"
            value={gp.masteringReference}
            onChange={e => gp.setMasteringReference(e.target.value)}
          >
            <option value="">{t('mastering.selectReference')}</option>
            {references.map(r => (
              <option key={r.name} value={r.name}>
                {r.name} ({formatFileSize(r.size)})
              </option>
            ))}
          </select>
        ) : (
          <div className="text-xs text-zinc-500 italic px-1">
            {t('mastering.noReferencesYet')}
          </div>
        )}
      </div>

      {/* Selected reference info + delete */}
      {gp.masteringReference && (
        <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-amber-500/5 border border-amber-500/10">
          <Music2 size={14} className="text-amber-400 flex-shrink-0" />
          <span className="text-xs text-amber-300 truncate flex-1">{gp.masteringReference}</span>
          <button
            onClick={() => handleDelete(gp.masteringReference)}
            className="p-1 rounded hover:bg-red-500/10 text-zinc-500 hover:text-red-400 transition-colors flex-shrink-0"
            title={t('mastering.deleteReference')}
          >
            <Trash2 size={12} />
          </button>
        </div>
      )}

      {/* Upload button */}
      <div className="flex items-center gap-2">
        <input
          type="file"
          accept="audio/*"
          id="mastering-ref-upload-pp"
          className="hidden"
          onChange={handleUpload}
        />
        <label
          htmlFor="mastering-ref-upload-pp"
          className={`flex items-center gap-2 px-3 py-2 text-xs font-semibold rounded-xl border cursor-pointer transition-all ${
            uploading
              ? 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 border-zinc-200 dark:border-white/5 cursor-wait'
              : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border-zinc-300 dark:border-white/10 hover:border-amber-500/30 hover:text-amber-400'
          }`}
        >
          {uploading ? (
            <><span className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" /> Uploading...</>
          ) : (
            <><Upload size={14} /> {t('mastering.uploadReference')}</>
          )}
        </label>
      </div>

      {/* Timbre reference toggle */}
      {gp.masteringReference && (
        gp.timbreAudioPath ? (
          <div className="flex items-center gap-1.5 mt-1 px-2 py-1.5 rounded-lg bg-teal-500/5 border border-teal-500/10">
            <Music2 size={14} className="text-teal-400" />
            <span className="text-[10px] text-teal-400">Timbre: using dedicated reference ({gp.timbreAudioPath.split(/[\\/]/).pop()})</span>
          </div>
        ) : (
          <div className="flex items-center justify-between mt-1">
            <div className="flex items-center gap-1.5">
              <Music2 size={14} className="text-teal-400" />
              <span className="text-sm text-zinc-600 dark:text-zinc-400">{t('mastering.alsoTimbreRef')}</span>
            </div>
            <ToggleSwitch checked={gp.timbreReference} onChange={gp.setTimbreReference} accentColor="amber" />
          </div>
        )
      )}
      {gp.timbreReference && gp.masteringReference && !gp.timbreAudioPath && (
        <p className="text-[10px] text-zinc-600 leading-relaxed">
          The reference track will be VAE-encoded and fed into the timbre conditioning pipeline,
          guiding the generation&apos;s tone and texture to match the reference.
        </p>
      )}

      <p className="text-[10px] text-zinc-600 leading-relaxed">
        Match the RMS level, frequency spectrum, and dynamic characteristics of the reference.
      </p>
    </div>
  );
};

// ── Main Dropdown ───────────────────────────────────────────────

export const PostProcessingDropdown: React.FC = () => {
  const { t } = useTranslation();
  const gp = useGlobalParams();
  const { chain } = useVstChainStore();
  const vstEnabled = chain.filter(p => p.enabled).length;

  // PP-VAE availability — auto-detect from models directory
  const [ppVaeAvailable, setPpVaeAvailable] = useState(false);
  useEffect(() => {
    fetch('/api/models/pp-vae')
      .then(r => r.json())
      .then(data => setPpVaeAvailable(!!data.available))
      .catch(() => setPpVaeAvailable(false));
  }, []);

  return (
    <div className="space-y-2">
      {/* LRC Subtitle Generation Toggle */}
      <div className="flex items-center justify-between px-1 py-1.5">
        <div className="flex items-center gap-2">
          <AudioWaveform size={14} className={gp.skipLrc ? 'text-zinc-500' : 'text-sky-400'} />
          <div className="flex flex-col">
            <span className="text-sm text-zinc-700 dark:text-zinc-300 font-medium">Lyric Timestamps (LRC)</span>
            <span className="text-[10px] text-zinc-500 leading-tight">Synchronized lyric alignment for karaoke-style playback</span>
          </div>
        </div>
        <ToggleSwitch checked={!gp.skipLrc} onChange={v => gp.setSkipLrc(!v)} accentColor="sky" />
      </div>
      {/* 0. PP-VAE Re-encode (only visible when PP-VAE model is available) */}
      {ppVaeAvailable && (
        <Accordion
          icon={<AudioWaveform size={14} />}
          label={t('pp.ppVaeReencode')}
          accentColor="emerald"
          persistKey="hs-ppAccordion-ppvae"
          toggle={{ checked: gp.ppVaeReencode, onChange: gp.setPpVaeReencode }}
        >
          <div className="space-y-2 mt-2">
            <p className="text-[10px] text-zinc-500 leading-relaxed">
              Neural VAE re-encode pass. Runs the decoded audio through a
              higher-fidelity autoencoder to clean up spectral artifacts,
              fizz, and high-frequency noise from the primary VAE.
              Adds ~1–2s processing time.
            </p>
            {gp.ppVaeReencode && (
              <div className="space-y-2 pt-1">
                <EditableSlider
                  label="Original Blend"
                  value={gp.ppVaeBlend}
                  min={0} max={1.0} step={0.01}
                  onChange={gp.setPpVaeBlend}
                  formatDisplay={v => v === 0 ? 'Full PP-VAE' : v >= 1 ? 'Original' : (v * 100).toFixed(0) + '% original'}
                  tooltip="Blend original audio back into the PP-VAE output. 0% = fully processed, 100% = fully original."
                />
                <p className="text-[10px] text-emerald-400/60 leading-relaxed">
                  ✓ PP-VAE model detected — will re-encode after VAE decode.
                </p>
              </div>
            )}
          </div>
        </Accordion>
      )}

      {/* 1. Spectral Lifter */}
      <Accordion
        icon={<Zap size={14} />}
        label={t('pp.spectralLifter')}
        accentColor="cyan"
        persistKey="hs-ppAccordion-sl"
        toggle={{ checked: gp.spectralLifterEnabled, onChange: gp.setSpectralLifterEnabled }}
      >
        <div className="space-y-3 mt-2">
          <p className="text-[10px] text-zinc-500 leading-relaxed">
            Removes AI shimmer artifacts, reduces spectral noise, and extends
            high-frequency content. Native C++ processing — runs in the engine
            post-VAE pipeline.
          </p>

          {gp.spectralLifterEnabled && (
            <div className="space-y-2 pt-1">
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => {
                    gp.setSlDenoiseStrength(0.3);
                    gp.setSlNoiseFloor(0.1);
                    gp.setSlHfMix(0.0);
                    gp.setSlTransientBoost(0.0);
                    gp.setSlShimmerReduction(6.0);
                  }}
                  className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-cyan-400 transition-colors"
                  title="Reset Spectral Lifter parameters to defaults"
                >
                  <RotateCcw size={11} />
                  Reset
                </button>
              </div>
              <EditableSlider
                label="Denoise Strength"
                value={gp.slDenoiseStrength}
                min={0} max={1.0} step={0.01}
                onChange={gp.setSlDenoiseStrength}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Spectral gate aggressiveness. Higher = more noise removal. 0 = skip denoising entirely."
              />
              <EditableSlider
                label="Noise Floor"
                value={gp.slNoiseFloor}
                min={0.01} max={0.5} step={0.01}
                onChange={gp.setSlNoiseFloor}
                formatDisplay={v => (v * 100).toFixed(0) + '%'}
                tooltip="Minimum signal that passes through the gate. Higher = gentler, less musical noise artifacts."
              />
              <EditableSlider
                label="HF Extension"
                value={gp.slHfMix}
                min={0} max={0.5} step={0.01}
                onChange={gp.setSlHfMix}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Blend amount for synthesized high-frequency content above 16kHz via spectral mirroring."
              />
              <EditableSlider
                label="Transient Boost"
                value={gp.slTransientBoost}
                min={0} max={1.0} step={0.01}
                onChange={gp.setSlTransientBoost}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Percussive enhancement via harmonic-percussive separation. Adds punch to drums and transients."
              />
              <EditableSlider
                label="Shimmer Reduction"
                value={gp.slShimmerReduction}
                min={0} max={12} step={0.5}
                onChange={gp.setSlShimmerReduction}
                formatDisplay={v => v === 0 ? 'Off' : v.toFixed(1) + ' dB'}
                tooltip="dB reduction applied to the 10–14kHz shimmer band. Higher = more aggressive shimmer suppression."
              />
            </div>
          )}
        </div>
      </Accordion>

      {/* 2. Vocal Naturalizer */}
      <Accordion
        icon={<Mic2 size={14} />}
        label={t('pp.vocalNaturalizer')}
        accentColor="pink"
        persistKey="hs-ppAccordion-naturalizer"
        toggle={{ checked: gp.vocalNaturalizerEnabled, onChange: gp.setVocalNaturalizerEnabled }}
      >
        <div className="space-y-2 mt-2">
          <p className="text-[10px] text-zinc-500 leading-relaxed">
            Applies 5-stage DSP humanisation to reduce robotic/auto-tune
            artifacts. Processes the full mix using frequency-band-targeted
            filters that primarily affect vocal content.
          </p>
          {gp.vocalNaturalizerEnabled && (
            <div className="flex items-start gap-2 px-2.5 py-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <span className="text-amber-400 text-xs mt-px">⚠</span>
              <p className="text-[10px] text-amber-300/80 leading-relaxed">
                <strong>Experimental.</strong> This feature may subtly degrade audio quality
                and interfere with downstream VST/mastering processing. A/B test
                with it disabled to verify it&apos;s improving your output.
              </p>
            </div>
          )}
          {gp.vocalNaturalizerEnabled && (
            <div className="space-y-2 pt-1">
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => {
                    gp.setNaturalizeAmount(0.5);
                    gp.setNatVibratoRate(4.5);
                    gp.setNatVibratoDepth(1.0);
                    gp.setNatFormantStrength(1.0);
                    gp.setNatMetallicReduction(1.0);
                    gp.setNatQuantizationMask(0.0);
                    gp.setNatTransitionSmooth(1.0);
                  }}
                  className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-pink-400 transition-colors"
                  title="Reset Vocal Naturalizer parameters to defaults"
                >
                  <RotateCcw size={11} />
                  Reset
                </button>
              </div>
              <EditableSlider
                label="Amount"
                value={gp.naturalizeAmount}
                min={0} max={1.0} step={0.01}
                onChange={gp.setNaturalizeAmount}
                formatDisplay={v => v === 0 ? 'Off' : v <= 0.3 ? 'Subtle' : v <= 0.6 ? 'Moderate' : v <= 0.8 ? 'Strong' : 'Maximum'}
                tooltip="Master intensity — scales all 5 naturalisation stages proportionally."
              />
              <EditableSlider
                label="Vibrato Rate"
                value={gp.natVibratoRate}
                min={3.0} max={7.0} step={0.1}
                onChange={gp.setNatVibratoRate}
                formatDisplay={v => v.toFixed(1) + ' Hz'}
                tooltip="Vibrato speed for pitch variation. Natural human vibrato is ~4–6 Hz."
              />
              <EditableSlider
                label="Vibrato Depth"
                value={gp.natVibratoDepth}
                min={0} max={1.0} step={0.01}
                onChange={gp.setNatVibratoDepth}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Pitch variation intensity. Breaks rigid pitch quantization from auto-tune."
              />
              <EditableSlider
                label="Formant Humanize"
                value={gp.natFormantStrength}
                min={0} max={1.0} step={0.01}
                onChange={gp.setNatFormantStrength}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Adds subtle variation to the 200–3000 Hz formant band to humanize locked timbre."
              />
              <EditableSlider
                label="Metallic Cut"
                value={gp.natMetallicReduction}
                min={0} max={1.0} step={0.01}
                onChange={gp.setNatMetallicReduction}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Reduces harsh digital artifacts in the 6–10 kHz range."
              />
              <EditableSlider
                label="Quantization Mask"
                value={gp.natQuantizationMask}
                min={0} max={1.0} step={0.01}
                onChange={gp.setNatQuantizationMask}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Shaped noise (1–4 kHz) to mask pitch 'stair-stepping' from quantization."
              />
              <EditableSlider
                label="Transition Smooth"
                value={gp.natTransitionSmooth}
                min={0} max={1.0} step={0.01}
                onChange={gp.setNatTransitionSmooth}
                formatDisplay={v => v === 0 ? 'Off' : (v * 100).toFixed(0) + '%'}
                tooltip="Smooths abrupt pitch transitions into natural glides between notes."
              />
            </div>
          )}
        </div>
      </Accordion>

      {/* 3. VST Chain */}
      <Accordion
        icon={<Sparkles size={14} />}
        label={t('pp.vstChain')}
        accentColor="violet"
        persistKey="hs-ppAccordion-vst"
        badge={vstEnabled > 0 ? (
          <span className="text-[10px] text-violet-400/60 font-mono">
            {vstEnabled} active
          </span>
        ) : undefined}
      >
        <div className="mt-2">
          <VstChainDropdown />
        </div>
      </Accordion>

      {/* 3. Mastering */}
      <Accordion
        icon={<AudioWaveform size={14} />}
        label={t('pp.mastering')}
        accentColor="amber"
        persistKey="hs-ppAccordion-master"
        toggle={{ checked: gp.masteringEnabled, onChange: gp.setMasteringEnabled }}
        badge={gp.masteringEnabled && gp.masteringReference ? (
          <span className="text-[10px] text-amber-400/60 font-mono truncate max-w-[100px]">
            {formatReferenceName(gp.masteringReference)}
          </span>
        ) : undefined}
      >
        <MasteringContent />
      </Accordion>

      {/* 4. Cover Art */}
      <Accordion
        icon={<Image size={14} />}
        label="Cover Art"
        accentColor="violet"
        persistKey="hs-ppAccordion-coverart"
        toggle={{ checked: gp.coverArtEnabled, onChange: gp.setCoverArtEnabled }}
        badge={gp.coverArtEnabled ? (
          <CoverArtBadge />
        ) : undefined}
      >
        <CoverArtContent />
      </Accordion>

      {/* 5. Quality Evaluator */}
      <Accordion
        icon={<BarChart3 size={14} />}
        label={t('pp.qualityEval')}
        accentColor="emerald"
        persistKey="hs-ppAccordion-quality"
        toggle={{ checked: gp.qualityEvalEnabled, onChange: gp.setQualityEvalEnabled }}
        badge={gp.qualityEvalEnabled ? (
          <span className="text-[10px] text-emerald-400/60 font-mono">
            {gp.qualityEvalTarget === 'both' ? 'Raw+Master' : gp.qualityEvalTarget === 'mastered' ? 'Mastered' : 'Unmastered'}
          </span>
        ) : undefined}
      >
        <div className="space-y-3 py-2">
          <p className="text-xs text-zinc-500 leading-relaxed">
            {t('pp.qualityEvalDesc')}
          </p>
          <div>
            <label className="text-[11px] font-medium text-zinc-500 mb-1.5 block">{t('pp.qualityEvalTarget')}</label>
            <div className="flex rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden">
              {(['unmastered', 'mastered', 'both'] as const).map(opt => (
                <button
                  key={opt}
                  onClick={() => gp.setQualityEvalTarget(opt)}
                  className={`
                    flex-1 px-2 py-1.5 text-[11px] font-semibold transition-all
                    ${gp.qualityEvalTarget === opt
                      ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                      : 'text-zinc-500 hover:text-zinc-300 hover:bg-white/5'
                    }
                    ${opt !== 'unmastered' ? 'border-l border-zinc-200 dark:border-white/10' : ''}
                  `}
                >
                  {t(`pp.qualityEval${opt.charAt(0).toUpperCase() + opt.slice(1)}`)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Accordion>
    </div>
  );
};

// ── Badge ───────────────────────────────────────────────────────

export const PostProcessingBadge: React.FC = () => {
  const { masteringEnabled, masteringReference, spectralLifterEnabled, ppVaeReencode, coverArtEnabled, vocalNaturalizerEnabled, qualityEvalEnabled } = useGlobalParams();
  const { chain } = useVstChainStore();
  const vstEnabled = chain.filter(p => p.enabled).length;

  const parts: string[] = [];
  if (ppVaeReencode) parts.push('PP-VAE');
  if (spectralLifterEnabled) parts.push('SL');
  if (vocalNaturalizerEnabled) parts.push('Nat');
  if (vstEnabled > 0) parts.push(`${vstEnabled} VST${vstEnabled !== 1 ? 's' : ''}`);
  if (masteringEnabled && masteringReference) parts.push('Master');
  if (coverArtEnabled) parts.push('Cover');
  if (qualityEvalEnabled) parts.push('QE');

  if (parts.length === 0) return null;

  return (
    <span className="text-[10px] text-amber-400/60 font-mono truncate">
      {parts.join(' + ')}
    </span>
  );
};
