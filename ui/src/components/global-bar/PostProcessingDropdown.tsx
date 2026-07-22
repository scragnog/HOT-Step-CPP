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
import { PluginControls } from './PluginControls';
import { EditableSlider } from '../shared/EditableSlider';

// LUFS normalization presets
const LUFS_PRESETS = [
  { id: 'spotify', label: 'Spotify / YouTube Music', lufs: -14 },
  { id: 'apple', label: 'Apple Music / Tidal', lufs: -16 },
  { id: 'ebu', label: 'EBU R128 (Broadcast)', lufs: -23 },
  { id: 'club', label: 'Club / DJ Playback', lufs: -8 },
  { id: 'custom', label: 'Custom', lufs: null },
] as const;

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
    sky: 'border-sky-500/20 bg-sky-500/5',
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

  // StableStep (SA3) availability — auto-detect ONNX (models/onnx/sa3) and
  // GGML (root GGUFs) backend installs from the extended availability endpoint.
  const [stableStepBackends, setStableStepBackends] = useState<{ onnx: boolean; gguf: boolean }>({ onnx: false, gguf: false });
  const [stableStepAvailable, setStableStepAvailable] = useState(false);
  useEffect(() => {
    fetch('/api/models/stablestep')
      .then(r => r.json())
      .then(data => {
        setStableStepAvailable(!!data.available);
        setStableStepBackends({ onnx: !!data.backends?.onnx, gguf: !!data.backends?.gguf });
      })
      .catch(() => { setStableStepAvailable(false); setStableStepBackends({ onnx: false, gguf: false }); });
  }, []);

  // Postprocess plugin availability — fetch from /api/plugins
  const [postprocessPlugins, setPostprocessPlugins] = useState<any[]>([]);
  useEffect(() => {
    fetch('/api/plugins')
      .then(r => r.json())
      .then(data => {
        const pp = data.postprocess || [];
        setPostprocessPlugins(pp);
        // Auto-select first plugin if none selected
        if (pp.length > 0 && !gp.postprocessPlugin) {
          gp.setPostprocessPlugin(pp[0].name);
        }
      })
      .catch(() => setPostprocessPlugins([]));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedPostprocessPlugin = postprocessPlugins.find((p: any) => p.name === gp.postprocessPlugin) || postprocessPlugins[0] || null;

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

      {/* Whisper Lyrics Transcription */}
      <Accordion
        icon={<Mic2 size={14} />}
        label="Whisper Lyrics"
        accentColor="sky"
        persistKey="hs-ppAccordion-whisper"
        toggle={{ checked: gp.whisperLyricsEnabled, onChange: gp.setWhisperLyricsEnabled }}
      >
        <div className="space-y-3 mt-2">
          <p className="text-[10px] text-zinc-500 leading-relaxed">
            Transcribes actual sung lyrics using Whisper AI with word-level timestamps.
            Uses source lyrics as a spelling guide. Requires a Whisper model from the Model Manager.
          </p>
          {gp.whisperLyricsEnabled && (
            <div className="space-y-2 pt-1">
              {/* Model selector */}
              <div>
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1">Model</label>
                <select
                  className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 outline-none transition-colors cursor-pointer"
                  value={gp.whisperModel}
                  onChange={e => gp.setWhisperModel(e.target.value)}
                >
                  <option value="">Auto-detect</option>
                  <option value="ggml-large-v3-turbo.bin">Large v3 Turbo (recommended)</option>
                  <option value="ggml-large-v3.bin">Large v3 (best accuracy)</option>
                  <option value="ggml-medium.bin">Medium</option>
                  <option value="ggml-base.bin">Base (fastest)</option>
                </select>
              </div>
              {/* Language */}
              <div>
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1">Language</label>
                <select
                  className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-sky-500/50 focus:ring-1 focus:ring-sky-500/20 outline-none transition-colors cursor-pointer"
                  value={gp.whisperLanguage}
                  onChange={e => gp.setWhisperLanguage(e.target.value)}
                >
                  <option value="auto">Auto-detect</option>
                  <option value="en">English</option>
                  <option value="ja">Japanese</option>
                  <option value="zh">Chinese</option>
                  <option value="ko">Korean</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                  <option value="ru">Russian</option>
                </select>
              </div>
              {/* Beam size */}
              <EditableSlider
                label="Beam Size"
                value={gp.whisperBeamSize}
                min={1} max={10} step={1}
                onChange={gp.setWhisperBeamSize}
                formatDisplay={v => v === 1 ? 'Greedy' : `${v} beams`}
                tooltip="Higher = more accurate but slower. 5 is recommended."
              />
              {/* Vocal isolation toggle */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Mic2 size={14} className="text-sky-400" />
                  <span className="text-sm text-zinc-600 dark:text-zinc-400">Isolate vocals first</span>
                </div>
                <ToggleSwitch checked={gp.whisperIsolateVocals} onChange={gp.setWhisperIsolateVocals} accentColor="sky" />
              </div>
              {gp.whisperIsolateVocals && (
                <p className="text-[10px] text-sky-400/60 leading-relaxed">
                  Runs stem separation to isolate vocals before transcription. May improve accuracy for busy mixes.
                </p>
              )}
            </div>
          )}
        </div>
      </Accordion>

      {/* 0. Postprocess Plugin (Tiled Decoder) — runs before PP-VAE in pipeline */}
      {postprocessPlugins.length > 0 && (
        <Accordion
          icon={<Zap size={14} />}
          label="Tiled Decoder"
          accentColor="cyan"
          persistKey="hs-ppAccordion-tiled"
          toggle={{ checked: gp.postprocessEnabled, onChange: gp.setPostprocessEnabled }}
        >
          <div className="space-y-2 mt-2">
            <p className="text-[10px] text-zinc-500 leading-relaxed">
              Replaces the built-in VAE decoder with an advanced tiled decode pipeline.
              Features OLA crossfading, optional dual-pass merge, latent channel suppression,
              and a DSP chain (hum notch, stereo width, soft clip).
            </p>
            {gp.postprocessEnabled && (
              <div className="space-y-2 pt-1">
                {postprocessPlugins.length > 1 && (
                  <select
                    className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/20 outline-none transition-colors cursor-pointer"
                    value={gp.postprocessPlugin}
                    onChange={e => gp.setPostprocessPlugin(e.target.value)}
                  >
                    {postprocessPlugins.map((p: any) => (
                      <option key={p.name} value={p.name}>{p.display || p.name}</option>
                    ))}
                  </select>
                )}
                {selectedPostprocessPlugin && (
                  <p className="text-[10px] text-cyan-400/60 leading-relaxed">
                    ✓ {selectedPostprocessPlugin.display || selectedPostprocessPlugin.name} — {selectedPostprocessPlugin.description || 'Active'}
                  </p>
                )}
                {selectedPostprocessPlugin?.params?.length > 0 && (
                  <PluginControls
                    pluginName={selectedPostprocessPlugin.name}
                    displayName={selectedPostprocessPlugin.display || selectedPostprocessPlugin.name}
                    accent={selectedPostprocessPlugin.accent || 'cyan'}
                    params={selectedPostprocessPlugin.params}
                    values={gp.pluginParams}
                    onChange={gp.setPluginParam}
                    onReset={() => gp.resetPluginParams(selectedPostprocessPlugin.name)}
                  />
                )}
              </div>
            )}
          </div>
        </Accordion>
      )}

      {/* 1. PP-VAE Re-encode (only visible when PP-VAE model is available) */}
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
                {/* Backend selector: ONNX (ORT/TRT) vs GGUF (GGML) */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <Zap size={14} className={gp.ppVaeUseOnnx ? 'text-emerald-400' : 'text-zinc-500'} />
                    <span className="text-sm text-zinc-600 dark:text-zinc-400">ONNX (ORT/TRT)</span>
                  </div>
                  <ToggleSwitch checked={gp.ppVaeUseOnnx} onChange={gp.setPpVaeUseOnnx} accentColor="emerald" />
                </div>
                <p className="text-[10px] text-zinc-500 leading-relaxed">
                  {gp.ppVaeUseOnnx
                    ? 'Using ONNX Runtime with TensorRT acceleration. Falls back to GGUF if ONNX models are missing.'
                    : 'Using GGUF (GGML) backend. Slower but proven stable.'}
                </p>
              </div>
            )}
          </div>
        </Accordion>
      )}

      {/* 1.5. StableStep — SA3 refine of the instrumental */}
      <Accordion
        icon={<Sparkles size={14} />}
        label="StableStep"
        accentColor="sky"
        persistKey="hs-ppAccordion-stablestep"
        toggle={stableStepAvailable
          ? { checked: gp.stableStepOn, onChange: gp.setStableStepOn }
          : undefined}
        badge={!stableStepAvailable ? (
          <span className="text-[10px] text-zinc-500 font-mono">not installed</span>
        ) : undefined}
      >
        <div className="space-y-2 mt-2">
          <p className="text-[10px] text-zinc-500 leading-relaxed">
            Re-renders the instrumental through Stable Audio 3 to replace VAE fizz
            with real detail; vocals are split out, cleaned with PP-VAE, and remixed.
            {' '}<span className="text-zinc-600">Powered by Stability AI.</span>
          </p>
          {!stableStepAvailable && (
            <div className="flex items-start gap-2 px-2.5 py-2 rounded-lg bg-sky-500/10 border border-sky-500/20">
              <span className="text-sky-400 text-xs mt-px">ⓘ</span>
              <p className="text-[10px] text-sky-300/80 leading-relaxed">
                StableStep models are not installed — download a backend set in the
                Model Manager (StableStep tab; GGML ~5.8 GB or ONNX ~12 GB) to
                enable this feature.
              </p>
            </div>
          )}
          {stableStepAvailable && gp.stableStepOn && (
            <div className="space-y-2 pt-1">
              <EditableSlider
                label="Refine strength"
                value={gp.stableStepStrength}
                min={0.10} max={0.60} step={0.05}
                onChange={gp.setStableStepStrength}
                formatDisplay={v => (v * 100).toFixed(0) + '%'}
                tooltip="How much of the instrumental is re-rendered. Higher values re-interpret the instrumentation more; lower values stay closer to the original."
              />
              <p className="text-[10px] text-zinc-500 leading-relaxed">
                Higher values re-interpret the instrumentation more. 30% is a good
                balance between cleanup and faithfulness.
              </p>
              {/* Backend selector: Auto / ONNX (TensorRT) / GGML */}
              <div>
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1">Backend</label>
                <div className="flex rounded-xl overflow-hidden border border-zinc-300 dark:border-white/10 bg-zinc-100 dark:bg-zinc-800">
                  {([
                    { value: 'auto' as const, label: 'Auto', installed: true },
                    { value: 'onnx' as const, label: 'ONNX (TensorRT)', installed: stableStepBackends.onnx },
                    { value: 'gguf' as const, label: 'GGML', installed: stableStepBackends.gguf },
                  ]).map((opt, idx) => {
                    const selected = gp.stableStepBackend === opt.value;
                    return (
                      <button
                        key={opt.value}
                        type="button"
                        disabled={!opt.installed}
                        onClick={() => gp.setStableStepBackend(opt.value)}
                        title={!opt.installed
                          ? `${opt.label} models not installed — download in Model Manager (StableStep tab)`
                          : opt.value === 'auto'
                            ? 'Let the engine pick the best installed backend'
                            : opt.value === 'onnx'
                              ? 'ONNX Runtime with TensorRT acceleration (NVIDIA only)'
                              : 'GGML backend — CUDA, Vulkan or CPU'}
                        className={`flex-1 px-2 py-1.5 text-xs transition-colors ${idx > 0 ? 'border-l border-zinc-300 dark:border-white/10' : ''} ${
                          selected
                            ? 'bg-sky-500/20 text-sky-600 dark:text-sky-300 font-medium'
                            : opt.installed
                              ? 'text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700 cursor-pointer'
                              : 'text-zinc-400 dark:text-zinc-600 cursor-not-allowed'
                        }`}
                      >
                        {opt.label}
                        {!opt.installed && (
                          <span className="block text-[9px] font-normal text-zinc-400 dark:text-zinc-600">not installed</span>
                        )}
                      </button>
                    );
                  })}
                </div>
                <p className="mt-1 text-[10px] text-zinc-500 leading-relaxed">
                  {gp.stableStepBackend === 'onnx'
                    ? 'ONNX Runtime with TensorRT (NVIDIA). First run per song-length bucket builds the TensorRT engine (slow once, then cached).'
                    : gp.stableStepBackend === 'gguf'
                      ? 'GGML backend — runs on CUDA, Vulkan or CPU. Fastest option on NVIDIA in current testing.'
                      : 'Auto lets the engine pick the best installed backend.'}
                </p>
              </div>
            </div>
          )}
        </div>
      </Accordion>

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

      {/* 2.5. Pre-VST Gain Offset */}
      <Accordion
        icon={<AudioWaveform size={14} />}
        label="Gain Offset"
        accentColor="amber"
        persistKey="hs-ppAccordion-gain"
        badge={gp.gainOffsetDb !== 0 ? (
          <span className="text-[10px] text-amber-400/60 font-mono">
            {gp.gainOffsetDb > 0 ? '+' : ''}{gp.gainOffsetDb} dB
          </span>
        ) : undefined}
      >
        <div className="space-y-2 mt-2">
          <p className="text-[10px] text-zinc-500 leading-relaxed">
            Apply a volume offset to the unmastered track before the VST chain
            and mastering stages. Use negative values to reduce volume (headroom
            for VST processing), positive to boost.
          </p>
          <EditableSlider
            label="Offset"
            value={gp.gainOffsetDb}
            min={-10} max={10} step={0.5}
            onChange={gp.setGainOffsetDb}
            formatDisplay={v => v === 0 ? '0 dB (bypass)' : `${v > 0 ? '+' : ''}${v.toFixed(1)} dB`}
            tooltip="dB gain applied to the audio before VST chain. 0 = no change."
          />
          {gp.gainOffsetDb !== 0 && (
            <button
              type="button"
              onClick={() => gp.setGainOffsetDb(0)}
              className="flex items-center gap-1 text-[10px] text-zinc-500 hover:text-amber-400 transition-colors"
            >
              <RotateCcw size={11} />
              Reset to 0 dB
            </button>
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

      {/* 3.5. LUFS Normalization (only visible when mastering is enabled) */}
      {gp.masteringEnabled && (
        <Accordion
          icon={<AudioWaveform size={14} />}
          label={t('pp.lufsNormalize')}
          accentColor="amber"
          persistKey="hs-ppAccordion-lufs"
          toggle={{ checked: gp.lufsEnabled, onChange: gp.setLufsEnabled }}
          badge={gp.lufsEnabled ? (
            <span className="text-[10px] text-amber-400/60 font-mono">
              {gp.lufsTarget} LUFS
            </span>
          ) : undefined}
        >
          <div className="space-y-3 mt-2">
            <p className="text-[10px] text-zinc-500 leading-relaxed">
              Measures integrated loudness (ITU-R BS.1770-4) and adjusts gain to hit
              the target &mdash; boosting quiet tracks and reducing loud ones. Includes a
              true-peak limiter at -1 dBTP to prevent clipping.
            </p>

            {gp.lufsEnabled && (
              <div className="space-y-2 pt-1">
                {/* Preset selector */}
                <div>
                  <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
                    {t('pp.lufsPreset')}
                  </label>
                  <select
                    className="w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 outline-none transition-colors cursor-pointer"
                    value={gp.lufsPreset}
                    onChange={e => gp.setLufsPreset(e.target.value)}
                  >
                    {LUFS_PRESETS.map(p => (
                      <option key={p.id} value={p.id}>
                        {p.label}{p.lufs !== null ? ` (${p.lufs} LUFS)` : ''}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Custom slider — only when preset is 'custom' */}
                {gp.lufsPreset === 'custom' && (
                  <EditableSlider
                    label={t('pp.lufsTarget')}
                    value={gp.lufsTarget}
                    min={-30} max={-5} step={0.5}
                    onChange={gp.setLufsTarget}
                    formatDisplay={v => `${v.toFixed(1)} LUFS`}
                    tooltip="Target integrated loudness. Lower = quieter, higher = louder."
                  />
                )}

                {/* Info about current target */}
                <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-amber-500/5 border border-amber-500/10">
                  <AudioWaveform size={14} className="text-amber-400 flex-shrink-0" />
                  <span className="text-[10px] text-amber-300">
                    Target: {gp.lufsTarget} LUFS &middot; True-peak ceiling: -1.0 dBTP
                  </span>
                </div>
              </div>
            )}
          </div>
        </Accordion>
      )}

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
  const { masteringEnabled, masteringReference, spectralLifterEnabled, ppVaeReencode, stableStepOn, coverArtEnabled, vocalNaturalizerEnabled, gainOffsetDb, qualityEvalEnabled, postprocessEnabled, postprocessPlugin, whisperLyricsEnabled, lufsEnabled, lufsTarget } = useGlobalParams();
  const { chain } = useVstChainStore();
  const vstEnabled = chain.filter(p => p.enabled).length;

  const parts: string[] = [];
  if (postprocessEnabled && postprocessPlugin) parts.push('TD');
  if (ppVaeReencode) parts.push('PP-VAE');
  if (stableStepOn) parts.push('StableStep');
  if (spectralLifterEnabled) parts.push('SL');
  if (vocalNaturalizerEnabled) parts.push('Nat');
  if (gainOffsetDb !== 0) parts.push(`${gainOffsetDb > 0 ? '+' : ''}${gainOffsetDb}dB`);
  if (vstEnabled > 0) parts.push(`${vstEnabled} VST${vstEnabled !== 1 ? 's' : ''}`);
  if (masteringEnabled && masteringReference) parts.push('Master');
  if (masteringEnabled && lufsEnabled) parts.push(`${lufsTarget} LUFS`);
  if (coverArtEnabled) parts.push('Cover');
  if (qualityEvalEnabled) parts.push('QE');
  if (whisperLyricsEnabled) parts.push('Whisper');

  if (parts.length === 0) return null;

  return (
    <span className="text-[10px] text-amber-400/60 font-mono truncate">
      {parts.join(' + ')}
    </span>
  );
};
