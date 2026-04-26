// PostProcessingDropdown.tsx — Unified post-processing panel for the global bar
//
// Three collapsible accordion sections:
//   1. Spectral Lifter — AI artifact removal (toggle only, no config)
//   2. VST Chain       — Plugin chain management (existing VstChainDropdown content)
//   3. Mastering       — Reference-based mastering (existing MasteringDropdown content)

import React, { useState, useEffect, useCallback } from 'react';
import {
  Sparkles, AudioWaveform, Zap,
  Upload, Trash2, Music2,
  ChevronDown,
} from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { masteringApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { useVstChainStore } from '../../stores/vstChainStore';
import { ToggleSwitch } from './BarSection';
import { formatReferenceName } from './modelLabels';
import { VstChainDropdown } from './VstChainDropdown';

// ── Accordion Section ───────────────────────────────────────────

interface AccordionProps {
  icon: React.ReactNode;
  label: string;
  badge?: React.ReactNode;
  accentColor: string;
  toggle?: { checked: boolean; onChange: (v: boolean) => void };
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const Accordion: React.FC<AccordionProps> = ({
  icon, label, badge, accentColor, toggle, defaultOpen, children,
}) => {
  const [open, setOpen] = useState(defaultOpen ?? false);

  const accentMap: Record<string, string> = {
    cyan: 'border-cyan-500/20 bg-cyan-500/5',
    violet: 'border-violet-500/20 bg-violet-500/5',
    amber: 'border-amber-500/20 bg-amber-500/5',
  };
  const activeStyle = toggle?.checked ? (accentMap[accentColor] || '') : '';

  return (
    <div className={`rounded-xl border transition-all ${activeStyle || 'border-white/5 bg-zinc-800/30'}`}>
      {/* Header */}
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left"
      >
        <span className={`flex-shrink-0 ${toggle?.checked ? `text-${accentColor}-400` : 'text-zinc-500'}`}>
          {icon}
        </span>
        <span className="text-sm text-zinc-300 font-medium flex-1">{label}</span>
        {badge && <span className="flex-shrink-0">{badge}</span>}
        {toggle && (
          <span onClick={e => e.stopPropagation()}>
            <ToggleSwitch checked={toggle.checked} onChange={toggle.onChange} accentColor={accentColor} />
          </span>
        )}
        <ChevronDown
          size={14}
          className={`text-zinc-500 transition-transform flex-shrink-0 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Content */}
      {open && (
        <div className="px-3 pb-3 pt-0 border-t border-white/5">
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
        Enable the toggle above to configure mastering.
      </div>
    );
  }

  return (
    <div className="space-y-3 mt-2">
      {/* Reference selector */}
      <div>
        <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
          Reference Track
        </label>
        {references.length > 0 ? (
          <select
            className="w-full px-3 py-2 rounded-xl bg-zinc-800 border border-white/10 text-sm text-zinc-200 focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 outline-none transition-colors cursor-pointer"
            value={gp.masteringReference}
            onChange={e => gp.setMasteringReference(e.target.value)}
          >
            <option value="">Select a reference...</option>
            {references.map(r => (
              <option key={r.name} value={r.name}>
                {r.name} ({formatFileSize(r.size)})
              </option>
            ))}
          </select>
        ) : (
          <div className="text-xs text-zinc-500 italic px-1">
            No reference tracks uploaded yet
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
            title="Delete reference"
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
              ? 'bg-zinc-800 text-zinc-500 border-white/5 cursor-wait'
              : 'bg-zinc-800 text-zinc-400 border-white/10 hover:border-amber-500/30 hover:text-amber-400'
          }`}
        >
          {uploading ? (
            <><span className="w-3 h-3 border-2 border-zinc-500 border-t-transparent rounded-full animate-spin" /> Uploading...</>
          ) : (
            <><Upload size={14} /> Upload Reference</>
          )}
        </label>
      </div>

      {/* Timbre reference toggle */}
      {gp.masteringReference && (
        <div className="flex items-center justify-between mt-1">
          <div className="flex items-center gap-1.5">
            <Music2 size={14} className="text-teal-400" />
            <span className="text-sm text-zinc-400">Also use as timbre reference</span>
          </div>
          <ToggleSwitch checked={gp.timbreReference} onChange={gp.setTimbreReference} accentColor="amber" />
        </div>
      )}
      {gp.timbreReference && gp.masteringReference && (
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
  const gp = useGlobalParams();
  const { chain } = useVstChainStore();
  const vstEnabled = chain.filter(p => p.enabled).length;

  return (
    <div className="space-y-2">
      {/* 1. Spectral Lifter */}
      <Accordion
        icon={<Zap size={14} />}
        label="Spectral Lifter"
        accentColor="cyan"
        toggle={{ checked: gp.spectralLifterEnabled, onChange: gp.setSpectralLifterEnabled }}
      >
        <p className="text-[10px] text-zinc-500 leading-relaxed mt-2">
          Removes AI shimmer artifacts, reduces spectral noise, and extends
          high-frequency content above 16kHz. Runs as a Python subprocess
          after generation.
        </p>
      </Accordion>

      {/* 2. VST Chain */}
      <Accordion
        icon={<Sparkles size={14} />}
        label="VST Chain"
        accentColor="violet"
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
        label="Mastering"
        accentColor="amber"
        toggle={{ checked: gp.masteringEnabled, onChange: gp.setMasteringEnabled }}
        badge={gp.masteringEnabled && gp.masteringReference ? (
          <span className="text-[10px] text-amber-400/60 font-mono truncate max-w-[100px]">
            {formatReferenceName(gp.masteringReference)}
          </span>
        ) : undefined}
      >
        <MasteringContent />
      </Accordion>
    </div>
  );
};

// ── Badge ───────────────────────────────────────────────────────

export const PostProcessingBadge: React.FC = () => {
  const { masteringEnabled, masteringReference, spectralLifterEnabled } = useGlobalParams();
  const { chain } = useVstChainStore();
  const vstEnabled = chain.filter(p => p.enabled).length;

  const parts: string[] = [];
  if (spectralLifterEnabled) parts.push('SL');
  if (vstEnabled > 0) parts.push(`${vstEnabled} VST${vstEnabled !== 1 ? 's' : ''}`);
  if (masteringEnabled && masteringReference) parts.push('Master');

  if (parts.length === 0) return null;

  return (
    <span className="text-[10px] text-amber-400/60 font-mono truncate">
      {parts.join(' + ')}
    </span>
  );
};
