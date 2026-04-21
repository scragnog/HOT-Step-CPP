// MasteringSection.tsx — Reference-based mastering controls
//
// Static section with:
// - Toggle: enable/disable reference mastering
// - Reference track selector with upload support

import React, { useState, useEffect, useCallback } from 'react';
import { Upload, Trash2, Sparkles, Music2 } from 'lucide-react';
import { masteringApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { ToggleSwitch } from '../shared/ToggleSwitch';

interface MasteringSectionProps {
  masteringEnabled: boolean;
  onMasteringEnabledChange: (v: boolean) => void;
  masteringReference: string;
  onMasteringReferenceChange: (v: string) => void;
  timbreReference: boolean;
  onTimbreReferenceChange: (v: boolean) => void;
}

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

export const MasteringSection: React.FC<MasteringSectionProps> = ({
  masteringEnabled,
  onMasteringEnabledChange,
  masteringReference,
  onMasteringReferenceChange,
  timbreReference,
  onTimbreReferenceChange,
}) => {
  const { token } = useAuth();
  const [references, setReferences] = useState<ReferenceTrack[]>([]);
  const [uploading, setUploading] = useState(false);

  // Load references on mount
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
      onMasteringReferenceChange(result.name);
      // Refresh list
      const data = await masteringApi.listReferences();
      setReferences(data.references);
    } catch (err) {
      console.error('[Mastering] Upload failed:', err);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  }, [token, onMasteringReferenceChange]);

  const handleDelete = useCallback(async (name: string) => {
    if (!token) return;
    try {
      await masteringApi.deleteReference(name, token);
      if (masteringReference === name) onMasteringReferenceChange('');
      const data = await masteringApi.listReferences();
      setReferences(data.references);
    } catch (err) {
      console.error('[Mastering] Delete failed:', err);
    }
  }, [token, masteringReference, onMasteringReferenceChange]);

  return (
    <div className="pt-3 border-t border-white/5">
      <div className="flex items-center justify-between px-3 py-2.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-zinc-200 uppercase tracking-wider">Mastering</span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
            masteringEnabled ? 'bg-amber-500/20 text-amber-400' : 'bg-zinc-700 text-zinc-400'
          }`}>
            {masteringEnabled ? 'ON' : 'OFF'}
          </span>
        </div>
      </div>

      <div className="px-3 pb-3 space-y-3">
        {/* Enable toggle */}
        <ToggleSwitch
          checked={masteringEnabled}
          onChange={onMasteringEnabledChange}
          label="Apply Reference Mastering"
          icon={<Sparkles size={14} className="text-amber-400" />}
        />

        {masteringEnabled && (
          <>
            {/* Reference selector */}
            <div>
              <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
                Reference Track
              </label>
              {references.length > 0 ? (
                <select
                  className="w-full px-3 py-2 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 outline-none transition-colors cursor-pointer"
                  value={masteringReference}
                  onChange={e => onMasteringReferenceChange(e.target.value)}
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
            {masteringReference && (
              <div className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-amber-500/5 border border-amber-500/10">
                <Music2 size={14} className="text-amber-400 flex-shrink-0" />
                <span className="text-xs text-amber-300 truncate flex-1">{masteringReference}</span>
                <button
                  onClick={() => handleDelete(masteringReference)}
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
                id="mastering-ref-upload"
                className="hidden"
                onChange={handleUpload}
              />
              <label
                htmlFor="mastering-ref-upload"
                className={`flex items-center gap-2 px-3 py-2 text-xs font-semibold rounded-xl border cursor-pointer transition-all ${
                  uploading
                    ? 'bg-zinc-800 text-zinc-500 border-white/5 cursor-wait'
                    : 'bg-zinc-900 text-zinc-400 border-white/10 hover:border-amber-500/30 hover:text-amber-400'
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
            {masteringReference && (
              <ToggleSwitch
                checked={timbreReference}
                onChange={onTimbreReferenceChange}
                label="Also use as timbre reference"
                icon={<Music2 size={14} className="text-teal-400" />}
              />
            )}
            {timbreReference && masteringReference && (
              <p className="text-[10px] text-zinc-600 leading-relaxed pl-11">
                The reference track will be VAE-encoded and fed into the timbre conditioning pipeline,
                guiding the generation&apos;s tone and texture to match the reference.
              </p>
            )}

            {/* Info */}
            <p className="text-[10px] text-zinc-600 leading-relaxed pl-11">
              The generated audio will be mastered to match the RMS level, frequency spectrum,
              and dynamic characteristics of the reference track.
            </p>
          </>
        )}
      </div>
    </div>
  );
};
