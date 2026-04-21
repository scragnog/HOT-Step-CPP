// MasteringDropdown.tsx — Mastering config for the global param bar
//
// The on/off toggle is in the bar header (ToggleSwitch).
// This dropdown shows reference track selection and options when mastering is enabled.

import React, { useState, useEffect, useCallback } from 'react';
import { Upload, Trash2, Music2 } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { masteringApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import { ToggleSwitch } from './BarSection';
import { formatReferenceName } from './modelLabels';

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

export const MasteringDropdown: React.FC = () => {
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
      <div className="text-xs text-zinc-500 italic text-center py-2">
        Mastering is disabled. Toggle it on in the bar above.
      </div>
    );
  }

  return (
    <div className="space-y-3">
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
          id="mastering-ref-upload-bar"
          className="hidden"
          onChange={handleUpload}
        />
        <label
          htmlFor="mastering-ref-upload-bar"
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

      {/* Info */}
      <p className="text-[10px] text-zinc-600 leading-relaxed">
        The generated audio will be mastered to match the RMS level, frequency spectrum,
        and dynamic characteristics of the reference track.
      </p>
    </div>
  );
};

/** Summary badge for the Mastering section */
export const MasteringBadge: React.FC = () => {
  const { masteringEnabled, masteringReference } = useGlobalParams();
  if (!masteringEnabled) return null;
  const refName = formatReferenceName(masteringReference);
  return (
    <span className="text-[10px] text-amber-400/60 font-mono truncate">
      {refName || 'No ref'}
    </span>
  );
};
