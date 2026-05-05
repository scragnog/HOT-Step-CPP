// LatentImport.tsx — Reusable latent file import with HSLAT metadata display
//
// Accepts .latent files, uploads to /api/upload/latent,
// shows embedded metadata, and emits the latent URL for generation.

import React, { useCallback, useRef, useState } from 'react';
import { FileAudio, X, Loader2, Layers } from 'lucide-react';

export interface LatentMetadata {
  bpm?: number;
  key?: string;
  lyrics?: string;
  caption?: string;
  seed?: number;
  adapter?: string;
  duration?: number;
  [key: string]: unknown;
}

interface LatentImportProps {
  /** Current latent URL (empty = no latent loaded) */
  latentUrl: string;
  /** Called when a latent is uploaded successfully */
  onLatentLoaded: (url: string, metadata: LatentMetadata) => void;
  /** Called when the latent is cleared */
  onClear: () => void;
  /** Optional className for the wrapper */
  className?: string;
}

export const LatentImport: React.FC<LatentImportProps> = ({
  latentUrl, onLatentLoaded, onClear, className = '',
}) => {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [loadedFilename, setLoadedFilename] = useState('');
  const [loadedMeta, setLoadedMeta] = useState<LatentMetadata | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setError('');
    setIsUploading(true);
    try {
      const fd = new FormData();
      fd.append('latent', file);
      const res = await fetch('/api/upload/latent', { method: 'POST', body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(err.error || `Upload failed: ${res.status}`);
      }
      const data = await res.json();
      setLoadedFilename(data.filename || file.name);
      setLoadedMeta(data.metadata || null);
      onLatentLoaded(data.latent_url, data.metadata || {});
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, [onLatentLoaded]);

  const handleClear = useCallback(() => {
    setLoadedFilename('');
    setLoadedMeta(null);
    setError('');
    onClear();
  }, [onClear]);

  if (latentUrl) {
    // Loaded state — show compact metadata card
    return (
      <div className={`rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-2.5 space-y-1.5 ${className}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 min-w-0">
            <Layers className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
            <span className="text-[10px] font-bold text-cyan-300 uppercase tracking-wider">Source Latent</span>
          </div>
          <button
            onClick={handleClear}
            className="p-0.5 rounded text-zinc-500 hover:text-red-400 transition-colors"
            title="Remove latent"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
        <p className="text-[10px] text-zinc-400 truncate" title={loadedFilename}>{loadedFilename}</p>
        {loadedMeta && (
          <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[9px] text-zinc-500">
            {loadedMeta.bpm != null && <span>BPM: <b className="text-zinc-300">{loadedMeta.bpm}</b></span>}
            {loadedMeta.key && <span>Key: <b className="text-zinc-300">{loadedMeta.key}</b></span>}
            {loadedMeta.duration != null && <span>Dur: <b className="text-zinc-300">{loadedMeta.duration.toFixed(1)}s</b></span>}
            {loadedMeta.seed != null && <span>Seed: <b className="text-zinc-300">{loadedMeta.seed}</b></span>}
            {loadedMeta.adapter && <span>Adapter: <b className="text-zinc-300 truncate max-w-[80px] inline-block align-bottom">{loadedMeta.adapter}</b></span>}
          </div>
        )}
      </div>
    );
  }

  // Empty state — show import button
  return (
    <div className={className}>
      <input
        ref={inputRef}
        type="file"
        accept=".latent"
        className="hidden"
        onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
        className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg border border-dashed border-white/10 hover:border-cyan-500/30 hover:bg-cyan-500/5 text-zinc-500 hover:text-cyan-400 text-xs transition-all disabled:opacity-50"
      >
        {isUploading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Uploading latent…
          </>
        ) : (
          <>
            <FileAudio className="w-3.5 h-3.5" />
            Import Latent (.hslat)
          </>
        )}
      </button>
      {error && <p className="text-[10px] text-red-400 mt-1">{error}</p>}
    </div>
  );
};
