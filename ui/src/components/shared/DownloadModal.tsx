// DownloadModal.tsx — Format selection modal for downloading tracks
//
// Supports: WAV, FLAC, Opus, MP3 with configurable bitrates.
// Shows mastered/original/both toggle when mastered version exists.

import React, { useState } from 'react';
import { X, Download, FileAudio, Sparkles, Music2 } from 'lucide-react';
import type { Song } from '../../types';

interface DownloadModalProps {
  song: Song;
  isOpen: boolean;
  onClose: () => void;
}

type AudioFormat = 'wav' | 'flac' | 'opus' | 'mp3';
type DownloadVersion = 'original' | 'mastered' | 'both';

const FORMAT_INFO: Record<AudioFormat, { label: string; desc: string; lossy: boolean }> = {
  wav:  { label: 'WAV',  desc: 'Lossless, uncompressed', lossy: false },
  flac: { label: 'FLAC', desc: 'Lossless, compressed (~50% smaller)', lossy: false },
  opus: { label: 'Opus', desc: 'High quality at low bitrates', lossy: true },
  mp3:  { label: 'MP3',  desc: 'Universal compatibility', lossy: true },
};

const BITRATES = [128, 192, 256, 320];

export const DownloadModal: React.FC<DownloadModalProps> = ({ song, isOpen, onClose }) => {
  const [format, setFormat] = useState<AudioFormat>('flac');
  const [bitrate, setBitrate] = useState(192);
  const [version, setVersion] = useState<DownloadVersion>('original');
  const [downloading, setDownloading] = useState(false);

  const hasMastered = !!(song.masteredAudioUrl);
  const isLossy = FORMAT_INFO[format].lossy;

  const triggerDownload = async (dlVersion: 'original' | 'mastered') => {
    const params = new URLSearchParams({
      format,
      version: dlVersion,
      ...(isLossy ? { bitrate: String(bitrate) } : {}),
    });
    const url = `/api/download/${song.id}?${params}`;

    // Use a hidden link to trigger browser download
    const a = document.createElement('a');
    a.href = url;
    a.download = ''; // Let server set the filename
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      if (version === 'both' && hasMastered) {
        await triggerDownload('original');
        // Small delay between downloads so browser doesn't block the second
        await new Promise(r => setTimeout(r, 500));
        await triggerDownload('mastered');
      } else if (version === 'mastered' && hasMastered) {
        await triggerDownload('mastered');
      } else {
        await triggerDownload('original');
      }
    } finally {
      setTimeout(() => setDownloading(false), 1000);
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 transition-opacity"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div
          className="bg-zinc-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-md pointer-events-auto animate-in fade-in zoom-in-95 duration-200"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-white/5">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20">
                <Download size={18} className="text-emerald-400" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white">Download</h2>
                <p className="text-xs text-zinc-500 truncate max-w-[240px]">{song.title || 'Untitled'}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Body */}
          <div className="px-5 py-4 space-y-4">
            {/* Format Selection */}
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                Format
              </label>
              <div className="grid grid-cols-4 gap-2">
                {(Object.keys(FORMAT_INFO) as AudioFormat[]).map(f => (
                  <button
                    key={f}
                    onClick={() => setFormat(f)}
                    className={`flex flex-col items-center gap-1 px-3 py-2.5 rounded-xl border transition-all ${
                      format === f
                        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                        : 'bg-zinc-800/50 border-white/5 text-zinc-400 hover:border-white/10 hover:text-zinc-300'
                    }`}
                  >
                    <FileAudio size={16} />
                    <span className="text-xs font-bold">{FORMAT_INFO[f].label}</span>
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-[10px] text-zinc-500">{FORMAT_INFO[format].desc}</p>
            </div>

            {/* Bitrate (lossy formats only) */}
            {isLossy && (
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Bitrate
                </label>
                <div className="grid grid-cols-4 gap-2">
                  {BITRATES.map(b => (
                    <button
                      key={b}
                      onClick={() => setBitrate(b)}
                      className={`px-3 py-2 rounded-xl border text-xs font-mono font-bold transition-all ${
                        bitrate === b
                          ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                          : 'bg-zinc-800/50 border-white/5 text-zinc-400 hover:border-white/10 hover:text-zinc-300'
                      }`}
                    >
                      {b}k
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Version (if mastered exists) */}
            {hasMastered && (
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Version
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {([
                    { key: 'original', label: 'Original', icon: <Music2 size={14} /> },
                    { key: 'mastered', label: 'Mastered', icon: <Sparkles size={14} /> },
                    { key: 'both', label: 'Both', icon: <Download size={14} /> },
                  ] as const).map(v => (
                    <button
                      key={v.key}
                      onClick={() => setVersion(v.key)}
                      className={`flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl border text-xs font-semibold transition-all ${
                        version === v.key
                          ? v.key === 'mastered'
                            ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                            : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                          : 'bg-zinc-800/50 border-white/5 text-zinc-400 hover:border-white/10 hover:text-zinc-300'
                      }`}
                    >
                      {v.icon}
                      {v.label}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-5 py-4 border-t border-white/5">
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-semibold text-sm transition-all duration-200 hover:shadow-lg hover:shadow-emerald-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {downloading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Converting...
                </>
              ) : (
                <>
                  <Download size={16} />
                  Download {FORMAT_INFO[format].label}
                  {isLossy ? ` (${bitrate}k)` : ''}
                  {hasMastered && version === 'both' ? ' × 2' : ''}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};
