// SourcePanel.tsx — Left panel: source audio upload + metadata + analysis
import React, { useCallback, useState } from 'react';
import { Upload, Music, Loader2, X } from 'lucide-react';
import type { AudioMetadata, AudioAnalysis } from './coverStudioUtils';

interface SourcePanelProps {
  sourceFileName: string;
  metadata: AudioMetadata | null;
  analysis: AudioAnalysis | null;
  isUploading: boolean;
  isAnalyzing: boolean;
  onFileSelected: (file: File) => void;
  onClear: () => void;
  bpmCorrection: number;
  onBpmCorrectionChange: (v: number) => void;
}

export const SourcePanel: React.FC<SourcePanelProps> = ({
  sourceFileName, metadata, analysis, isUploading, isAnalyzing,
  onFileSelected, onClear,
  bpmCorrection, onBpmCorrectionChange,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && /\.(mp3|wav|flac|ogg|m4a|opus|aac)$/i.test(file.name)) {
      onFileSelected(file);
    }
  }, [onFileSelected]);

  const correctedBpm = analysis?.bpm ? Math.round(analysis.bpm * bpmCorrection) : null;

  return (
    <div className="w-[320px] flex-shrink-0 overflow-y-auto scrollbar-hide border-r border-zinc-200 dark:border-white/5 p-4 space-y-4">
      {/* Upload zone */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          <Upload className="w-4 h-4 text-cyan-400" />
          Source Audio
        </div>
        <div
          onDrop={handleDrop}
          onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-300
            ${isDragging
              ? 'border-cyan-400 bg-cyan-500/10 scale-[1.02]'
              : sourceFileName
                ? 'border-cyan-500/30 bg-cyan-500/5'
                : 'border-zinc-300 dark:border-zinc-700 hover:border-cyan-400/50 hover:bg-cyan-500/5'}
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp3,.wav,.flac,.ogg,.m4a,.opus,.aac"
            className="hidden"
            onChange={e => { if (e.target.files?.[0]) onFileSelected(e.target.files[0]); }}
          />
          {isUploading ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
              <span className="text-xs text-cyan-400">Uploading...</span>
            </div>
          ) : isAnalyzing ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="w-8 h-8 text-teal-400 animate-spin" />
              <span className="text-xs text-teal-400">Analyzing BPM & Key...</span>
            </div>
          ) : sourceFileName ? (
            <div className="flex flex-col items-center gap-2">
              <Music className="w-8 h-8 text-cyan-400" />
              <span className="text-xs text-zinc-600 dark:text-zinc-400 truncate max-w-full">{sourceFileName}</span>
              <span className="text-[10px] text-zinc-500">Click or drop to replace</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Upload className="w-8 h-8 text-zinc-400" />
              <span className="text-xs text-zinc-500">Drop audio file or click to browse</span>
              <span className="text-[10px] text-zinc-600">MP3, WAV, FLAC, OGG, M4A</span>
            </div>
          )}
        </div>
      </div>

      {/* Metadata display */}
      {metadata && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-medium text-zinc-500 uppercase">Metadata</span>
            <button onClick={onClear} className="text-zinc-500 hover:text-red-400 transition-colors" title="Clear">
              <X className="w-3 h-3" />
            </button>
          </div>
          <div className="rounded-lg bg-black/5 dark:bg-white/5 p-3 space-y-1">
            {metadata.artist && <MetaRow label="Artist" value={metadata.artist} />}
            {metadata.title && <MetaRow label="Title" value={metadata.title} />}
            {metadata.album && <MetaRow label="Album" value={metadata.album} />}
            {metadata.duration != null && (
              <MetaRow label="Duration" value={`${Math.floor(metadata.duration / 60)}:${String(Math.floor(metadata.duration % 60)).padStart(2, '0')}`} />
            )}
          </div>
        </div>
      )}

      {/* Analysis display */}
      {analysis && (
        <div className="space-y-2">
          <span className="text-[10px] font-medium text-zinc-500 uppercase">Analysis</span>
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-gradient-to-br from-cyan-500/10 to-teal-500/10 border border-cyan-500/20 p-3 text-center">
              <span className="text-[10px] text-zinc-500 block">BPM</span>
              <span className="text-lg font-bold text-cyan-400">{correctedBpm ?? analysis.bpm}</span>
              {bpmCorrection !== 1 && (
                <span className="text-[9px] text-zinc-500 block">
                  (detected: {analysis.bpm})
                </span>
              )}
            </div>
            <div className="rounded-lg bg-gradient-to-br from-teal-500/10 to-emerald-500/10 border border-teal-500/20 p-3 text-center">
              <span className="text-[10px] text-zinc-500 block">Key</span>
              <span className="text-lg font-bold text-teal-400">{analysis.key}</span>
            </div>
          </div>
          {/* BPM correction — Essentia sometimes halves or doubles the tempo */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 whitespace-nowrap">Tempo fix:</span>
            <div className="flex gap-1 flex-1">
              {([
                { label: '÷2', value: 0.5 },
                { label: 'Detected', value: 1 },
                { label: '×2', value: 2 },
              ] as const).map(opt => (
                <button
                  key={opt.value}
                  onClick={() => onBpmCorrectionChange(opt.value)}
                  className={`flex-1 px-2 py-1 rounded-md text-[10px] font-medium transition-all ${
                    bpmCorrection === opt.value
                      ? 'bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-500/40'
                      : 'bg-white/5 text-zinc-500 hover:bg-white/10 hover:text-zinc-300'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const MetaRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex items-center gap-2">
    <span className="text-[10px] text-zinc-500 w-14 flex-shrink-0">{label}</span>
    <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate">{value}</span>
  </div>
);
