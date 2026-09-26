// SampleRow.tsx — one row of the review grid
//
// Cells are controlled inputs seeded from samplesById merged with
// pendingEdits. Typing is optimistic; the store debounces the PATCH by 500 ms
// per (sampleId, field), flushes on blur/Enter and reverts on Escape.
// Rows the job is currently rewriting (labelStatus === 'processing') are
// read-only — the job is about to overwrite them.

import React from 'react';
import { AlertCircle, CheckSquare, Copy, FileWarning, Loader2, Maximize2, Pause, Play, Square } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { PatchSampleInput, SampleLabelStatus, TrainingSample } from '../../services/trainingApi';
import { useTrainingStore, type EditableField } from '../../stores/trainingStore';
import { Toggle } from '../shared/Toggle';
import type { ColDef } from './SampleGrid';

const STATUS_META: Record<SampleLabelStatus, { dot: string; key: string }> = {
  unlabeled:  { dot: 'bg-zinc-400 dark:bg-zinc-600', key: 'trainingStudio.grid.statusUnlabeled' },
  labeled:    { dot: 'bg-emerald-500',               key: 'trainingStudio.grid.statusLabeled' },
  pending:    { dot: 'bg-sky-500',                   key: 'trainingStudio.grid.statusPending' },
  processing: { dot: 'bg-amber-500',                 key: 'trainingStudio.grid.statusProcessing' },
  error:      { dot: 'bg-red-500',                   key: 'trainingStudio.grid.statusError' },
};

function formatDuration(seconds: number): string {
  if (!seconds || !Number.isFinite(seconds)) return '—';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface SampleRowProps {
  /** Server truth merged with any un-saved optimistic edit. */
  sample: TrainingSample;
  columns: ColDef[];
  selected: boolean;
  saving: boolean;
  hasSaveError: boolean;
  isPreviewing: boolean;
  onToggleSelect: () => void;
  onOpenDrawer: () => void;
  onTogglePreview: () => void;
}

export const SampleRow: React.FC<SampleRowProps> = ({
  sample, columns, selected, saving, hasSaveError, isPreviewing,
  onToggleSelect, onOpenDrawer, onTogglePreview,
}) => {
  const { t } = useTranslation();
  const editSample = useTrainingStore(s => s.editSample);
  const datasetTag = useTrainingStore(s => s.detail?.customTag ?? '');
  const flushSample = useTrainingStore(s => s.flushSample);
  const revertSampleField = useTrainingStore(s => s.revertSampleField);
  const toggleExcluded = useTrainingStore(s => s.toggleExcluded);

  const readOnly = sample.labelStatus === 'processing' || sample.fileMissing;

  const commitKeys = (field: EditableField) => (e: React.KeyboardEvent) => {
    // Enter blurs; the blur handler is what flushes (avoids a double PATCH).
    if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
    if (e.key === 'Escape') { revertSampleField(sample.sampleId, field); (e.target as HTMLInputElement).blur(); }
  };

  const inputCls = `w-full bg-transparent border rounded px-1.5 py-0.5 text-xs text-zinc-800 dark:text-zinc-200 outline-none transition-colors ${
    hasSaveError ? 'border-red-500' : 'border-transparent hover:border-zinc-300 dark:hover:border-white/10 focus:border-amber-500'
  } disabled:opacity-50 disabled:cursor-not-allowed`;

  const textCell = (field: EditableField, value: string, placeholder?: string) => (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      disabled={readOnly}
      onChange={(e) => void editSample(sample.sampleId, { [field]: e.target.value } as PatchSampleInput)}
      onBlur={() => void flushSample(sample.sampleId)}
      onKeyDown={commitKeys(field)}
      className={inputCls}
    />
  );

  const numberCell = (field: 'bpm' | 'repeat', value: number | null, min?: number) => (
    <input
      type="number"
      min={min}
      value={value ?? ''}
      disabled={readOnly}
      onChange={(e) => {
        const raw = e.target.value;
        const parsed = raw === '' ? null : Number(raw);
        if (field === 'repeat') void editSample(sample.sampleId, { repeat: parsed === null || Number.isNaN(parsed) ? 1 : Math.max(1, Math.trunc(parsed)) });
        else void editSample(sample.sampleId, { bpm: parsed === null || Number.isNaN(parsed) ? null : parsed });
      }}
      onBlur={() => void flushSample(sample.sampleId)}
      onKeyDown={commitKeys(field)}
      className={`${inputCls} font-mono`}
    />
  );

  const status = STATUS_META[sample.labelStatus] ?? STATUS_META.unlabeled;
  const lyricLines = sample.lyrics ? sample.lyrics.split('\n').filter(l => l.trim()).length : 0;

  const cells: Record<string, React.ReactNode> = {
    select: selected
      ? <CheckSquare size={14} className="text-amber-500" />
      : <Square size={14} className="text-zinc-400 dark:text-zinc-600" />,

    status: (
      <div className="flex items-center gap-1.5" title={sample.error || undefined}>
        {sample.labelStatus === 'processing'
          ? <Loader2 size={11} className="animate-spin text-amber-500 flex-shrink-0" />
          : <span className={`w-2 h-2 rounded-full flex-shrink-0 ${status.dot}`} />}
        <span className="text-[10px] text-zinc-600 dark:text-zinc-400 truncate">{t(status.key)}</span>
        {sample.labelStatus === 'error' && <AlertCircle size={11} className="text-red-500 flex-shrink-0" />}
      </div>
    ),

    filename: (
      <div className="flex items-center gap-1.5 min-w-0" title={sample.relPath}>
        {sample.fileMissing && <FileWarning size={12} className="text-red-500 flex-shrink-0" />}
        <span className={`text-xs truncate ${sample.fileMissing ? 'text-red-400' : 'text-zinc-700 dark:text-zinc-300'}`}>
          {sample.filename}
        </span>
        {sample.fileMissing && (
          <span className="text-[9px] font-semibold px-1 py-0.5 rounded bg-red-500/10 text-red-500 border border-red-500/20 flex-shrink-0"
            title={t('trainingStudio.grid.missingTitle')}>
            {t('trainingStudio.grid.missing')}
          </span>
        )}
      </div>
    ),

    duration: <span className="text-[11px] font-mono text-zinc-500">{formatDuration(sample.duration)}</span>,

    caption: (
      <div onDoubleClick={onOpenDrawer}>
        {textCell('caption', sample.caption)}
      </div>
    ),

    genre: textCell('genre', sample.genre),
    bpm: numberCell('bpm', sample.bpm),
    key: textCell('key', sample.key),
    signature: textCell('signature', sample.signature),
    language: textCell('language', sample.language),

    instrumental: (
      <Toggle
        size="sm"
        accent="amber"
        checked={sample.isInstrumental}
        disabled={readOnly}
        onChange={(checked) => { void editSample(sample.sampleId, { isInstrumental: checked }); void flushSample(sample.sampleId); }}
        aria-label={t('trainingStudio.grid.instrumental')}
        title={t('trainingStudio.grid.instrumentalInfo')}
      />
    ),

    lyrics: (
      <button onClick={onOpenDrawer} className="w-full text-left flex items-center gap-1.5 min-w-0">
        <span className="text-[11px] text-zinc-500 truncate flex-1">
          {sample.lyrics ? sample.lyrics.split('\n')[0] : t('trainingStudio.grid.noLyrics')}
        </span>
        {lyricLines > 1 && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-zinc-200/70 dark:bg-white/5 text-zinc-500 flex-shrink-0">
            {t('trainingStudio.grid.lyricLines', { count: lyricLines })}
          </span>
        )}
      </button>
    ),

    // Empty = inherits the dataset trigger tag (applied at build time); the
    // inherited value is shown greyed so the inheritance is visible.
    tag: textCell('customTag', sample.customTag, datasetTag || undefined),
    repeat: numberCell('repeat', sample.repeat, 1),

    // Deliberately NOT gated on fileMissing: excluding a vanished file is the
    // documented way to unblock Build (§5.5). The store routes that case
    // through the bulk endpoint, which the 409-on-PATCH rule exempts.
    exclude: (
      <Toggle
        size="sm"
        accent="amber"
        checked={sample.excluded}
        disabled={sample.labelStatus === 'processing'}
        onChange={(checked) => void toggleExcluded(sample.sampleId, checked)}
        aria-label={t('trainingStudio.grid.exclude')}
        title={t('trainingStudio.grid.excludeInfo')}
      />
    ),

    spacer: null,

    actions: (
      <div className="flex items-center justify-end gap-0.5 flex-nowrap">
        <button
          onClick={(e) => { e.stopPropagation(); onTogglePreview(); }}
          disabled={sample.fileMissing}
          className="p-1 rounded text-zinc-500 hover:text-amber-500 hover:bg-amber-500/10 disabled:opacity-30 transition-colors"
          title={sample.filename}
        >
          {isPreviewing ? <Pause size={12} /> : <Play size={12} />}
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onOpenDrawer(); }}
          className="p-1 rounded text-zinc-500 hover:text-white hover:bg-white/10 transition-colors"
          title={t('trainingStudio.grid.caption')}
        >
          <Maximize2 size={12} />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            try { void navigator.clipboard.writeText(sample.audioPath); } catch { /* ignore */ }
          }}
          className="p-1 rounded text-zinc-500 hover:text-white hover:bg-white/10 transition-colors"
          title={sample.audioPath}
        >
          <Copy size={12} />
        </button>
      </div>
    ),
  };

  return (
    <tr
      className={`group border-t border-zinc-100 dark:border-white/5 transition-colors hover:bg-black/[0.02] dark:hover:bg-white/5 ${
        selected ? 'bg-amber-500/10' : ''
      } ${sample.excluded ? 'opacity-50' : ''}`}
    >
      {columns.map(c => (
        <td
          key={c.id}
          onClick={c.id === 'select' ? onToggleSelect : undefined}
          className={`px-2 py-1.5 overflow-hidden ${c.align === 'center' ? 'text-center' : c.align === 'right' ? 'text-right' : ''} ${
            c.id === 'select' ? 'cursor-pointer' : ''
          } ${
            c.id === 'actions'
              ? 'sticky right-0 z-10 bg-white dark:bg-suno-card group-hover:bg-zinc-50 dark:group-hover:bg-zinc-900 border-l border-zinc-200 dark:border-white/10'
              : ''
          }`}
        >
          {c.id === 'status' && saving
            ? <div className="flex items-center gap-1.5"><Loader2 size={11} className="animate-spin text-amber-500" />{cells[c.id]}</div>
            : cells[c.id]}
        </td>
      ))}
    </tr>
  );
};

export default SampleRow;
