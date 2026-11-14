// SampleGrid.tsx — the review table
//
// Derived from the SongTable pattern in components/library/SongList.tsx:
// ColDef list, <colgroup> + tableLayout:'fixed', drag-resize handles, a
// right-pinned sticky actions column, a columns-visibility dropdown and
// localStorage persistence (keys are training-specific).
//
// No virtualisation in v1 — the scan is capped at 5000 files, and above 800
// rows the grid defaults to the unlabeled-first filtered view.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CheckSquare, Columns3, RefreshCw, Square, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { sampleAudioUrl, type TrainingSample } from '../../services/trainingApi';
import { mergedSample, useTrainingStore } from '../../stores/trainingStore';
import { toast } from 'react-toastify';
import { SampleRow } from './SampleRow';
import { SampleDrawer } from './SampleDrawer';

export type ColAlign = 'left' | 'center' | 'right';
export interface ColDef {
  id: string;
  labelKey: string;
  defaultW: number;
  minW: number;
  align: ColAlign;
  resizable: boolean;
  defaultHidden?: boolean;
}

/** §5.4 grid spec. `spacer` absorbs leftover width so `actions` hugs the right. */
const BASE_COLS: ColDef[] = [
  { id: 'select',       labelKey: '',                                 defaultW: 32,  minW: 32,  align: 'left',   resizable: false },
  { id: 'status',       labelKey: 'trainingStudio.grid.status',       defaultW: 90,  minW: 60,  align: 'left',   resizable: true },
  { id: 'filename',     labelKey: 'trainingStudio.grid.file',         defaultW: 240, minW: 90,  align: 'left',   resizable: true },
  { id: 'duration',     labelKey: 'trainingStudio.grid.duration',     defaultW: 64,  minW: 44,  align: 'left',   resizable: true },
  { id: 'caption',      labelKey: 'trainingStudio.grid.caption',      defaultW: 360, minW: 120, align: 'left',   resizable: true },
  { id: 'genre',        labelKey: 'trainingStudio.grid.genre',        defaultW: 140, minW: 60,  align: 'left',   resizable: true },
  { id: 'bpm',          labelKey: 'trainingStudio.grid.bpm',          defaultW: 150, minW: 125, align: 'left',   resizable: true },
  { id: 'key',          labelKey: 'trainingStudio.grid.key',          defaultW: 110, minW: 50,  align: 'left',   resizable: true },
  { id: 'signature',    labelKey: 'trainingStudio.grid.signature',    defaultW: 60,  minW: 44,  align: 'left',   resizable: true },
  { id: 'language',     labelKey: 'trainingStudio.grid.language',     defaultW: 60,  minW: 44,  align: 'left',   resizable: true },
  { id: 'instrumental', labelKey: 'trainingStudio.grid.instrumental', defaultW: 56,  minW: 44,  align: 'center', resizable: true },
  { id: 'lyrics',       labelKey: 'trainingStudio.grid.lyrics',       defaultW: 200, minW: 80,  align: 'left',   resizable: true },
  { id: 'tag',          labelKey: 'trainingStudio.grid.tag',          defaultW: 120, minW: 60,  align: 'left',   resizable: true },
  { id: 'repeat',       labelKey: 'trainingStudio.grid.repeat',       defaultW: 48,  minW: 40,  align: 'left',   resizable: true, defaultHidden: true },
  { id: 'exclude',      labelKey: 'trainingStudio.grid.exclude',      defaultW: 40,  minW: 40,  align: 'center', resizable: false },
  { id: 'spacer',       labelKey: '',                                 defaultW: 0,   minW: 0,   align: 'left',   resizable: false },
  { id: 'actions',      labelKey: '',                                 defaultW: 88,  minW: 88,  align: 'right',  resizable: false },
];

const MANDATORY_COLS = new Set(['select', 'status', 'filename', 'spacer', 'actions']);

const WIDTHS_KEY = 'hs-training-colWidths';
const VIS_KEY = 'hs-training-colVisibility';
/** Above this many rows the grid opens on the unlabeled-first filtered view. */
const BIG_GRID_ROWS = 800;

function loadColWidths(): Record<string, number> {
  try { const v = localStorage.getItem(WIDTHS_KEY); return v ? JSON.parse(v) : {}; } catch { return {}; }
}
function saveColWidths(w: Record<string, number>) {
  try { localStorage.setItem(WIDTHS_KEY, JSON.stringify(w)); } catch { /* ignore */ }
}
function loadColVisibility(): Record<string, boolean> {
  try { const v = localStorage.getItem(VIS_KEY); return v ? JSON.parse(v) : {}; } catch { return {}; }
}
function saveColVisibility(v: Record<string, boolean>) {
  try { localStorage.setItem(VIS_KEY, JSON.stringify(v)); } catch { /* ignore */ }
}
function colDefaultVisible(id: string): boolean {
  if (MANDATORY_COLS.has(id)) return true;
  return !BASE_COLS.find(c => c.id === id)?.defaultHidden;
}

type RowFilter = 'all' | 'unlabeled';

export const SampleGrid: React.FC = () => {
  const { t } = useTranslation();
  const datasetId = useTrainingStore(s => s.selectedDatasetId);
  const samplesById = useTrainingStore(s => s.samplesById);
  const sampleOrder = useTrainingStore(s => s.sampleOrder);
  const pendingEdits = useTrainingStore(s => s.pendingEdits);
  const savingSampleIds = useTrainingStore(s => s.savingSampleIds);
  const saveError = useTrainingStore(s => s.saveError);
  const clearSaveError = useTrainingStore(s => s.clearSaveError);
  // A failed save shows on the app's toast stack; the row stays marked until it closes.
  useEffect(() => {
    if (!saveError) return;
    const id = toast.error(t('trainingStudio.grid.saveFailed', { message: saveError.message }), {
      // Only clear the error this toast was for; a newer one keeps its own toast.
      onClose: () => { if (useTrainingStore.getState().saveError === saveError) clearSaveError(); },
    });
    return () => toast.dismiss(id);
  }, [saveError, clearSaveError, t]);
  const selectedSampleIds = useTrainingStore(s => s.selectedSampleIds);
  const toggleSampleSelected = useTrainingStore(s => s.toggleSampleSelected);
  const setSelectedSampleIds = useTrainingStore(s => s.setSelectedSampleIds);
  const clearSelection = useTrainingStore(s => s.clearSelection);
  const openSampleId = useTrainingStore(s => s.openSampleId);
  const setOpenSampleId = useTrainingStore(s => s.setOpenSampleId);
  const bulkSet = useTrainingStore(s => s.bulkSet);
  const rescan = useTrainingStore(s => s.rescan);
  const startLabel = useTrainingStore(s => s.startLabel);
  const startGenius = useTrainingStore(s => s.startGenius);
  const startCaption = useTrainingStore(s => s.startCaption);
  const caps = useTrainingStore(s => s.capabilities);

  const [colVisible, setColVisible] = useState<Record<string, boolean>>(() => loadColVisibility());
  const [colMenuOpen, setColMenuOpen] = useState(false);
  const [colWidths, setColWidths] = useState<Record<string, number>>(() => {
    const saved = loadColWidths();
    const merged: Record<string, number> = {};
    for (const c of BASE_COLS) merged[c.id] = saved[c.id] ?? c.defaultW;
    return merged;
  });
  const [filter, setFilter] = useState<RowFilter>('all');
  const [previewId, setPreviewId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Big folders open filtered so the un-virtualised table stays responsive.
  const bigGridApplied = useRef('');
  useEffect(() => {
    if (!datasetId || bigGridApplied.current === datasetId) return;
    if (sampleOrder.length > BIG_GRID_ROWS) { setFilter('unlabeled'); bigGridApplied.current = datasetId; }
    else if (sampleOrder.length > 0) { bigGridApplied.current = datasetId; }
  }, [datasetId, sampleOrder.length]);

  const isColVisible = useCallback(
    (id: string) => MANDATORY_COLS.has(id) || (colVisible[id] ?? colDefaultVisible(id)),
    [colVisible],
  );

  const toggleCol = (id: string) => setColVisible(prev => {
    const cur = prev[id] ?? colDefaultVisible(id);
    const next = { ...prev, [id]: !cur };
    saveColVisibility(next);
    return next;
  });

  const columns = useMemo(() => BASE_COLS.filter(c => isColVisible(c.id)), [isColVisible]);
  const toggleableCols = useMemo(() => BASE_COLS.filter(c => !MANDATORY_COLS.has(c.id) && c.labelKey), []);

  const getW = (id: string) => colWidths[id] ?? BASE_COLS.find(c => c.id === id)?.defaultW ?? 80;
  const totalW = columns.reduce((s, c) => s + (c.id === 'spacer' ? 0 : getW(c.id)), 0);

  const handleResizeStart = useCallback((colId: string, minW: number, startX: number) => {
    const startW = colWidths[colId] ?? BASE_COLS.find(c => c.id === colId)?.defaultW ?? 80;
    const onMove = (e: MouseEvent) => {
      const newW = Math.max(minW, startW + (e.clientX - startX));
      setColWidths(prev => { const next = { ...prev, [colId]: newW }; saveColWidths(next); return next; });
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, [colWidths]);

  // Rows: server truth + optimistic edits, in display order.
  const rows = useMemo(() => {
    const list: TrainingSample[] = [];
    for (const id of sampleOrder) {
      const merged = mergedSample(samplesById[id], pendingEdits[id]);
      if (merged) list.push(merged);
    }
    return list;
  }, [sampleOrder, samplesById, pendingEdits]);

  const visibleRows = useMemo(
    () => (filter === 'unlabeled' ? rows.filter(r => r.labelStatus !== 'labeled') : rows),
    [rows, filter],
  );

  // Audio preview lives here, deliberately outside the global playback store —
  // these are files on disk, not library songs.
  useEffect(() => {
    if (!audioRef.current) return;
    if (previewId) { audioRef.current.play().catch(() => setPreviewId(null)); }
    else { audioRef.current.pause(); }
  }, [previewId]);

  const selCount = selectedSampleIds.size;
  const allVisibleSelected = visibleRows.length > 0 && visibleRows.every(r => selectedSampleIds.has(r.sampleId));

  const bulkBtn = 'px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors';

  return (
    <div className="flex flex-col gap-2">
      {/* Toolbar */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <div className="relative">
            <button
              onClick={() => setColMenuOpen(o => !o)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-white/10 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
            >
              <Columns3 size={13} /> {t('trainingStudio.grid.columns')}
            </button>
            {colMenuOpen && (
              <>
                <div className="fixed inset-0 z-40" onClick={() => setColMenuOpen(false)} />
                <div className="absolute left-0 top-full mt-1 z-50 min-w-[180px] max-h-80 overflow-y-auto py-1 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 shadow-xl">
                  {toggleableCols.map(c => (
                    <button
                      key={c.id}
                      onClick={() => toggleCol(c.id)}
                      className="w-full flex items-center gap-2.5 px-3 py-1.5 text-xs text-zinc-700 dark:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
                    >
                      {isColVisible(c.id)
                        ? <CheckSquare size={14} className="text-amber-500 flex-shrink-0" />
                        : <Square size={14} className="text-zinc-500 flex-shrink-0" />}
                      {t(c.labelKey)}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          <button
            onClick={() => void rescan()}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-white/10 text-xs text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
          >
            <RefreshCw size={13} /> {t('trainingStudio.grid.rescan')}
          </button>
        </div>

        <div className="flex items-center gap-2">
          {(filter === 'unlabeled' || rows.length > BIG_GRID_ROWS) && (
            <span className="text-[11px] text-zinc-500">
              {t('trainingStudio.grid.showing', { visible: visibleRows.length, total: rows.length })}
            </span>
          )}
          <div className="flex items-center rounded-lg border border-zinc-200 dark:border-white/10 overflow-hidden">
            {(['all', 'unlabeled'] as RowFilter[]).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-2.5 py-1 text-[11px] transition-colors ${
                  filter === f ? 'bg-amber-500/20 text-amber-500' : 'text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300'
                }`}
              >
                {f === 'all' ? t('trainingStudio.grid.filterAll') : t('trainingStudio.grid.statusUnlabeled')}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Bulk bar */}
      {selCount > 0 && (
        <div className="flex items-center gap-2 flex-wrap px-3 py-2 rounded-xl bg-zinc-100/80 dark:bg-zinc-800/60 border border-zinc-200 dark:border-white/5">
          <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
            {t('trainingStudio.grid.selected', { count: selCount })}
          </span>
          <button className={bulkBtn} onClick={() => void bulkSet({ excluded: true })}>{t('trainingStudio.grid.bulkExclude')}</button>
          <button className={bulkBtn} onClick={() => void bulkSet({ excluded: false })}>{t('trainingStudio.grid.bulkInclude')}</button>
          <button className={bulkBtn} onClick={() => void bulkSet({ isInstrumental: true })}>{t('trainingStudio.grid.bulkInstrumental')}</button>
          <button
            className={bulkBtn}
            onClick={() => {
              const genre = window.prompt(t('trainingStudio.grid.bulkGenre'));
              if (genre !== null) void bulkSet({ genre });
            }}
          >
            {t('trainingStudio.grid.bulkGenre')}
          </button>
          <button
            className={bulkBtn}
            onClick={() => {
              const tag = window.prompt(t('trainingStudio.grid.bulkTag'));
              if (tag !== null) void bulkSet({ customTag: tag });
            }}
          >
            {t('trainingStudio.grid.bulkTag')}
          </button>
          <button
            className={bulkBtn}
            onClick={() => void startLabel({ sampleIds: Array.from(selectedSampleIds) })}
          >
            {t('trainingStudio.grid.bulkLabel')}
          </button>
          {/* §5.4-7 — scoped enhance jobs. Two EXPLICIT buttons: "Enhance" was
              ambiguous between lyrics (Genius) and captions (LLM), and users
              ran the wrong job. Each is hidden when its key is missing. */}
          {caps?.genius.configured && (
            <button
              className={bulkBtn}
              onClick={() => void startGenius({ sampleIds: Array.from(selectedSampleIds) })}
            >
              {t('trainingStudio.grid.bulkGenius')}
            </button>
          )}
          {caps?.llm.configured && (
            <button
              className={bulkBtn}
              onClick={() => void startCaption({ sampleIds: Array.from(selectedSampleIds) })}
            >
              {t('trainingStudio.grid.bulkCaption')}
            </button>
          )}
          <button
            onClick={clearSelection}
            className="ml-auto flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
          >
            <X size={12} /> {t('trainingStudio.grid.clearSelection')}
          </button>
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-zinc-200 dark:border-white/5">
        <table className="text-xs" style={{ tableLayout: 'fixed', width: '100%', minWidth: totalW }}>
          <colgroup>
            {columns.map(c => <col key={c.id} style={{ width: c.id === 'spacer' ? 'auto' : getW(c.id) }} />)}
          </colgroup>
          <thead>
            <tr className="bg-zinc-100/80 dark:bg-zinc-800/80 text-zinc-500 text-left">
              {columns.map(c => (
                <th
                  key={c.id}
                  className={`relative px-2 py-2 font-semibold select-none ${
                    c.align === 'center' ? 'text-center' : c.align === 'right' ? 'text-right' : ''
                  } ${
                    c.id === 'actions'
                      ? 'sticky right-0 z-20 bg-zinc-100 dark:bg-zinc-800 border-l border-zinc-200 dark:border-white/10'
                      : ''
                  }`}
                >
                  {c.id === 'select' ? (
                    <button
                      onClick={() => allVisibleSelected ? clearSelection() : setSelectedSampleIds(visibleRows.map(r => r.sampleId))}
                      className="text-zinc-500 hover:text-amber-500 transition-colors"
                    >
                      {allVisibleSelected ? <CheckSquare size={14} className="text-amber-500" /> : <Square size={14} />}
                    </button>
                  ) : (c.labelKey ? t(c.labelKey) : '')}
                  {c.resizable && (
                    <div
                      className="absolute top-0 right-0 w-1.5 h-full cursor-col-resize z-10 group/resize hover:bg-amber-500/30"
                      onMouseDown={(e) => { e.preventDefault(); handleResizeStart(c.id, c.minW, e.clientX); }}
                    >
                      <div className="w-px h-full mx-auto bg-zinc-300 dark:bg-zinc-600 group-hover/resize:bg-amber-400 transition-colors" />
                    </div>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map(row => (
              <SampleRow
                key={row.sampleId}
                sample={row}
                columns={columns}
                selected={selectedSampleIds.has(row.sampleId)}
                saving={savingSampleIds.has(row.sampleId)}
                hasSaveError={saveError?.sampleId === row.sampleId}
                isPreviewing={previewId === row.sampleId}
                onToggleSelect={() => toggleSampleSelected(row.sampleId)}
                onOpenDrawer={() => setOpenSampleId(row.sampleId)}
                onTogglePreview={() => setPreviewId(prev => (prev === row.sampleId ? null : row.sampleId))}
              />
            ))}
          </tbody>
        </table>
        {visibleRows.length === 0 && (
          <div className="py-10 text-center text-xs text-zinc-500">{t('trainingStudio.wizard.foundNone')}</div>
        )}
      </div>

      {/* Standalone preview element — never routed through the playback store. */}
      {datasetId && previewId && (
        <audio
          ref={audioRef}
          src={sampleAudioUrl(datasetId, previewId)}
          onEnded={() => setPreviewId(null)}
          controls
          className="w-full h-8"
        />
      )}

      {openSampleId && <SampleDrawer sampleId={openSampleId} />}

    </div>
  );
};

export default SampleGrid;
