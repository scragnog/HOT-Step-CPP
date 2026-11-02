// TakeRow.tsx - one take in a Compare session: waveform, metrics, seeds, scores.

import React, { useEffect, useState } from 'react';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { useTranslation } from 'react-i18next';
import { Trash2 } from 'lucide-react';
import type { CompareAnalysis, CompareCriterion, CompareTake } from '../../services/compareApi';
import { useRemoveCompareTake, useSetCompareScore } from '../../queries/compare/writes';
import type { AbEngine } from './useAbEngine';
import { Waveform } from './Waveform';
import { Toggle } from '../shared/Toggle';
import { QueryError } from '../shared/QueryError';

interface Props {
  /** Session the take belongs to; commands patch its cache entry. */
  sessionId: string;
  index: number;
  /** 'A' or 'B' when this take is in the A/B quick-switch pair. */
  abSlot?: 'A' | 'B' | null;
  /** Show the Set A / Set B buttons (2+ takes). */
  showAbButtons?: boolean;
  /** Make this the playing take (may update the A/B pair). */
  onActivate?: () => void;
  /** Put this take into slot A or B. */
  onSetSlot?: (slot: 'a' | 'b') => void;
  take: CompareTake;
  criteria: CompareCriterion[];
  analysis: CompareAnalysis | null;
  analysisLoading: boolean;
  /** Set when this take's analysis read failed. */
  analysisError?: unknown;
  onAnalysisRetry?: () => void;
  engine: AbEngine;
  duration: number;
  isBaseline: boolean;
}

const fmt = (v: number | null | undefined, d = 1, unit = ''): string =>
  typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(d)}${unit}` : '-';

const Chip: React.FC<{ label: string; value: string; warn?: boolean; title?: string }> = ({ label, value, warn, title }) => (
  <span title={title} className={`inline-flex items-baseline gap-1 px-2 py-0.5 rounded-md text-[11px] border ${warn
    ? 'border-red-500/40 bg-red-500/10 text-red-400'
    : 'border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-zinc-800/60 text-zinc-700 dark:text-zinc-300'}`}>
    <span className="text-zinc-500">{label}</span><span className="font-mono">{value}</span>
  </span>
);

export const TakeRow: React.FC<Props> = ({ sessionId, index, take, criteria, analysis, analysisLoading, analysisError, onAnalysisRetry, engine, duration, isBaseline, abSlot, showAbButtons, onActivate, onSetSlot }) => {
  const { t } = useTranslation();
  const score = useSetCompareScore();
  const remove = useRemoveCompareTake();
  const active = engine.activeId === take.songId;
  const scoreOf = (c: string) => take.scores.find(s => s.criterion === c);
  const scored = take.scores.some(s => s.value !== null || s.keep !== null || (s.note ?? '').length > 0);
  const noteCrit = criteria.find(c => c.type === 'note');
  const savedNote = noteCrit ? scoreOf(noteCrit.id)?.note ?? '' : '';
  const [note, setNote] = useState(savedNote);
  const [confirmRemove, setConfirmRemove] = useState(false);
  useEffect(() => { setNote(savedNote); }, [savedNote]);

  const activate = () => { if (onActivate) { onActivate(); return; } void engine.unlock(); engine.setActive(take.songId); };
  const tp = analysis?.truePeakDbtp;
  const gain = engine.levelMatch ? engine.gainsDb[take.songId] : undefined;

  return (
    <>
      <ConfirmDialog
        isOpen={confirmRemove}
        title={t('compare.removeTakeTitle', 'Remove take')}
        message={t('compare.removeTakeConfirm', 'Remove this take from the comparison?')}
        danger
        onConfirm={() => { setConfirmRemove(false); remove.mutate({ sessionId, songId: take.songId }); }}
        onCancel={() => setConfirmRemove(false)}
      />
    <div
      onClick={activate}
      className={`rounded-xl border p-3 transition-colors cursor-pointer ${active
        ? 'border-pink-500/60 bg-pink-500/5'
        : 'border-zinc-200 dark:border-white/5 bg-white/60 dark:bg-zinc-900/40 hover:border-zinc-400 dark:hover:border-white/20'}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className={`flex items-center justify-center w-9 h-9 shrink-0 rounded-lg text-sm font-bold ${active ? 'bg-pink-500 text-white' : 'bg-zinc-200 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300'}`}>
          {index + 1}
        </span>
        {showAbButtons && onSetSlot && (['a', 'b'] as const).map(slot => {
          const on = abSlot === slot.toUpperCase();
          return (
            <button
              key={slot}
              onClick={e => { e.stopPropagation(); onSetSlot(slot); }}
              className={`min-w-[48px] h-12 sm:h-8 sm:min-w-[40px] px-2 shrink-0 rounded-lg text-xs font-bold border-2 ${on
                ? 'bg-pink-500 border-pink-500 text-white'
                : 'border-zinc-300 dark:border-white/15 text-zinc-500 hover:border-pink-400 hover:text-pink-500'}`}
              aria-pressed={on}
              title={t('compare.setSlotInfo', 'Put this take in the A/B quick switch slot')}
            >
              {slot === 'a' ? t('compare.setA', 'Set A') : t('compare.setB', 'Set B')}
            </button>
          );
        })}
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold truncate text-zinc-900 dark:text-zinc-100">{take.title || take.songId}</div>
          <div className="text-[11px] text-zinc-500 flex flex-wrap gap-x-3">
            {isBaseline && <span className="text-sky-400 font-semibold">{t('compare.baseline', 'baseline')}</span>}
            {take.backend && <span>{take.backend}</span>}
            <span>{t('compare.seed', 'seed')} {take.seed ?? '-'}</span>
            {take.seedB !== null && <span>{t('compare.seedB', 'seed B')} {take.seedB}</span>}
            {take.seedBlend !== null && <span>{t('compare.blend', 'blend')} {take.seedBlend}</span>}
            {gain !== undefined && <span>{t('compare.gain', 'gain')} {fmt(gain, 1, ' dB')}</span>}
            {engine.errors[take.songId] && <span className="text-red-400">{engine.errors[take.songId]}</span>}
          </div>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${scored ? 'bg-emerald-500/15 text-emerald-400' : 'bg-zinc-500/15 text-zinc-500'}`}>
          {scored ? t('compare.scored', 'scored') : t('compare.unscored', 'unscored')}
        </span>
        <button
          onClick={e => { e.stopPropagation(); setConfirmRemove(true); }}
          className="p-2 rounded-lg text-zinc-500 hover:text-red-400"
          title={t('compare.removeTake', 'Remove take')}
          aria-label={t('compare.removeTake', 'Remove take')}
        >
          <Trash2 size={16} />
        </button>
      </div>

      <Waveform
        peaks={analysis?.peaks ?? null}
        duration={duration}
        engine={engine}
        active={active}
        onTap={sec => { activate(); engine.seek(sec); }}
      />

      <div className="flex flex-wrap gap-1.5 mt-2">
        {(analysisError && !analysis) ? (
          <QueryError error={analysisError} onRetry={onAnalysisRetry} compact />
        ) : null}
        {analysisLoading && !analysis && <span className="text-[11px] text-zinc-500">{t('compare.analyzing', 'Analysing...')}</span>}
        {analysis && <>
          <Chip label="LUFS" value={fmt(analysis.integratedLufs)} />
          <Chip label={t('compare.stMax', 'ST max')} value={fmt(analysis.shortTermMaxLufs)} />
          <Chip label="TP" value={fmt(tp, 1, ' dBTP')} warn={typeof tp === 'number' && tp > 0} title={t('compare.tpInfo', 'True peak. Red above 0 dBTP: likely to clip after conversion.')} />
          <Chip label="RMS" value={fmt(analysis.rmsDbfs, 1, ' dB')} />
          <Chip label={t('compare.crest', 'crest')} value={fmt(analysis.crestDb, 1, ' dB')} />
          <Chip label={t('compare.centroid', 'centroid')} value={fmt(analysis.centroidHz, 0, ' Hz')} />
          <Chip label={t('compare.flatness', 'flat')} value={fmt(analysis.flatness, 3)} />
          <Chip label={t('compare.corr', 'corr')} value={fmt(analysis.stereoCorr, 2)} />
        </>}
      </div>

      <div className="mt-3 flex flex-col gap-2" onClick={e => e.stopPropagation()}>
        {criteria.filter(c => c.type === 'scale').map(c => {
          const cur = scoreOf(c.id)?.value ?? null;
          const lo = c.min ?? 1, hi = c.max ?? 5;
          const opts = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
          return (
            <div key={c.id} className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-zinc-500 w-24 shrink-0">{c.label}</span>
              {opts.map(v => (
                <button
                  key={v}
                  onClick={() => score.mutate({ sessionId, items: [{ songId: take.songId, criterion: c.id, value: cur === v ? null : v }] })}
                  className={`min-w-[40px] h-10 rounded-lg text-sm font-semibold border ${cur === v
                    ? 'bg-pink-500 border-pink-500 text-white'
                    : 'border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-pink-400'}`}
                  aria-pressed={cur === v}
                >{v}</button>
              ))}
            </div>
          );
        })}
        {criteria.filter(c => c.type === 'keep').map(c => (
          <Toggle
            key={c.id}
            accent="emerald"
            label={c.label}
            checked={scoreOf(c.id)?.keep === true}
            onChange={v => score.mutate({ sessionId, items: [{ songId: take.songId, criterion: c.id, keep: v }] })}
          />
        ))}
        {noteCrit && (
          <input
            type="text"
            value={note}
            placeholder={noteCrit.label}
            onChange={e => setNote(e.target.value)}
            onBlur={() => { if (note !== savedNote) score.mutate({ sessionId, items: [{ songId: take.songId, criterion: noteCrit.id, note }] }); }}
            onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); }}
            className="w-full h-10 px-3 rounded-lg text-sm bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-zinc-900 dark:text-zinc-100"
          />
        )}
      </div>
    </div>
    </>
  );
};
