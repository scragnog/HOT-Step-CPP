// LadderReview.tsx — the listening-test parts every engine's checkpoint ladder
// shows: the blind-rungs toggle, the album score, a rung's likeness/corruption
// scores and notes, the floating scoreboard and the cleanup modal after "Use
// this …". Presentation only: Yue2LadderReview and Mm3CheckpointLadder own the
// data and the writes, so the two engines differ in data source, not in look.
import { ChevronDown, ChevronUp } from 'lucide-react';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { usePersistedState, writePersistedState } from '../../hooks/usePersistedState';
import type { Yue2CleanupChoice, Yue2CleanupItem, Yue2TrainedDirection } from '../../services/trainingApi';
import { Toggle } from '../settings/SettingsPrimitives';
import { ParamLabel } from '../shared/ParamLabel';

const input = 'px-2 py-1.5 rounded-lg text-xs bg-white/70 dark:bg-black/20 border border-zinc-300/70 dark:border-white/10 text-zinc-800 dark:text-zinc-100';
export const mib = (b: number) => b >= 1073741824 ? `${(b / 1073741824).toFixed(2)} GiB` : `${(b / 1048576).toFixed(0)} MiB`;

const BLIND_KEY = 'hs-yue2-blind-rungs';
/** One "Blind rungs" preference for every engine's ladder. */
export const useBlindRungs = () => usePersistedState(BLIND_KEY, true)[0];

export const BlindRungsToggle: React.FC<{ id: string; checked: boolean }> = ({ id, checked }) => {
  const { t } = useTranslation();
  return <div className="flex items-center gap-2 text-xs">
    <Toggle id={id} checked={checked} onChange={value => writePersistedState(BLIND_KEY, value)} />
    <ParamLabel label={t('trainingStudio.refine.blindToggle', 'Blind rungs')}
      info={t('trainingStudio.refine.blindToggleInfo', 'Show lettered rungs while listening and scoring. The chosen step is revealed after cleanup finishes.')} />
  </div>;
};

export interface AlbumVerdict { score: number | null; instruments: Yue2TrainedDirection | null; vocals: Yue2TrainedDirection | null; notes: string }
export type AlbumPatch = { score?: number | null; instruments?: Yue2TrainedDirection | null; vocals?: Yue2TrainedDirection | null; notes?: string };

/** The run's album verdict: 1-5, which way each half missed, and notes. */
export const AlbumScoreBar: React.FC<{ album: AlbumVerdict | null; pending: boolean; onScore: (patch: AlbumPatch) => void;
  note: string | undefined; onNote: (note: string) => void }> = ({ album, pending, onScore, note, onNote }) => {
  const { t } = useTranslation();
  return <div className="rounded-lg border border-violet-500/40 bg-violet-500/5 p-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px]">
    <ParamLabel label={t('trainingStudio.refine.albumScore', 'How well did this album train?')}
      info={t('trainingStudio.refine.albumScoreInfo', 'One score for the whole run, once you have heard enough of its ladder: 1 = the adapter never really caught the album, 5 = as good as the best albums you have trained. Score it against your other albums, not against this run\'s own rungs. The rung scores show how fast an album trains; this is the only score that shows how well, and Dataset-Calibrated Training learns from it.')} />
    <div className="flex items-center gap-1">
      {[1, 2, 3, 4, 5].map(n => <button key={n} type="button" disabled={pending} onClick={() => onScore({ score: album?.score === n ? null : n })}
        className={`w-6 h-6 rounded border text-[11px] font-semibold ${album?.score === n ? 'bg-violet-500/20 border-violet-500 text-violet-700 dark:text-violet-300' : 'border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10'}`}>{n}</button>)}
    </div>
    {(['instruments', 'vocals'] as const).map(half => <div key={half} className="flex items-center gap-1">
      <ParamLabel label={half === 'instruments' ? t('trainingStudio.refine.albumInstruments', 'Instruments') : t('trainingStudio.refine.albumVocals', 'Vocals')}
        className="text-[11px] text-zinc-500 w-20"
        info={half === 'instruments'
          ? t('trainingStudio.refine.albumInstrumentsInfo', 'Which way the instruments missed at the rung you would pick. Under: the band\'s sound, playing and arrangement never really arrived. Right: nothing to change. Over: overcooked, memorised riffs or the same parts in every song, a smeared or brittle mix. The score says how well; this says whether the next run should train more or less.')
          : t('trainingStudio.refine.albumVocalsInfo', 'Which way the vocals missed at the rung you would pick. Under: the singer\'s voice and delivery never really arrived. Right: nothing to change. Over: overcooked, garbled words, one phrase or melody in every song, a strained or breaking voice. The score says how well; this says whether the next run should train more or less.')} />
      {(['under', 'right', 'over'] as const).map(d => <button key={d} type="button" disabled={pending} onClick={() => onScore({ [half]: album?.[half] === d ? null : d })}
        className={`px-2 h-6 rounded border text-[11px] font-semibold ${album?.[half] === d ? (d === 'right' ? 'bg-emerald-500/20 border-emerald-500 text-emerald-700 dark:text-emerald-300' : 'bg-violet-500/20 border-violet-500 text-violet-700 dark:text-violet-300') : 'border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10'}`}>
        {d === 'under' ? t('trainingStudio.refine.dirUnder', 'Under') : d === 'right' ? t('trainingStudio.refine.dirRight', 'Right') : t('trainingStudio.refine.dirOver', 'Over')}</button>)}
    </div>)}
    <input className={`${input} flex-1 min-w-[240px]`} placeholder={t('trainingStudio.refine.albumNotes', 'Notes: what this album got right or never got')}
      value={note ?? album?.notes ?? ''} onChange={e => onNote(e.target.value)}
      onBlur={e => { if (e.target.value !== (album?.notes ?? '')) onScore({ notes: e.target.value }); }} />
  </div>;
};

export type RungPatch = { likeness?: number | null; corruption?: number | null; notes?: string };

/** A rung's likeness and corruption (1-5, click again to clear) and its notes. */
export const RungScoreRow: React.FC<{ score: { likeness?: number | null; corruption?: number | null; notes?: string } | undefined;
  note: string | undefined; onNote: (note: string) => void; onScore: (patch: RungPatch) => void }> = ({ score, note, onNote, onScore }) => {
  const { t } = useTranslation();
  return <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-[11px]">
    {(['likeness', 'corruption'] as const).map(key => <div key={key} className="flex items-center gap-1">
      <span className="text-zinc-500 w-16">{key === 'likeness' ? t('trainingStudio.refine.scoreLikeness', 'Likeness') : t('trainingStudio.refine.scoreCorruption', 'Corruption')}</span>
      {[1, 2, 3, 4, 5].map(n => <button key={n} type="button" onClick={() => onScore({ [key]: score?.[key] === n ? null : n })}
        className={`w-6 h-6 rounded border text-[11px] font-semibold ${score?.[key] === n ? (key === 'corruption' ? 'bg-red-500/20 border-red-500 text-red-700 dark:text-red-300' : 'bg-amber-500/20 border-amber-500 text-amber-700 dark:text-amber-300') : 'border-zinc-300/70 dark:border-white/10 text-zinc-500 hover:bg-zinc-500/10'}`}>{n}</button>)}
    </div>)}
    <input className={`${input} flex-1 min-w-[240px]`} placeholder={t('trainingStudio.refine.notes', 'Notes: what you heard, and where (m:ss)')}
      value={note ?? score?.notes ?? ''} onChange={e => onNote(e.target.value)}
      onBlur={e => { if (e.target.value !== (score?.notes ?? '')) onScore({ notes: e.target.value }); }} />
  </div>;
};

export interface ScoreboardRow {
  step: number; kl?: number; likeness: number | null; corruption: number | null;
  /** Null until scored on both axes. No `replansPerTake`: the engine has no replans. */
  overall: { overall: number; replansPerTake?: number } | null;
}

/** Best scored first (or by blind letter), each row scrolling to its rung. */
export const LadderScoreboard: React.FC<{ rows: ScoreboardRow[]; bestStep: number; blind: boolean; rungName: (step: number) => string;
  idPrefix: string; formula: string }> = ({ rows, bestStep, blind, rungName, idPrefix, formula }) => {
  const { t } = useTranslation();
  const [collapsed, setCollapsed] = useState(false);
  return <div className="hidden md:block fixed right-4 bottom-4 z-40 max-w-xs rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white/90 dark:bg-zinc-900/90 shadow-xl p-3">
    <div className="flex items-center justify-between gap-2">
      <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-100">{t('trainingStudio.refine.scoreboardTitle', 'Scoreboard')}</span>
      <button type="button" onClick={() => setCollapsed(v => !v)} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200">
        {collapsed ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
      </button>
    </div>
    {!collapsed && <>
      <p className="mt-1 text-[10px] text-zinc-500">{formula}</p>
      <div className="mt-2 flex flex-col gap-1 max-h-64 overflow-y-auto">
        {rows.map(row => (
          <button key={row.step} type="button"
            onClick={() => document.getElementById(`${idPrefix}-${row.step}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
            className={`w-full text-left px-2 py-1 rounded-lg text-[11px] ${row.overall ? (row.step === bestStep ? 'bg-sky-500/15 text-sky-800 dark:text-sky-200' : 'hover:bg-zinc-500/10 text-zinc-700 dark:text-zinc-300') : 'text-zinc-500'}`}>
            {row.overall ? <>
              <div className="flex items-center justify-between gap-2">
                <span>{blind ? rungName(row.step) : t('trainingStudio.refine.scoreboardStep', 'step {{step}}', { step: row.step })}
                  {row.step === bestStep && <span className="ml-1 px-1 py-0.5 rounded bg-sky-500/20 text-[9px] font-semibold">{t('trainingStudio.refine.bestChip', 'best')}</span>}</span>
                <span className="font-bold font-mono">{row.overall.overall.toFixed(2)}</span>
              </div>
              <div className="text-[10px] text-zinc-500">{!blind && row.kl !== undefined ? `KL ${row.kl.toFixed(2)} · ` : ''}L {row.likeness} · C {row.corruption}{row.overall.replansPerTake !== undefined ? ` · ${t('trainingStudio.refine.scoreboardReplans', 'replans {{n}}/take', { n: row.overall.replansPerTake.toFixed(2) })}` : ''}</div>
            </> : blind ? `${rungName(row.step)} · ${t('trainingStudio.refine.unscored', 'unscored')}` : t('trainingStudio.refine.scoreboardUnscored', 'step {{step}} · unscored', { step: row.step })}
          </button>
        ))}
      </div>
    </>}
  </div>;
};

export type CleanupItemRow = { key: keyof Yue2CleanupChoice; label: string; item?: Yue2CleanupItem };

/** "<rung> is now the adapter. Clean up around it?": one toggle per item with
 *  its count and size, "Keep everything" and "Delete selected". */
export const LadderCleanupModal: React.FC<{
  title: string; intro: string; items: CleanupItemRow[]; blind: boolean; idPrefix: string;
  choice: Yue2CleanupChoice; onChoice: (choice: Yue2CleanupChoice) => void;
  cleaning: boolean; onClose: () => void; onCleanup: (choice: Yue2CleanupChoice) => void;
}> = ({ title, intro, items, blind, idPrefix, choice, onChoice, cleaning, onClose, onCleanup }) => {
  const { t } = useTranslation();
  const totalBytes = items.reduce((s, i) => s + (choice[i.key] && i.item ? i.item.bytes : 0), 0);
  const keepAll: Yue2CleanupChoice = { caches: false, otherCheckpoints: false, otherRuns: false, resume: false, otherPreviews: false };
  return <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => !cleaning && onClose()}>
    <div className="w-full max-w-xl rounded-xl border border-zinc-300/70 dark:border-white/10 bg-white dark:bg-zinc-900 p-5 shadow-xl" onClick={e => e.stopPropagation()}>
      <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">{title}</div>
      <p className="mt-1 text-[12px] text-zinc-600 dark:text-zinc-400">{intro}</p>
      <div className="mt-3 flex flex-col gap-2">
        {items.map(i => <label key={i.key} className="flex items-start gap-3 text-xs text-zinc-700 dark:text-zinc-300">
          <Toggle id={`cleanup-${idPrefix}-${i.key}`} checked={!!choice[i.key] && !!i.item?.count} onChange={v => onChoice({ ...choice, [i.key]: v })} />
          <span className={`flex-1 ${!i.item?.count ? 'opacity-50' : ''}`}>{i.label}<span className="ml-2 text-zinc-500 tabular-nums">{i.item ? `${i.item.count} · ${mib(i.item.bytes)}` : ''}</span>
            {!blind && i.item?.detail?.length ? <span className="block text-[10px] text-zinc-500 truncate">{i.item.detail.join(', ')}</span> : null}</span>
        </label>)}
      </div>
      <div className="mt-4 flex items-center justify-between gap-3">
        <span className="text-[11px] text-zinc-500 tabular-nums">{t('trainingStudio.refine.cleanupTotal', 'About {{size}} to free', { size: mib(totalBytes) })}</span>
        <div className="flex gap-2">
          <button type="button" disabled={cleaning} onClick={() => onCleanup(keepAll)} className="px-3 py-1.5 rounded-lg text-xs border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10">{t('trainingStudio.refine.cleanupSkip', 'Keep everything')}</button>
          <button type="button" disabled={cleaning || totalBytes === 0} onClick={() => onCleanup(choice)} className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-500 text-white hover:bg-red-400 disabled:opacity-40">{cleaning ? t('trainingStudio.refine.cleaning', 'Cleaning…') : t('trainingStudio.refine.cleanupGo', 'Delete selected')}</button>
        </div>
      </div>
    </div>
  </div>;
};

/** Overall = (likeness + (6 − corruption)) / 2, to two places; null until
 *  both are scored. YuE2 subtracts a replan penalty on top (yue2BestRung.ts). */
export function plainOverall(likeness: number | null | undefined, corruption: number | null | undefined): number | null {
  if (typeof likeness !== 'number' || typeof corruption !== 'number') return null;
  return Math.round((likeness + 6 - corruption) / 2 * 100) / 100;
}
