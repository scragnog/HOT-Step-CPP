// SessionView.tsx - one Compare session: sticky transport, take rows, diff trace.
// Keys: Space play/pause, 1-9 switch active take, B (or Tab) A/B quick switch.

import React, { useEffect, useMemo, useState } from 'react';
import { ConfirmDialog } from '../shared/ConfirmDialog';
import { useTranslation } from 'react-i18next';
import { Play, Pause, ArrowLeft, Archive, ArchiveRestore, Trash2, X } from 'lucide-react';
import { useQueries, useQuery } from '@tanstack/react-query';
import { compareQueries } from '../../queries/compare/queries';
import { useArchiveCompareSession, useDeleteCompareSession } from '../../queries/compare/writes';
import type { CompareAnalysis, CompareSession } from '../../services/compareApi';
import { useAbEngine, type AbEngine } from './useAbEngine';
import { TakeRow } from './TakeRow';
import { DiffTrace } from './DiffTrace';
import { LoopOverlay, Playhead } from './Waveform';
import { Toggle } from '../shared/Toggle';
import { QueryError } from '../shared/QueryError';
import { initialPair, normalizePair, setSlot, tapTake, toggleTarget, type AbPair } from './abPair';

const mmss = (s: number) => {
  const v = Math.max(0, s || 0);
  return `${Math.floor(v / 60)}:${String(Math.floor(v % 60)).padStart(2, '0')}`;
};

const TimeLabel: React.FC<{ engine: AbEngine; duration: number }> = ({ engine, duration }) => {
  const t = engine.useCurrentTime();
  return <span className="font-mono text-xs text-zinc-500 tabular-nums">{mmss(t)} / {mmss(duration)}</span>;
};

const SeekBar: React.FC<{ engine: AbEngine; duration: number }> = ({ engine, duration }) => {
  const seekAt = (e: React.PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    if (!duration) return;
    engine.seek(Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1) * duration);
  };
  return (
    <div
      role="slider"
      aria-valuemin={0}
      aria-valuemax={duration}
      className="relative flex-1 h-8 rounded-md bg-zinc-200 dark:bg-zinc-800 cursor-pointer touch-none overflow-hidden"
      onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); seekAt(e); }}
      onPointerMove={e => { if (e.buttons) seekAt(e); }}
    >
      <LoopOverlay loop={engine.loop} duration={duration} />
      <Playhead engine={engine} duration={duration} />
    </div>
  );
};

interface Props { session: CompareSession; onBack: () => void; }

export const SessionView: React.FC<Props> = ({ session, onBack }) => {
  const { t } = useTranslation();
  const takes = session.takes;
  const [confirmDelete, setConfirmDelete] = useState(false);
  const archive = useArchiveCompareSession();
  const remove = useDeleteCompareSession();

  // One analysis entry per take, shared with any other view that asked for
  // the same song. A failed analysis stays local to its take row.
  const analysisQueries = useQueries({
    queries: useMemo(() => takes.map(tk => compareQueries.analysis(tk.songId)), [takes]),
  });
  const analysis = useMemo(() => {
    const out: Record<string, CompareAnalysis> = {};
    takes.forEach((tk, i) => {
      const data = analysisQueries[i]?.data;
      if (data) out[tk.songId] = data;
    });
    return out;
  }, [takes, analysisQueries]);

  const abTakes = useMemo(() => takes.filter(tk => tk.audioUrl).map(tk => ({
    id: tk.songId, url: tk.audioUrl as string, integratedLufs: analysis[tk.songId]?.integratedLufs ?? null,
  })), [takes, analysis]);
  const engine = useAbEngine(abTakes);

  const duration = engine.duration || Math.max(0, ...takes.map(tk => analysis[tk.songId]?.durationSec ?? tk.duration ?? 0));
  const baselineId = session.baselineSongId ?? takes[0]?.songId ?? null;
  const activeId = engine.activeId;

  const diffQuery = useQuery({
    ...compareQueries.diff(activeId ?? '', baselineId ?? ''),
    enabled: !!activeId && !!baselineId && activeId !== baselineId,
  });

  // A/B quick switch: explicit {a, b} slots, normalized so a !== b with 2+ takes.
  const takeIds = useMemo(() => takes.map(tk => tk.songId), [takes]);
  const [rawPair, setPair] = useState<AbPair>(() => initialPair(takeIds, baselineId));
  const pair = useMemo(() => normalizePair(rawPair, takeIds, baselineId), [rawPair, takeIds, baselineId]);
  const activateTake = (id: string) => {
    setPair(p => tapTake(normalizePair(p, takeIds, baselineId), activeId, id));
    void engine.unlock(); engine.setActive(id);
  };
  const setAbSlot = (slot: 'a' | 'b', id: string) => setPair(p => setSlot(normalizePair(p, takeIds, baselineId), slot, id));
  const abSwitch = () => {
    const target = toggleTarget(pair, activeId);
    if (target) { void engine.unlock(); engine.setActive(target); }
  };
  const numOf = (id: string | null) => { const i = id ? takeIds.indexOf(id) : -1; return i >= 0 ? String(i + 1) : '-'; };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      if (e.code === 'Space') { e.preventDefault(); void engine.unlock().then(() => engine.toggle()); return; }
      if ((e.key === 'b' || e.key === 'B' || e.key === 'Tab') && !e.ctrlKey && !e.metaKey && !e.altKey && takes.length > 1) {
        e.preventDefault(); abSwitch(); return;
      }
      const n = parseInt(e.key, 10);
      if (n >= 1 && n <= 9 && takes[n - 1]) { e.preventDefault(); activateTake(takes[n - 1].songId); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine, takes, pair, activeId]);

  const setIn = () => {
    const now = engine.getCurrentTime();
    const end = engine.loop && engine.loop.end > now + 0.1 ? engine.loop.end : duration;
    if (end - now > 0.1) engine.setLoop({ start: now, end });
  };
  const setOut = () => {
    const now = engine.getCurrentTime();
    const start = engine.loop && engine.loop.start < now - 0.1 ? engine.loop.start : 0;
    if (now - start > 0.1) engine.setLoop({ start, end: now });
  };

  const titleOf = (id: string | null) => takes.find(tk => tk.songId === id)?.title || id || '';
  const btn = 'h-10 px-3 rounded-lg text-xs font-semibold border border-zinc-300 dark:border-white/10 text-zinc-700 dark:text-zinc-300 hover:border-pink-400';

  return (
    <div className="flex flex-col min-h-0 h-full">
      <ConfirmDialog
        isOpen={confirmDelete}
        title={t('compare.deleteTitle', 'Delete comparison')}
        message={t('compare.deleteConfirm', 'Delete this comparison and its scores? Songs stay in the library.')}
        danger
        onConfirm={() => { setConfirmDelete(false); remove.mutate(session.id, { onSuccess: () => onBack() }); }}
        onCancel={() => setConfirmDelete(false)}
      />
      <div className="sticky top-0 z-10 px-3 sm:px-4 py-3 bg-white/90 dark:bg-zinc-950/90 backdrop-blur border-b border-zinc-200 dark:border-white/5">
        <div className="flex items-center gap-2 mb-2">
          <button onClick={onBack} className="p-2 rounded-lg text-zinc-500 hover:text-pink-400" aria-label={t('compare.back', 'Back to sessions')}>
            <ArrowLeft size={18} />
          </button>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-bold truncate text-zinc-900 dark:text-zinc-100">{session.name}</div>
            <div className="text-[11px] text-zinc-500">{session.group || t('compare.ungrouped', 'Ungrouped')} · {session.scored}/{session.total} {t('compare.scored', 'scored')}</div>
          </div>
          <button onClick={() => archive.mutate({ id: session.id, archived: !session.archived })} className="p-2 rounded-lg text-zinc-500 hover:text-pink-400"
            title={session.archived ? t('compare.unarchive', 'Unarchive') : t('compare.archive', 'Archive')}
            aria-label={session.archived ? t('compare.unarchive', 'Unarchive') : t('compare.archive', 'Archive')}>
            {session.archived ? <ArchiveRestore size={18} /> : <Archive size={18} />}
          </button>
          <button onClick={() => { setConfirmDelete(true); }}
            className="p-2 rounded-lg text-zinc-500 hover:text-red-400" title={t('compare.delete', 'Delete')} aria-label={t('compare.delete', 'Delete')}>
            <Trash2 size={18} />
          </button>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void engine.unlock().then(() => engine.toggle())}
            className="w-12 h-12 shrink-0 rounded-full bg-pink-500 hover:bg-pink-400 text-white flex items-center justify-center"
            aria-label={engine.isPlaying ? t('compare.pause', 'Pause') : t('compare.play', 'Play')}
          >
            {engine.isPlaying ? <Pause size={22} /> : <Play size={22} />}
          </button>
          {takes.length > 1 && (
            <button
              onClick={abSwitch}
              className="h-12 min-w-[48px] px-3 shrink-0 rounded-xl border-2 border-pink-500 text-pink-500 hover:bg-pink-500/10 flex flex-col items-center justify-center leading-tight"
              title={t('compare.abSwitchInfo', 'Switch between the takes in slot A and slot B at the same playhead (B or Tab). Use Set A / Set B on a take to choose them.')}
              aria-label={t('compare.abSwitch', 'A/B switch')}
            >
              <span className="text-sm font-black">A/B</span>
              <span className="text-[10px] font-mono whitespace-nowrap">
                {t('compare.abSlots', 'A: {{a}} / B: {{b}}', { a: numOf(pair.a), b: numOf(pair.b) })}
              </span>
            </button>
          )}
          <SeekBar engine={engine} duration={duration} />
          <span className="hidden sm:inline"><TimeLabel engine={engine} duration={duration} /></span>
        </div>
        <div className="flex flex-wrap items-center gap-2 mt-2">
          <button className={btn} onClick={setIn}>{t('compare.loopIn', 'In')}</button>
          <button className={btn} onClick={setOut}>{t('compare.loopOut', 'Out')}</button>
          {engine.loop && (
            <button className={`${btn} flex items-center gap-1`} onClick={() => engine.setLoop(null)}>
              <X size={12} /> {t('compare.loopClear', 'Clear loop')} {mmss(engine.loop.start)}-{mmss(engine.loop.end)}
            </button>
          )}
          <span className="sm:hidden"><TimeLabel engine={engine} duration={duration} /></span>
          <div className="ml-auto">
            <Toggle
              size="sm"
              accent="pink"
              checked={engine.levelMatch}
              onChange={engine.setLevelMatch}
              label={t('compare.levelMatch', 'Level match')}
              info={t('compare.levelMatchInfo', 'Turns every take down to the loudness of the quietest one, so you judge tone and character instead of volume. Louder nearly always sounds better at first, so leave this on for fair scoring.')}
            />
          </div>
        </div>
        <div className="text-[10px] text-zinc-500 mt-1 hidden sm:block">{t('compare.keysHint', 'Space play/pause, 1-9 switch take, B or Tab A/B switch, drag a waveform to set a loop')}</div>
      </div>

      <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-3 flex flex-col gap-3">
        {takes.length === 0 && <div className="text-sm text-zinc-500">{t('compare.noTakes', 'No takes in this comparison.')}</div>}
        {takes.map((tk, i) => (
          <TakeRow
            key={tk.songId}
            sessionId={session.id}
            index={i}
            take={tk}
            criteria={session.criteria}
            analysis={analysis[tk.songId] ?? null}
            analysisLoading={!!analysisQueries[i]?.isPending}
            analysisError={analysisQueries[i]?.error}
            onAnalysisRetry={() => void analysisQueries[i]?.refetch()}
            engine={engine}
            duration={duration}
            isBaseline={tk.songId === baselineId}
            abSlot={takes.length > 1 ? (tk.songId === pair.a ? 'A' : tk.songId === pair.b ? 'B' : null) : null}
            showAbButtons={takes.length > 1}
            onActivate={() => activateTake(tk.songId)}
            onSetSlot={slot => setAbSlot(slot, tk.songId)}
          />
        ))}
        {takes.length > 1 && (
          <>
            <DiffTrace
              diff={diffQuery.data && diffQuery.data.a === activeId && diffQuery.data.b === baselineId ? diffQuery.data : null}
              activeTitle={titleOf(activeId)}
              baselineTitle={titleOf(baselineId)}
              same={!activeId || activeId === baselineId}
            />
            {diffQuery.isError && (
              <QueryError error={diffQuery.error} onRetry={() => void diffQuery.refetch()} />
            )}
          </>
        )}
      </div>
    </div>
  );
};
