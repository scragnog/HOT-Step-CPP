// Prepare, step 2: assign lyrics. The pasted words are blocks on the left;
// every section of the confirmed score is a row on the right. A block goes to
// a row by a click, a drag or the Assign menu; a row that already has words
// asks whether to append or swap. The score is locked here: nothing in this
// view can change a section, a boundary or a label (the hook refuses it too).
// Suggestions from tags and order are asked for on arrival, once per score;
// the recording is used only when asked. They are proposals the user accepts
// row by row.
//
// Ready (the last step) shows the ordered pair that Generate and Build send.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Play, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { assignmentProblem } from '../../../../../server/src/contracts/yue2Prepare';
import { sectionColours } from '@/utils/abc/abcSections';
import { exportPrepare, structureConfirmed } from '@/utils/lyrics/yue2Prepare';
import type { Yue2AssignmentSuggestions } from '@/services/yue2CoverApi';
import type { PrepareSuggestions } from '@/components/cover-studio/build/useYue2Prepare';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { Toggle } from '@/components/shared/controls/Toggle';
import { clock, barTimes, sectionSpans, spanBars, spanName, spanSeconds, type Prepare, type ScoreRange, type SectionSpan } from '@/components/cover-studio/build/prepareView';

const btn = 'rounded-md border border-zinc-300 dark:border-white/10 px-2 py-1 text-xs text-zinc-700 dark:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-white/5 disabled:opacity-40';

interface Props {
  prep: Prepare;
  /** The pasted lyrics (the studio's lyrics field: dataset, Genius or typed). */
  lyrics: string;
  onLyricsText: (text: string) => void;
  /** Asks the server for proposals for the current document. */
  suggest: (useRecording: boolean) => Promise<Yue2AssignmentSuggestions>;
  scorePlaying: boolean;
  onPlayScore: (range: ScoreRange | null) => void;
  /** The score section now playing, to mark its row. */
  activeSection: number | null;
  /** Two columns, when the panel is wide enough. */
  wide: boolean;
}

/** Score structures whose suggestions were already asked for on arrival. */
const suggestedOnArrival = new Set<string>();

interface Proposal { guard: ReturnType<Prepare['captureGuard']>; reply: Yue2AssignmentSuggestions; off: string[] }

const firstLine = (text: string) => text.split('\n').find(line => line.trim())?.trim() ?? '';

export const PrepareLyrics: React.FC<Props> = ({ prep, lyrics, onLyricsText, suggest, scorePlaying, onPlayScore, activeSection, wide }) => {
  const { t } = useTranslation();
  const { doc } = prep;
  const spans = useMemo(() => sectionSpans(doc.abc, doc.sectionIds), [doc.abc, doc.sectionIds]);
  const list: SectionSpan[] = typeof spans === 'string' ? [] : spans;
  const colours = sectionColours(list.map(s => s.label));
  const clockMap = useMemo(() => barTimes(doc.abc), [doc.abc]);
  const spanOf = (id: string) => list.find(s => s.id === id);
  const blockOf = (id: string) => doc.blocks.find(b => b.id === id);
  const ready = prep.step === 'ready';
  const locked = prep.editingStructure || !structureConfirmed(doc);

  // Where each block sits now.
  const home = new Map<string, string>();
  for (const [sectionId, a] of Object.entries(doc.assignments)) if (a.kind === 'words') a.blockIds.forEach(b => home.set(b, sectionId));

  const [selected, setSelected] = useState('');
  const [dragging, setDragging] = useState('');
  const [pending, setPending] = useState<{ blockId: string; sectionId: string } | null>(null);
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null);
  const [message, setMessage] = useState('');
  const err = prep.error || message;
  const run = (ok: boolean) => { setMessage(''); return ok; };

  const assign = (blockId: string, sectionId: string) => {
    const target = doc.assignments[sectionId];
    if (target?.kind === 'words' && !target.blockIds.includes(blockId)) { setPending({ blockId, sectionId }); return; }
    setPending(null); setSelected('');
    run(prep.applyLyricOp({ kind: 'assign', blockId, sectionId }));
  };
  const resolve = (mode: 'append' | 'swap') => {
    if (!pending) return;
    run(prep.applyLyricOp({ kind: 'assign', blockId: pending.blockId, sectionId: pending.sectionId, mode }));
    setPending(null); setSelected('');
  };
  const instrumental = (sectionId: string) => {
    const a = doc.assignments[sectionId];
    if (a?.kind === 'instrumental') { run(prep.applyLyricOp({ kind: 'clear', sectionId })); return; }
    if (a?.kind === 'words' && !prep.applyLyricOp({ kind: 'clear', sectionId })) return;
    run(prep.applyLyricOp({ kind: 'instrumental', sectionId }));
  };

  const markEmpty = () => {
    for (const id of doc.sectionIds) if (!doc.assignments[id] && !prep.applyLyricOp({ kind: 'instrumental', sectionId: id })) return;
    setMessage('');
  };

  // ── Paste ──
  const pasteStale = lyrics !== doc.paste;
  // The first time a score with no words yet meets pasted text, make its blocks.
  const seeded = useRef('');
  useEffect(() => {
    if (locked || seeded.current === doc.sourceKey || doc.blocks.length || !lyrics.trim()) return;
    seeded.current = doc.sourceKey;
    prep.applyLyricOp({ kind: 'paste', paste: lyrics });
  }, [locked, doc.sourceKey, doc.blocks.length, lyrics]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Suggestions ──
  const [asking, setAsking] = useState(false);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const ask = async (useRecording: boolean) => {
    const guard = prep.captureGuard();
    setAsking(true); setMessage('');
    try {
      const reply = await suggest(useRecording && prep.recordingAlignmentAvailable);
      if (reply.sourceKey !== guard.sourceKey || reply.structureHash !== guard.structureHash || reply.lyricRevision !== guard.lyricRevision) {
        setMessage(t('cover.prepare.lyrics.staleReply')); return;
      }
      setProposal({ guard, reply, off: reply.uncertain.map(id => `row:${id}`) });
    } catch (e) { setMessage((e as Error).message); }
    finally { setAsking(false); }
  };
  const stale = !!proposal && !prep.matchesGuard(proposal.guard);
  // On arrival, with blocks to place and a section still empty, ask once for this score.
  const arrivalKey = `${doc.sourceKey}|${prep.captureGuard().structureHash}`;
  const anyEmpty = doc.sectionIds.some(id => !doc.assignments[id]);
  useEffect(() => {
    if (locked || ready || asking || proposal || !doc.blocks.length || !anyEmpty || suggestedOnArrival.has(arrivalKey)) return;
    suggestedOnArrival.add(arrivalKey);
    void ask(false);
  }, [locked, ready, doc.blocks.length, anyEmpty, arrivalKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const reuseGroups = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const r of proposal?.reply.reuse ?? []) groups.set(r.blockId, [...(groups.get(r.blockId) ?? []), r.sectionId]);
    return [...groups].map(([blockId, sectionIds]) => ({ blockId, sectionIds }));
  }, [proposal]);
  const flip = (key: string) => setProposal(p => p && { ...p, off: p.off.includes(key) ? p.off.filter(k => k !== key) : [...p.off, key] });
  const accept = () => {
    if (!proposal || stale) return;
    const picked: PrepareSuggestions = {};
    for (const [sectionId, value] of Object.entries(proposal.reply.assignments)) if (!proposal.off.includes(`row:${sectionId}`)) picked[sectionId] = value;
    reuseGroups.forEach(({ blockId, sectionIds }) => {
      if (!proposal.off.includes(`reuse:${blockId}`)) sectionIds.forEach(sectionId => { picked[sectionId] = [blockId]; });
    });
    // In score order, so a repeated block is first placed where it was proposed first.
    const ordered: PrepareSuggestions = {};
    doc.sectionIds.forEach(id => { if (picked[id]) ordered[id] = picked[id]; });
    if (run(prep.applySuggestions(proposal.guard, ordered, proposal.reply.recording))) setProposal(null);
  };

  // ── Confirm ──
  const problem = assignmentProblem(doc);
  const preview = ready ? exportPrepare(doc) : null;
  const sectionOptions = list.map(s => ({ value: s.id, label: `${spanName(s)} · ${spanBars(s)}` }));

  const word = (s: SectionSpan) => {
    const seconds = spanSeconds(clockMap, s.start, s.end);
    return `${t('cover.prepare.structure.bars', { bars: spanBars(s) })}${seconds !== null ? ` · ${clock(seconds)}` : ''}`;
  };

  // The score's sections in order with the words each will sing: what Generate and Build send.
  const orderedPreview = <ol className="space-y-1">{list.map((s, i) => {
    const a = doc.assignments[s.id];
    return <li key={s.id} style={{ borderColor: colours[i] }} className="rounded border-l-4 bg-zinc-500/5 px-2 py-1">
      <span className="font-semibold">{spanName(s)}</span> <span className="text-zinc-500">{word(s)}</span>
      <div className="whitespace-pre-wrap text-zinc-600 dark:text-zinc-300">{!a ? t('cover.prepare.lyrics.needsWords') : a.kind === 'words'
        ? a.blockIds.map(id => firstLine(blockOf(id)?.text ?? '')).join(' / ') : t('cover.prepare.lyrics.instrumental')}</div>
    </li>;
  })}</ol>;

  if (ready) return (
    <div className="space-y-3 text-xs text-zinc-700 dark:text-zinc-200">
      <p className="font-medium text-emerald-600 dark:text-emerald-400">{t('cover.prepare.ready.title')}</p>
      {preview && 'error' in preview ? <p role="alert" className="text-red-500">{preview.error}</p> : orderedPreview}
      <div className="flex gap-2">
        <button type="button" className={btn} onClick={() => prep.showStep('lyrics')}>{t('cover.prepare.ready.editLyrics')}</button>
        <button type="button" className={btn} onClick={() => { prep.beginStructureEdit(); }}>{t('cover.prepare.structure.edit')}</button>
      </div>
    </div>
  );

  return (
    <div className="space-y-3 text-xs text-zinc-700 dark:text-zinc-200">
      {locked && <p role="status" className="text-amber-600 dark:text-amber-400">{t('cover.prepare.lyrics.needStructure')}</p>}
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className={btn} disabled={!prep.canUndo} onClick={() => prep.undo()}>{t('cover.prepare.undo')}</button>
        <button type="button" className={btn} disabled={!prep.canRedo} onClick={() => prep.redo()}>{t('cover.prepare.redo')}</button>
        <button type="button" className={btn} onClick={() => { prep.showStep('structure'); }}>{t('cover.prepare.lyrics.viewStructure')}</button>
        <button type="button" className={btn} onClick={() => { prep.beginStructureEdit(); }}>{t('cover.prepare.structure.edit')}</button>
      </div>
      {err && <p role="alert" className="text-red-500">{err}</p>}

      <div className={`grid gap-3 ${wide ? 'grid-cols-2' : 'grid-cols-1'} items-start`}>
        {/* Left: the words */}
        <div className="min-w-0 space-y-2">
          <div className="space-y-1">
            <textarea aria-label={t('cover.prepare.lyrics.pasteLabel')} value={lyrics} onChange={e => onLyricsText(e.target.value)}
              placeholder={t('cover.lyricsPlaceholder')} rows={12} disabled={locked}
              className="w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 font-mono text-xs text-zinc-900 dark:text-white" />
            <button type="button" className={btn} disabled={locked || !pasteStale}
              onClick={() => run(prep.applyLyricOp({ kind: 'paste', paste: lyrics }))}>{t('cover.prepare.lyrics.updateBlocks')}</button>
            <span className="ml-2 text-[11px] text-zinc-500">{t('cover.prepare.lyrics.updateHint')}</span>
          </div>
          {doc.blocks.length === 0 && <p className="text-zinc-500">{t('cover.prepare.lyrics.noBlocks')}</p>}
          {doc.blocks.length > 0 && <details className="space-y-2">
          <summary className="cursor-pointer font-medium">{t('cover.prepare.lyrics.blocks')} ({doc.blocks.length})</summary>
          {doc.blocks.map(block => {
            const at = home.get(block.id), span = at ? spanOf(at) : undefined;
            const excluded = doc.excluded.includes(block.id);
            const lines = block.text.split('\n');
            return <div key={block.id} draggable={!locked} onDragStart={e => { e.dataTransfer.setData('text/plain', block.id); setDragging(block.id); }}
              onDragEnd={() => setDragging('')} onClick={() => !locked && setSelected(s => s === block.id ? '' : block.id)}
              className={`space-y-1 rounded-lg border p-2 ${selected === block.id ? 'border-cyan-500 ring-1 ring-cyan-500' : 'border-zinc-200 dark:border-white/10'} ${excluded ? 'opacity-60' : ''}`}>
              <div className="flex flex-wrap items-center gap-1 text-[11px]">
                <span className="rounded bg-zinc-500/20 px-1">{block.tag ?? t('cover.prepare.lyrics.untagged')}</span>
                {block.copyOf && <span className="rounded bg-cyan-500/20 px-1">{t('cover.prepare.lyrics.copy')}</span>}
                <span className={span ? 'text-emerald-600 dark:text-emerald-400' : excluded ? 'text-zinc-500' : 'text-amber-600 dark:text-amber-400'}>
                  {span ? `→ ${spanName(span)}` : excluded ? t('cover.prepare.lyrics.excluded') : t('cover.prepare.lyrics.unassigned')}</span>
              </div>
              {editing?.id === block.id
                ? <div className="space-y-1" onClick={e => e.stopPropagation()}>
                  <textarea value={editing.text} onChange={e => setEditing({ id: block.id, text: e.target.value })} rows={Math.min(8, lines.length + 1)}
                    className="w-full rounded border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-1 font-mono text-xs text-zinc-900 dark:text-white" />
                  <button type="button" className={btn} onClick={() => { if (run(prep.applyLyricOp({ kind: 'text', blockId: block.id, text: editing.text }))) setEditing(null); }}>{t('cover.prepare.save')}</button>
                  <button type="button" className={btn} onClick={() => setEditing(null)}>{t('cover.prepare.cancel')}</button>
                </div>
                : <pre className="max-h-28 overflow-y-auto whitespace-pre-wrap font-sans">{block.text}</pre>}
              <div className="flex flex-wrap items-center gap-1" onClick={e => e.stopPropagation()}>
                <StyledSelect accent="cyan" size="sm" value="" placeholder={t('cover.prepare.lyrics.assignTo')} disabled={locked || excluded}
                  onChange={id => id && assign(block.id, id)} options={sectionOptions} className="w-44" />
                {excluded
                  ? <button type="button" className={btn} disabled={locked} onClick={() => run(prep.applyLyricOp({ kind: 'include', blockId: block.id }))}>{t('cover.prepare.lyrics.include')}</button>
                  : <button type="button" className={btn} disabled={locked} onClick={() => run(prep.applyLyricOp({ kind: 'exclude', blockId: block.id }))}>{t('cover.prepare.lyrics.exclude')}</button>}
                <button type="button" className={btn} disabled={locked} onClick={() => setEditing({ id: block.id, text: block.text })}>{t('cover.prepare.lyrics.edit')}</button>
                {lines.length > 1 && <StyledSelect accent="cyan" size="sm" value="" placeholder={t('cover.prepare.lyrics.splitBefore')} disabled={locked}
                  onChange={line => line && run(prep.applyLyricOp({ kind: 'splitBlock', blockId: block.id, line: Number(line) }))}
                  options={lines.slice(1).map((line, k) => ({ value: String(k + 1), label: `${k + 2}: ${line.trim().slice(0, 28)}` }))} className="w-40" />}
                {doc.blocks.indexOf(block) < doc.blocks.length - 1 &&
                  <button type="button" className={btn} disabled={locked} onClick={() => run(prep.applyLyricOp({ kind: 'joinNext', blockId: block.id }))}>{t('cover.prepare.lyrics.joinNext')}</button>}
              </div>
            </div>;
          })}
          </details>}
        </div>

        {/* Right: the score's sections */}
        <div className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className={btn} disabled={locked || asking || !doc.blocks.length}
              onClick={() => void ask(false)} title={t('cover.prepare.lyrics.suggestAgainInfo')}>{t('cover.prepare.lyrics.suggestAgain')}</button>
            <button type="button" className={btn} disabled={locked || asking || !doc.blocks.length || !prep.recordingAlignmentAvailable}
              onClick={() => void ask(true)} title={t('cover.prepare.lyrics.useRecordingInfo')}>{t('cover.prepare.lyrics.useRecording')}</button>
            {asking && <span role="status" className="text-zinc-500">{t('cover.prepare.lyrics.suggesting')}</span>}
            <button type="button" className={btn} disabled={locked || doc.sectionIds.every(id => doc.assignments[id])} onClick={markEmpty}>
              {t('cover.prepare.lyrics.allInstrumental')}</button>
          </div>
          {!prep.recordingAlignmentAvailable && <p className="text-[11px] text-zinc-500">{t('cover.prepare.lyrics.noRecording')}</p>}

          {proposal && <div className="space-y-1 rounded-lg border border-cyan-500/40 p-2">
            <p className="font-medium">{proposal.reply.recording ? t('cover.prepare.lyrics.proposalRecording') : t('cover.prepare.lyrics.proposalInstant')}</p>
            {proposal.reply.recordingError && <p role="status" className="text-amber-600 dark:text-amber-400">{t('cover.prepare.lyrics.recordingFailed', { error: proposal.reply.recordingError })}</p>}
            {stale && <p role="alert" className="text-red-500">{t('cover.prepare.lyrics.proposalStale')}</p>}
            {doc.sectionIds.filter(id => proposal.reply.assignments[id]).map(id => {
              const value = proposal.reply.assignments[id], s = spanOf(id);
              return <div key={id} className="flex items-start gap-2">
                <Toggle size="sm" accent="cyan" checked={!proposal.off.includes(`row:${id}`)} onChange={() => flip(`row:${id}`)} aria-label={s ? spanName(s) : id} />
                <span><span className="font-semibold">{s ? spanName(s) : id}</span>{' ← '}
                  {value === 'instrumental' ? t('cover.prepare.lyrics.instrumental') : value.map(b => firstLine(blockOf(b)?.text ?? '')).join(' / ')}
                  {proposal.reply.uncertain.includes(id) && <span className="ml-1 text-amber-600 dark:text-amber-400">{t('cover.prepare.lyrics.unsure')}</span>}</span>
              </div>;
            })}
            {reuseGroups.map(({ blockId, sectionIds }) => <div key={blockId} className="flex items-start gap-2">
              <Toggle size="sm" accent="cyan" checked={!proposal.off.includes(`reuse:${blockId}`)} onChange={() => flip(`reuse:${blockId}`)} aria-label={t('cover.prepare.lyrics.reuse')} />
              <span><span className="font-semibold">{t('cover.prepare.lyrics.reuseFor', { names: sectionIds.map(id => { const s = spanOf(id); return s ? spanName(s) : id; }).join(' & ') })}</span>
                {sectionIds.map(id => { const s = spanOf(id); return <span key={id} className="block text-zinc-500">
                  <span className="font-medium">{s ? spanName(s) : id}:</span> <span className="whitespace-pre-wrap">{(blockOf(blockId)?.text ?? '').split(/\r?\n/).slice(0, 3).join(' / ')}</span></span>; })}
                <span className="block text-[11px] text-zinc-500">{t('cover.prepare.lyrics.reuseHint')}</span></span>
            </div>)}
            {proposal.reply.unplaced.length > 0 && <p className="text-zinc-500">{t('cover.prepare.lyrics.unplaced', {
              blocks: proposal.reply.unplaced.map(b => firstLine(blockOf(b)?.text ?? '')).join(' / ') })}</p>}
            <div className="flex gap-2">
              <button type="button" className={btn} disabled={stale} onClick={accept}>{t('cover.prepare.lyrics.accept')}</button>
              <button type="button" className={btn} onClick={() => setProposal(null)}>{t('cover.prepare.lyrics.discard')}</button>
            </div>
          </div>}

          {list.map((s, i) => {
            const a = doc.assignments[s.id];
            const flagged = doc.flagged.includes(s.id);
            const asked = pending?.sectionId === s.id;
            return <div key={s.id} style={{ borderColor: colours[i] }}
              onDragOver={e => { if (dragging && !locked) e.preventDefault(); }}
              onDrop={e => { e.preventDefault(); const id = e.dataTransfer.getData('text/plain') || dragging; setDragging(''); if (id && !locked) assign(id, s.id); }}
              className={`space-y-1 rounded-lg border-l-4 border-y border-r border-y-zinc-200 border-r-zinc-200 p-2 dark:border-y-white/10 dark:border-r-white/10 ${activeSection === i ? 'bg-cyan-500/10' : ''} ${dragging ? 'outline-dashed outline-1 outline-cyan-500/50' : ''}`}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold">{spanName(s)}</span>
                <span className="text-zinc-500">{word(s)}</span>
                <button type="button" className={btn} aria-label={t('cover.prepare.lyrics.hear', { name: spanName(s) })}
                  title={t('cover.prepare.lyrics.hear', { name: spanName(s) })} onClick={() => onPlayScore(scorePlaying ? null : { from: s.start, to: s.end, loop: false })}>
                  <Play className="h-3 w-3" /></button>
                {flagged && <span className="rounded bg-amber-500/20 px-1 text-amber-700 dark:text-amber-300">{t('cover.prepare.lyrics.review')}</span>}
                {!a && <span className="rounded bg-red-500/10 px-1 text-red-500">{t('cover.prepare.lyrics.needsWords')}</span>}
              </div>
              {a?.kind === 'instrumental' && <p className="text-zinc-500">{t('cover.prepare.lyrics.instrumental')}</p>}
              {a?.kind === 'words' && a.blockIds.map(id => <div key={id} className="flex items-start gap-1 rounded bg-zinc-500/5 px-1">
                <span className="min-w-0 flex-1 truncate" title={blockOf(id)?.text}>{firstLine(blockOf(id)?.text ?? '')}</span>
                <button type="button" aria-label={t('cover.prepare.lyrics.remove')} title={t('cover.prepare.lyrics.remove')} disabled={locked}
                  onClick={() => run(prep.applyLyricOp({ kind: 'unassign', blockId: id }))} className="text-zinc-500 hover:text-red-500 disabled:opacity-40"><X className="h-3 w-3" /></button>
              </div>)}
              {asked && <div role="alert" className="flex flex-wrap items-center gap-2 rounded border border-amber-500/40 p-1">
                <span>{t('cover.prepare.lyrics.occupied')}</span>
                <button type="button" className={btn} onClick={() => resolve('append')}>{t('cover.prepare.lyrics.append')}</button>
                <button type="button" className={btn} onClick={() => resolve('swap')}>{t('cover.prepare.lyrics.swap')}</button>
                <button type="button" className={btn} onClick={() => setPending(null)}>{t('cover.prepare.cancel')}</button>
              </div>}
              <div className="flex flex-wrap items-center gap-1">
                {selected && !locked && !asked && home.get(selected) !== s.id &&
                  <button type="button" className={btn} onClick={() => assign(selected, s.id)}>{t('cover.prepare.lyrics.putHere')}</button>}
                <button type="button" className={btn} disabled={locked} onClick={() => instrumental(s.id)}>
                  {a?.kind === 'instrumental' ? t('cover.prepare.lyrics.notInstrumental') : t('cover.prepare.lyrics.markInstrumental')}</button>
                {a?.kind === 'words' && <button type="button" className={btn} disabled={locked} onClick={() => run(prep.applyLyricOp({ kind: 'clear', sectionId: s.id }))}>{t('cover.prepare.lyrics.clear')}</button>}
              </div>
            </div>;
          })}
        </div>
      </div>

      {!locked && list.length > 0 && <details className="space-y-1">
        <summary className="cursor-pointer font-medium">{t('cover.prepare.lyrics.order')}</summary>
        {orderedPreview}
      </details>}
      {/* Stuck to the bottom of the scrolling column, so it is always in view. */}
      <div className="sticky -bottom-4 z-10 -mx-4 -mb-4 flex flex-wrap items-center gap-3 border-t border-zinc-200 dark:border-white/10 bg-white dark:bg-zinc-950 px-4 py-2">
        <button type="button" disabled={locked || !!problem} onClick={() => run(prep.confirmLyrics())}
          className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-40">{t('cover.prepare.lyrics.confirm')}</button>
        {problem && <span className="text-[11px] text-amber-700 dark:text-amber-300">{problem}</span>}
        {!problem && <span className="text-[11px] text-zinc-500">{t('cover.prepare.lyrics.confirmHint')}</span>}
      </div>
    </div>
  );
};
