import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { abcChart, addChord, setChord, setTempo, type ChartBar, type ChartNote } from '@/utils/abc/abcChart';
import { sectionColours } from '@/utils/abc/abcSections';

interface Props {
  abc: string;
  disabled: boolean;
  height?: number;
  /** The bar playing now, highlighted and kept in view. */
  playingBar: number | null;
  onAbcChange: (value: string) => void;
  /** Without it (a preview) the bars are not clickable. */
  onPlayBar?: (bar: number) => void;
}

/** A text field that commits on Enter or blur and drops the edit on Escape. */
function Inline({ value, onCommit, onCancel, className }: {
  value: string; onCommit: (value: string) => void; onCancel: () => void; className: string;
}) {
  const [text, setText] = useState(value);
  // Leaving the field blurs it; only the first of Enter, Escape or blur counts.
  const done = useRef(false);
  const finish = (commit: boolean) => {
    if (done.current) return;
    done.current = true;
    if (commit) onCommit(text); else onCancel();
  };
  return <input autoFocus value={text} onChange={e => setText(e.target.value)} className={className}
    onClick={e => e.stopPropagation()}
    onKeyDown={e => {
      if (e.key === 'Enter') finish(true);
      if (e.key === 'Escape') finish(false);
    }}
    onBlur={() => finish(true)} />;
}

/** The melody's shape across the bar, pitch against time. */
function Contour({ notes, range, colour }: { notes: ChartNote[]; range: [number, number]; colour: string }) {
  const total = notes.reduce((sum, note) => sum + note.length, 0);
  if (!total || !notes.some(note => note.midi !== null)) return <div className="h-5" />;
  const [low, high] = range;
  const y = (midi: number) => 18 - ((midi - low) / Math.max(1, high - low)) * 16;
  let at = 0;
  const segments = notes.map(note => {
    const x1 = (at / total) * 100;
    at += note.length;
    return note.midi === null ? null
      : <line key={x1} x1={x1} x2={(at / total) * 100 - 1} y1={y(note.midi)} y2={y(note.midi)}
        stroke={colour} strokeWidth={2} strokeLinecap="round" />;
  });
  return <svg viewBox="0 0 100 20" preserveAspectRatio="none" className="h-5 w-full">{segments}</svg>;
}

const noteText = (notes: ChartNote[]) => notes
  .filter(note => note.midi !== null && !note.held).map(note => note.name).join(' ');

/** The score as a song chart: the header, then one row of bar cards per
 *  section. Chords and tempo are edited here; the ABC stays the source and
 *  each edit changes only the characters it touches. A click on a bar plays
 *  the preview from it. */
export const AbcChart: React.FC<Props> = ({ abc, disabled, height, playingBar, onAbcChange, onPlayBar }) => {
  const { t } = useTranslation();
  const chart = useMemo(() => abcChart(abc), [abc]);
  const [editing, setEditing] = useState<string | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  // Keep the playing bar in view inside this box, without scrolling the page.
  useEffect(() => {
    const box = boxRef.current, card = box?.querySelector<HTMLElement>(`[data-bar="${playingBar}"]`);
    if (!box || !card) return;
    const top = card.offsetTop - box.offsetTop, bottom = top + card.offsetHeight;
    if (top < box.scrollTop || bottom > box.scrollTop + box.clientHeight) box.scrollTop = Math.max(0, top - 24);
  }, [playingBar]);

  if (typeof chart === 'string') return <div style={{ height }} className="rounded-lg border border-amber-500/40 p-3 text-xs text-amber-600 dark:text-amber-400">
    The chart cannot show this score: {chart} Switch to ABC to edit it.</div>;

  const labelled = chart.sections.filter(section => section.index >= 0);
  const colours = sectionColours(labelled.map(section => section.label));
  const pitches = chart.sections.flatMap(s => s.bars.flatMap(b => [...b.vocal, ...(b.ins ?? [])]))
    .map(note => note.midi).filter((midi): midi is number => midi !== null);
  const range: [number, number] = pitches.length ? [Math.min(...pitches), Math.max(...pitches)] : [60, 72];
  const commit = (next: string) => { setEditing(null); if (next !== abc) onAbcChange(next); };
  const field = 'rounded border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/30 px-1 text-xs text-zinc-900 dark:text-white';

  const card = (bar: ChartBar, colour: string) => {
    const rest = !bar.vocal.some(note => note.midi !== null);
    return <div key={bar.number} data-bar={bar.number} onClick={onPlayBar && (() => onPlayBar(bar.number))}
      title={onPlayBar && `Play from bar ${bar.number}`}
      className={`w-32 ${onPlayBar ? 'cursor-pointer' : ''} rounded-md border bg-white dark:bg-black/20 p-1.5 space-y-0.5 ${playingBar === bar.number
        ? 'border-amber-500 ring-2 ring-amber-500/40' : 'border-zinc-200 dark:border-white/10 hover:border-zinc-400'}`}>
      <div className="flex items-center gap-1 min-h-[20px]">
        <span className="text-[10px] text-zinc-400 tabular-nums">{bar.number}</span>
        {bar.chords.map((chord, i) => editing === `${bar.number}:${i}`
          ? <Inline key={i} value={chord.text} className={`${field} w-14`} onCancel={() => setEditing(null)}
            onCommit={text => commit(setChord(abc, chord, text))} />
          : <ParamLabel key={i} label={chord.text} info={t('coverStudio.chart.editChord', 'Edit chord; clear it to remove')}>
            <button type="button" disabled={disabled}
              onClick={e => { e.stopPropagation(); setEditing(`${bar.number}:${i}`); }}
              className="rounded px-1 text-xs font-semibold text-zinc-800 dark:text-zinc-100 hover:bg-zinc-200 dark:hover:bg-white/10">{chord.text}</button>
          </ParamLabel>)}
        {bar.chordAt !== null && !bar.chords.length && (editing === `${bar.number}:new`
          ? <Inline value="" className={`${field} w-14`} onCancel={() => setEditing(null)}
            onCommit={text => commit(addChord(abc, bar.chordAt!, text))} />
          : <ParamLabel label={t('coverStudio.chart.addChordLabel', 'Chord')} info={t('coverStudio.chart.addChord', 'Add a chord at the start of this bar')}>
            <button type="button" disabled={disabled}
              onClick={e => { e.stopPropagation(); setEditing(`${bar.number}:new`); }}
              className="text-[10px] text-zinc-400 hover:text-cyan-600">+ chord</button>
          </ParamLabel>)}
      </div>
      <Contour notes={bar.vocal} range={range} colour={colour} />
      <p className="truncate font-mono text-[10px] text-zinc-600 dark:text-zinc-300" title={noteText(bar.vocal)}>
        {rest ? <span className="text-zinc-400">{bar.vocal.some(note => note.held) ? 'held' : 'rest'}</span> : noteText(bar.vocal)}
      </p>
      {bar.ins && <div className="opacity-60" title={`Ins: ${noteText(bar.ins) || 'rest'}`}>
        <Contour notes={bar.ins} range={range} colour="#71717a" /></div>}
    </div>;
  };

  return (
    <div ref={boxRef} style={{ height }} className="overflow-y-auto rounded-lg border border-zinc-300 dark:border-white/10 bg-zinc-50 dark:bg-black/20 p-2 space-y-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-600 dark:text-zinc-300">
        <label className="flex items-center gap-1">Tempo
          {editing === 'tempo'
            ? <Inline value={String(chart.tempo ?? '')} className={`${field} w-14`} onCancel={() => setEditing(null)}
              onCommit={text => {
                const bpm = Number(text);
                if (Number.isFinite(bpm) && bpm >= 20 && bpm <= 300) commit(setTempo(abc, bpm)); else setEditing(null);
              }} />
            : <ParamLabel label={t('coverStudio.chart.tempoLabel', 'Tempo')} info={t('coverStudio.chart.tempoInfo', 'Edit the tempo (20–300 BPM)')}>
              <button type="button" disabled={disabled} onClick={() => setEditing('tempo')}
                className="rounded px-1 font-semibold text-zinc-900 dark:text-white hover:bg-zinc-200 dark:hover:bg-white/10">{chart.tempo ?? 'not set'}</button>
            </ParamLabel>} BPM
        </label>
        <ParamLabel label={t('coverStudio.chart.meterLabel', 'Meter')} info={chart.meter === '2/4'
          ? t('coverStudio.chart.meter2_4Info', 'Convert 2/4 to 4/4 (under Score sections) re-bars the score')
          : t('coverStudio.chart.meterInfo', 'Changing the meter means re-barring the score; only 2/4 to 4/4 is supported')}>
          <span>Meter <b className="text-zinc-900 dark:text-white">{chart.meter || '4/4'}</b></span>
        </ParamLabel>
        <ParamLabel label={t('coverStudio.chart.keyLabel', 'Key')} info={t('coverStudio.chart.keyInfo', 'Changing the key means transposing every note, which comes with the bar tools')}>
          <span>Key <b className="text-zinc-900 dark:text-white">{chart.key || 'C'}</b></span>
        </ParamLabel>
      </div>
      {chart.sections.map(section => {
        const colour = section.index >= 0 ? colours[section.index] : '#a1a1aa';
        const first = section.bars[0]?.number, last = section.bars.at(-1)?.number;
        return <section key={`${section.index}-${first}`} className="space-y-1">
          <h4 className="border-l-4 pl-2 text-xs font-semibold text-zinc-800 dark:text-zinc-100" style={{ borderColor: colour }}>
            {section.label}{first !== undefined && <span className="font-normal text-zinc-500"> · bars {first}{last !== first ? `–${last}` : ''}</span>}
          </h4>
          <div className="flex flex-wrap gap-1">{section.bars.map(bar => card(bar, colour))}</div>
        </section>;
      })}
    </div>
  );
};
