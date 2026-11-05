import React, { useEffect, useRef, useState } from 'react';
import abcjs from 'abcjs';
import 'abcjs/abcjs-audio.css';
import { useTranslation } from 'react-i18next';
import { followNoteInBox } from '../../utils/abcFollow';
import { abcSectionMarks, sectionAtChar, sectionColours, withBarNumbers } from '../../utils/abcSections';
import { withChoirVocal } from '../../utils/abcChoirVocal';
import { downloadAbcMidi } from '../../utils/abcMidi';
import { barAtChar, barRanges } from '../../utils/abcSectionMove';
import { AbcChart } from './AbcChart';
import { MusicXmlImport } from './MusicXmlImport';
import { Toggle } from '../shared/Toggle';
import { startPlayer, stopPlayer, type AbcPlayer, type ScoreRange } from './prepareView';
import type { Yue2CoverJob, Yue2CoverReadiness } from '../../services/yue2CoverApi';

interface Props {
  height: number;
  /** ABC beside the rendered score instead of above it. */
  wide: boolean;
  sourceReady: boolean;
  sourceMode: 'audio' | 'musicxml';
  readiness: Yue2CoverReadiness | null;
  job: Yue2CoverJob | null;
  preparing: boolean;
  error: string;
  abc: string;
  sourceKey: string;
  scoreSource?: string | null;
  chordQuality?: 'estimated' | 'transcribed' | 'provided' | null;
  /** The confirmed score has been approved for rendering. */
  approved: boolean;
  /** One edit of the score: the Apply button of the ABC text, a chart edit or an import. */
  onAbcChange: (value: string) => void;
  onTranscribe: () => void;
  onCancel: () => void;
  /** The score is locked (Prepare's lyrics step, or a confirmed structure):
   *  no ABC, chart or import edits and no dragging a section marker. */
  readOnly: boolean;
  /** A section marker dropped on a bar. */
  onMoveSection: (index: number, bar: number) => void;
  /** The bars (1-based, inclusive) selected in Prepare's strip, drawn highlighted. */
  highlightBars: [number, number] | null;
  /** Play these bars, or (range null) stop the player however it was started.
   *  A new object each time, so the same request is acted on again. */
  play: { range: ScoreRange | null } | null;
  /** Whether a range started by `play` is sounding. */
  onPlaying: (playing: boolean) => void;
  /** The score section now playing or just picked. */
  onSection?: (index: number) => void;
}

type Tune = ReturnType<typeof abcjs.renderAbc>[number];
/** abcjs's drum track: one wood-block hit at the start of every full bar, so a bar line off the song's downbeat is heard. */
const BAR_CLICK = { drum: 'd 76 110', drumBars: 1, drumIntro: 0 };
type TimingEvent = InstanceType<typeof abcjs.TimingCallbacks>['noteTimings'][number];

/** Colours each rendered note and bar by its score section and draws a
 *  labelled line where each section starts; clicking a label plays from it.
 *  Sections come from the ABC's `%Label` lines by character offset, the same
 *  offsets abcjs records on each element, so multi-bar rests (Z4) and the
 *  interleaved Vocal/Ins voices need no bar counting. Returns each section's
 *  first timed event, the seek target, or null for a section with no notes.
 *  Without `onMove` (a locked score) a label only plays. */
function markSections(tune: Tune, abc: string, host: HTMLElement, onPick: (index: number) => void,
  onMove: ((index: number, bar: number) => void) | null): Array<TimingEvent | null> {
  const marks = abcSectionMarks(abc);
  if (!marks.length) return [];
  const colours = sectionColours(marks.map(mark => mark.label));
  type Item = { startChar?: number; abselem?: { elemset?: Array<SVGElement | null> } };
  for (const line of tune.lines ?? [])
    for (const staff of line.staff ?? [])
      for (const voice of (staff.voices ?? []) as unknown as Item[][])
        for (const item of voice) {
          if (typeof item.startChar !== 'number') continue;
          const index = sectionAtChar(marks, item.startChar);
          // An attribute, not a style: the playing note's highlight class must still win.
          if (index >= 0) item.abselem?.elemset?.forEach(node => node?.setAttribute('fill', colours[index]));
        }
  // Timings are sorted by time, so the first event to touch a section is its start.
  const starts: Array<TimingEvent | null> = marks.map(() => null);
  const events: TimingEvent[] = [];
  for (const event of new abcjs.TimingCallbacks(tune, {}).noteTimings) {
    if (event.type !== 'event' || event.left == null) continue;
    events.push(event);
    for (const char of event.startCharArray ?? []) {
      const index = sectionAtChar(marks, char);
      if (index >= 0 && !starts[index]) starts[index] = event;
    }
  }
  const svg = host.querySelector('svg');
  const ns = 'http://www.w3.org/2000/svg';
  starts.forEach((start, index) => {
    if (!svg || !start || start.left == null || start.top == null || start.height == null) return;
    const x = start.left - 3;
    const g = document.createElementNS(ns, 'g');
    g.setAttribute('class', onMove ? 'cursor-grab' : 'cursor-pointer');
    // Click plays from the section; a drag drops its start on another bar.
    g.addEventListener('pointerdown', down => {
      const ctm = svg.getScreenCTM()?.inverse();
      if (!ctm || down.button !== 0) return;
      down.preventDefault();
      const toSvg = (e: PointerEvent) => new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm);
      const origin = toSvg(down);
      let dragging = false;
      const move = (e: PointerEvent) => {
        if (!onMove) return;
        if (!dragging && Math.hypot(e.clientX - down.clientX, e.clientY - down.clientY) < 4) return;
        dragging = true;
        const at = toSvg(e);
        g.setAttribute('transform', `translate(${at.x - origin.x} ${at.y - origin.y})`);
      };
      const up = (e: PointerEvent) => {
        window.removeEventListener('pointermove', move);
        window.removeEventListener('pointerup', up);
        g.removeAttribute('transform');
        if (!dragging || !onMove) { onPick(index); return; }
        const at = toSvg(e);
        const char = dropChar(events, x + at.x - origin.x, start.top! + start.height! / 2 + at.y - origin.y);
        const bar = char === null ? null : barAtChar(abc, char);
        if (bar !== null) onMove(index, bar);
      };
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });
    const rule = document.createElementNS(ns, 'line');
    Object.entries({ x1: x, x2: x, y1: start.top, y2: start.top + start.height, stroke: colours[index], 'stroke-width': 2 })
      .forEach(([key, value]) => rule.setAttribute(key, String(value)));
    const text = document.createElementNS(ns, 'text');
    Object.entries({ x: x + 3, y: Math.max(10, start.top - 2), fill: colours[index], 'font-size': 11, 'font-weight': 700 })
      .forEach(([key, value]) => text.setAttribute(key, String(value)));
    text.textContent = `▶ ${marks[index].label}`;
    const title = document.createElementNS(ns, 'title');
    title.textContent = onMove ? `Click to play from ${marks[index].label}; drag onto a bar to move where it starts`
      : `Click to play from ${marks[index].label}`;
    // A wide invisible stroke so the 2px rule is easy to grab.
    const grip = rule.cloneNode() as SVGLineElement;
    grip.setAttribute('stroke', 'transparent');
    grip.setAttribute('stroke-width', '10');
    g.append(title, grip, rule, text);
    svg.appendChild(g);
  });
  return starts;
}

/** The ABC offset of the first note at or right of `x` on the system nearest
 *  `y` (SVG units), or of the next system's first note past its end. */
function dropChar(events: TimingEvent[], x: number, y: number): number | null {
  const systems = [...new Set(events.map(event => event.line ?? 0))].sort((a, b) => a - b);
  const centre = (line: number) => {
    const first = events.find(event => (event.line ?? 0) === line)!;
    return (first.top ?? 0) + (first.height ?? 0) / 2;
  };
  const nearest = systems.reduce((best, line) => Math.abs(centre(line) - y) < Math.abs(centre(best) - y) ? line : best, systems[0]);
  const pick = (line: number, fromX: number) => events
    .filter(event => (event.line ?? 0) === line && (event.left ?? 0) >= fromX)
    .sort((a, b) => (a.left ?? 0) - (b.left ?? 0))[0];
  const after = systems[systems.indexOf(nearest) + 1];
  const hit = pick(nearest, x - 4) ?? (after === undefined ? undefined : pick(after, -Infinity));
  return hit?.startCharArray?.find(char => typeof char === 'number') ?? null;
}

export const Yue2CoverScore: React.FC<Props> = p => {
  const { t } = useTranslation();
  const scoreRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLDivElement>(null);
  const [previewValid, setPreviewValid] = useState(false);
  // The ABC text is typed freely and applied as one edit, so Prepare's undo has one step per Apply.
  const [typed, setTyped] = useState({ for: p.abc, text: p.abc });
  if (typed.for !== p.abc) setTyped({ for: p.abc, text: p.abc });
  const draft = typed.text;
  // Plays from score section i, or from score bar n; replaced on every render of the ABC.
  const playFromRef = useRef<(index: number) => Promise<void>>(async () => {});
  const playBarRef = useRef<(bar: number) => Promise<void>>(async () => {});
  // Prepare's Hear score: play a bar range, or stop.
  const playRangeRef = useRef<(range: ScoreRange) => Promise<void>>(async () => {});
  const stopRangeRef = useRef<() => void>(() => {});
  // The left pane shows the raw ABC or the same score as a song chart.
  const [view, setView] = useState<'abc' | 'chart'>(() =>
    localStorage.getItem('cover-studio-abcView') === 'chart' ? 'chart' : 'abc');
  useEffect(() => { localStorage.setItem('cover-studio-abcView', view); }, [view]);
  const [click, setClick] = useState(() => localStorage.getItem('cover-studio-barClick') !== 'off');
  useEffect(() => { localStorage.setItem('cover-studio-barClick', click ? 'on' : 'off'); }, [click]);
  const [playingBar, setPlayingBar] = useState<number | null>(null);
  const onSectionRef = useRef(p.onSection);
  const onMoveRef = useRef(p.onMoveSection);
  const onPlayingRef = useRef(p.onPlaying);
  const readOnlyRef = useRef(p.readOnly);
  useEffect(() => {
    onSectionRef.current = p.onSection; onMoveRef.current = p.onMoveSection;
    onPlayingRef.current = p.onPlaying; readOnlyRef.current = p.readOnly;
  });
  useEffect(() => {
    if (!p.play) return;
    if (p.play.range) void playRangeRef.current(p.play.range); else stopRangeRef.current();
  }, [p.play]);
  // The selected bars: each bar's nodes are collected when the score renders.
  const applyHighlightRef = useRef<() => void>(() => {});
  const highlightRef = useRef(p.highlightBars);
  const firstBar = p.highlightBars?.[0], lastBar = p.highlightBars?.[1];
  useEffect(() => {
    highlightRef.current = firstBar === undefined || lastBar === undefined ? null : [firstBar, lastBar];
    applyHighlightRef.current();
  }, [firstBar, lastBar]);
  const busy = p.preparing || p.job?.status === 'queued' || p.job?.status === 'running';
  // Side by side, the ABC and the score share the panel's height, less its
  // heading, the import row and the approve button.
  const boxHeight = p.wide ? Math.max(128, p.height - 158) : undefined;
  const viewToggle = <div className="inline-flex items-center gap-2">
    <div className="inline-flex rounded-md border border-zinc-300 dark:border-white/10 text-[11px]">
      {(['abc', 'chart'] as const).map(mode => <button key={mode} type="button" onClick={() => setView(mode)}
        className={`px-2 py-0.5 ${view === mode ? 'bg-cyan-500/20 text-cyan-700 dark:text-cyan-300' : 'text-zinc-500'}`}>
        {mode === 'abc' ? 'ABC' : 'Chart'}</button>)}
    </div>
    <button type="button" onClick={() => downloadAbcMidi(p.abc, `score-${p.sourceKey || 'cover'}`)}
      title={t('cover.prepare.score.downloadMidiInfo')}
      className="text-[11px] text-cyan-700 dark:text-cyan-300 hover:underline">{t('cover.prepare.score.downloadMidi')}</button>
    <Toggle size="sm" accent="cyan" checked={click} onChange={setClick} label="Bar click"
      info="Plays a click on beat 1 of every bar, so a score whose bar lines are a beat off the song is easy to hear." />
  </div>;
  const editor = view === 'chart' && p.abc.trim()
    ? <AbcChart abc={p.abc} disabled={busy || p.readOnly} height={boxHeight ?? 320} playingBar={playingBar}
      onAbcChange={p.onAbcChange} onPlayBar={bar => void playBarRef.current(bar)} />
    : <div className="space-y-1">
      <textarea aria-label="Editable ABC score" value={draft} disabled={busy} readOnly={p.readOnly} onChange={e => setTyped({ for: p.abc, text: e.target.value })}
        placeholder="Paste an ABC score here to skip transcription" style={{ height: p.wide && boxHeight ? boxHeight - 24 : boxHeight }}
        className={`w-full ${p.wide ? 'resize-none' : 'h-28 resize-y'} rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-xs font-mono text-zinc-900 dark:text-white`} />
      {!p.readOnly && <button type="button" onClick={() => p.onAbcChange(draft)} disabled={busy || draft === p.abc}
        className="text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">{t('cover.prepare.score.applyAbc')}</button>}
    </div>;
  const transcribe = <button onClick={p.onTranscribe} disabled={!p.sourceReady || busy || p.readiness?.ready === false}
    className="rounded-lg bg-cyan-500/20 px-3 py-1.5 text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">
    {p.abc ? 'Retranscribe' : 'Transcribe full score'}
  </button>;
  // The ABC and the score, under the import row so both start level. The
  // score scales to its column (responsive: 'resize'), so the half-width one
  // zooms out. Wrapping it in MusicXmlImport or not (and a new sourceKey)
  // remounts scoreRef's div, which is why the render effect keys on those too.
  const sheet = <div className={`grid gap-3 items-start ${p.wide ? 'grid-cols-2' : 'grid-cols-1'}`}>
    {/* Imported scores have no import row, so the toggle sits on the box. */}
    <div className="min-w-0">{(p.sourceMode !== 'audio' || p.readOnly) && p.abc.trim() && <div className="mb-1 flex justify-end">{viewToggle}</div>}{editor}</div>
    <div className={`${p.abc.trim() ? '' : 'hidden'} min-w-0 bg-white rounded-lg border border-zinc-200 overflow-y-auto ${p.wide ? '' : 'max-h-48'}`}
      style={{ height: boxHeight }}>
      <div ref={scoreRef} className="text-black p-2 [&_svg]:fill-current [&_.prep-selected]:fill-fuchsia-600 [&_.abcjs-highlight]:fill-amber-500" />
    </div>
    <div className={p.wide ? '' : 'order-last'}>
      {p.approved && <p className="text-xs text-emerald-600 dark:text-emerald-400">{t('cover.prepare.score.approved')}</p>}
    </div>
    <div ref={audioRef} className={`min-w-0 text-xs ${previewValid ? '' : 'hidden'}`} />
  </div>;

  useEffect(() => {
    if (!scoreRef.current) return;
    scoreRef.current.innerHTML = '';
    if (audioRef.current) audioRef.current.innerHTML = '';
    setPreviewValid(false);
    applyHighlightRef.current = () => {};
    if (!p.abc.trim()) return;
    let tunes: ReturnType<typeof abcjs.renderAbc>;
    // Every bar numbered, so a section can be moved to a bar read off the sheet. Only the
    // rendered copy carries the directive; offsets below all index this copy.
    const abc = withBarNumbers(withChoirVocal(p.abc));
    try { tunes = abcjs.renderAbc(scoreRef.current, abc, { responsive: 'resize', add_classes: true }); }
    catch { return; }
    if (!tunes[0]) return;
    setPreviewValid(true);
    const box = scoreRef.current.parentElement;
    let control: InstanceType<typeof abcjs.synth.SynthController> | null = null;
    // Bars by character offset in the preview copy, which abcjs's events index.
    const ranges = barRanges(abc);
    const barOf = (char: number | undefined) => char === undefined ? null
      : ranges.find(range => char >= range.start && char <= range.end)?.bar ?? null;
    const charOf = (event: { startCharArray?: number[] }) => event.startCharArray?.find(value => typeof value === 'number');
    // A locked score cannot move a section marker (the click still plays from it).
    const starts = markSections(tunes[0], abc, scoreRef.current, index => void playFromRef.current(index),
      readOnlyRef.current ? null : (index, bar) => onMoveRef.current(index, bar));
    // The nodes of each bar, for the Prepare strip's selection.
    type RenderedItem = { startChar?: number; abselem?: { elemset?: Array<SVGElement | null> } };
    const barNodes: Array<{ bar: number; nodes: SVGElement[] }> = [];
    for (const line of tunes[0].lines ?? [])
      for (const staff of line.staff ?? [])
        for (const voice of (staff.voices ?? []) as unknown as RenderedItem[][])
          for (const item of voice) {
            const bar = barOf(item.startChar);
            const nodes = (item.abselem?.elemset ?? []).filter((n): n is SVGElement => !!n);
            if (bar !== null && nodes.length) barNodes.push({ bar, nodes });
          }
    let lit: Element[] = [];
    let selected: SVGElement[] = [];
    applyHighlightRef.current = () => {
      selected.forEach(node => node.classList.remove('prep-selected'));
      const bars = highlightRef.current;
      selected = bars ? barNodes.filter(entry => entry.bar >= bars[0] && entry.bar <= bars[1]).flatMap(entry => entry.nodes) : [];
      selected.forEach(node => node.classList.add('prep-selected'));
      if (selected[0]) followNoteInBox(selected[0], box);
    };
    applyHighlightRef.current();
    const events = new abcjs.TimingCallbacks(tunes[0], {}).noteTimings.filter(event => event.type === 'event');
    playBarRef.current = async bar => {
      // The first event, in time, in this bar or the nearest one after it.
      const start = events.find(event => barOf(charOf(event)) === bar)
        ?? events.find(event => (barOf(charOf(event)) ?? 0) > bar);
      if (!start) return;
      setPlayingBar(bar);
      const index = sectionAtChar(abcSectionMarks(abc), charOf(start) ?? -1);
      await seekTo(start, index);
    };
    playFromRef.current = async index => {
      const start = starts[index];
      if (start) await seekTo(start, index);
    };
    // Hear score: play from the range's first bar and stop (or loop) when a note past its last bar begins.
    let range: ScoreRange | null = null;
    const player = () => control as unknown as AbcPlayer | null;
    // Stops whatever the player is doing, a range or the embedded player's own play.
    const endRange = () => {
      range = null;
      const current = player();
      if (current) void stopPlayer(current);
      onPlayingRef.current(false);
    };
    playRangeRef.current = async next => {
      const start = events.find(event => barOf(charOf(event)) === next.from)
        ?? events.find(event => (barOf(charOf(event)) ?? 0) > next.from);
      if (!start) return;
      range = next;
      onPlayingRef.current(true);
      await seekTo(start, -1);
    };
    stopRangeRef.current = endRange;
    const seekTo = async (start: TimingEvent, index: number) => {
      if (index >= 0) onSectionRef.current?.(index);
      followNoteInBox(start.elements?.[0]?.[0], box);
      const current = player();
      if (!current) return;
      // seek() only moves a player that is sounding; a stopped or paused one is started first.
      await startPlayer(current);
      current.seek(start.milliseconds / 1000, 'seconds');
    };
    if (!audioRef.current || !abcjs.synth.supportsAudio()) return;
    const synth = control = new abcjs.synth.SynthController();
    // Report a section once, when playback crosses into it.
    const marks = abcSectionMarks(abc);
    let section = -1, playingNow: number | null = null;
    synth.load(audioRef.current, {
      onStart() { lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = []; onPlayingRef.current(true); },
      onEvent(ev: { elements?: Element[][]; startCharArray?: number[] }) {
        const char = ev.startCharArray?.find(value => typeof value === 'number');
        const index = char === undefined ? -1 : sectionAtChar(marks, char);
        if (index >= 0 && index !== section) onSectionRef.current?.(section = index);
        const bar = barOf(char);
        if (bar !== null && bar !== playingNow) setPlayingBar(playingNow = bar);
        if (range && bar !== null && bar > range.to) {
          if (range.loop) void playRangeRef.current(range); else endRange();
          return;
        }
        lit.forEach(el => el.classList.remove('abcjs-highlight'));
        lit = (ev.elements ?? []).flat();
        lit.forEach(el => el.classList.add('abcjs-highlight'));
        followNoteInBox(lit[0], box);
      },
      onFinished() {
        lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = [];
        if (range?.loop) void playRangeRef.current(range); else endRange();
      },
    }, { displayLoop: false, displayRestart: true, displayPlay: true, displayProgress: true, displayWarp: false });
    void synth.setTune(tunes[0], false, click ? BAR_CLICK : {}).catch(() => {});
    return () => { range = null; try { synth.pause(); } catch { /* the player was never started */ } onPlayingRef.current(false); };
  }, [p.abc, p.sourceMode, p.sourceKey, p.readOnly, click]);

  return (
    <div className="min-h-0 flex-shrink-0 border-t border-zinc-200 dark:border-white/10 p-4 space-y-3 overflow-y-auto max-h-[calc(100%-160px)]" style={{ height: p.height }}>
      <div>
        <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">YuE2 lead sheet</h3>
        <p className="text-xs text-zinc-500">{p.sourceMode === 'audio'
          ? 'Transcribe the source melody and chords or paste ABC, then fix its sections and confirm the structure.'
          : 'Review the imported ABC score, then fix its sections and confirm the structure.'}</p>
      </div>
      {p.sourceMode === 'audio' && p.readiness?.message && <p className="text-xs text-amber-600 dark:text-amber-400">{p.readiness.message}</p>}
      {busy && (
        <div className="space-y-1 text-xs text-zinc-500">
          <p>{p.preparing ? 'Preparing source audio…' : `${p.job?.phase || 'Queued'}${p.job?.total ? ` · ${p.job.done}/${p.job.total}` : ''}`}</p>
          <button onClick={p.onCancel} className="text-red-500">Cancel transcription</button>
        </div>
      )}
      {p.error && <p role="alert" className="text-xs text-red-500">{p.error}</p>}
      {p.chordQuality && <p className="text-xs text-zinc-500">{t('cover.chordQuality', 'Chords')}: {p.chordQuality === 'estimated'
        ? t('cover.chordEstimated', 'estimated, review before approving') : p.chordQuality === 'provided'
        ? t('cover.chordProvided', 'provided') : t('cover.chordTranscribed', 'transcribed')}</p>}
      {p.scoreSource === 'dataset' && <p className="text-xs text-cyan-600 dark:text-cyan-300">Score from dataset · editable</p>}
      {p.abc.trim() && !previewValid && <p className="text-xs text-amber-500">The ABC could not be previewed. Check its notation before confirming.</p>}
      {p.sourceMode === 'audio' && !p.readOnly
        ? <MusicXmlImport key={p.sourceKey} abc={p.abc} disabled={busy} onAbcChange={p.onAbcChange} action={transcribe}
          wide={p.wide} toggle={p.abc.trim() ? viewToggle : undefined}>{sheet}</MusicXmlImport>
        : sheet}
    </div>
  );
};
