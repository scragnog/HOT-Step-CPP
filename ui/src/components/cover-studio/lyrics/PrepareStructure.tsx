// Prepare, step 1: fix the score's structure. A scrolling strip of the score's
// sections (select one or several adjacent ones; the clicked one offers
// playback in a popup), then two collapsed panels: the score-only edits on the
// selection and bar regrouping with a preview. Playback is approximate from
// the recording and exact from the score. Lyrics are not needed here and
// are never touched; every edit goes through the Prepare hook, which owns undo
// and the confirmation (useYue2Prepare.ts, utils/lyrics/yue2Prepare.ts).

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Redo2, Undo2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { previewRegroup, type RegroupResult } from '@/utils/abc/abcRebar';
import { cutSection, mergeScoreSections, resectionScore } from '@/utils/abc/abcSectionMove';
import type { Yue2ScorePlan } from '@/services/yue2CoverApi';
import { sectionColours } from '@/utils/abc/abcSections';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { AbcChart } from './AbcChart';
import { SectionPlayPopup, useHoverPopup } from './SectionPlayback';
import { barTimes, clock, createPlaybackOwner, sectionSpans, shortBars, spanBars, spanName, spanSeconds, type Prepare, type ScoreRange } from '@/components/cover-studio/build/prepareView';
import { useInterval } from '@/hooks/useInterval';


const clockText = (seconds: number | null) => seconds === null ? '?' : clock(seconds);
const METERS = ['4/4', '3/4', '2/4', '6/8', '5/4', '3/8', '12/8'];
const btn = 'rounded-md border border-zinc-300 dark:border-white/10 px-2 py-1 text-xs text-zinc-700 dark:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-white/5 disabled:opacity-40';
const field = 'rounded bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 px-1 py-0.5 text-xs text-zinc-900 dark:text-white';

interface Props {
  prep: Prepare;
  /** The recording, for Hear recording; empty for a MusicXML source. */
  sourceAudioUrl: string;
  scorePlaying: boolean;
  /** Counts every start of the score player, including the embedded player's own play button. */
  scoreStarts: number;
  /** Plays the range; null stops the score player however it was started. */
  onPlayScore: (range: ScoreRange | null) => void;
  /** The selected bars, for the score view to light up. */
  onHighlight: (bars: [number, number] | null) => void;
  /** After the structure is confirmed, to approve the score for rendering. */
  onConfirmed: () => void;
  /** Where the recording sings each lyric block, as score sections (the
   *  server's Match score to lyrics); absent without lyrics or a recording. */
  planFromLyrics?: () => Promise<Yue2ScorePlan>;
  /** Extra tools the studio keeps under the strip (the 2/4 to 4/4 converter). */
  children?: React.ReactNode;
}

export const PrepareStructure: React.FC<Props> = ({ prep, sourceAudioUrl, scorePlaying, scoreStarts, onPlayScore, onHighlight, onConfirmed, planFromLyrics, children }) => {
  const { t } = useTranslation();
  const { doc } = prep;
  const editing = prep.editingStructure && prep.step === 'structure';
  const spans = useMemo(() => sectionSpans(doc.abc, doc.sectionIds), [doc.abc, doc.sectionIds]);
  const list = typeof spans === 'string' ? [] : spans;
  const colours = sectionColours(list.map(s => s.label));
  const clockMap = useMemo(() => barTimes(doc.abc), [doc.abc]);
  const short = useMemo(() => shortBars(doc.abc), [doc.abc]);
  const totalBars = list.length ? list[list.length - 1].end : 0;

  // The selection is kept as section ids, so an edit that removes one drops it.
  const [selected, setSelected] = useState<string[]>([]);
  const chosen = list.map((s, i) => selected.includes(s.id) ? i : -1).filter(i => i >= 0);
  const range: [number, number] | null = chosen.length ? [chosen[0], chosen[chosen.length - 1]] : null;
  const picked = range ? list.slice(range[0], range[1] + 1) : [];
  const single = picked.length === 1 ? picked[0] : null;
  const bars: [number, number] | null = range ? [list[range[0]].start, list[range[1]].end] : null;
  useEffect(() => { onHighlight(bars); }, [bars?.[0], bars?.[1]]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => onHighlight(null), []); // eslint-disable-line react-hooks/exhaustive-deps

  // The section last clicked: hovering it (and only it) brings back its playback popup.
  const [anchor, setAnchor] = useState('');
  const popup = useHoverPopup<string>();
  const select = (i: number, extend: boolean) => {
    if (!extend || !range) { setSelected([list[i].id]); return; }
    const [a, b] = [Math.min(range[0], i), Math.max(range[1], i)];
    setSelected(list.slice(a, b + 1).map(s => s.id));
  };
  const onKey = (e: React.KeyboardEvent, i: number) => {
    const to = e.key === 'ArrowRight' ? i + 1 : e.key === 'ArrowLeft' ? i - 1 : -1;
    if (to < 0 || to >= list.length) return;
    e.preventDefault();
    if (e.shiftKey && range) {
      // Grow or shrink the selection from the end the arrow points at.
      const [a, b] = range;
      const [na, nb] = e.key === 'ArrowRight' ? (i === a && a < b ? [a + 1, b] : [a, to]) : (i === b && a < b ? [a, b - 1] : [to, b]);
      setSelected(list.slice(Math.min(na, nb), Math.max(na, nb) + 1).map(s => s.id));
    } else setSelected([list[to].id]);
    (e.currentTarget.parentElement?.children[to] as HTMLElement | undefined)?.focus();
  };

  const [message, setMessage] = useState('');
  const fail = (text: string) => { setMessage(text); return false; };
  const ok = (done: boolean) => { if (done) setMessage(''); return done; };
  const err = prep.error || message;

  // ── Edits on the selection ──
  // The fields start afresh whenever the selection, or the bars it covers, change (state adjusted during render).
  const formKey = `${selected.join(',')}|${bars?.join('-')}`;
  const [form, setForm] = useState({ key: '', label: '', newLabel: '', bar: '', from: '', to: '', deleting: false });
  if (form.key !== formKey) setForm({ key: formKey, label: picked[0]?.label ?? '', newLabel: single?.label ?? '', bar: '',
    from: bars ? String(bars[0]) : form.from, to: bars ? String(bars[1]) : form.to, deleting: false });
  const { label, newLabel, bar, from, to, deleting } = form;
  const change = (patch: Partial<typeof form>) => setForm(f => ({ ...f, ...patch }));
  const pick = Number(bar);
  const merge = () => {
    if (!range || range[0] === range[1]) return;
    // A dry run first, so a refusal halfway cannot leave a partial merge.
    let abc = doc.abc;
    for (let k = range[1] - 1; k >= range[0]; k--) {
      const out = mergeScoreSections(abc, k, 'first');
      if ('error' in out) { fail(out.error); return; }
      abc = out.abc;
    }
    for (let k = range[1] - 1; k >= range[0]; k--)
      if (!prep.applyStructuralOp({ kind: 'merge', index: k, keep: 'first' })) return;
    const name = label.trim();
    if (name && name !== list[range[0]].label) prep.applyStructuralOp({ kind: 'relabel', index: range[0], label: name });
    setMessage('');
  };
  const needBar = (low: number, high: number) => Number.isInteger(pick) && pick >= low && pick <= high ? null
    : t('cover.prepare.structure.barRange', { low, high });
  const split = () => {
    if (!single) return;
    const problem = needBar(single.start + 1, single.end);
    if (problem) { fail(problem); return; }
    ok(prep.applyStructuralOp({ kind: 'split', index: list.indexOf(single), bar: pick, label: newLabel }));
  };
  const move = () => {
    if (!single) return;
    const i = list.indexOf(single);
    if (i === 0) { fail(t('cover.prepare.structure.firstFixed')); return; }
    const problem = needBar(list[i - 1].start + 1, single.end);
    if (problem) { fail(problem); return; }
    ok(prep.applyStructuralOp({ kind: 'move', index: i, bar: pick }));
  };
  const rename = () => { if (single) ok(prep.applyStructuralOp({ kind: 'relabel', index: list.indexOf(single), label })); };

  // ── Sections from lyrics ──
  // Each sung lyric block opens a section where the recording sings it, named
  // with its own tag; sections with no words stay. One replace, so one undo.
  const [planning, setPlanning] = useState(false);
  const [planNote, setPlanNote] = useState('');
  const abcNow = useRef(doc.abc);
  useEffect(() => { abcNow.current = doc.abc; });
  const fromLyrics = async () => {
    if (!planFromLyrics) return;
    const abc = doc.abc;
    setPlanning(true); setMessage(''); setPlanNote('');
    try {
      const plan = await planFromLyrics();
      if (abcNow.current !== abc) { fail(t('cover.prepare.structure.fromLyricsStale')); return; }
      const next = resectionScore(abc, plan.sections.map(s => ({ label: s.tag ?? s.label, bar: s.bar })));
      if ('error' in next) { fail(next.error); return; }
      if (next.abc === abc) { setPlanNote(t('cover.prepare.structure.fromLyricsSame')); return; }
      if (ok(prep.applyStructuralOp({ kind: 'replace', abc: next.abc })))
        setPlanNote(t('cover.prepare.structure.fromLyricsDone', { count: plan.sections.length, unsure: plan.unsure.length }));
    } catch (e) { fail(String((e as Error)?.message ?? e)); }
    finally { setPlanning(false); }
  };

  // ── Delete (advanced) ──
  const remove = () => {
    if (!single) return;
    const out = cutSection(doc.abc, list.indexOf(single));
    if ('error' in out) { fail(out.error); return; }
    ok(prep.applyStructuralOp({ kind: 'cut', abc: out.abc }));
    change({ deleting: false });
  };

  // ── Bar grouping ──
  const [meter, setMeter] = useState('4/4');
  const [regroup, setRegroup] = useState<{ source: string; from: number; to: number; meter: string; result: RegroupResult } | null>(null);
  const preview = () => {
    const [a, b] = [Number(from), Number(to)];
    setRegroup({ source: doc.abc, from: a, to: b, meter, result: previewRegroup(doc.abc, [a, b], meter) });
  };
  const live = regroup?.source === doc.abc ? regroup : null;
  const applyRegroup = () => {
    if (!live || 'error' in live.result) return;
    if (ok(prep.applyStructuralOp({ kind: 'replace', abc: live.result.abc }))) setRegroup(null);
  };
  const fixShort = (n: number) => { change({ from: String(n), to: String(n), bar: String(n) }); setPanel('regroup'); };

  // ── Playback: one source at a time, whoever starts it ──
  const [loop, setLoop] = useState(false);
  const [hearing, setHearing] = useState(false);
  const audio = useRef<HTMLAudioElement>(null);
  const stops = useRef({ recording: () => {}, score: () => {} });
  useEffect(() => { stops.current = { recording: () => audio.current?.pause(), score: () => onPlayScore(null) }; });
  const owner = useMemo(() => createPlaybackOwner({ recording: () => stops.current.recording(), score: () => stops.current.score() }), []); // eslint-disable-line react-hooks/refs -- the callbacks only run from events
  // The embedded score player starting (its own play button included) silences the recording;
  // the score going quiet gives up its claim.
  useEffect(() => { if (scoreStarts) owner.start('score'); }, [scoreStarts, owner]);
  useEffect(() => { if (!scorePlaying) owner.ended('score'); }, [scorePlaying, owner]);
  useEffect(() => () => { audio.current?.pause(); }, []);
  // One bar before the selection, to the end of it; null when the clock does not reach the selected bars.
  const lead = bars ? spanSeconds(clockMap, bars[0], bars[0]) : null;
  const window_ = bars && clockMap && lead !== null && spanSeconds(clockMap, bars[0], bars[1]) !== null
    ? { from: Math.max(0, clockMap.start[bars[0] - 1] - lead), to: clockMap.end[bars[1] - 1] } : null;
  useInterval(() => {
    const a = audio.current;
    if (!a || !window_) return;
    if (a.ended || a.currentTime >= window_.to) { if (loop) a.currentTime = window_.from; else a.pause(); }
  }, 60, !!(hearing && window_));
  const hearRecording = () => {
    if (!audio.current || !window_) return;
    audio.current.currentTime = window_.from;
    void audio.current.play().catch(e => fail(String(e?.message ?? e)));
  };
  const hearScore = () => { if (bars) { owner.start('score'); onPlayScore({ from: bars[0], to: bars[1], loop }); } };
  const stop = () => { audio.current?.pause(); onPlayScore(null); };

  const confirmDisabled = !editing || typeof spans === 'string';
  const doConfirm = () => { if (prep.confirmStructure()) { setMessage(''); onConfirmed(); } };
  const section = 'space-y-2 rounded-lg border border-zinc-200 dark:border-white/10 p-2';
  const [panel, setPanel] = useState<'edits' | 'regroup' | null>(null);

  return (
    <div className="space-y-3 text-xs text-zinc-700 dark:text-zinc-200">
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className={btn} disabled={!prep.canUndo} onClick={() => prep.undo()}
          title={t('cover.prepare.undo')} aria-label={t('cover.prepare.undo')}><Undo2 className="h-3 w-3" /></button>
        <button type="button" className={btn} disabled={!prep.canRedo} onClick={() => prep.redo()}
          title={t('cover.prepare.redo')} aria-label={t('cover.prepare.redo')}><Redo2 className="h-3 w-3" /></button>
        {editing && planFromLyrics && <button type="button" className={btn} disabled={planning} onClick={() => void fromLyrics()}
          title={t('cover.prepare.structure.fromLyricsInfo')}>{planning ? t('cover.prepare.structure.fromLyricsBusy') : t('cover.prepare.structure.fromLyrics')}</button>}
        {!editing && <>
          <span className="text-emerald-600 dark:text-emerald-400">{t('cover.prepare.structure.locked')}</span>
          <button type="button" className={btn} onClick={() => { prep.beginStructureEdit(); }}>{t('cover.prepare.structure.edit')}</button>
          <button type="button" className={btn} disabled={!prep.doc.confirmed.structure} onClick={() => prep.showStep('lyrics')}>{t('cover.prepare.structure.toLyrics')}</button>
        </>}
      </div>

      {!doc.abc.trim() ? <p className="text-zinc-500">{t('cover.prepare.structure.noScore')}</p>
        : typeof spans === 'string' && <p role="alert" className="text-amber-600 dark:text-amber-400">{spans}</p>}
      {!list.length && doc.abc.trim() && typeof spans !== 'string' && <p className="text-zinc-500">{t('cover.prepare.structure.empty')}</p>}

      {/* The strip: a section per button, wider for more bars, scrolling sideways when the song is long. */}
      {list.length > 0 && <div className="flex gap-1 overflow-x-auto pb-1" role="group" aria-label={t('cover.prepare.structure.strip')}
        onScroll={popup.hide}>
        {list.map((s, i) => {
          const on = selected.includes(s.id);
          const seconds = spanSeconds(clockMap, s.start, s.end);
          const length = s.end - s.start + 1;
          const name = picked.length === 1 ? spanName(picked[0]) : t('cover.prepare.structure.selection', { count: picked.length });
          return <div key={s.id} className="flex" style={{ flex: `${length} 0 ${Math.max(96, length * 10)}px` }}
            onMouseEnter={e => { if (s.id === anchor && on) popup.show(s.id, e.currentTarget); }}
            onMouseLeave={() => { if (popup.open?.key === s.id) popup.close(); }}>
            <button type="button" aria-pressed={on} aria-haspopup="dialog" onKeyDown={e => onKey(e, i)}
              onClick={e => { select(i, e.shiftKey); setAnchor(s.id); popup.show(s.id, e.currentTarget.parentElement!); }}
              style={{ borderColor: colours[i], backgroundColor: on ? `${colours[i]}40` : `${colours[i]}14` }}
              className={`w-full rounded border-l-4 px-2 py-1 text-left ${on ? 'ring-2 ring-inset ring-cyan-500' : ''}`}>
              <span className="block truncate font-semibold">{s.label}{s.repeats > 1 && <span className="ml-1 rounded bg-zinc-500/20 px-1 text-[10px] font-normal">{s.occurrence}/{s.repeats}</span>}</span>
              <span className="block truncate text-[10px] text-zinc-500">{t('cover.prepare.structure.bars', { bars: spanBars(s) })}{seconds !== null && ` · ${clock(seconds)}`}</span>
            </button>
            {popup.open?.key === s.id && range && <SectionPlayPopup at={popup.open} name={name} onMouseEnter={popup.keep} onMouseLeave={popup.close}
              onHearScore={hearScore} onHearRecording={hearRecording}
              recordingDisabled={!prep.recordingSeekAvailable || !window_ || !sourceAudioUrl}
              recordingTitle={prep.recordingSeekAvailable ? t('cover.prepare.structure.hearRecordingInfo') : t('cover.prepare.structure.noRecording')}
              playing={hearing || scorePlaying} onStop={stop} loop={loop} onLoop={setLoop} />}
          </div>;
        })}
      </div>}
      {list.length > 0 && <p className="text-[11px] text-zinc-500">{t('cover.prepare.structure.selectHint')}</p>}

      {short.length > 0 && <p role="status" className="text-amber-700 dark:text-amber-300">
        {t('cover.prepare.structure.short', { count: short.length })}{' '}
        {short.slice(0, 12).map(n => <button key={n} type="button" disabled={!editing} onClick={() => fixShort(n)}
          className="mr-1 rounded border border-amber-500/40 px-1 disabled:opacity-60">{n}</button>)}
        {short.length > 12 && '…'}
      </p>}

      {err && <p role="alert" className="text-red-500">{err}</p>}
      {planNote && !err && <p role="status" className="text-zinc-500">{planNote}</p>}

      {sourceAudioUrl && <audio ref={audio} src={sourceAudioUrl} preload="auto" className="hidden"
        onPlay={() => { owner.start('recording'); setHearing(true); }}
        onPause={() => { owner.ended('recording'); setHearing(false); }}
        onEnded={() => { owner.ended('recording'); setHearing(false); }} />}

      {/* Two collapsed panels side by side; the open one spans the width beneath them. */}
      {list.length > 0 && <div className="grid grid-cols-2 gap-2">
        {(['edits', 'regroup'] as const).map(id => <button key={id} type="button" aria-expanded={panel === id}
          onClick={() => setPanel(open => open === id ? null : id)}
          className="flex items-center justify-between rounded-lg border border-zinc-200 dark:border-white/10 px-2 py-1.5 text-left font-medium hover:bg-zinc-100 dark:hover:bg-white/5">
          {t(`cover.prepare.structure.${id}`)}<ChevronDown className={`h-3 w-3 transition-transform ${panel === id ? 'rotate-180' : ''}`} /></button>)}
      </div>}

      {panel === 'edits' && !range && <p className="text-zinc-500">{t('cover.prepare.structure.selectFirst')}</p>}
      {panel === 'edits' && range && <fieldset disabled={!editing} className={`${section} ${editing ? '' : 'opacity-60'}`}>
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-0.5"><ParamLabel label={single ? t('cover.prepare.structure.name') : t('cover.prepare.structure.mergedName')}
            info={single ? t('cover.prepare.structure.nameInfo') : t('cover.prepare.structure.mergedNameInfo')} />
            <input value={label} onChange={e => change({ label: e.target.value })} className={`${field} w-32`} /></label>
          {single && <button type="button" className={btn} onClick={rename}>{t('cover.prepare.structure.rename')}</button>}
          {!single && <button type="button" className={btn} onClick={merge}>{t('cover.prepare.structure.merge', { count: picked.length })}</button>}
        </div>
        {single && <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-0.5"><ParamLabel label={t('cover.prepare.structure.atBar')} info={t('cover.prepare.structure.atBarInfo')} />
            <input value={bar} inputMode="numeric" onChange={e => change({ bar: e.target.value })} className={`${field} w-16`} /></label>
          <label className="flex flex-col gap-0.5"><ParamLabel label={t('cover.prepare.structure.newName')} info={t('cover.prepare.structure.newNameInfo')} />
            <input value={newLabel} onChange={e => change({ newLabel: e.target.value })} className={`${field} w-32`} /></label>
          <button type="button" className={btn} onClick={split}>{t('cover.prepare.structure.split')}</button>
          <button type="button" className={btn} onClick={move}>{t('cover.prepare.structure.move')}</button>
        </div>}
        {/* The barlines of the selection, to pick where a split or move lands. */}
        {bars && bars[1] - bars[0] < 80 && <div className="flex flex-wrap gap-0.5" aria-label={t('cover.prepare.structure.barPicker')}>
          {Array.from({ length: bars[1] - bars[0] + 1 }, (_, k) => bars[0] + k).map(n =>
            <button key={n} type="button" onClick={() => change({ bar: String(n) })} aria-pressed={pick === n}
              className={`h-5 min-w-[1.5rem] rounded border px-1 text-[10px] ${pick === n ? 'border-cyan-500 bg-cyan-500/20' : short.includes(n) ? 'border-amber-500/60' : 'border-zinc-300 dark:border-white/10'}`}>{n}</button>)}
        </div>}
        {single && <div>
          {!deleting
            ? <button type="button" className="text-zinc-500 hover:text-red-500" onClick={() => change({ deleting: true })}>{t('cover.prepare.structure.delete')}</button>
            : <p className="space-x-2 text-red-500">
              <span>{t('cover.prepare.structure.deleteWarn', { bars: single.end - single.start + 1,
                time: clockText(spanSeconds(clockMap, single.start, single.end)) })}</span>
              <button type="button" className={btn} onClick={remove}>{t('cover.prepare.structure.deleteConfirm')}</button>
              <button type="button" className={btn} onClick={() => change({ deleting: false })}>{t('cover.prepare.cancel')}</button></p>}
        </div>}
      </fieldset>}

      {panel === 'regroup' && list.length > 0 && <fieldset disabled={!editing} className={`${section} ${editing ? '' : 'opacity-60'}`}>
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-0.5"><ParamLabel label={t('cover.prepare.structure.fromBar')} info={t('cover.prepare.structure.fromBarInfo')} />
            <input value={from} inputMode="numeric" onChange={e => change({ from: e.target.value })} className={`${field} w-14`} /></label>
          <label className="flex flex-col gap-0.5"><ParamLabel label={t('cover.prepare.structure.toBar')} info={t('cover.prepare.structure.toBarInfo')} />
            <input value={to} inputMode="numeric" onChange={e => change({ to: e.target.value })} className={`${field} w-14`} /></label>
          <div className="flex flex-col gap-0.5"><ParamLabel label={t('cover.prepare.structure.meter')} info={t('cover.prepare.structure.meterInfo')} />
            <StyledSelect accent="cyan" size="sm" value={meter} onChange={setMeter} options={METERS.map(m => ({ value: m, label: m }))} className="w-24" /></div>
          <button type="button" className={btn} onClick={preview} disabled={!from || !to || totalBars === 0}>{t('cover.prepare.structure.preview')}</button>
        </div>
        {live && ('error' in live.result
          ? <p role="alert" className="text-red-500">{live.result.error}</p>
          : <div className="space-y-2">
            <p className="text-zinc-500">{t('cover.prepare.structure.previewResult', { from: live.from, to: live.to, meter: live.meter,
              first: live.result.bars[0], last: live.result.bars[1] })}</p>
            <AbcChart abc={live.result.abc} disabled height={220} playingBar={null} onAbcChange={() => {}} />
            <div className="flex gap-2">
              <button type="button" className={btn} onClick={applyRegroup}>{t('cover.prepare.structure.apply')}</button>
              <button type="button" className={btn} onClick={() => setRegroup(null)}>{t('cover.prepare.cancel')}</button>
            </div></div>)}
      </fieldset>}

      {editing && children}

      {/* Stuck to the bottom of the scrolling column, so it is always in view. */}
      <div className="sticky -bottom-4 z-10 -mx-4 -mb-4 flex items-center gap-3 border-t border-zinc-200 dark:border-white/10 bg-white dark:bg-zinc-950 px-4 py-2">
        <button type="button" disabled={confirmDisabled} onClick={doConfirm}
          className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-40">{t('cover.prepare.structure.confirm')}</button>
        <span className="text-[11px] text-zinc-500">{t('cover.prepare.structure.confirmHint')}</span>
      </div>
    </div>
  );
};
