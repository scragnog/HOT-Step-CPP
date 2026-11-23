// A section's playback popup (Prepare's Fix structure strip and the Build
// page's source bar), and a player for pages without the lead-sheet panel:
// the score's bars through abcjs's synth, or the recording at the same bars
// by the score's own clock (approximate, one bar early).

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Square } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import abcjs from 'abcjs';
import { withChoirVocal } from '@/utils/abc/abcChoirVocal';
import { Toggle } from '@/components/shared/controls/Toggle';
import { barTimes, spanSeconds } from '@/components/cover-studio/build/prepareView';

const btn = 'rounded-md border border-zinc-300 dark:border-white/10 px-2 py-1 text-xs text-zinc-700 dark:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-white/5 disabled:opacity-40';

interface PopupProps {
  /** Viewport position of the popup's top-left corner. */
  at: { left: number; top: number };
  name: string;
  onHearScore: () => void;
  onHearRecording: () => void;
  recordingDisabled: boolean;
  recordingTitle: string;
  /** Something is sounding, so Stop is live. */
  playing: boolean;
  onStop: () => void;
  loop: boolean;
  onLoop: (loop: boolean) => void;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

/** In a portal on the body: an ancestor's transform would otherwise move a fixed box off its section. */
export const SectionPlayPopup: React.FC<PopupProps> = p => {
  const { t } = useTranslation();
  return createPortal(
    <div role="dialog" aria-label={p.name} className="fixed z-50 pt-1 text-xs text-zinc-700 dark:text-zinc-200"
      style={{ left: Math.max(8, Math.min(p.at.left, window.innerWidth - 264)), top: p.at.top }}
      onMouseEnter={p.onMouseEnter} onMouseLeave={p.onMouseLeave}>
      <div className="w-64 space-y-2 rounded-lg border border-zinc-200 dark:border-white/10 bg-white dark:bg-zinc-900 p-2 shadow-xl">
        <p className="font-medium">{p.name}</p>
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" className={btn} onClick={p.onHearScore}>{t('cover.prepare.structure.hearScore')}</button>
          <button type="button" className={btn} disabled={p.recordingDisabled} onClick={p.onHearRecording} title={p.recordingTitle}>
            {t('cover.prepare.structure.hearRecording')}</button>
          <button type="button" className={btn} disabled={!p.playing} onClick={p.onStop}
            title={t('cover.prepare.structure.stop')} aria-label={t('cover.prepare.structure.stop')}><Square className="h-3 w-3" /></button>
          <Toggle size="sm" accent="cyan" checked={p.loop} onChange={p.onLoop} label={t('cover.prepare.structure.loop')}
            info={t('cover.prepare.structure.loopInfo')} />
        </div>
        <p className="text-[11px] text-zinc-500">{t('cover.prepare.structure.approx')}</p>
      </div>
    </div>, document.body);
};

/** Opens a popup on a section and closes it 200 ms after the pointer leaves both, so it can cross between them. */
export function useHoverPopup<K>() {
  const [open, setOpen] = useState<{ key: K; left: number; top: number } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const keep = () => { if (timer.current) clearTimeout(timer.current); timer.current = null; };
  const close = () => { keep(); timer.current = setTimeout(() => setOpen(null), 200); };
  useEffect(() => keep, []);
  const show = (key: K, el: HTMLElement) => {
    keep();
    const r = el.getBoundingClientRect();
    setOpen({ key, left: r.left, top: r.bottom });
  };
  return { open, show, keep, close, hide: () => { keep(); setOpen(null); } };
}

type Synth = InstanceType<typeof abcjs.synth.CreateSynth>;

/** Plays bars `from`..`to` (1-based, Vocal numbering) of `abc` with the synth, or of the
 *  recording by the score's clock from one bar early. One at a time; `loop` repeats. */
export function useSectionPlayer(abc: string, recordingUrl: string) {
  const clock = useMemo(() => barTimes(abc), [abc]);
  const [playing, setPlaying] = useState<'score' | 'recording' | null>(null);
  const synth = useRef<{ abc: string; ready: Promise<Synth> } | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const run = useRef(0);

  const stop = () => {
    run.current++;
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
    audio.current?.pause();
    void synth.current?.ready.then(s => { try { s.stop(); } catch { /* never started */ } });
    setPlaying(null);
  };
  useEffect(() => stop, [abc, recordingUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const primed = () => {
    if (synth.current?.abc !== abc) {
      const tune = abcjs.renderAbc(document.createElement('div'), withChoirVocal(abc), {})[0];
      const s = new abcjs.synth.CreateSynth();
      synth.current = { abc, ready: s.init({ visualObj: tune }).then(() => s.prime()).then(() => s) };
    }
    return synth.current.ready;
  };

  const window_ = (from: number, to: number, lead: boolean) => {
    if (!clock || spanSeconds(clock, from, to) === null) return null;
    const start = clock.start[from - 1], early = lead ? spanSeconds(clock, from, from) ?? 0 : 0;
    return { from: Math.max(0, start - early), to: clock.end[to - 1] };
  };

  const hearScore = async (from: number, to: number, loop: boolean) => {
    stop();
    const w = window_(from, to, false);
    if (!w || !abcjs.synth.supportsAudio()) return;
    const mine = run.current;
    const s = await primed();
    if (run.current !== mine) return;
    const begin = () => { s.start(); s.seek(w.from, 'seconds'); };
    begin();
    setPlaying('score');
    const started = performance.now();
    let at = started;
    timer.current = setInterval(() => {
      if ((performance.now() - at) / 1000 < w.to - w.from) return;
      if (loop) { at = performance.now(); begin(); } else stop();
    }, 50);
  };

  const hearRecording = (from: number, to: number, loop: boolean) => {
    stop();
    const w = window_(from, to, true);
    if (!w || !recordingUrl) return;
    if (!audio.current || audio.current.src !== new URL(recordingUrl, location.href).href) audio.current = new Audio(recordingUrl);
    const a = audio.current;
    a.currentTime = w.from;
    void a.play().catch(() => stop());
    setPlaying('recording');
    timer.current = setInterval(() => {
      if (!a.ended && a.currentTime < w.to) return;
      if (loop) { a.currentTime = w.from; void a.play().catch(() => stop()); } else stop();
    }, 60);
  };

  return { playing, stop, hearScore, hearRecording, canHearRecording: !!recordingUrl && !!clock };
}
