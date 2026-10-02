import React, { useEffect, useRef, useState } from 'react';
import abcjs from 'abcjs';
import 'abcjs/abcjs-audio.css';
import { followNoteInBox } from '../../utils/abcFollow';
import type { Yue2CoverJob, Yue2CoverReadiness } from '../../services/yue2CoverApi';

interface Props {
  sourceReady: boolean;
  readiness: Yue2CoverReadiness | null;
  job: Yue2CoverJob | null;
  preparing: boolean;
  error: string;
  abc: string;
  scoreSource?: 'dataset' | null;
  approved: boolean;
  onAbcChange: (value: string) => void;
  onTranscribe: () => void;
  onCancel: () => void;
  onApprove: () => void;
}

export const Yue2CoverScore: React.FC<Props> = p => {
  const scoreRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLDivElement>(null);
  const [previewValid, setPreviewValid] = useState(false);
  const busy = p.preparing || p.job?.status === 'queued' || p.job?.status === 'running';

  useEffect(() => {
    if (!scoreRef.current) return;
    scoreRef.current.innerHTML = '';
    if (audioRef.current) audioRef.current.innerHTML = '';
    setPreviewValid(false);
    if (!p.abc.trim()) return;
    let tunes: ReturnType<typeof abcjs.renderAbc>;
    try { tunes = abcjs.renderAbc(scoreRef.current, p.abc, { responsive: 'resize', add_classes: true }); }
    catch { return; }
    if (!tunes[0]) return;
    setPreviewValid(true);
    if (!audioRef.current || !abcjs.synth.supportsAudio()) return;
    const control = new abcjs.synth.SynthController();
    const box = scoreRef.current.parentElement;
    let lit: Element[] = [];
    control.load(audioRef.current, {
      onStart() { lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = []; },
      onEvent(ev: { elements?: Element[][] }) {
        lit.forEach(el => el.classList.remove('abcjs-highlight'));
        lit = (ev.elements ?? []).flat();
        lit.forEach(el => el.classList.add('abcjs-highlight'));
        followNoteInBox(lit[0], box);
      },
      onFinished() { lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = []; },
    }, { displayLoop: false, displayRestart: true, displayPlay: true, displayProgress: true, displayWarp: false });
    void control.setTune(tunes[0], false).catch(() => {});
    return () => { try { control.pause(); } catch {} };
  }, [p.abc]);

  return (
    <div className="border-t border-zinc-200 dark:border-white/10 p-4 space-y-3 overflow-y-auto max-h-[55%]">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">YuE2 lead sheet</h3>
          <p className="text-xs text-zinc-500">Transcribe the source melody and chords or paste ABC, then review and approve it.</p>
        </div>
        <button onClick={p.onTranscribe} disabled={!p.sourceReady || busy || p.readiness?.ready === false}
          className="rounded-lg bg-cyan-500/20 px-3 py-1.5 text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">
          {p.abc ? 'Retry transcription' : 'Transcribe full score'}
        </button>
      </div>
      {p.readiness?.message && <p className="text-xs text-amber-600 dark:text-amber-400">{p.readiness.message}</p>}
      {busy && (
        <div className="space-y-1 text-xs text-zinc-500">
          <p>{p.preparing ? 'Preparing source audio…' : `${p.job?.phase || 'Queued'}${p.job?.total ? ` · ${p.job.done}/${p.job.total}` : ''}`}</p>
          <button onClick={p.onCancel} className="text-red-500">Cancel transcription</button>
        </div>
      )}
      {p.error && <p role="alert" className="text-xs text-red-500">{p.error}</p>}
      {p.scoreSource && <p className="text-xs text-cyan-600 dark:text-cyan-300">Score from dataset · editable</p>}
      <textarea aria-label="Editable ABC score" value={p.abc} disabled={busy} onChange={e => p.onAbcChange(e.target.value)}
        placeholder="Paste an ABC score here to skip transcription"
        className="w-full h-28 resize-y rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-xs font-mono text-zinc-900 dark:text-white" />
      {p.abc.trim() && !previewValid && <p className="text-xs text-amber-500">The ABC could not be previewed. Check its notation before approval.</p>}
      <div className={`${p.abc.trim() ? '' : 'hidden'} bg-white rounded-lg border border-zinc-200 overflow-y-auto max-h-48`}>
        <div ref={scoreRef} className="text-black p-2 [&_svg]:fill-current [&_.abcjs-highlight]:fill-amber-500" />
      </div>
      <div ref={audioRef} className={`text-xs ${previewValid ? '' : 'hidden'}`} />
      <button onClick={p.onApprove} disabled={!p.sourceReady || !previewValid || busy || p.approved}
        className="rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-40">
        {p.approved ? 'Score approved' : 'Approve score'}
      </button>
    </div>
  );
};
