// Yue2ScorePreviewModal.tsx — look at (and listen to) the planner's lead sheet
// before any audio is rendered.
//
// The plan stage takes seconds; the render takes minutes. A runaway plan —
// intro > verse for two hundred bars, no singing — is obvious on the score,
// so this is where to catch it. Three ways out: Continue renders THIS score
// (the engine skips its own plan stage), Retry plans again with a fresh seed,
// Cancel renders nothing.
//
// Playback is abcjs's own SynthController, the same one the Training Studio
// sheet preview uses; the first Play click is the user gesture the browser
// needs before an AudioContext may make sound, and it also fetches abcjs's
// soundfont from its default CDN. Nothing is bundled.

import React, { useEffect, useRef } from 'react';
import ReactDOM from 'react-dom';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, CheckCircle2, Loader2, RefreshCw, X } from 'lucide-react';
import abcjs from 'abcjs';
import 'abcjs/abcjs-audio.css';

export interface Yue2ScorePreviewData {
  abc: string;
  seed: number;
  end_reason: string;
  health: { verdict: string; reason: string; bars: number; estSeconds?: number; sections: string[] };
}

interface Props {
  open: boolean;
  /** null while the plan is still running. */
  data: Yue2ScorePreviewData | null;
  error: string | null;
  onContinue: () => void;
  onRetry: () => void;
  onCancel: () => void;
}

export const Yue2ScorePreviewModal: React.FC<Props> = ({ open, data, error, onContinue, onRetry, onCancel }) => {
  const { t } = useTranslation();
  const scoreRef = useRef<HTMLDivElement | null>(null);
  const audioRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open || !data?.abc || !scoreRef.current) return;
    scoreRef.current.innerHTML = '';
    let tunes: ReturnType<typeof abcjs.renderAbc> | undefined;
    try {
      tunes = abcjs.renderAbc(scoreRef.current, data.abc, { responsive: 'resize', add_classes: true });
    } catch {
      return;
    }
    if (!tunes?.[0] || !audioRef.current || !abcjs.synth.supportsAudio()) return;
    audioRef.current.innerHTML = '';
    const control = new abcjs.synth.SynthController();
    const box = scoreRef.current;
    let lit: Element[] = [];
    control.load(audioRef.current, {
      onStart() { lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = []; },
      onEvent(ev: { elements?: Element[][] }) {
        lit.forEach(el => el.classList.remove('abcjs-highlight'));
        lit = (ev.elements ?? []).flat();
        lit.forEach(el => el.classList.add('abcjs-highlight'));
        const first = lit[0] as (Element & { scrollIntoView?: (o: ScrollIntoViewOptions) => void }) | undefined;
        if (first && box) first.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
      },
      onFinished() { lit.forEach(el => el.classList.remove('abcjs-highlight')); lit = []; },
    }, { displayLoop: false, displayRestart: true, displayPlay: true, displayProgress: true, displayWarp: false });
    control.setTune(tunes[0], false).catch(() => { /* the score still renders without audio */ });
    // Continue/Cancel/Retry all close or replace the sheet; the synth keeps
    // playing on its own AudioContext unless it is told to stop.
    return () => { try { control.pause(); } catch { /* nothing was playing */ } };
  }, [open, data?.abc]);

  if (!open) return null;

  const verdict = data?.health.verdict;
  const verdictTone = verdict === 'runaway' ? 'text-red-600 dark:text-red-400'
    : verdict === 'long' ? 'text-amber-600 dark:text-amber-400'
    : 'text-emerald-600 dark:text-emerald-400';

  return ReactDOM.createPortal(
    <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" data-portal-layer>
      <div className="w-full max-w-4xl max-h-[90vh] flex flex-col rounded-2xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-white/10 shadow-2xl">
        <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-200 dark:border-white/10">
          <div>
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              {t('createPanel.yue2ScorePreviewTitle', 'Lead sheet preview')}
            </h3>
            <p className="text-[11px] text-zinc-500">
              {t('createPanel.yue2ScorePreviewHint', 'The planner wrote this score for your caption and lyrics. Continue renders exactly this; Retry plans again with a new seed.')}
            </p>
          </div>
          <button onClick={onCancel} className="p-1 rounded-lg text-zinc-500 hover:text-zinc-900 dark:hover:text-white" title={t('common.cancel', 'Cancel')}>
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-auto px-5 py-3 space-y-3">
          {!data && !error && (
            <div className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400 py-8 justify-center">
              <Loader2 size={14} className="animate-spin" /> {t('createPanel.yue2ScorePlanning', 'Planning the lead sheet…')}
            </div>
          )}
          {error && (
            <div className="flex items-start gap-2 text-xs text-red-600 dark:text-red-400">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" /> <span>{error}</span>
            </div>
          )}
          {data && (
            <>
              <div className={`flex items-start gap-2 text-xs ${verdictTone}`}>
                {verdict === 'healthy' ? <CheckCircle2 size={14} className="mt-0.5 shrink-0" /> : <AlertTriangle size={14} className="mt-0.5 shrink-0" />}
                <span>
                  <span className="font-semibold uppercase tracking-wider">{verdict}</span> — {data.health.reason}
                  {data.health.estSeconds ? ` · ~${Math.floor(data.health.estSeconds / 60)}:${String(data.health.estSeconds % 60).padStart(2, '0')}` : ''}
                  {` · seed ${data.seed}`}
                  {data.health.sections.length ? ` · ${data.health.sections.join(' → ')}` : ''}
                </span>
              </div>
              {/* Same paper as the Training Studio sheet viewer: white, black
                  ink, the playing note lit amber. Two nested divs on purpose —
                  abcjs's responsive mode rewrites the inline style of the div
                  it renders into, so the scroll box must be its parent. */}
              <div className="bg-white rounded-lg border border-zinc-200 dark:border-white/10 overflow-y-auto overflow-x-hidden" style={{ maxHeight: '50vh' }}>
                <div ref={scoreRef} className="text-black p-2 [&_svg]:fill-current [&_.abcjs-highlight]:fill-amber-500 [&_.abcjs-highlight]:stroke-amber-500" />
              </div>
              <div ref={audioRef} className="text-xs" />
              <details className="text-[11px] text-zinc-500">
                <summary className="cursor-pointer">{t('createPanel.yue2ScoreRaw', 'Raw ABC')}</summary>
                <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[10px] text-zinc-600 dark:text-zinc-400">{data.abc}</pre>
              </details>
            </>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-zinc-200 dark:border-white/10">
          <button onClick={onCancel}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-zinc-300 dark:border-white/10 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-white/5">
            {t('common.cancel', 'Cancel')}
          </button>
          <button onClick={onRetry} disabled={!data && !error}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40 flex items-center gap-1.5">
            <RefreshCw size={12} /> {t('createPanel.yue2ScoreRetry', 'Retry (new seed)')}
          </button>
          <button onClick={onContinue} disabled={!data}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-pink-600 to-purple-600 text-white hover:from-pink-500 hover:to-purple-500 disabled:opacity-40">
            {t('createPanel.yue2ScoreContinue', 'Continue — render this score')}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
};
