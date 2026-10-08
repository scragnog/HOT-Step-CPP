// AuditionPlayer.tsx — one side of a codes audition
//
// Renders a single AuditionSideResult: label chip, <audio>, the determinism
// receipt (codesSha1) and the plan the LM handed back. Shared by the A/B card,
// the SampleDrawer and the milestone flow.
//
// The honest hint (C20) is rendered HERE, by the player itself, so it cannot be
// forgotten at a call site. It is never collapsible and never a tooltip.

import React, { useState } from 'react';
import { AlertTriangle, Music4, SendHorizonal, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import type { AuditionPreview, AuditionSideResult } from '../../services/trainingApi';
import { sendAuditionToCustomGen, type AuditionRenderCell } from './sendAuditionToCustomGen';

interface AuditionPlayerProps {
  side: AuditionSideResult;
  hintPosition?: 'above' | 'below';
  /** True when the OTHER side of this A/B reported the same codesSha1. The
   *  single most likely silent failure of the whole feature (§6.4). */
  identicalCodes?: boolean;
  /** The preview this side belongs to. When present (the A/B card), each
   *  render row gets a "Send to Custom-Gen" button that replays this exact
   *  run on the Create page via the mirrored-settings recipe. The
   *  SampleDrawer's stored-codes preview passes nothing — there is no LM run
   *  to replay. */
  preview?: AuditionPreview;
}

const CHIP = 'text-[10px] font-semibold px-1.5 py-0.5 rounded-full border tabular-nums';

export const AuditionPlayer: React.FC<AuditionPlayerProps> = ({
  side, hintPosition = 'above', identicalCodes = false, preview,
}) => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  // The handoff navigates to Create itself, and only once the server draft exists.
  const send = async (cell: AuditionRenderCell) => {
    if (!preview) return;
    setSending(true); setSendError(null);
    try {
      if (!token) throw new Error(t('createPanel.trainingDraft.signIn', 'Sign in to open this audition setup.'));
      await sendAuditionToCustomGen(preview, side, cell, token);
    } catch (e) {
      setSendError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  };

  const sendButton = (cell: AuditionRenderCell) => preview && (
    <button
      onClick={() => send(cell)}
      disabled={sending}
      title={t('trainingStudio.audition.sendToCustomGenHint')}
      className="ml-auto flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] font-semibold normal-case tracking-normal text-zinc-500 hover:text-amber-500 hover:bg-amber-500/10 disabled:opacity-50 transition-colors flex-shrink-0"
    >
      <SendHorizonal size={11} />
      {t('trainingStudio.audition.sendToCustomGen')}
    </button>
  );

  const accent = side.slot === 'adapter'
    ? 'text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/25'
    : 'text-zinc-600 dark:text-zinc-300 bg-zinc-500/10 border-zinc-500/25';

  const hint = (
    <p className="text-[11px] text-zinc-500 dark:text-zinc-400 italic">
      {t('trainingStudio.audition.hint')}
    </p>
  );

  const hasPlan = !!(side.caption || side.lyrics || side.bpm || side.keyscale || side.timesignature);

  return (
    <div className="flex flex-col gap-2 min-w-0">
      {hintPosition === 'above' && hint}

      <div className="flex items-center gap-2 flex-wrap">
        <span className={`${CHIP} ${accent}`}>{side.label || side.slot}</span>
        {side.lmAdapter
          ? (
            <span
              className="text-[10px] font-mono text-zinc-500 truncate max-w-[220px]"
              title={side.lmAdapter}
            >
              {side.lmAdapter}
            </span>
          )
          : <span className="text-[10px] text-zinc-500">{t('trainingStudio.audition.baseLm')}</span>}
        {side.ok && side.lmAdapter !== '' && side.lmAdapterScale !== 1 && (
          <span className="text-[10px] text-zinc-500 tabular-nums">×{side.lmAdapterScale}</span>
        )}
      </div>

      {!side.ok ? (
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">
          <XCircle size={13} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{side.error || t('trainingStudio.audition.sideFailed')}</span>
        </div>
      ) : (
        <>
          {/* ok:true with an EMPTY audioUrl is a real server state, not a
              contradiction: auditionService.decodeStoredCodes and auditionRunner
              both blank every audioUrl when writePreview() fails (read-only
              previews root, disk full) while leaving ok true and putting the
              explanation in .error. Rendering <audio src=""> makes the browser
              resolve the CURRENT DOCUMENT URL, fetch the HTML page as audio and
              fail silently, with the server's explanation shown nowhere.
              The codes still exist, so the meta line and the planned-output
              details below stay — the plan is half the evidence (C13) and it is
              still valid even when the audio could not be written. */}
          {/* Opt-in DiT render — the codes played as MUSIC. Listed first when
              present: it is the human-comparable artifact; the sketch below
              stays for the structural view. */}
          {side.renderUrl && (
            <div className="flex flex-col gap-1">
              <span className="flex items-center text-[10px] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
                {t('trainingStudio.audition.renderedLabel')}
                {!!side.renderMs && (
                  <span className="ml-2 normal-case font-normal text-zinc-500 tabular-nums">
                    {t('trainingStudio.audition.renderMs', { ms: Math.round(side.renderMs) })}
                  </span>
                )}
                {sendButton('bare')}
              </span>
              <audio src={side.renderUrl} controls preload="none" className="w-full h-9" />
            </div>
          )}
          {side.renderError && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-600 dark:text-amber-400">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              <span className="min-w-0 break-words">
                {t('trainingStudio.audition.renderFailed', { error: side.renderError })}
              </span>
            </div>
          )}

          {/* DiT-adapter render — the same codes through the dataset's trained
              sound adapter. With the bare render above this side shows both
              halves of its row in the 2×2 matrix. */}
          {side.renderAdapterUrl && (
            <div className="flex flex-col gap-1">
              <span className="flex items-center text-[10px] font-semibold uppercase tracking-wide text-sky-600 dark:text-sky-400">
                {t('trainingStudio.audition.renderedAdapterLabel')}
                {!!side.renderAdapterMs && (
                  <span className="ml-2 normal-case font-normal text-zinc-500 tabular-nums">
                    {t('trainingStudio.audition.renderMs', { ms: Math.round(side.renderAdapterMs) })}
                  </span>
                )}
                {sendButton('adapter')}
              </span>
              <audio src={side.renderAdapterUrl} controls preload="none" className="w-full h-9" />
            </div>
          )}
          {side.renderAdapterError && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-600 dark:text-amber-400">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              <span className="min-w-0 break-words">
                {t('trainingStudio.audition.renderAdapterFailed', { error: side.renderAdapterError })}
              </span>
            </div>
          )}

          {side.audioUrl ? (
            /* Local to the card — preview audio is not a library song, so it
               never goes through the global playback store (same rule as
               SampleDrawer). */
            <div className="flex flex-col gap-1">
              {side.renderUrl && (
                <span className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">
                  {t('trainingStudio.audition.sketchLabel')}
                </span>
              )}
              <audio src={side.audioUrl} controls preload="none" className="w-full h-9" />
            </div>
          ) : (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              <span className="min-w-0 break-words">
                {side.error || t('trainingStudio.audition.noPreviewFile')}
              </span>
            </div>
          )}

          {identicalCodes && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-red-500/30 bg-red-500/10 text-xs font-semibold text-red-500 dark:text-red-400">
              <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.audition.identicalCodes')}
            </div>
          )}

          <div className="flex items-center gap-2 flex-wrap text-[10px] text-zinc-500 tabular-nums">
            <span>{t('trainingStudio.audition.codesCount', { count: side.codesCount })}</span>
            <span className="text-zinc-400 dark:text-zinc-600">·</span>
            <span>{side.durationSec.toFixed(1)}s</span>
            {side.lmMs > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span>{t('trainingStudio.audition.lmMs', { ms: Math.round(side.lmMs) })}</span>
              </>
            )}
            {side.decodeMs > 0 && (
              <>
                <span className="text-zinc-400 dark:text-zinc-600">·</span>
                <span>{t('trainingStudio.audition.decodeMs', { ms: Math.round(side.decodeMs) })}</span>
              </>
            )}
            {side.codesSha1 && (
              <span
                className={`${CHIP} text-sky-500 bg-sky-500/10 border-sky-500/20 font-mono`}
                title={side.codesSha1}
              >
                {side.codesSha1.slice(0, 8)}
              </span>
            )}
            {/* No render on this preview → the send lives here instead; the
                Create page then produces the render this audition skipped. */}
            {!side.renderUrl && !side.renderAdapterUrl && sendButton('bare')}
          </div>

          {hasPlan && (
            <details className="rounded-lg border border-zinc-200 dark:border-white/10 px-3 py-2">
              <summary className="cursor-pointer text-[11px] font-semibold text-zinc-600 dark:text-zinc-400 flex items-center gap-1.5">
                <Music4 size={12} className="text-amber-500" />
                {t('trainingStudio.audition.plannedTitle')}
              </summary>
              <div className="flex flex-col gap-2 mt-2">
                {side.caption && (
                  <p className="text-[11px] text-zinc-700 dark:text-zinc-300 break-words">{side.caption}</p>
                )}
                <div className="flex items-center gap-2 flex-wrap text-[10px] text-zinc-500 tabular-nums">
                  {side.bpm > 0 && <span>{side.bpm} BPM</span>}
                  {side.keyscale && <span>{side.keyscale}</span>}
                  {side.timesignature && <span>{side.timesignature}</span>}
                </div>
                {side.lyrics && (
                  <pre className="max-h-56 overflow-y-auto whitespace-pre-wrap break-words rounded-lg bg-zinc-100 dark:bg-black/30 px-3 py-2 font-mono text-[10px] leading-relaxed text-zinc-600 dark:text-zinc-400">
                    {side.lyrics}
                  </pre>
                )}
              </div>
            </details>
          )}
        </>
      )}

      {sendError && (
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-[11px] text-red-500 dark:text-red-400">
          <XCircle size={12} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{sendError}</span>
        </div>
      )}

      {hintPosition === 'below' && hint}
    </div>
  );
};

export default AuditionPlayer;
