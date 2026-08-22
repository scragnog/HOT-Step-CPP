// Mm3ComposeButton.tsx — plain English in the caption box -> MM3 Structured Caption
//
// The MiniMax-Music3 counterpart to "Generate with AI", except nothing is
// generated: the caption is assembled from MiniMax's own 1,000 reference
// captions, so it can only ever contain vocabulary the target genre actually
// uses. No provider, no API key, no model. See services/lireek/mm3Compose.ts.
//
// The disclosure below the button is the point, not decoration — a wrong genre
// family is this feature's real failure mode, so the routed genre, the number of
// source templates and every note (control/prompt conflicts, controls MM3 cannot
// express) are shown rather than swallowed.

import React, { useState } from 'react';
import { Wand2, Loader2, AlertTriangle, Dices } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { composeMm3Caption, type Mm3ComposeControls, type Mm3ComposeResult } from '../../services/lireekApi';

interface Mm3ComposeButtonProps {
  /** Current caption box contents — used as the brief. */
  brief: string;
  controls: Mm3ComposeControls;
  onComposed: (caption: string) => void;
}

export const Mm3ComposeButton: React.FC<Mm3ComposeButtonProps> = ({ brief, controls, onComposed }) => {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Mm3ComposeResult | null>(null);
  const [seed, setSeed] = useState(1);

  const trimmed = brief.trim();
  // A caption that already starts with the MM3 heading is a composed one, not a
  // brief — re-composing would route on Structured Caption prose, not intent.
  const alreadyComposed = /^Global Metadata\b/.test(trimmed);

  const run = async (nextSeed: number) => {
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await composeMm3Caption(trimmed, controls, nextSeed);
      setResult(r);
      setSeed(nextSeed);
      onComposed(r.caption);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <button
          onClick={() => void run(1)}
          disabled={busy || !trimmed || alreadyComposed}
          title={!trimmed ? t('mm3Compose.empty') : t('mm3Compose.tooltip')}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-cyan-400 hover:text-cyan-300 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/20 hover:border-cyan-500/30 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {busy ? <Loader2 size={13} className="animate-spin" /> : <Wand2 size={13} />}
          {t('mm3Compose.button')}
        </button>

        {result && (
          <button
            onClick={() => void run(seed + 1)}
            disabled={busy}
            title={t('mm3Compose.reroll')}
            className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-xs text-zinc-400 hover:text-zinc-200 bg-white/5 hover:bg-white/10 border border-white/10 transition-all disabled:opacity-40"
          >
            <Dices size={13} />
          </button>
        )}
      </div>

      {error && (
        <p className="text-[11px] text-red-400 flex items-start gap-1">
          <AlertTriangle size={12} className="mt-0.5 shrink-0" />
          {error}
        </p>
      )}

      {result && !error && (
        <div className="text-[11px] leading-snug space-y-1 rounded-lg bg-white/5 border border-white/10 px-2.5 py-2">
          <p className="text-zinc-300">
            <span className="text-zinc-500">Genre </span>
            <span className="font-medium text-cyan-300">{result.genre}</span>
            <span className="text-zinc-500"> · {result.family.replace(/-/g, ' ')}</span>
            <span className="text-zinc-500"> · {result.sourceCount} reference captions</span>
          </p>

          {result.fallback && (
            <p className="text-amber-400">
              No genre was recognised in the description — fell back to the general pop/ballad family.
              Naming a style (e.g. &ldquo;post-hardcore&rdquo;, &ldquo;nu-disco&rdquo;) will route it properly.
            </p>
          )}

          {result.notes.map((n, i) => (
            <p key={i} className="text-amber-400/90">{n}</p>
          ))}

          {result.validation.length > 0 && (
            <p className="text-red-400">Format check: {result.validation.join(' · ')}</p>
          )}
        </div>
      )}
    </div>
  );
};
