import React from 'react';
import type { Song } from '../../types';
import { useAuth } from '../../context/AuthContext';
import { yue2CoverApi, type Yue2CoverDriftResult } from '../../services/yue2CoverApi';

const measuredThisSession = new Map<string, Yue2CoverDriftResult>();

function seconds(value: number): string { return `${value.toFixed(1)}s`; }
function span(value: { start: number; end: number } | null): string {
  return value ? `${seconds(value.start)}–${seconds(value.end)}` : 'Unscored';
}

const unscoredText: Record<NonNullable<Yue2CoverDriftResult['sections'][number]['unscoredReason']>, string> = {
  no_matching_lyric_tag: 'No matching sung tag',
  no_aligned_words: 'No aligned words',
  low_word_confidence: 'Low word confidence',
  mix_stem_disagreement: 'Mix/stem disagreement',
};

/** An explicit measurement: opening a library item never starts MMS_FA. */
export const Yue2CoverDrift: React.FC<{ song: Song }> = ({ song }) =>
  <Yue2CoverDriftBody key={song.id} song={song} />;

const Yue2CoverDriftBody: React.FC<{ song: Song }> = ({ song }) => {
  const { token } = useAuth();
  const params = song.generationParams || song.generation_params || {};
  const cached = params.yue2CoverDrift as Yue2CoverDriftResult | undefined;
  const saved = cached?.metricVersion === 3 ? cached : undefined;
  const [result, setResult] = React.useState<Yue2CoverDriftResult | null>(measuredThisSession.get(song.id) ?? saved ?? null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');

  if (!params.yue2Cover || !params.yue2Abc) return null;

  const measure = async () => {
    if (!token) return;
    setBusy(true);
    setError('');
    try {
      const measured = await yue2CoverApi.measureDrift(song.id, token);
      measuredThisSession.set(song.id, measured);
      setResult(measured);
    }
    catch (err) { setError(err instanceof Error ? err.message : 'Could not measure drift.'); }
    finally { setBusy(false); }
  };

  return <section className="space-y-2 rounded-xl border border-zinc-200 dark:border-white/10 p-3 text-xs">
    <div className="flex items-center justify-between gap-2">
      <h4 className="font-semibold text-zinc-800 dark:text-zinc-200">Lyric drift</h4>
      <button type="button" onClick={() => void measure()} disabled={busy || !token}
        className="text-cyan-700 dark:text-cyan-300 disabled:opacity-40">
        {busy ? 'Aligning words…' : result ? 'Refresh drift' : 'Measure drift'}
      </button>
    </div>
    {!result && <p className="text-zinc-500">Compare sung word times with the score’s section bars.</p>}
    {error && <p role="alert" className="text-red-600 dark:text-red-400">{error}</p>}
    {result && <>
      <p className="font-medium text-zinc-800 dark:text-zinc-200">
        Mean absolute start offset: {result.meanAbsoluteOffsetBars === null ? 'No aligned sections' : `${result.meanAbsoluteOffsetBars.toFixed(2)} bars`}
      </p>
      <p className="text-zinc-500">
        {result.tempoBpm} BPM · {result.meter} · {result.tempoSource === 'source-score-fallback'
          ? 'Source tempo fallback: rendered score had free tempo' : 'Rendered score tempo'}
      </p>
      <p className="text-zinc-500">{result.stemChecked ? 'Mix and vocal stem compared' : 'Mix only; vocal stem not checked'}
        {' · '}{result.sections.filter(section => section.offsetBars !== null).length}/{result.sections.length} sections scored
      </p>
      {result.sectionWarning && <p className="text-amber-700 dark:text-amber-300">{result.sectionWarning}</p>}
      {result.unmatchedLyricBlocks.length > 0 && <p className="text-amber-700 dark:text-amber-300">
        Unmatched sung lyric tags: {result.unmatchedLyricBlocks.map(block => `${block.index} [${block.label}]`).join(', ')}
      </p>}
      {result.firstOverOneBar && <p className="text-amber-700 dark:text-amber-300">
        First over one bar: section {result.firstOverOneBar.index} ({result.firstOverOneBar.label})
      </p>}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[11px]">
          <thead><tr className="text-zinc-500"><th className="pr-2">Section / bars</th><th className="pr-2">Expected</th><th className="pr-2">Sung</th><th>Start offset</th></tr></thead>
          <tbody>{result.sections.map((section, index) => <tr key={index} className="border-t border-zinc-200 dark:border-white/10">
            <td className="py-1 pr-2">{section.scoreLabel} · {section.startBar}–{section.endBar}{section.lyricTag && section.lyricTag.toLowerCase() !== section.scoreLabel.toLowerCase() ? ` / [${section.lyricTag}]` : ''}</td>
            <td className="pr-2">{span(section.expected)}</td>
            <td className="pr-2">{span(section.sung)}</td>
            <td>{section.offsetBars === null
              ? section.unscoredReason ? unscoredText[section.unscoredReason] : 'Unscored'
              : `${section.offsetBars >= 0 ? '+' : ''}${section.offsetBars.toFixed(2)} bars`}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </>}
  </section>;
};
