import React from 'react';
import { useMutation } from '@tanstack/react-query';
import type { Song } from '@/types';
import { useAuth } from '@/context/AuthContext';
import { yue2CoverApi, type Yue2CoverDriftResult } from '@/services/yue2CoverApi';
import { formatDecimal } from '@/utils/format/format';

const measuredThisSession = new Map<string, Yue2CoverDriftResult>();

function seconds(value: number): string { return `${formatDecimal(value, 1)}s`; }
function span(value: { start: number; end: number } | null): string {
  return value ? `${seconds(value.start)}–${seconds(value.end)}` : 'Unscored';
}
function expectedSpan(value: { start: number | null; end: number }): string {
  return value.start === null ? 'No vocal note' : span({ start: value.start, end: value.end });
}

const unscoredText: Record<NonNullable<Yue2CoverDriftResult['sections'][number]['unscoredReason']>, string> = {
  no_matching_lyric_tag: 'Instrumental / N/A: no lyrics',
  no_vocal_note: 'Instrumental / N/A: no vocal note',
  no_aligned_words: 'No aligned words',
  low_word_confidence: 'Low word confidence',
  mix_stem_disagreement: 'Mix/stem disagreement',
  whisper_transcript_unreliable: 'Whisper transcript unreliable',
  no_whisper_match: 'No Whisper word match',
  whisper_disagreement: 'Whisper disagreement',
};

/** An explicit measurement: opening a library item never starts MMS_FA. */
export const Yue2CoverDrift: React.FC<{ song: Song }> = ({ song }) =>
  <Yue2CoverDriftBody key={song.id} song={song} />;

const Yue2CoverDriftBody: React.FC<{ song: Song }> = ({ song }) => {
  const { token } = useAuth();
  const params = song.generationParams || song.generation_params || {};
  const cached = params.yue2CoverDrift as Yue2CoverDriftResult | undefined;
  const saved = cached?.metricVersion === 5 ? cached : undefined;
  const session = measuredThisSession.get(song.id);
  const [result, setResult] = React.useState<Yue2CoverDriftResult | null>(
    (session?.metricVersion === 5 ? session : undefined) ?? saved ?? null);

  const skipped = (section: Yue2CoverDriftResult['sections'][number]) =>
    section.unscoredReason === 'no_vocal_note' || section.unscoredReason === 'no_matching_lyric_tag';
  const skippedCount = result?.sections.filter(skipped).length ?? 0;

  const { mutate: measure, isPending: busy, error: measureError } = useMutation({
    mutationFn: () => yue2CoverApi.measureDrift(song.id),
    onSuccess: (measured) => {
      measuredThisSession.set(song.id, measured);
      setResult(measured);
    },
  });
  const error = measureError?.message ?? '';

  // Hook order must not depend on the song's params: a generation finishing
  // while this card is mounted can flip these on the same instance.
  if (!params.yue2Cover || !params.yue2Abc) return null;

  return <section className="space-y-2 rounded-xl border border-zinc-200 dark:border-white/10 p-3 text-xs">
    <div className="flex items-center justify-between gap-2">
      <h4 className="font-semibold text-zinc-800 dark:text-zinc-200">Lyric drift</h4>
      <button type="button" onClick={() => measure()} disabled={busy || !token}
        className="text-cyan-700 dark:text-cyan-300 disabled:opacity-40">
        {busy ? 'Aligning words…' : result ? 'Refresh drift' : 'Measure drift'}
      </button>
    </div>
    {!result && <p className="text-zinc-500">Compare sung word times with the score’s first vocal notes.</p>}
    {error && <p role="alert" className="text-red-600 dark:text-red-400">{error}</p>}
    {result && <>
      <p className="font-medium text-zinc-800 dark:text-zinc-200">
        Mean absolute start offset: {result.meanAbsoluteOffsetBars === null ? 'No aligned sections' : `${formatDecimal(result.meanAbsoluteOffsetBars, 2)} bars`}
      </p>
      <p className="text-zinc-500">
        {result.tempoBpm} BPM · {result.meter} · {result.tempoSource === 'source-score-fallback'
          ? 'Source tempo fallback: rendered score had free tempo' : 'Rendered score tempo'}
      </p>
      <p className="text-zinc-500">{result.stemChecked ? 'Mix and vocal stem compared' : 'Mix only; vocal stem not checked'}
        {' · '}{result.sections.filter(section => section.offsetBars !== null).length}/{result.sections.length - skippedCount} vocal sections scored
        {skippedCount > 0 && <> · {skippedCount} instrumental / N/A</>}
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
          <thead><tr className="text-zinc-500"><th className="pr-2">Section / bars</th><th className="pr-2">Expected note–end</th><th className="pr-2">Boundary–end</th><th className="pr-2">Sung</th><th>Start offset</th></tr></thead>
          <tbody>{result.sections.map((section, index) => <tr key={index} className="border-t border-zinc-200 dark:border-white/10">
            <td className="py-1 pr-2">{section.scoreLabel} · {section.startBar}–{section.endBar}{section.lyricTag && section.lyricTag.toLowerCase() !== section.scoreLabel.toLowerCase() ? ` / [${section.lyricTag}]` : ''}</td>
            <td className="pr-2">{expectedSpan(section.expected)}</td>
            <td className="pr-2">{span(section.boundary)}</td>
            <td className="pr-2">{skipped(section) ? 'N/A' : span(section.sung)}</td>
            <td>{section.offsetBars === null
              ? section.unscoredReason ? unscoredText[section.unscoredReason] : 'Unscored'
              : `${section.offsetBars >= 0 ? '+' : ''}${formatDecimal(section.offsetBars, 2)} bars`}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </>}
  </section>;
};
