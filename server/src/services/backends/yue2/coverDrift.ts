import type { Yue2AlignWord } from './align.js';
import { classifyYue2Score, lyricSectionTags } from './scoreHealth.js';
import { scoreSections } from './scoreSections.js';
import { scoreBarClock, scoreBarPosition } from './scoreClock.js';

export const COVER_DRIFT_METRIC_VERSION = 3;
const MIN_WORD_CONFIDENCE = 0.2;

export interface CoverDriftSection {
  scoreLabel: string;
  lyricTag: string | null;
  startBar: number;
  endBar: number;
  expected: { start: number; end: number };
  sung: { start: number; end: number } | null;
  offsetBars: number | null;
  unscoredReason: 'no_matching_lyric_tag' | 'no_aligned_words' | 'low_word_confidence' |
    'mix_stem_disagreement' | null;
  disagreementBars: number | null;
}

export interface CoverDriftResult {
  tempoBpm: number;
  meter: string;
  secondsPerBar: number;
  tempoSource: 'rendered-score' | 'source-score-fallback';
  stemChecked: boolean;
  sectionWarning: string | null;
  sungLyricBlocks: number;
  unmatchedLyricBlocks: Array<{ index: number; label: string }>;
  sections: CoverDriftSection[];
  meanAbsoluteOffsetBars: number | null;
  firstOverOneBar: { index: number; label: string; offsetBars: number } | null;
}

function lyricBlocks(lyrics: string): Array<{ label: string; start: number; end: number; hasLyrics: boolean }> {
  // The same S10 tag matcher decides which lines are section headers. Work in
  // codepoints because MMS_FA's char0/char1 offsets are not UTF-16 indices.
  const normalized = lyrics.replace(/\r\n?/g, '\n');
  const lines = normalized.split('\n');
  const length = Array.from(normalized).length;
  const blocks: Array<{ label: string; start: number; end: number; hasLyrics: boolean }> = [];
  let offset = 0;
  for (const line of lines) {
    const label = lyricSectionTags(line)[0];
    if (label) {
      if (blocks.length) blocks[blocks.length - 1].end = offset;
      blocks.push({ label, start: offset, end: length, hasLyrics: false });
    }
    offset += Array.from(line).length + 1;
  }
  const codepoints = Array.from(normalized);
  return blocks.map(block => {
    const body = codepoints.slice(block.start, block.end).join('').replace(/^\s*\[[^\]\n]+\]/, '');
    return { ...block, hasLyrics: /[\p{L}\p{N}]/u.test(body) };
  });
}

function sectionKind(label: string): string {
  const normalized = label.toLowerCase().replace(/[_–—]/g, '-');
  if (/\bpre[ -]?chorus\b/.test(normalized)) return 'pre-chorus';
  return /\b(verse|chorus|bridge|intro|outro|interlude)\b/.exec(normalized)?.[1] ?? normalized.trim();
}

function instrumental(label: string): boolean {
  return /\b(instrumental|solo|break|interlude)\b/i.test(label);
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function alignedBlock(block: { start: number; end: number }, words: Yue2AlignWord[]) {
  const timed = words.filter(word => word.char0 >= block.start && word.char0 < block.end &&
    Number.isFinite(word.start) && Number.isFinite(word.end) && word.end > word.start &&
    Number.isFinite(word.score));
  if (!timed.length) return { sung: null, reason: 'no_aligned_words' as const };
  // The first words locate the section; the whole block must also sound like
  // speech to avoid force-fitted tails after the vocals stop.
  if (median(timed.slice(0, 5).map(word => word.score)) < MIN_WORD_CONFIDENCE ||
      median(timed.map(word => word.score)) < MIN_WORD_CONFIDENCE) {
    return { sung: null, reason: 'low_word_confidence' as const };
  }
  const confident = timed.filter(word => word.score >= MIN_WORD_CONFIDENCE);
  return { sung: { start: confident[0].start, end: confident[confident.length - 1].end }, reason: null };
}

/** Compare score sections with like-named sung lyric blocks in order. */
export function measureCoverDrift(
  renderedScore: string, fullScore: string, lyrics: string, words: Yue2AlignWord[],
  stemWords?: Yue2AlignWord[],
): CoverDriftResult {
  const sections = scoreSections(renderedScore);
  const totalBars = classifyYue2Score(renderedScore).bars;
  if (!sections.length || !totalBars) throw new Error('The rendered score has no labelled Vocal bars.');
  const renderedClock = /^Q:/m.test(renderedScore);
  const bars = scoreBarClock(renderedClock ? renderedScore : fullScore);
  if (bars.length !== totalBars) throw new Error('Rendered and timed scores have different Vocal bar counts.');
  const meter = bars[0].meter;
  const secondsPerBar = bars[0].end - bars[0].start;
  const tags = lyricBlocks(lyrics);
  let nextTag = 0;
  const usedTags = new Set<number>();
  const rows = sections.map((section, index): CoverDriftSection => {
    const endBar = index + 1 < sections.length ? sections[index + 1].startBar - 1 : totalBars;
    const match = instrumental(section.label) ? -1 : tags.findIndex((tag, tagIndex) =>
      tagIndex >= nextTag && tag.hasLyrics && !instrumental(tag.label) &&
      sectionKind(tag.label) === sectionKind(section.label));
    const block = match < 0 ? undefined : tags[match];
    if (match >= 0) { nextTag = match + 1; usedTags.add(match); }
    const aligned = block ? alignedBlock(block, words) : null;
    const stem = block && stemWords ? alignedBlock(block, stemWords) : null;
    const expected = { start: bars[section.startBar - 1].start, end: bars[endBar - 1].end };
    const disagreementBars = aligned?.sung && stem?.sung
      ? Math.abs(scoreBarPosition(bars, aligned.sung.start) - scoreBarPosition(bars, stem.sung.start)) : null;
    const unscoredReason = !block ? 'no_matching_lyric_tag' :
      aligned?.reason ?? stem?.reason ??
      (disagreementBars !== null && disagreementBars > 0.5 ? 'mix_stem_disagreement' : null);
    const sung = unscoredReason ? null : aligned?.sung ?? null;
    return { scoreLabel: section.label, lyricTag: block?.label ?? null, startBar: section.startBar,
      endBar, expected, sung, offsetBars: sung ? scoreBarPosition(bars, sung.start) - (section.startBar - 1) : null,
      unscoredReason, disagreementBars };
  });
  const sungLyricBlocks = tags.filter(tag => tag.hasLyrics).length;
  const unmatchedLyricBlocks = tags.flatMap((tag, index) =>
    tag.hasLyrics && !usedTags.has(index)
      ? [{ index: index + 1, label: tag.label }] : []);
  const unmatchedScoreSections = rows.filter(row => row.unscoredReason === 'no_matching_lyric_tag').length;
  const measured = rows.filter(row => row.offsetBars !== null);
  const meanAbsoluteOffsetBars = measured.length
    ? measured.reduce((sum, row) => sum + Math.abs(row.offsetBars!), 0) / measured.length : null;
  const first = rows.findIndex(row => row.offsetBars !== null && Math.abs(row.offsetBars) > 1);
  return { tempoBpm: bars[0].bpm, meter, secondsPerBar, stemChecked: stemWords !== undefined,
    tempoSource: renderedClock ? 'rendered-score' : 'source-score-fallback',
    sectionWarning: unmatchedScoreSections || unmatchedLyricBlocks.length
      ? `${unmatchedScoreSections} score sections lack a matching sung tag; ${unmatchedLyricBlocks.length} sung lyric tags unused.`
      : null, sungLyricBlocks, unmatchedLyricBlocks, sections: rows, meanAbsoluteOffsetBars,
    firstOverOneBar: first < 0 ? null : { index: first + 1, label: rows[first].scoreLabel,
      offsetBars: rows[first].offsetBars! } };
}
