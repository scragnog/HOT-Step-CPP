import type { Yue2AlignWord } from './align.js';
import { classifyYue2Score, lyricSectionTags } from './scoreHealth.js';
import { lintScoreSections, scoreSections } from './scoreSections.js';

export interface CoverDriftSection {
  scoreLabel: string;
  lyricTag: string | null;
  startBar: number;
  endBar: number;
  expected: { start: number; end: number };
  sung: { start: number; end: number } | null;
  offsetBars: number | null;
}

export interface CoverDriftResult {
  tempoBpm: number;
  meter: string;
  secondsPerBar: number;
  tempoSource: 'rendered-score' | 'source-score-fallback';
  sectionWarning: string | null;
  sections: CoverDriftSection[];
  meanAbsoluteOffsetBars: number | null;
  firstOverOneBar: { index: number; label: string; offsetBars: number } | null;
}

function scoreClock(score: string): { bpm: number; beat: number } | null {
  const field = score.match(/^Q:\s*(?:(\d+)\s*\/\s*(\d+)\s*=\s*)?(\d+(?:\.\d+)?)\s*$/m);
  if (!field) return null;
  const beat = field[1] ? Number(field[1]) / Number(field[2]) : 1 / 4;
  const bpm = Number(field[3]);
  return beat > 0 && Number.isFinite(beat) && bpm > 0 && Number.isFinite(bpm) ? { bpm, beat } : null;
}

function lyricBlocks(lyrics: string): Array<{ label: string; start: number; end: number }> {
  // The same S10 tag matcher decides which lines are section headers. Work in
  // codepoints because MMS_FA's char0/char1 offsets are not UTF-16 indices.
  const normalized = lyrics.replace(/\r\n?/g, '\n');
  const lines = normalized.split('\n');
  const length = Array.from(normalized).length;
  const blocks: Array<{ label: string; start: number; end: number }> = [];
  let offset = 0;
  for (const line of lines) {
    const label = lyricSectionTags(line)[0];
    if (label) {
      if (blocks.length) blocks[blocks.length - 1].end = offset;
      blocks.push({ label, start: offset, end: length });
    }
    offset += Array.from(line).length + 1;
  }
  return blocks;
}

/** Compare the score's bar grid with the rendered words, in section order. */
export function measureCoverDrift(
  renderedScore: string, fullScore: string, lyrics: string, words: Yue2AlignWord[],
): CoverDriftResult {
  const sections = scoreSections(renderedScore);
  const totalBars = classifyYue2Score(renderedScore).bars;
  if (!sections.length || !totalBars) throw new Error('The rendered score has no labelled Vocal bars.');
  const meterField = renderedScore.match(/^M:\s*(\d+)\s*\/\s*(\d+)\s*$/m);
  if (!meterField || !Number(meterField[2])) throw new Error('The rendered score has no supported M: meter.');
  const meter = `${meterField[1]}/${meterField[2]}`;
  const barWholeNotes = Number(meterField[1]) / Number(meterField[2]);
  const renderedClock = scoreClock(renderedScore);
  const clock = renderedClock ?? scoreClock(fullScore);
  if (!clock) throw new Error('Neither rendered nor reviewed score has a usable Q: tempo.');
  const secondsPerBar = barWholeNotes / clock.beat * 60 / clock.bpm;
  const tags = lyricBlocks(lyrics);
  const lint = lintScoreSections(sections, lyrics);
  const rows = sections.map((section, index): CoverDriftSection => {
    const endBar = index + 1 < sections.length ? sections[index + 1].startBar - 1 : totalBars;
    const block = tags[index];
    const timed = block ? words.filter(word => word.char0 >= block.start && word.char0 < block.end &&
      Number.isFinite(word.start) && Number.isFinite(word.end)) : [];
    const sung = timed.length ? { start: Math.min(...timed.map(word => word.start)),
      end: Math.max(...timed.map(word => word.end)) } : null;
    const expected = { start: (section.startBar - 1) * secondsPerBar, end: endBar * secondsPerBar };
    return { scoreLabel: section.label, lyricTag: block?.label ?? null, startBar: section.startBar,
      endBar, expected, sung, offsetBars: sung ? (sung.start - expected.start) / secondsPerBar : null };
  });
  const measured = rows.filter(row => row.offsetBars !== null);
  const meanAbsoluteOffsetBars = measured.length
    ? measured.reduce((sum, row) => sum + Math.abs(row.offsetBars!), 0) / measured.length : null;
  const first = rows.findIndex(row => row.offsetBars !== null && Math.abs(row.offsetBars) > 1);
  return { tempoBpm: clock.bpm, meter, secondsPerBar,
    tempoSource: renderedClock ? 'rendered-score' : 'source-score-fallback',
    sectionWarning: lint.ok ? null : lint.message, sections: rows, meanAbsoluteOffsetBars,
    firstOverOneBar: first < 0 ? null : { index: first + 1, label: rows[first].scoreLabel,
      offsetBars: rows[first].offsetBars! } };
}
