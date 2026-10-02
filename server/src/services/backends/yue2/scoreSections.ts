import { barsIn, isVocalVoice, lyricSectionTags, scoreBarSegments, withoutLyricSectionTags } from './scoreHealth.js';

export interface Yue2ScoreSection { label: string; startBar: number }

/** Count Vocal bars the same way scoreHealth does; Ins mirrors them. */
export function scoreSections(abc: string): Yue2ScoreSection[] {
  const sections: Yue2ScoreSection[] = [];
  let inVocal = false;
  let bars = 0;
  for (const raw of abc.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('%')) {
      const label = line.slice(1).trim();
      if (label) sections.push({ label, startBar: bars + 1 });
      continue;
    }
    if (/^V:/.test(line)) {
      inVocal = isVocalVoice(line);
      continue;
    }
    if (/^[A-Za-z]:/.test(line) || !inVocal) continue;
    for (const segment of scoreBarSegments(line)) bars += barsIn(segment);
  }
  return sections;
}

export interface Yue2SectionLint {
  ok: boolean;
  scoreCount: number;
  lyricCount: number;
  message: string;
}

export function lintScoreSections(sections: Yue2ScoreSection[], lyrics: string): Yue2SectionLint {
  const tags = lyricSectionTags(lyrics);
  const scoreCount = sections.length, lyricCount = tags.length;
  const mismatch = sections.findIndex((section, index) =>
    section.label.toLowerCase() !== (tags[index] ?? '').toLowerCase());
  const firstMismatch = mismatch >= 0 ? mismatch : tags.length > sections.length ? sections.length : -1;
  const countText = `score has ${scoreCount} section${scoreCount === 1 ? '' : 's'}, lyrics have ${lyricCount} tag${lyricCount === 1 ? '' : 's'}`;
  if (scoreCount === lyricCount && firstMismatch < 0) return { ok: true, scoreCount, lyricCount, message: 'Score sections match lyric tags.' };
  const detail = firstMismatch < 0 ? ''
    : `; score section ${firstMismatch + 1} is ${sections[firstMismatch]?.label ?? 'missing'}, lyric tag ${firstMismatch + 1} is ${tags[firstMismatch] ?? 'missing'}`;
  return { ok: false, scoreCount, lyricCount, message: countText + detail };
}

/** Place every lyric line below the first new tag for manual redistribution. */
export function insertScoreSectionTags(sections: Yue2ScoreSection[], lyrics: string): string {
  if (!sections.length) return lyrics;
  const body = withoutLyricSectionTags(lyrics).trim();
  const [first, ...rest] = sections.map(section => `[${section.label}]`);
  return `${first}${body ? `\n${body}` : ''}${rest.length ? `\n\n${rest.join('\n\n')}` : ''}`;
}
