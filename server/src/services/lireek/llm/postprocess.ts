// llm/postprocess.ts — Lyric post-processing pipeline (ported from HOT-Step 9000)

/** Strip thinking/reasoning blocks from LLM output */
export function stripThinkingBlocks(text: string): string {
  let result = text.replace(/<think>[\s\S]*?<\/think>/g, '');
  result = result.replace(/<analysis>[\s\S]*?<\/analysis>/g, '');
  result = result.replace(/<reasoning>[\s\S]*?<\/reasoning>/g, '');
  result = result.replace(/<reflection>[\s\S]*?<\/reflection>/g, '');
  result = result.replace(/<thought>[\s\S]*?<\/thought>/g, '');
  result = result.replace(/<\|channel>thought[\s\S]*?<channel\|>/g, '');
  result = result.replace(/<(?:think|analysis|reasoning|reflection|thought)>[\s\S]*/g, '');
  result = result.replace(/<\|channel>thought[\s\S]*/g, '');
  const cotMatch = result.match(/^(?:\s*\*+\s*)?(?:Thinking Process|Thought Process|Thinking|Reasoning):\s*[\s\S]*?(?:---|[*]{3,}|={3,})\s*/i);
  if (cotMatch) result = result.slice(cotMatch[0].length);
  return result.trim();
}

const SECTION_KEYWORDS = [
  'Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus',
  'Bridge', 'Interlude', 'Outro', 'Hook', 'Refrain',
];

export const SECTION_LINE_RE = new RegExp(
  '^\\[?(' + SECTION_KEYWORDS.map(k => k.replace(/[-/]/g, '\\$&')).join('|') + ')\\s*(\\d*)\\]?\\s*$',
  'i'
);

const PUNCTUATION_ENDINGS = new Set('.,!?;:-…)"\'');

export function postprocessLyrics(text: string): string {
  const resultLines: string[] = [];
  for (const line of text.split('\n')) {
    const stripped = line.trim();
    if (!stripped) { resultLines.push(''); continue; }
    const m = SECTION_LINE_RE.exec(stripped);
    if (m) {
      const sectionName = m[1].charAt(0).toUpperCase() + m[1].slice(1).toLowerCase();
      const sectionNum = m[2];
      resultLines.push(sectionNum ? `[${sectionName} ${sectionNum}]` : `[${sectionName}]`);
      continue;
    }
    if (/^\[.+\]$/.test(stripped)) { resultLines.push(stripped); continue; }
    if (stripped && !PUNCTUATION_ENDINGS.has(stripped[stripped.length - 1])) {
      resultLines.push(stripped + ',');
    } else {
      resultLines.push(stripped);
    }
  }
  return resultLines.join('\n');
}

export function fixSectionLabels(text: string): string {
  const INVALID_TO_VALID: Record<string, string> = {
    'x': 'Interlude', 'breakdown': 'Bridge', 'drop': 'Chorus',
    'solo': 'Interlude', 'hook': 'Chorus', 'rap': 'Verse', 'spoken': 'Verse',
  };
  const lines = text.split('\n');
  const result: string[] = [];
  const sectionHeaders: { lineIdx: number; header: string }[] = [];
  for (const line of lines) {
    const stripped = line.trim();
    const m = stripped.match(/^\[(.+?)(?:\s+\d+)?\]$/);
    if (m) {
      let label = m[1].trim().toLowerCase();
      let newStripped = stripped;
      if (INVALID_TO_VALID[label]) {
        const newLabel = INVALID_TO_VALID[label];
        const numMatch = stripped.match(/\d+/);
        newStripped = numMatch ? `[${newLabel} ${numMatch[0]}]` : `[${newLabel}]`;
      }
      sectionHeaders.push({ lineIdx: result.length, header: newStripped });
      result.push(newStripped);
    } else {
      result.push(stripped.startsWith('[') && stripped.endsWith(']') ? stripped : line);
    }
  }
  const bridgeIndices = sectionHeaders.map((h, i) => ({ i, h })).filter(x => x.h.header.toLowerCase().includes('bridge'));
  const chorusExists = sectionHeaders.some(h => h.header.toLowerCase().includes('chorus'));
  if (!chorusExists && bridgeIndices.length >= 2) {
    for (const bi of bridgeIndices.slice(0, -1)) {
      result[sectionHeaders[bi.i].lineIdx] = '[Chorus]';
    }
  }
  return result.join('\n');
}

// Musical phrases resolve in even numbers of lines, so odd-length verses and
// choruses (5, 7, 9 lines) clash with the music model's phrasing. Rather than
// forcing every section to exactly 4/8 lines (the old behaviour, which deleted
// up to 3 lines of content), only trim ONE line when the count is odd.
// Even counts of any reasonable size pass through untouched.
export function enforceLineCounts(text: string): string {
  const sections: { header: string; lines: string[] }[] = [];
  let currentHeader = '';
  let currentLines: string[] = [];
  for (const line of text.split('\n')) {
    const stripped = line.trim();
    if (/^\[.+\]$/.test(stripped)) {
      if (currentHeader || currentLines.length) sections.push({ header: currentHeader, lines: currentLines });
      currentHeader = stripped;
      currentLines = [];
    } else { currentLines.push(line); }
  }
  if (currentHeader || currentLines.length) sections.push({ header: currentHeader, lines: currentLines });
  const resultParts: string[] = [];
  for (const { header, lines } of sections) {
    const lyricLines = lines.filter(l => l.trim());
    const count = lyricLines.length;
    const headerLower = header.toLowerCase();
    const isVerse = headerLower.includes('verse');
    const isChorus = headerLower.includes('chorus') || headerLower.includes('hook');
    let target: number | null = null;
    if ((isVerse || isChorus) && count >= 3 && count % 2 !== 0) target = count - 1;
    let finalLines = lines;
    if (target !== null && target < count) {
      let kept = 0; finalLines = [];
      for (const l of lines) {
        if (l.trim()) { if (kept < target) { finalLines.push(l); kept++; } }
        else { if (kept < target) finalLines.push(l); }
      }
    }
    if (header) resultParts.push(header);
    resultParts.push(...finalLines);
  }
  return resultParts.join('\n');
}

const BAD_A_PREFIX_RE = /\ba-(?!\w+ing\b)(?!\w+in'\b)/gi;
export function fixAPrefix(text: string): string { return text.replace(BAD_A_PREFIX_RE, ''); }

export function stripLyricQuotes(text: string): string { return text.replace(/'[^']{4,}'/g, '[quote removed]'); }

export function estimateDuration(lyrics: string, bpm: number): number {
  if (!lyrics.trim() || bpm <= 0) return 0;
  const barDuration = 240.0 / Math.max(bpm, 40);
  const lines = lyrics.trim().split('\n');
  let sectionCount = 0, lyricLineCount = 0;
  for (const line of lines) {
    const stripped = line.trim();
    if (!stripped) continue;
    if (SECTION_LINE_RE.test(stripped) || (stripped.startsWith('[') && stripped.endsWith(']'))) sectionCount++;
    else lyricLineCount++;
  }
  return Math.max(90, Math.min(Math.floor(lyricLineCount * 3.5 + Math.max(sectionCount - 1, 0) * 4 * barDuration), 360));
}

// selectBestBlueprint was removed — structure selection now lives in
// ../prompts.ts (pickBlueprint), which samples from the artist's observed
// blueprints instead of deterministically picking the same one every time.
