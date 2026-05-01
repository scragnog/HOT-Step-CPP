/**
 * estimateDuration.ts — BPM + lyric-aware duration estimation.
 *
 * Mirrors the server-side estimateDuration() in llmService.ts.
 * Used when the "Use LLM Duration" setting is disabled to override
 * potentially inaccurate LLM duration estimates with a heuristic
 * calculation based on actual lyric content and tempo.
 */

const SECTION_RE = /^\[.+\]$/;

export function estimateDuration(lyrics: string, bpm: number): number {
  if (!lyrics.trim() || bpm <= 0) return 0;

  const barDuration = 240.0 / Math.max(bpm, 40);
  const lines = lyrics.trim().split('\n');
  let sectionCount = 0;
  let lyricLineCount = 0;

  for (const line of lines) {
    const stripped = line.trim();
    if (!stripped) continue;
    if (SECTION_RE.test(stripped)) {
      sectionCount++;
    } else {
      lyricLineCount++;
    }
  }

  const vocalSeconds = lyricLineCount * 3.5;
  const breakSeconds = Math.max(sectionCount - 1, 0) * 4 * barDuration;
  const estimated = Math.floor(vocalSeconds + breakSeconds);

  return Math.max(90, Math.min(estimated, 360));
}

/** localStorage key for the "Use LLM Duration" toggle */
export const LLM_DURATION_KEY = 'lireek-useLlmDuration';

/** Read the persisted preference (defaults to true = use LLM duration) */
export function getUseLlmDuration(): boolean {
  try {
    const raw = localStorage.getItem(LLM_DURATION_KEY);
    return raw !== null ? JSON.parse(raw) : true;
  } catch {
    return true;
  }
}

/**
 * Resolve the duration for a generation: returns the LLM's duration
 * if the setting is enabled, otherwise recalculates from lyrics + BPM.
 */
export function resolveDuration(
  llmDuration: number | undefined,
  lyrics: string,
  bpm: number,
  fallback = 180,
): number {
  if (getUseLlmDuration() && llmDuration && llmDuration > 0) {
    return llmDuration;
  }
  const estimated = estimateDuration(lyrics, bpm);
  return estimated > 0 ? estimated : fallback;
}
