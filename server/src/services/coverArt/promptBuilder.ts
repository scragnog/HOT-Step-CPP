// promptBuilder.ts — Build text-to-image prompts from song metadata
//
// Ported from HOT-Step 9000's acestep/core/cover_art.py
//
// When `subject` is provided (from Lireek metadata), it's used as the
// primary prompt for more evocative imagery. Otherwise falls back to
// keyword extraction from lyrics.

const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
  'of', 'with', 'by', 'from', 'is', 'it', 'its', 'are', 'was', 'were',
  'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
  'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
  'not', 'no', 'so', 'if', 'up', 'out', 'just', 'like', 'my', 'me',
  'we', 'you', 'your', 'they', 'them', 'he', 'she', 'her', 'his',
  'i', 'im', 'ive', 'dont', 'that', 'this', 'all', 'got', 'get',
  'when', 'what', 'where', 'how', 'why', 'oh', 'yeah', 'ya', 'na',
  'la', 'da', 'uh', 'ah', 'ooh', 'hey', 'go', 'know', 'come', 'take',
  'make', 'see', 'let', 'say', 'one', 'way', 'back', 'now',
  'more', 'than', 'into', 'over', 'down', 'been',
]);

/** Extract the most common meaningful words from lyrics. */
export function extractThemeKeywords(lyrics: string, maxKeywords = 5): string[] {
  if (!lyrics?.trim()) return [];

  // Strip section headers like [Verse 1]
  let cleaned = lyrics.replace(/\[.*?\]/g, '');
  // Remove punctuation, lowercase
  cleaned = cleaned.replace(/[^\w\s]/g, '').toLowerCase();

  const words = cleaned.split(/\s+/)
    .filter(w => w.length > 3 && !STOP_WORDS.has(w));

  if (words.length === 0) return [];

  // Count frequencies
  const freq: Record<string, number> = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;

  // Sort by frequency, take top N
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxKeywords)
    .map(([word]) => word);
}

export interface CoverArtPromptOpts {
  title?: string;
  style?: string;
  lyrics?: string;
  subject?: string;
}

/**
 * Build a text-to-image prompt from song metadata.
 *
 * Keeps total prompt under ~60 words for CLIP's 77-token limit.
 */
export function buildCoverArtPrompt(opts: CoverArtPromptOpts): string {
  const parts: string[] = [];

  if (opts.subject?.trim()) {
    // Rich subject path: use the curated description directly
    parts.push(opts.subject.trim());

    // Add a couple of genre words for visual tone
    if (opts.style) {
      const styleWords = opts.style.split(',').map(w => w.trim()).filter(Boolean);
      const shortStyle = styleWords.slice(0, 2).join(', ');
      if (shortStyle) parts.push(shortStyle);
    }
  } else {
    // Fallback: keyword extraction
    parts.push('Album cover art');

    if (opts.style) {
      const styleWords = opts.style.split(',').map(w => w.trim()).filter(Boolean);
      const shortStyle = styleWords.slice(0, 3).join(', ');
      if (shortStyle) parts.push(`for a ${shortStyle} song`);
    }

    if (opts.title) {
      let cleanTitle = opts.title.trim();
      // Strip "Artist - " prefix if present
      if (cleanTitle.includes(' - ')) {
        cleanTitle = cleanTitle.split(' - ').pop()?.trim() || cleanTitle;
      }
      parts.push(`called "${cleanTitle}"`);
    }

    const keywords = extractThemeKeywords(opts.lyrics || '', 4);
    if (keywords.length > 0) {
      parts.push(`themes of ${keywords.join(', ')}`);
    }
  }

  // Art direction suffix (always)
  parts.push('digital art, cinematic, professional album artwork');

  return parts.join(', ');
}
