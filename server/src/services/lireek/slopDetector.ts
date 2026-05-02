// slopDetector.ts — AI-Slop Detection and Prevention System
//
// 7-layer defense to ensure generated lyrics feel authentic.
// Direct port from Python slop_detector.py — all patterns preserved.
// Layer 7 added based on corpus analysis of 608 generations (2026-05-02).

// ── Layer 1 — Blacklisted Words ─────────────────────────────────────────────

export const BLACKLISTED_WORDS = new Set([
  // Visual clichés
  'neon', 'streetlights', 'streetlight', 'silhouette', 'silhouettes',
  'tapestry', 'mosaic', 'kaleidoscope', 'prism',
  // Action clichés
  'yearning', 'beckons', 'beckoning', 'cascading', 'cascade',
  'unfurling', 'unfurl',
  // Emotional clichés
  'bittersweet', 'melancholy', 'wistful', 'poignant', 'ethereal', 'ephemeral',
  // Abstract concepts
  'symphony', 'harmonize', 'harmonizing', 'crossroads',
  // Overused metaphors
  'phoenix', 'labyrinth', 'soaring',
  // Generic intensity
  'pulsing', 'pulsating', 'throbbing', 'vibrant', 'vivid', 'luminous',
  'radiant', 'shimmering',
  // Time clichés
  'hourglass', 'timeless',
  // Nature clichés
  'tempest',
  // Existential clichés
  'essence', 'consciousness', 'realm', 'dimension',
  // Modern AI clichés (2024-2026 patterns)
  'unraveling', 'unravel', 'ember', 'embers', 'ignite', 'ignites',
  'resonate', 'resonates', 'reverberate', 'reverberates',
  // Faux-poetic
  'amidst', 'entwined', 'intertwined', 'ablaze',
  // Cosmic clichés
  'constellation', 'constellations', 'cosmos', 'infinite', 'infinity', 'void',
  // Over-emotional constructions
  'shattering', 'hollowed',
  // Synesthesia clichés
  'crimson sky', 'velvet night',
  // Generic AI title/filler words
  'static', 'catalyst', 'paradox', 'paradigm', 'mantra', 'epitome',
  'chronicles', 'solace', 'juxtaposition', 'serenity', 'resilience',
  'dichotomy', 'transcend', 'transcendence', 'metamorphosis', 'pinnacle',
  // Lighting clichés
  'fluorescent', 'halogen',
  // Corpus-analysis additions (2026-05-02) — overused across 608 generations
  'wreckage', 'jagged', 'bitter', 'hollow',
  'steel', 'metal', 'transmission', 'dashboard', 'gears', 'gloom',
]);

// ── Layer 1b — Overused Words (soft-ban: penalized per-occurrence, not banned) ──
// These words aren't inherently bad but the model leans on them like a crutch.
// "heavy" alone appeared 811 times across 64.3% of 608 songs.

export const OVERUSED_WORDS = new Set([
  'heavy', 'broken', 'cold', 'dust', 'ghost', 'machine',
  'nothing', 'nowhere', 'searching', 'wreckage', 'losing',
]);

// ── Layer 2 — Blacklisted Phrases ───────────────────────────────────────────

export const BLACKLISTED_PHRASES = new Set([
  'reaching up to the sky', 'beneath the streetlights', 'under the streetlights',
  'neon lights', 'neon glow', 'neon dreams', 'echoes in the night',
  'whispers in the dark', 'shadows dance', 'dancing shadows',
  'tapestry of dreams', 'symphony of', 'kaleidoscope of',
  'mosaic of emotions', 'bittersweet memories', 'fleeting moments',
  'sands of time', 'rising from the ashes', 'like a phoenix',
  'tangled web', 'labyrinth of', 'journey begins', 'path ahead',
  'crossroads of', 'fabric of reality', 'threads of fate',
  'ocean of tears', 'sea of faces', 'waves of emotion',
  'storm within', 'tempest raging', 'essence of', 'realm of',
  'universe within', 'consciousness expands', 'vivid dreams',
  'radiant light', 'pulsing with', 'cascading down',
  'ethereal beauty', 'melancholy mood', 'beckons me', 'yearning for',
  // Expanded set
  'paint the sky', 'written in the stars', 'dance with the devil',
  'scream into the void', 'drown in your eyes', 'heart on my sleeve',
  'break the chains', 'find my voice', 'lost in the moment',
  'through the fire', 'edge of forever', 'weight of the world',
  'paint a picture', 'piece by piece', 'shattered glass',
  'hollow eyes', 'burning bridges', 'chase the sun',
  'bleeding heart', 'silent scream', 'torn apart',
  'crumbling walls', 'whisper your name', 'dust settles',
  'ghost of you', 'ashes to ashes', 'taste of freedom',
  'colors of the wind', 'sound of silence',
  'in this moment', 'against the tide', 'into the unknown',
  'carry the weight', 'unravel the truth', 'embers glow',
  'spark ignites', 'constellations align', 'resonates within',
  'let it all go', 'rise above it all',
  // Corpus-analysis additions (2026-05-02)
  'nothing left', 'nowhere left', 'nothing left to',
  'the weight of', 'the wreckage', 'same old',
  'every single', 'cold and heavy', 'cold and dark',
  'cold and empty', 'heavy and cold', 'heavy and dark',
  'pulling me down', 'dragging me down',
]);

// ── Layer 3 — Regex Patterns ────────────────────────────────────────────────

const AI_PATTERNS = [
  /\b(tapestry|fabric|symphony|kaleidoscope|mosaic|labyrinth|maze|void)\s+of\s+\w+\b/gi,
  /\blike\s+a\s+(phoenix|symphony|kaleidoscope|constellation|ember)\b/gi,
  /\b\w+ing\s+\w+ing\b/gi,
  /\bthe\s+\w+\s+of\s+my\s+\w+\b/gi,
  /^In\s+the\s+(darkness|silence|shadows|distance|stillness|emptiness)\b/gim,
  // Corpus-analysis additions — "cold and [X]" / "heavy [noun]" overuse patterns
  /\bcold\s+and\s+\w+\b/gi,
  /\bheavy\s+(weight|hand|heart|air|load|chain|sky|dust|night|crown|veil|rain|door|stone|iron|fog|clouds?)\b/gi,
];

// ── Layer 4 — Structural Analysis ───────────────────────────────────────────

const FUNCTION_WORDS = new Set([
  'the', 'a', 'an', 'of', 'in', 'to', 'and', 'is', 'it', 'for', 'on',
  'with', 'as', 'at', 'by', 'from', 'or', 'but', 'not', 'be', 'are',
  'was', 'were', 'been', 'this', 'that', 'which', 'who', 'what', 'if',
  'so', 'my', 'your', 'we', 'they', 'i', 'you', 'he', 'she', 'me', 'us',
]);

function extractWords(text: string): string[] {
  return (text.toLowerCase().match(/\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b/g) ?? []);
}

function analyzeStructure(lines: string[]): { issues: string[]; score: number } {
  const issues: string[] = [];
  let score = 0;
  if (lines.length < 4) return { issues, score };

  const wordCounts = lines.filter(l => l.trim()).map(l => l.split(/\s+/).length);
  if (!wordCounts.length) return { issues, score };

  const mean = wordCounts.reduce((a, b) => a + b, 0) / wordCounts.length;
  const variance = wordCounts.reduce((a, wc) => a + (wc - mean) ** 2, 0) / wordCounts.length;
  const stddev = Math.sqrt(variance);

  if (wordCounts.length >= 6 && stddev < 1.0) {
    issues.push(`Line lengths are suspiciously uniform (stddev=${stddev.toFixed(2)})`);
    score += 10;
  }

  // First word repetition
  const firstWords = lines.filter(l => l.trim()).map(l => l.trim().split(/\s+/)[0]?.toLowerCase()).filter(Boolean);
  if (firstWords.length) {
    const freq = new Map<string, number>();
    for (const w of firstWords) freq.set(w, (freq.get(w) ?? 0) + 1);
    let maxWord = '', maxCount = 0;
    for (const [w, c] of freq) { if (c > maxCount) { maxWord = w; maxCount = c; } }
    const ratio = maxCount / firstWords.length;
    if (ratio > 0.5 && maxCount >= 3) {
      issues.push(`Over-repetitive line starter '${maxWord}' (${maxCount}/${firstWords.length} = ${(ratio * 100).toFixed(0)}%)`);
      score += 8;
    }
  }

  return { issues, score };
}

function detectAnomalies(text: string): { issues: string[]; score: number } {
  const issues: string[] = [];
  let score = 0;
  const words = extractWords(text);
  if (words.length < 20) return { issues, score };

  const freq = new Map<string, number>();
  for (const w of words) freq.set(w, (freq.get(w) ?? 0) + 1);
  const unique = freq.size;

  // Hapax ratio
  let hapax = 0;
  for (const c of freq.values()) if (c === 1) hapax++;
  const hapaxRatio = unique ? hapax / unique : 0;
  if (hapaxRatio > 0.85 && unique > 20) {
    issues.push(`Very high hapax ratio (${hapaxRatio.toFixed(2)}) — too many unique-once words`);
    score += 6;
  }

  // Function word density
  let funcCount = 0;
  for (const fw of FUNCTION_WORDS) funcCount += freq.get(fw) ?? 0;
  const funcRatio = words.length ? funcCount / words.length : 0;
  if (funcRatio > 0.38) {
    issues.push(`High function word density (${funcRatio.toFixed(2)}) — reads more like prose than lyrics`);
    score += 5;
  }

  // Line opener variety
  const lyricLines = text.split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('[') && !l.startsWith('('));
  if (lyricLines.length >= 6) {
    const openers = lyricLines.map(l => l.split(/\s+/)[0]?.toLowerCase()).filter(Boolean);
    const openerUnique = new Set(openers).size / openers.length;
    if (openerUnique < 0.3) {
      issues.push(`Very low line opener variety (${openerUnique.toFixed(2)})`);
      score += 5;
    } else if (openerUnique > 0.95 && openers.length > 8) {
      issues.push('Suspiciously perfect line opener variety — real lyrics naturally repeat some starters');
      score += 3;
    }
  }

  return { issues, score };
}

// ── Main Scan Function ──────────────────────────────────────────────────────

export interface SlopScanResult {
  ai_score: number;
  severity: 'high' | 'medium' | 'low';
  is_likely_ai: boolean;
  layers: {
    blacklisted_words: { score: number; found: string[] };
    blacklisted_phrases: { score: number; found: string[] };
    pattern_matches: { score: number; found: [string, string][] };
    structural: { score: number; raw_score: number; issues: string[] };
    fingerprint: { score: number; raw_score: number; issues: string[] };
    statistical: { score: number; raw_score: number; issues: string[] };
    overuse: { score: number; found: { word: string; count: number }[] };
  };
}

export function scanForSlop(
  text: string,
  fingerprint?: Record<string, any> | null,
  statisticalWeight = 1.0,
): SlopScanResult {
  // Strip section tags and performance notes
  const clean = text.replace(/\[.*?\]/g, '').replace(/\(.*?\)/g, '');
  const words = extractWords(clean);
  const lines = clean.split('\n').map(l => l.trim()).filter(l => l.length > 1);

  // Layer 1: Blacklisted words
  const badWords = words.filter(w => BLACKLISTED_WORDS.has(w));
  const l1Score = badWords.length * 10;

  // Layer 2: Blacklisted phrases
  const textLower = clean.toLowerCase();
  const badPhrases = [...BLACKLISTED_PHRASES].filter(p => textLower.includes(p));
  const l2Score = badPhrases.length * 20;

  // Layer 3: Regex patterns
  const badPatterns: [string, string][] = [];
  for (const pat of AI_PATTERNS) {
    pat.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = pat.exec(clean)) !== null) {
      badPatterns.push([pat.source, m[0]]);
    }
  }
  const l3Score = badPatterns.length * 5;

  // Layer 4: Structural
  const l4 = analyzeStructure(lines);

  // Layer 5: Fingerprint (basic comparison if provided)
  const l5 = { issues: [] as string[], score: 0 };
  if (fingerprint) {
    const safeVocab = new Set<string>(fingerprint.safe_vocabulary ?? []);
    if (safeVocab.size) {
      const wordSet = new Set(words);
      let overlap = 0;
      for (const w of wordSet) if (safeVocab.has(w)) overlap++;
      const overlapRatio = wordSet.size ? overlap / wordSet.size : 0;
      if (overlapRatio < 0.5) {
        l5.issues.push(`Low vocabulary overlap with source artist (${(overlapRatio * 100).toFixed(0)}% vs expected ≥50%)`);
        l5.score += 12;
      }
    }
    const artistTtr = fingerprint.type_token_ratio;
    if (artistTtr != null) {
      const genTtr = words.length ? new Set(words).size / words.length : 0;
      const diff = Math.abs(genTtr - artistTtr);
      if (diff > 0.15) {
        l5.issues.push(`TTR mismatch: generated=${genTtr.toFixed(3)}, artist=${artistTtr.toFixed(3)}`);
        l5.score += 8;
      }
    }
  }

  // Layer 6: Statistical anomalies
  const l6 = detectAnomalies(clean);

  // Layer 7: Overused word detection (soft-ban)
  // Penalize +3 per occurrence after the first — these aren't banned but the model
  // uses them as a crutch across genres ("heavy" alone: 811x in 608 songs).
  const overuseFound: { word: string; count: number }[] = [];
  let l7Score = 0;
  const wordFreq = new Map<string, number>();
  for (const w of words) {
    if (OVERUSED_WORDS.has(w)) wordFreq.set(w, (wordFreq.get(w) ?? 0) + 1);
  }
  for (const [word, count] of wordFreq) {
    if (count > 1) {
      const penalty = (count - 1) * 3; // +3 per occurrence after the first
      l7Score += penalty;
      overuseFound.push({ word, count });
    }
  }

  const sw = Math.max(0, Math.min(1, statisticalWeight));
  const total = l1Score + l2Score + l3Score +
    Math.floor(l4.score * sw) + Math.floor(l5.score * sw) + Math.floor(l6.score * sw) +
    l7Score;

  return {
    ai_score: total,
    severity: total > 30 ? 'high' : total > 15 ? 'medium' : 'low',
    is_likely_ai: total > 15,
    layers: {
      blacklisted_words: { score: l1Score, found: [...new Set(badWords)] },
      blacklisted_phrases: { score: l2Score, found: badPhrases },
      pattern_matches: { score: l3Score, found: badPatterns.slice(0, 10) },
      structural: { score: Math.floor(l4.score * sw), raw_score: l4.score, issues: l4.issues },
      fingerprint: { score: Math.floor(l5.score * sw), raw_score: l5.score, issues: l5.issues },
      statistical: { score: Math.floor(l6.score * sw), raw_score: l6.score, issues: l6.issues },
      overuse: { score: l7Score, found: overuseFound },
    },
  };
}
