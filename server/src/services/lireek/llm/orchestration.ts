// llm/orchestration.ts — High-level lyric generation and refinement functions

import * as slopDetector from '../slopDetector.js';
import {
  GENERATION_SYSTEM_PROMPT,
  SONG_METADATA_SYSTEM_PROMPT,
  REFINEMENT_SYSTEM_PROMPT,
  TITLE_DERIVATION_PROMPT
} from '../prompts.js';
import type { LyricsProfile } from '../profilerService.js';
import type { GenerationResponse, ChunkCallback } from './types.js';
import { getProvider } from './registry.js';
import {
  stripThinkingBlocks, postprocessLyrics, fixSectionLabels,
  enforceLineCounts, fixAPrefix, stripLyricQuotes,
  estimateDuration, selectBestBlueprint
} from './postprocess.js';

const BLUEPRINT_LABEL_NAMES: Record<string, string> = {
  V: 'Verse', C: 'Chorus', B: 'Bridge', PC: 'Pre-Chorus',
  POC: 'Post-Chorus', I: 'Intro', O: 'Outro', IL: 'Interlude',
};

// Words banned from titles — enforced programmatically since prompt-only bans leak
const BANNED_TITLE_WORDS = new Set([
  'glass', 'steel', 'plastic', 'concrete', 'midnight', 'mirror',
  'heavy', 'terminal', 'altar', 'confessional', 'ledger', 'gospel',
  'chrome', 'gilded', 'puppet', 'halo', 'protocol', 'eden', 'digital',
  'algorithm', 'code', 'circuit', 'grid', 'data', 'wire',
  'sanctuary', 'void', 'ethereal', 'neon', 'silhouette', 'static',
  'embers', 'fluorescent', 'shimmering', 'tapestry', 'weight',
  'skin', 'signal', 'platform',
]);

// Structural title formula patterns that produce repetitive titles
const BANNED_TITLE_PATTERNS: RegExp[] = [
  /^watch\s+(it|me|them|us|him|her)\b/i,
  /^let\s+it\s+(burn|fade|go|fall|bleed|break|rot|die|end)\b/i,
  /^burn\s+it\s+(all|down)\b/i,
  /^nothing\s+left/i,
  /^nowhere\s+left/i,
];

/**
 * Check if a title contains banned words or matches banned patterns.
 * Returns an array of issues found (empty = title is clean).
 */
function validateTitle(title: string): string[] {
  const issues: string[] = [];
  const words = title.toLowerCase().split(/\W+/).filter(Boolean);
  for (const w of words) {
    if (BANNED_TITLE_WORDS.has(w)) issues.push(`banned word: "${w}"`);
  }
  for (const pat of BANNED_TITLE_PATTERNS) {
    if (pat.test(title)) issues.push(`banned pattern: ${pat.source}`);
  }
  return issues;
}

/**
 * Append a unique nonce to a system prompt to bust oMLX's KV cache.
 * Without this, oMLX reuses cached prefix states for identical system prompts,
 * which can cause thinking models to skip reasoning on subsequent calls.
 */
function cacheBustPrompt(prompt: string): string {
  return `${prompt}\n\n(session ${Date.now()}-${Math.random().toString(36).slice(2, 8)})`;
}

async function planSongMetadata(
  profile: LyricsProfile,
  usedSubjects: string[],
  usedBpms: number[],
  usedKeys: string[],
  usedDurations: number[],
  providerName: string,
  modelName: string,
  onChunk?: ChunkCallback,
  userSubject?: string
): Promise<any> {
  const provider = getProvider(providerName);
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);
  if (profile.themes?.length) lines.push(`Themes: ${profile.themes.join(', ')}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
  if (profile.additional_notes) lines.push(`Additional notes: ${profile.additional_notes}`);
  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  if (profile.song_subjects && typeof profile.song_subjects === 'object') {
    lines.push('\nOriginal song subjects (for reference):');
    for (const [songTitle, subject] of Object.entries(profile.song_subjects)) {
      lines.push(`  • ${songTitle}: ${subject}`);
    }
  }
  if (profile.subject_categories?.length) {
    lines.push(`\nThematic categories: ${profile.subject_categories.join(', ')}`);
  }
  if (userSubject) {
    lines.push(`\nThe subject for this song has been chosen by the user: "${userSubject}"`);
    lines.push('Use this exact subject. Plan the BPM, key, caption, and duration to complement it.');
  } else {
    if (usedSubjects?.length) {
      lines.push('\nSubjects ALREADY USED (do NOT repeat these):');
      for (const s of usedSubjects) lines.push(`  ✗ ${s}`);
    }
  }
  if (usedKeys?.length) lines.push(`\nKeys ALREADY USED (try different ones): ${usedKeys.join(', ')}`);
  lines.push('\nPlan the metadata for the next song:');

  const prompt = lines.join('\n');
  console.log('[LLM] Planning song metadata via', providerName, modelName);
  const responseJsonStr = await provider.call(cacheBustPrompt(SONG_METADATA_SYSTEM_PROMPT), prompt, modelName, onChunk);
  const cleaned = stripThinkingBlocks(responseJsonStr);
  const cleanJson = cleaned.replace(/^```(?:json)?\s*|\s*```$/gm, '').trim();
  try {
    return JSON.parse(cleanJson);
  } catch (err) {
    const start = cleanJson.indexOf('{');
    if (start !== -1) {
      let depth = 0;
      for (let i = start; i < cleanJson.length; i++) {
        if (cleanJson[i] === '{') depth++;
        else if (cleanJson[i] === '}') { depth--; if (depth === 0) { try { return JSON.parse(cleanJson.slice(start, i + 1)); } catch {} break; } }
      }
    }
    console.error("Failed to parse metadata JSON:", cleanJson.slice(0, 300));
    return { subject: '', bpm: 0, key: '', caption: '', duration: 0 };
  }
}

function buildGenerationPrompt(profile: LyricsProfile, extraInstructions?: string, targetDuration?: number, bpm?: number): string {
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);

  lines.push('', '=== STYLISTIC PROFILE ===', '');
  lines.push(`Themes: ${(profile.themes || []).join(', ')}`);
  lines.push(`Common subjects / motifs: ${(profile.common_subjects || []).join(', ')}`);
  lines.push(`Rhyme schemes: ${(profile.rhyme_schemes || []).join(', ')}`);
  lines.push(`Average verse length: ${profile.avg_verse_lines} lines`);
  lines.push(`Average chorus length: ${profile.avg_chorus_lines} lines`);
  if (profile.vocabulary_notes) lines.push(`Vocabulary: ${stripLyricQuotes(profile.vocabulary_notes)}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${stripLyricQuotes(profile.tone_and_mood)}`);
  if (profile.structural_patterns) lines.push(`Structural patterns: ${stripLyricQuotes(profile.structural_patterns)}`);

  if (profile.structure_blueprints?.length) {
    const bp = selectBestBlueprint(profile.structure_blueprints);
    lines.push('', '=== SONG STRUCTURE (MANDATORY) ===');
    lines.push(`Blueprint: ${bp}`);
    const parts = bp.split('-');
    let verseNum = 0;
    const sectionList: string[] = [];
    for (const part of parts) {
      let name = BLUEPRINT_LABEL_NAMES[part] || part;
      if (part === 'V') { verseNum++; name = `Verse ${verseNum}`; }
      sectionList.push(`[${name}]`);
    }
    lines.push(`You MUST write these sections in this exact order: ${sectionList.join(' → ')}`);
    if (parts.includes('B')) lines.push("This artist uses bridges — you MUST include a [Bridge] section.");
  }

  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  const ms = profile.meter_stats;
  if (ms) {
    lines.push('', '=== LINE LENGTH & METER ===');
    lines.push(`Average: ~${ms.avg_syllables_per_line ?? '?'} syllables/line, ~${ms.avg_words_per_line ?? '?'} words/line`);
    lines.push(`Standard deviation: ±${ms.syllable_std_dev ?? '?'} syllables (VARY your line lengths!)`);
    const llv = ms.line_length_variation;
    if (llv?.histogram) {
      const histStr = Object.entries(llv.histogram).map(([k, v]) => `${k} syl: ${v}%`).join(', ');
      lines.push(`Syllable distribution: ${histStr}`);
      lines.push('Match this distribution — NOT all lines the same length!');
    }
  }

  const rs = profile.repetition_stats;
  if (rs) {
    lines.push('', '=== REPETITION & HOOKS ===');
    lines.push(`Chorus repetition: ${rs.chorus_repetition_pct ?? 0}% of chorus lines are repeats`);
    lines.push(`Pattern: ${rs.pattern || 'unknown'}`);
    if ((rs.chorus_repetition_pct ?? 0) >= 20) lines.push('You MUST use repeated lines in your chorus to create a hook effect.');
    if (rs.hook_examples?.length) lines.push(`Hook examples: ${rs.hook_examples.slice(0, 3).join('; ')}`);
  }

  const vs = profile.vocabulary_stats;
  if (vs) {
    lines.push('', '=== VOCABULARY ===');
    lines.push(`Level: ${vs.contraction_pct ?? 0}% contractions, ${vs.profanity_pct ?? 0}% profanity`);
    lines.push(`Type-token ratio: ${vs.type_token_ratio ?? '?'} (${vs.unique_words ?? '?'} unique / ${vs.total_words ?? '?'} total)`);
    if (vs.distinctive_words?.length) lines.push(`Use words like: ${vs.distinctive_words.slice(0, 10).join(', ')}`);
  }

  if (profile.rhyme_quality) {
    const rq = profile.rhyme_quality;
    const total = Object.values(rq).reduce((a, b) => a + b, 0);
    if (total > 0) {
      lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
    }
  }

  if (profile.narrative_techniques) lines.push(`Narrative techniques: ${stripLyricQuotes(profile.narrative_techniques)}`);
  if (profile.imagery_patterns) lines.push(`Imagery patterns: ${stripLyricQuotes(profile.imagery_patterns)}`);
  if (profile.signature_devices) lines.push(`Signature devices: ${stripLyricQuotes(profile.signature_devices)}`);
  if (profile.emotional_arc) lines.push(`Emotional arc: ${stripLyricQuotes(profile.emotional_arc)}`);

  if (profile.raw_summary) lines.push('', '=== PROSE SUMMARY ===', '', stripLyricQuotes(profile.raw_summary));
  if (extraInstructions) lines.push('', '=== EXTRA INSTRUCTIONS ===', '', extraInstructions);

  if (profile.representative_excerpts?.length) {
    lines.push('', '=== REPRESENTATIVE EXCERPTS (STYLE REFERENCE ONLY — DO NOT COPY) ===');
    lines.push(...profile.representative_excerpts.slice(0, 10).flatMap(e => [e, '---']));
  }

  if (targetDuration && targetDuration > 0 && bpm && bpm > 0) {
    const barSeconds = 240.0 / bpm;
    const totalBars = Math.round(targetDuration / barSeconds);
    // Reserve bars for instrumental gaps between sections (2-4 bars each)
    const blueprintParts = profile.structure_blueprints?.length
      ? selectBestBlueprint(profile.structure_blueprints).split('-')
      : ['I', 'V', 'C', 'V', 'C', 'B', 'C', 'O'];
    const sectionCount = blueprintParts.length;
    const transitionBars = (sectionCount - 1) * 3; // ~3 bars per transition
    const singableBars = totalBars - transitionBars;
    // At ~2 bars per lyric line (rough average), calculate max lyric lines
    const maxLyricLines = Math.max(12, Math.floor(singableBars / 2));
    const minutes = Math.floor(targetDuration / 60);
    const seconds = Math.round(targetDuration % 60);

    lines.push('', '=== DURATION BUDGET (CRITICAL — DO NOT EXCEED) ===');
    lines.push(`Target duration: ${targetDuration} seconds (${minutes}:${String(seconds).padStart(2, '0')})`);
    lines.push(`BPM: ${bpm} — one bar of 4/4 = ${barSeconds.toFixed(1)} seconds`);
    lines.push(`Total bars available: ~${totalBars} bars for the entire song`);
    lines.push(`After accounting for ~${transitionBars} bars of instrumental transitions between ${sectionCount} sections, you have ~${singableBars} singable bars.`);
    lines.push(`At roughly 2 bars per lyric line, aim for approximately ${maxLyricLines} total lyric lines (across ALL sections).`);
    lines.push('');
    lines.push('USE THIS TO DECIDE LINE COUNTS:');
    if (maxLyricLines <= 20) {
      lines.push('- This is a SHORT song. Use 4-line verses and 4-line choruses. Keep it tight.');
    } else if (maxLyricLines <= 32) {
      lines.push('- This is a STANDARD-length song. Use 4-line verses (or one 8-line verse). Choruses should be 4-6 lines.');
    } else {
      lines.push('- This is a LONGER song. You can use 8-line verses and 6-8 line choruses if the blueprint calls for it.');
    }
    lines.push(`- Count your total lyric lines before finalising. If you exceed ~${maxLyricLines} lines, the song will run over its target duration.`);
  }

  lines.push(
    '', '=== FINAL REMINDERS ===',
    '1. VERSE LINE COUNT: Exactly 4 or 8 lines per verse.',
    '2. CHORUS LINE COUNT: Exactly 4, 6, or 8 lines per chorus. Each chorus MUST have a hook line that repeats.',
    '3. *** ZERO TOLERANCE FOR COPYING ***',
    '4. NO SLOP: Do not use neon, fluorescent, embers, silhouette, static, void, ethereal, or any AI cliché.',
    '5. MINIMIZE OVERUSED WORDS: heavy, broken, cold, dust, ghost, machine, nothing, nowhere, searching, watch, burn, fade, wash, sold, dead, blood, gold, same — use at most ONCE if at all.',
    '6. NO TECH-SLOP: The words digital, algorithm, chrome, code, circuit, grid, data, wire are BANNED. Do not force tech/digital metaphors onto non-tech artists.',
    "7. VOCABULARY DIVERSITY: A Snoop Dogg song must NOT sound like a Joy Division song. Use THIS artist's actual vocabulary.",
    '8. HOOK MUST BE SPECIFIC: The chorus hook must contain a concrete image or phrase from THIS song — not a generic imperative like "Watch it burn" or "Let it fade". If the hook could fit in any song by any artist, rewrite it.',
    '',
    'Now write the song (lyrics only, starting with [Intro] or [Verse 1] — no title line):',
  );
  return lines.join('\n');
}

export async function generateLyricsStreaming(
  profile: LyricsProfile, providerName: string, model?: string,
  extraInstructions?: string, usedSubjects: string[] = [],
  usedBpms: number[] = [], usedKeys: string[] = [],
  usedTitles: string[] = [], usedDurations: number[] = [],
  onChunk?: ChunkCallback, onPhase?: (phase: string) => void,
  userSubject?: string
): Promise<GenerationResponse> {
  const provider = getProvider(providerName);
  const effectiveModel = model || provider.defaultModel;

  if (onPhase) onPhase("Planning song metadata…");
  let metadata = { subject: '', bpm: 0, key: '', caption: '', duration: 0 };
  if (profile.song_subjects || (profile.themes && profile.themes.length) || userSubject) {
    try {
      metadata = await planSongMetadata(profile, usedSubjects, usedBpms, usedKeys, usedDurations, providerName, effectiveModel, onChunk, userSubject);
      if (userSubject) metadata.subject = userSubject;
      console.log("Planned metadata:", metadata);
    } catch(e) { console.warn("Failed to plan metadata", e); }
  }
  if (userSubject && !metadata.subject) metadata.subject = userSubject;

  if (metadata.subject) extraInstructions = `The song must be about: ${metadata.subject}\n\n${extraInstructions || ''}`;
  if (onPhase) onPhase("Writing lyrics…");
  const userPrompt = buildGenerationPrompt(profile, extraInstructions, metadata.duration, metadata.bpm);
  let raw = await provider.call(cacheBustPrompt(GENERATION_SYSTEM_PROMPT), userPrompt, effectiveModel, onChunk);

  raw = stripThinkingBlocks(raw);
  raw = raw.replace(/<\|[a-z_]+\|>/g, '');
  raw = raw.replace(/\[?(System|User|Assistant)\]?:.*/gi, '');
  raw = raw.replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '');
  raw = raw.replace(/ +$/gm, '');

  const rawLines = raw.trim().split('\n');
  for (let i = 0; i < rawLines.length; i++) {
    const match = rawLines[i].match(/^(?:Title:\s*|#\s*)(.*)/i);
    if (match) {
      const rest = rawLines.slice(i + 1);
      while (rest.length && !rest[0].trim()) rest.shift();
      raw = rest.join('\n');
      break;
    }
    if (rawLines[i].trim().startsWith('[') || (rawLines[i].trim() && i > 2)) break;
  }

  raw = postprocessLyrics(raw);
  raw = fixSectionLabels(raw);
  raw = fixAPrefix(raw);
  raw = enforceLineCounts(raw);

  const slopResult = slopDetector.scanForSlop(raw);
  if (slopResult.ai_score > 0) {
    console.warn(`Generation slop scan: score=${slopResult.ai_score} severity=${slopResult.severity}`,
      'words:', slopResult.layers.blacklisted_words.found,
      'phrases:', slopResult.layers.blacklisted_phrases.found,
      'overuse:', slopResult.layers.overuse.found.map((o: any) => `${o.word}(${o.count}x)`).join(', ') || 'none',
      'hook_formulas:', slopResult.layers.hook_formulas.found.join(', ') || 'none');
  }

  if (onPhase) onPhase("Choosing title…");
  let title = '';
  try {
    const titleLines: string[] = [`Artist: ${profile.artist}`];
    if (profile.album) titleLines.push(`Album style: ${profile.album}`);
    if (usedTitles?.length) {
      titleLines.push('\nTitles already used (avoid these and their key words):');
      for (const t of usedTitles) titleLines.push(`  ✗ ${t}`);
    }
    titleLines.push('\n--- LYRICS ---', raw, '--- END LYRICS ---');
    titleLines.push('\nChoose the best title for this song:');
    let titleRaw = await provider.call(cacheBustPrompt(TITLE_DERIVATION_PROMPT), titleLines.join('\n'), effectiveModel, onChunk);
    titleRaw = stripThinkingBlocks(titleRaw).trim();
    titleRaw = titleRaw.replace(/^(?:Title:\s*|#\s*)/i, '').replace(/^["'`]|["'`]$/g, '').trim();
    title = titleRaw.split('\n')[0].trim();
    console.log('[LLM] Derived title:', title);

    // Validate title against banned words/patterns
    const titleIssues = validateTitle(title);
    if (titleIssues.length) {
      console.warn(`[LLM] Title "${title}" failed validation: ${titleIssues.join(', ')}. Requesting re-derivation.`);
      // Re-derive with explicit rejection guidance
      const retryLines = [...titleLines];
      retryLines.push(`\nThe title "${title}" is REJECTED because: ${titleIssues.join(', ')}.`);
      retryLines.push('Choose a DIFFERENT title that avoids these issues. Return ONLY the new title:');
      try {
        let retryRaw = await provider.call(cacheBustPrompt(TITLE_DERIVATION_PROMPT), retryLines.join('\n'), effectiveModel, onChunk);
        retryRaw = stripThinkingBlocks(retryRaw).trim();
        retryRaw = retryRaw.replace(/^(?:Title:\s*|#\s*)/i, '').replace(/^["'`]|["'`]$/g, '').trim();
        const retryTitle = retryRaw.split('\n')[0].trim();
        const retryIssues = validateTitle(retryTitle);
        if (!retryIssues.length) {
          console.log(`[LLM] Re-derived title: "${retryTitle}" (was: "${title}")`);
          title = retryTitle;
        } else {
          console.warn(`[LLM] Re-derived title "${retryTitle}" still failed: ${retryIssues.join(', ')}. Keeping original.`);
        }
      } catch (retryErr) {
        console.warn('[LLM] Title re-derivation failed:', retryErr);
      }
    }
  } catch (err) { console.warn('[LLM] Title derivation failed, falling back to empty:', err); }

  let duration = metadata.duration || 0;
  if (metadata.bpm > 0 && !duration) duration = estimateDuration(raw, metadata.bpm);

  return {
    lyrics: raw, provider: providerName, model: effectiveModel, title,
    subject: metadata.subject, bpm: metadata.bpm, key: metadata.key,
    caption: metadata.caption, duration,
    system_prompt: GENERATION_SYSTEM_PROMPT, user_prompt: userPrompt
  };
}

function buildRefinementPrompt(originalLyrics: string, artistName: string, title: string, profile?: LyricsProfile, originalSlop?: string[]): string {
  const lines = [`Artist: ${artistName}`, `Original Title: ${title}`, ''];
  if (profile) {
    lines.push('=== INTENDED LANE PROFILE (match this style) ===');
    lines.push(`Themes: ${(profile.themes || []).slice(0, 8).join(', ')}`);
    if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
    if (profile.vocabulary_notes) lines.push(`Vocabulary: ${profile.vocabulary_notes}`);
    if (profile.imagery_patterns) lines.push(`Imagery patterns: ${profile.imagery_patterns}`);
    if (profile.signature_devices) lines.push(`Signature devices: ${profile.signature_devices}`);
    if (profile.narrative_techniques) lines.push(`Narrative techniques: ${profile.narrative_techniques}`);
    if (profile.emotional_arc) lines.push(`Emotional arc: ${profile.emotional_arc}`);
    if (profile.structural_patterns) lines.push(`Structure: ${profile.structural_patterns}`);
    if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);
    if (profile.rhyme_schemes?.length) lines.push(`Rhyme schemes: ${profile.rhyme_schemes.join(', ')}`);
    if (profile.rhyme_quality) {
      const rq = profile.rhyme_quality;
      const total = Object.values(rq).reduce((a, b) => a + b, 0);
      if (total > 0) lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
    }
    const ms = profile.meter_stats;
    if (ms) lines.push(`Line density: ~${ms.avg_syllables_per_line ?? '?'} syl/line (σ=${ms.syllable_std_dev ?? '?'}), ~${ms.avg_words_per_line ?? '?'} words/line`);
    const rs = profile.repetition_stats;
    if (rs) {
      lines.push(`Hook behavior: ${rs.pattern || 'unknown'} (${rs.chorus_repetition_pct ?? 0}% chorus repetition)`);
      if ((rs.chorus_repetition_pct ?? 0) >= 20) lines.push('Calibration: This artist uses heavy chorus repetition — ensure hook lines repeat.');
      else if ((rs.chorus_repetition_pct ?? 0) < 15) lines.push('Calibration: This artist uses light repetition — be subtle with hooks.');
    }
    if (profile.avg_verse_lines || profile.avg_chorus_lines) lines.push(`Verse/chorus: avg ${profile.avg_verse_lines} verse lines, avg ${profile.avg_chorus_lines} chorus lines`);
    if (profile.song_subjects && typeof profile.song_subjects === 'object') {
      const titles = Object.keys(profile.song_subjects);
      if (titles.length) {
        lines.push('', '=== ORIGINAL SONG TITLES (check for plagiarism) ===');
        for (const t of titles) lines.push(`  • ${t}`);
      }
    }
    lines.push('');
  }
  if (originalSlop?.length) {
    lines.push('=== KNOWN ISSUES TO FIX ===');
    lines.push('The original lyrics contain the following AI-clichés or circular phrases that MUST be replaced:');
    lines.push(`Words/Phrases to Remove: ${originalSlop.join(', ')}`);
    lines.push('');
  }
  lines.push('=== ORIGINAL LYRICS ===', '', originalLyrics, '', '=== INSTRUCTIONS ===', '');
  lines.push('Refine the lyrics above according to the refinement rules.');
  lines.push('Keep as much of the original as possible — only change what genuinely needs fixing.');
  lines.push(`Maintain ${artistName}'s distinctive style throughout.`);
  lines.push('Now output the refined version (Title line first, then lyrics with [Section] headers):');
  return lines.join('\n');
}

export async function refineLyricsStreaming(
  originalLyrics: string, artistName: string, title: string,
  providerName: string, model?: string, profile?: LyricsProfile,
  onChunk?: ChunkCallback
): Promise<GenerationResponse> {
  const provider = getProvider(providerName);
  const effectiveModel = model || provider.defaultModel;
  const slopScan = slopDetector.scanForSlop(originalLyrics);
  const foundSlop = [...slopScan.layers.blacklisted_words.found, ...slopScan.layers.blacklisted_phrases.found];
  const userPrompt = buildRefinementPrompt(originalLyrics, artistName, title, profile, foundSlop);
  let raw = await provider.call(cacheBustPrompt(REFINEMENT_SYSTEM_PROMPT), userPrompt, effectiveModel, onChunk);

  raw = stripThinkingBlocks(raw);
  raw = raw.replace(/<\|[a-z_]+\|>/g, '');
  raw = raw.replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '');
  raw = raw.replace(/ +$/gm, '');

  let refinedTitle = title;
  const rLines = raw.trim().split('\n');
  for (let i = 0; i < rLines.length; i++) {
    const match = rLines[i].match(/^(?:Title:\s*|#\s*)(.*)/i);
    if (match) {
      refinedTitle = match[1].trim().replace(/^['"]|['"]$/g, '');
      const rest = rLines.slice(i + 1);
      while (rest.length && !rest[0].trim()) rest.shift();
      raw = rest.join('\n');
      break;
    }
  }

  raw = postprocessLyrics(raw);
  raw = fixSectionLabels(raw);
  raw = fixAPrefix(raw);
  raw = enforceLineCounts(raw);

  const slopResult = slopDetector.scanForSlop(raw);
  if (slopResult.ai_score > 0) {
    console.warn(`Refinement slop scan: score=${slopResult.ai_score} severity=${slopResult.severity}`,
      'words:', slopResult.layers.blacklisted_words.found,
      'phrases:', slopResult.layers.blacklisted_phrases.found,
      'overuse:', slopResult.layers.overuse.found.map((o: any) => `${o.word}(${o.count}x)`).join(', ') || 'none',
      'hook_formulas:', slopResult.layers.hook_formulas.found.join(', ') || 'none');
  }

  return {
    lyrics: raw, provider: providerName, model: effectiveModel, title: refinedTitle,
    subject: '', bpm: 0, key: '', caption: '', duration: 0,
    system_prompt: REFINEMENT_SYSTEM_PROMPT, user_prompt: userPrompt
  };
}
