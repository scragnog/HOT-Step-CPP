// llm/orchestration.ts — High-level lyric generation and refinement functions
//
// All prompt TEXT and prompt BUILDERS live in ../prompts.ts (the canonical
// single source, shared with tools/mcp-lyricstudio). This file owns the
// call orchestration: provider calls, JSON parsing, postprocessing, retries.

import * as slopDetector from '../slopDetector.js';
import {
  GENERATION_SYSTEM_PROMPT,
  SONG_METADATA_SYSTEM_PROMPT,
  REFINEMENT_SYSTEM_PROMPT,
  TITLE_DERIVATION_PROMPT,
  MM3_CAPTION_SYSTEM_PROMPT,
  YUE2_CAPTION_SYSTEM_PROMPT,
  buildMetadataPrompt,
  buildGenerationPrompt,
  buildRefinementPrompt,
  buildTitlePrompt,
  buildMm3CaptionPrompt,
  normalizeMm3Caption,
  validateMm3Caption,
  buildYue2CaptionPrompt,
  normalizeYue2Caption,
  validateYue2Caption,
  reconcileDurationToLyrics,
  ensureInstrumentalIntro,
} from '../prompts.js';
import type { LyricsProfile } from '../profilerService.js';
import { withModelSuffix } from '../modelName.js';
import type { GenerationResponse, ChunkCallback, CallOptions } from './types.js';
import { getProvider } from './registry.js';
import {
  stripThinkingBlocks, postprocessLyrics, fixSectionLabels,
  enforceLineCounts, fixAPrefix,
  estimateDuration
} from './postprocess.js';

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

interface PlannedMetadata {
  subject: string;
  bpm: number;
  key: string;
  caption: string;
  duration: number;
  structure?: string;
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
  userSubject?: string,
  callOptions?: CallOptions
): Promise<PlannedMetadata> {
  const provider = getProvider(providerName);
  const prompt = buildMetadataPrompt(profile, usedSubjects, usedBpms, usedKeys, usedDurations, userSubject);
  console.log('[LLM] Planning song metadata via', providerName, modelName);
  const responseJsonStr = await provider.call(cacheBustPrompt(SONG_METADATA_SYSTEM_PROMPT), prompt, modelName, onChunk, callOptions);
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

/**
 * Write the MiniMax-Music3 Structured Caption for a finished song.
 *
 * Its own call, deliberately, and deliberately LAST: the Arrangement section is
 * a section-by-section timeline of this specific song, so it needs the final
 * lyrics — including the instrumental intro applied above, which is a real
 * section of the arrangement. The metadata planner runs before any lyric exists
 * and structurally cannot write it.
 *
 * Never fatal. MM3 is the second backend; a caption failure here must not cost
 * the user a set of lyrics that is otherwise finished and saved.
 */
async function writeMm3Caption(
  profile: LyricsProfile,
  ctx: {
    lyrics: string; aceCaption?: string; subject?: string;
    bpm?: number; key?: string; signature?: string;
  },
  providerName: string,
  modelName: string,
  onChunk?: ChunkCallback,
  callOptions?: CallOptions,
): Promise<string> {
  const provider = getProvider(providerName);
  const facts = { bpm: ctx.bpm, key: ctx.key, signature: ctx.signature };
  const userPrompt = buildMm3CaptionPrompt(profile, { ...ctx, instrumental: !ctx.lyrics.trim() });

  const attempt = async (prompt: string): Promise<string> => {
    const raw = await provider.call(
      cacheBustPrompt(MM3_CAPTION_SYSTEM_PROMPT), prompt, modelName, onChunk, callOptions);
    return normalizeMm3Caption(stripThinkingBlocks(raw), facts);
  };

  try {
    let caption = await attempt(userPrompt);
    let issues = validateMm3Caption(caption);

    // One retry, with the specific defects named. Worth the extra call: the
    // format IS the adherence lever for MM3, and the common failures (dropped
    // labels, markdown headings, a 200-word sketch) are all things a model
    // fixes readily once told which one it committed.
    if (issues.length) {
      console.warn(`[LLM] MM3 caption failed validation: ${issues.join('; ')}. Retrying once.`);
      const retryPrompt = [
        userPrompt,
        '',
        `Your previous attempt was REJECTED for: ${issues.join('; ')}.`,
        'Write it again with every heading and every label present, plain text only, and the full length. Output the caption only.',
      ].join('\n');
      const retry = await attempt(retryPrompt);
      const retryIssues = validateMm3Caption(retry);
      if (retryIssues.length < issues.length) { caption = retry; issues = retryIssues; }
    }

    // A caption that still has defects is kept anyway — a partly-malformed
    // Structured Caption is closer to what MM3 wants than the ACE caption it
    // would otherwise fall back to, and the field is user-editable.
    if (issues.length) console.warn(`[LLM] MM3 caption kept with issues: ${issues.join('; ')}`);
    else console.log(`[LLM] MM3 caption written (${caption.split(/\s+/).length} words)`);
    return caption;
  } catch (err) {
    console.warn('[LLM] MM3 caption generation failed:', err);
    return '';
  }
}

export async function generateLyricsStreaming(
  profile: LyricsProfile, providerName: string, model?: string,
  extraInstructions?: string, usedSubjects: string[] = [],
  usedBpms: number[] = [], usedKeys: string[] = [],
  usedTitles: string[] = [], usedDurations: number[] = [],
  onChunk?: ChunkCallback, onPhase?: (phase: string) => void,
  userSubject?: string, callOptions?: CallOptions
): Promise<GenerationResponse> {
  const provider = getProvider(providerName);
  const effectiveModel = model || provider.defaultModel;

  if (onPhase) onPhase("Planning song metadata…");
  let metadata: PlannedMetadata = { subject: '', bpm: 0, key: '', caption: '', duration: 0 };
  if (profile.song_subjects || (profile.themes && profile.themes.length) || userSubject) {
    try {
      metadata = await planSongMetadata(profile, usedSubjects, usedBpms, usedKeys, usedDurations, providerName, effectiveModel, onChunk, userSubject, callOptions);
      if (userSubject) metadata.subject = userSubject;
      console.log("Planned metadata:", metadata);
    } catch(e) { console.warn("Failed to plan metadata", e); }
  }
  if (userSubject && !metadata.subject) metadata.subject = userSubject;

  if (metadata.subject) extraInstructions = `The song must be about: ${metadata.subject}\n\n${extraInstructions || ''}`;
  if (onPhase) onPhase("Writing lyrics…");
  const userPrompt = buildGenerationPrompt(profile, extraInstructions, metadata.duration, metadata.bpm, metadata.structure);
  let raw = await provider.call(cacheBustPrompt(GENERATION_SYSTEM_PROMPT), userPrompt, effectiveModel, onChunk, callOptions);

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
    const titleUserPrompt = buildTitlePrompt(raw, profile.artist, profile.album, usedTitles);
    let titleRaw = await provider.call(cacheBustPrompt(TITLE_DERIVATION_PROMPT), titleUserPrompt, effectiveModel, onChunk, callOptions);
    titleRaw = stripThinkingBlocks(titleRaw).trim();
    titleRaw = titleRaw.replace(/^(?:Title:\s*|#\s*)/i, '').replace(/^["'`]|["'`]$/g, '').trim();
    title = titleRaw.split('\n')[0].trim();
    console.log('[LLM] Derived title:', title);

    // Validate title against banned words/patterns
    const titleIssues = validateTitle(title);
    if (titleIssues.length) {
      console.warn(`[LLM] Title "${title}" failed validation: ${titleIssues.join(', ')}. Requesting re-derivation.`);
      // Re-derive with explicit rejection guidance
      const retryPrompt = [
        titleUserPrompt,
        `\nThe title "${title}" is REJECTED because: ${titleIssues.join(', ')}.`,
        'Choose a DIFFERENT title that avoids these issues. Return ONLY the new title:',
      ].join('\n');
      try {
        let retryRaw = await provider.call(cacheBustPrompt(TITLE_DERIVATION_PROMPT), retryPrompt, effectiveModel, onChunk, callOptions);
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

  // Tag the title with the model that wrote it, same as the MCP path does
  title = withModelSuffix(title, effectiveModel);

  // Most records open with a few bars before the first vocal, and the writer
  // ignores that instruction far more often than it follows it (91% of the
  // existing catalogue declared no instrumental section at all). Applied here
  // deterministically, seeded from artist+title so the same song always makes
  // the same choice — and BEFORE the duration is derived, since a declared
  // intro is instrumental time the derivation counts.
  const withIntro = ensureInstrumentalIntro(raw, `${profile.artist}|${title}`);
  if (withIntro !== null) {
    console.log(`[LLM] Instrumental intro applied to "${title}"`);
    raw = withIntro;
  }

  // Duration is derived from the FINAL lyrics at this artist's measured pacing,
  // not trusted from the plan: the LM renders at exactly the stated duration,
  // so duration==content must hold or the song cuts off / pads with filler.
  // The planner's number survives only when the lyrics already agree with it.
  let duration = metadata.duration || 0;
  if (metadata.bpm > 0 && !duration) duration = estimateDuration(raw, metadata.bpm);
  const reconciled = reconcileDurationToLyrics(
    raw, metadata.bpm, duration, (profile as any).audio_enrichment?.wordsPerSec);
  if (reconciled !== duration) {
    console.log(`[LLM] Duration reconciled to lyrics: ${duration}s -> ${reconciled}s` +
      ` (rate ${((profile as any).audio_enrichment?.wordsPerSec > 0) ? 'artist' : 'global'})`);
    duration = reconciled;
  }

  // MiniMax-Music3 wants a completely different caption from ACE-Step's, and
  // the two are not interchangeable in either direction (see the A/B in
  // prompts.ts). Written last so the Arrangement can follow the FINAL lyrics.
  if (onPhase) onPhase("Writing MM3 caption…");
  const caption_mm3 = await writeMm3Caption(
    profile,
    {
      lyrics: raw, aceCaption: metadata.caption, subject: metadata.subject,
      bpm: metadata.bpm, key: metadata.key,
      signature: (profile as any).audio_enrichment?.signatures?.[0],
    },
    providerName, effectiveModel, onChunk, callOptions,
  );

  // The YuE2 planner caption: one sentence, fixed order. Cheap (a short
  // completion) and, like MM3's, never fatal.
  if (onPhase) onPhase("Writing YuE2 caption…");
  const caption_yue2 = await writeYue2Caption(
    profile,
    { lyrics: raw, aceCaption: metadata.caption, subject: metadata.subject, bpm: metadata.bpm, key: metadata.key },
    providerName, effectiveModel, onChunk, callOptions,
  );

  return {
    lyrics: raw, provider: providerName, model: effectiveModel, title,
    subject: metadata.subject, bpm: metadata.bpm, key: metadata.key,
    caption: metadata.caption, caption_mm3, caption_yue2, duration,
    system_prompt: GENERATION_SYSTEM_PROMPT, user_prompt: userPrompt
  };
}

/**
 * Write the YuE2 planner caption for a finished song: ONE sentence in the
 * order language → genre → vocal → instruments → mood → production → BPM,
 * which is the shape the planner's training captions take (and what our own
 * `.yue2.txt` dataset captions follow). Validated and retried once, like the
 * MM3 caption; '' on failure so a lyric set is never lost to a caption.
 */
async function writeYue2Caption(
  profile: LyricsProfile,
  ctx: { lyrics: string; aceCaption?: string; subject?: string; bpm?: number; key?: string },
  providerName: string,
  modelName: string,
  onChunk?: ChunkCallback,
  callOptions?: CallOptions,
): Promise<string> {
  const provider = getProvider(providerName);
  const userPrompt = buildYue2CaptionPrompt(profile, { ...ctx, instrumental: !ctx.lyrics.trim() });
  const attempt = async (prompt: string): Promise<string> => {
    const raw = await provider.call(cacheBustPrompt(YUE2_CAPTION_SYSTEM_PROMPT), prompt, modelName, onChunk, callOptions);
    return normalizeYue2Caption(stripThinkingBlocks(raw), { bpm: ctx.bpm });
  };
  try {
    let caption = await attempt(userPrompt);
    let issues = validateYue2Caption(caption);
    if (issues.length) {
      console.warn(`[LLM] YuE2 caption failed validation: ${issues.join('; ')}. Retrying once.`);
      const retry = await attempt([userPrompt, '', `Your previous attempt was REJECTED for: ${issues.join('; ')}.`,
        'Write it again: one sentence, the fixed order, nothing else.'].join('\n'));
      const retryIssues = validateYue2Caption(retry);
      if (retryIssues.length < issues.length) { caption = retry; issues = retryIssues; }
    }
    if (issues.length) console.warn(`[LLM] YuE2 caption saved with issues: ${issues.join('; ')}`);
    return caption;
  } catch (err: any) {
    console.warn(`[LLM] YuE2 caption failed (${err?.message || err}) — generation saved without one`);
    return '';
  }
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

  refinedTitle = withModelSuffix(refinedTitle, effectiveModel);

  return {
    lyrics: raw, provider: providerName, model: effectiveModel, title: refinedTitle,
    subject: '', bpm: 0, key: '', caption: '', caption_mm3: '', duration: 0,
    system_prompt: REFINEMENT_SYSTEM_PROMPT, user_prompt: userPrompt
  };
}
