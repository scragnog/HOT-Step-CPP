// inspire.ts — Inspire API endpoint
//
// Two inspire paths:
//   1. POST /api/inspire      — engine's built-in LM (inspire mode)
//   2. POST /api/inspire/llm   — external LLM lyric generation
//
// Async job pattern mirrors generate.ts: submit → poll → result.

import { Router } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { aceClient, type AceRequest } from '../services/aceClient.js';
import { getUserId } from './auth.js';
import { engineReady, engineBootStatus } from '../engineState.js';
import { subscribeLines } from './logs.js';
import { translateParams } from '../services/generation/translateParams.js';
import { getProvider, listProviders } from '../services/lireek/llm/registry.js';
import { stripThinkingBlocks, postprocessLyrics, fixSectionLabels, enforceLineCounts, fixAPrefix } from '../services/lireek/llm/postprocess.js';
import { INSTAGEN_LYRIC_SYSTEM_PROMPT, INSTAGEN_FULL_SYSTEM_PROMPT } from '../services/lireek/prompts.js';
import { getSetting, setSetting } from '../db/lireekDb.js';

const router = Router();

/** Inspire job state */
interface InspireJob {
  id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  stage?: string;
  progress?: number;
  aceJobId?: string;
  result?: {
    caption: string;
    lyrics: string;
    bpm: number;
    duration: number;
    keyScale: string;
    timeSignature: string;
    vocalLanguage: string;
  };
  error?: string;
  createdAt: number;
}

const inspireJobs = new Map<string, InspireJob>();

// Cleanup old jobs after 10 minutes
setInterval(() => {
  const cutoff = Date.now() - 10 * 60 * 1000;
  for (const [id, job] of inspireJobs) {
    if (job.createdAt < cutoff && job.status !== 'running') {
      inspireJobs.delete(id);
    }
  }
}, 60_000);

/** Poll ace-server job until completion */
async function pollUntilDone(aceJobId: string, job: InspireJob, signal: AbortSignal): Promise<void> {
  const POLL_INTERVAL = 500;
  const MAX_POLLS = 600; // 5 minutes max (inspire is fast)

  for (let i = 0; i < MAX_POLLS; i++) {
    if (signal.aborted || job.status === 'cancelled') {
      await aceClient.cancelJob(aceJobId);
      throw new Error('Cancelled');
    }

    const status = await aceClient.pollJob(aceJobId);
    if (status.status === 'done') return;
    if (status.status === 'failed') throw new Error('Inspire failed on ace-server');
    if (status.status === 'cancelled') throw new Error('Cancelled by ace-server');

    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
  }

  throw new Error('Inspire timed out');
}

/** Run inspire pipeline */
async function runInspire(job: InspireJob, params: any): Promise<void> {
  if (job.status === 'cancelled') return;

  const aceReq = translateParams(params);
  const abortController = new AbortController();
  (job as any)._abort = abortController;

  try {
    job.status = 'running';
    job.stage = 'Generating lyrics & metadata...';
    job.progress = 10;

    // Log what reaches the engine
    console.log(`[Inspire] Job ${job.id} — caption: ${(params.caption || '').substring(0, 80)}, lang: ${params.vocalLanguage}`);

    // Subscribe to engine logs for LM Phase 1 progress
    const unsub = subscribeLines((line) => {
      if (line.source !== 'engine') return;
      const lm1 = line.text.match(/\[LM-Phase1\] Step (\d+).*?([\d.]+) tok\/s/);
      if (lm1) {
        job.stage = `Composing lyrics: Step ${lm1[1]} (${lm1[2]} tok/s)`;
        job.progress = 30;
        return;
      }
      if (line.text.includes('[LM-Phase1] Prefill')) {
        job.stage = 'Preparing language model...';
        job.progress = 15;
      } else if (line.text.includes('[Adapter]') && line.text.includes('Merge')) {
        job.stage = 'Loading adapter...';
      }
    });

    console.log(`[Inspire] Job ${job.id} — submitting LM inspire request`);

    const lmJobId = await aceClient.submitLm(aceReq, 'inspire');
    job.aceJobId = lmJobId;

    await pollUntilDone(lmJobId, job, abortController.signal);

    // Fetch inspire results
    const resultRes = await aceClient.getJobResult(lmJobId);
    const lmResults = await resultRes.json() as AceRequest[];

    unsub();

    if (!lmResults || lmResults.length === 0) {
      throw new Error('No results from inspire mode');
    }

    const first = lmResults[0];

    job.status = 'succeeded';
    job.progress = 100;
    job.stage = 'Done!';
    job.result = {
      caption: first.caption || aceReq.caption || '',
      lyrics: first.lyrics || '',
      bpm: first.bpm || 120,
      duration: first.duration || 120,
      keyScale: first.keyscale || 'C major',
      timeSignature: first.timesignature || '4',
      vocalLanguage: first.vocal_language || params.vocalLanguage || 'en',
    };

    console.log(`[Inspire] Job ${job.id} — complete. BPM=${job.result.bpm}, lang=${job.result.vocalLanguage}, caption=${job.result.caption.substring(0, 100)}, lyrics=${job.result.lyrics.substring(0, 200)}`);

  } catch (err: any) {
    if (err.message === 'Cancelled') {
      job.status = 'cancelled';
      job.stage = 'Cancelled';
    } else {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.stage = 'Failed';
      console.error(`[Inspire] Job ${job.id} failed:`, err.message);
    }
  }
}

// ── Serialization queue (shares the engine with generate) ──
// Inspire jobs go through the same single-GPU bottleneck.
// For now we run them independently — they're fast and only use the LM.
// If contention becomes an issue, we can merge with the generate queue.

// POST /api/inspire — start an inspire job
router.post('/', (req, res) => {
  if (!engineReady) {
    res.status(503).json({
      error: `Engine not ready: ${engineBootStatus}`,
      detail: 'Please wait for the engine to finish starting up.',
    });
    return;
  }

  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const job: InspireJob = {
    id: uuidv4(),
    status: 'pending',
    stage: 'Starting...',
    progress: 0,
    createdAt: Date.now(),
  };

  inspireJobs.set(job.id, job);
  runInspire(job, req.body);

  res.json({
    jobId: job.id,
    status: job.status,
  });
});

// GET /api/inspire/status/:id — poll inspire job status
router.get('/status/:id', (req, res) => {
  const job = inspireJobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  res.json({
    jobId: job.id,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    result: job.result,
    error: job.error,
    ace_job_id: job.aceJobId ?? null,
    ace_phase: (job as any).acePhase ?? null,
    ace_phase_progress: (job as any).acePhaseProgress ?? null,
  });
});

// POST /api/inspire/cancel/:id — cancel an inspire job
router.post('/cancel/:id', (req, res) => {
  const job = inspireJobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  job.status = 'cancelled';
  if (job.aceJobId) {
    aceClient.cancelJob(job.aceJobId).catch(() => {});
  }
  if ((job as any)._abort) {
    (job as any)._abort.abort();
  }

  res.json({ success: true, jobId: job.id });
});

// ── External LLM lyric generation ──────────────────────────────────
// POST /api/inspire/llm — generate lyrics via an external LLM provider.
// This is synchronous (not a job queue) since external LLMs respond
// in seconds. Returns { lyrics, caption } directly.

const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English', zh: 'Chinese', ja: 'Japanese', ko: 'Korean',
  es: 'Spanish', fr: 'French', de: 'German', it: 'Italian',
  pt: 'Portuguese', ru: 'Russian', ar: 'Arabic', hi: 'Hindi',
  tr: 'Turkish', vi: 'Vietnamese', th: 'Thai', sv: 'Swedish',
  pl: 'Polish', nl: 'Dutch',
};

router.post('/llm', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const { provider: providerName, model, genres, subject, language, systemPrompt: clientPrompt } = req.body as {
    provider: string;
    model?: string;
    genres: string[];     // e.g. ["Pop Punk", "Punk Rock"]
    subject: string;      // e.g. "a man tired from working 9 to 5"
    language?: string;    // e.g. "en"
    systemPrompt?: string; // Optional client-side prompt override (takes priority over DB custom)
  };

  if (!providerName) {
    res.status(400).json({ error: 'Missing provider' });
    return;
  }
  if (!subject?.trim()) {
    res.status(400).json({ error: 'Missing subject — external LLM mode requires a song subject' });
    return;
  }
  if (!genres?.length) {
    res.status(400).json({ error: 'Missing genres — select at least one genre' });
    return;
  }

  try {
    const provider = getProvider(providerName);
    const effectiveModel = model || provider.defaultModel;
    const langName = LANGUAGE_NAMES[language || 'en'] || language || 'English';
    const genreStr = genres.join(', ');

    // Resolve system prompt: client override → DB custom → default
    const dbCustom = getSetting('instagen_system_prompt');
    const systemPrompt = clientPrompt?.trim() || dbCustom || INSTAGEN_FULL_SYSTEM_PROMPT;

    // Build user prompt
    const userPrompt = [
      `Genre/Style: ${genreStr}`,
      `Subject: ${subject.trim()}`,
      `Language: ${langName}`,
      '',
      'Generate the complete song now:',
    ].join('\n');

    console.log(`[Inspire/LLM] Generating song via ${providerName}/${effectiveModel}`);
    console.log(`[Inspire/LLM] Genre: ${genreStr}, Subject: ${subject}, Language: ${langName}`);
    console.log(`[Inspire/LLM] Prompt source: ${clientPrompt ? 'client override' : dbCustom ? 'DB custom' : 'default'}`);

    let raw = await provider.call(systemPrompt, userPrompt, effectiveModel);

    // Strip thinking blocks first
    raw = stripThinkingBlocks(raw);
    raw = raw.replace(/<\|[a-z_]+\|>/g, '');

    // Try to parse as structured JSON response
    let structuredResult = parseStructuredLlmResponse(raw);

    if (structuredResult) {
      // ── Structured JSON path — LLM returned all metadata ──
      let lyrics = structuredResult.lyrics || '';
      lyrics = lyrics.replace(/\[?(System|User|Assistant)\]?:.*/gi, '');
      lyrics = lyrics.replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '');
      lyrics = lyrics.replace(/ +$/gm, '');
      lyrics = postprocessLyrics(lyrics);
      lyrics = fixSectionLabels(lyrics);
      lyrics = fixAPrefix(lyrics);
      lyrics = enforceLineCounts(lyrics);

      console.log(`[Inspire/LLM] Structured response: ${lyrics.split('\n').length} lines, BPM=${structuredResult.bpm}, Key=${structuredResult.key}, Title="${structuredResult.title}"`);

      res.json({
        lyrics,
        caption: structuredResult.tags || genreStr,
        title: structuredResult.title || undefined,
        bpm: structuredResult.bpm || undefined,
        key: structuredResult.key || undefined,
        timeSignature: structuredResult.time_signature || undefined,
        duration: structuredResult.duration || undefined,
        structured: true,  // Flag so frontend knows to skip inspire
        provider: providerName,
        model: effectiveModel,
      });
    } else {
      // ── Fallback: raw text path (legacy prompt or non-JSON response) ──
      console.log('[Inspire/LLM] Non-JSON response, falling back to lyrics-only parsing');

      raw = raw.replace(/\[?(System|User|Assistant)\]?:.*/gi, '');
      raw = raw.replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '');
      raw = raw.replace(/ +$/gm, '');

      // Extract title from the end of the response (Title: <song title>)
      let extractedTitle = '';
      const titleEndMatch = raw.match(/\n\s*Title:\s*(.+?)\s*$/im);
      if (titleEndMatch) {
        extractedTitle = titleEndMatch[1]
          .replace(/^["']+|["']+$/g, '')  // strip quotes
          .replace(/[.!?,;:]+$/, '')       // strip trailing punctuation
          .trim();
        raw = raw.replace(/\n\s*Title:\s*.+?\s*$/im, '').trimEnd();
      }

      // Strip any title line at the start
      const rawLines = raw.trim().split('\n');
      for (let i = 0; i < rawLines.length; i++) {
        const match = rawLines[i].match(/^(?:Title:\s*|#\s*)(.*)/i);
        if (match) {
          if (!extractedTitle) extractedTitle = match[1].replace(/^["']+|["']+$/g, '').trim();
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

      console.log(`[Inspire/LLM] Generated ${raw.split('\n').length} lines of lyrics${extractedTitle ? `, title: "${extractedTitle}"` : ''}`);

      res.json({
        lyrics: raw,
        caption: genreStr,
        title: extractedTitle || undefined,
        structured: false,
        provider: providerName,
        model: effectiveModel,
      });
    }
  } catch (err: any) {
    console.error(`[Inspire/LLM] Failed:`, err.message);
    res.status(500).json({ error: err.message || 'LLM lyric generation failed' });
  }
});

// GET /api/inspire/llm/providers — list available LLM providers
router.get('/llm/providers', async (_req, res) => {
  try {
    const providers = await listProviders();
    res.json(providers);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/inspire/llm/subject — generate a random song subject via LLM
router.post('/llm/subject', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const { provider: providerName, model, genres } = req.body as {
    provider: string;
    model?: string;
    genres?: string[];
  };

  if (!providerName) {
    res.status(400).json({ error: 'Missing provider' });
    return;
  }

  try {
    const provider = getProvider(providerName);
    const effectiveModel = model || provider.defaultModel;
    const genreStr = genres?.length ? genres.join(', ') : 'any genre';

    const systemPrompt = `You generate creative, specific song subjects for songwriters. Given a musical genre, suggest ONE vivid, concrete song subject. The subject should be a brief description (1-2 sentences max) that a songwriter can write lyrics about. Be specific and interesting — avoid generic topics. Do NOT write lyrics, only the subject/concept. Output ONLY the subject, nothing else.`;

    const userPrompt = `Genre/Style: ${genreStr}\n\nSuggest a creative song subject:`;

    console.log(`[Inspire/LLM] Generating random subject via ${providerName}/${effectiveModel}`);

    let raw = await provider.call(systemPrompt, userPrompt, effectiveModel);
    raw = stripThinkingBlocks(raw);
    // Clean up: remove quotes, "Subject:" prefix, etc.
    raw = raw.replace(/^["']|["']$/g, '').trim();
    raw = raw.replace(/^(?:Subject|Topic|Concept|Idea):\s*/i, '').trim();
    // Take only the first 1-2 sentences
    const sentences = raw.split(/(?<=[.!?])\s+/);
    raw = sentences.slice(0, 2).join(' ').trim();
    // Remove trailing period for cleaner look in input field
    raw = raw.replace(/\.\s*$/, '');

    console.log(`[Inspire/LLM] Generated subject: "${raw}"`);

    res.json({ subject: raw });
  } catch (err: any) {
    console.error(`[Inspire/LLM] Subject generation failed:`, err.message);
    res.status(500).json({ error: err.message || 'Subject generation failed' });
  }
});

// ── InstaGen system prompt CRUD ──────────────────────────────────────
// Reuses Lireek DB settings table (same as Lyric Studio's prompt editor).

router.get('/llm/prompt', (_req, res) => {
  try {
    const custom = getSetting('instagen_system_prompt') || null;
    res.json({
      name: 'instagen_system',
      default_content: INSTAGEN_FULL_SYSTEM_PROMPT,
      custom,
    });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.put('/llm/prompt', (req, res) => {
  try {
    const { value } = req.body;
    if (!value) { res.status(400).json({ error: 'value required' }); return; }
    setSetting('instagen_system_prompt', value);
    console.log('[Inspire/LLM] Custom InstaGen system prompt saved');
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

router.delete('/llm/prompt', (_req, res) => {
  try {
    setSetting('instagen_system_prompt', '');
    console.log('[Inspire/LLM] InstaGen system prompt reset to default');
    res.json({ success: true });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// ── Structured JSON response parser ──────────────────────────────────
// Extracts the JSON object from LLM output, handling code fences, thinking
// blocks, and other wrapping the LLM might add around the JSON.

interface StructuredLlmResult {
  tags?: string;
  lyrics: string;
  title?: string;
  bpm?: number;
  key?: string;
  time_signature?: string;
  duration?: number;
}

function parseStructuredLlmResponse(raw: string): StructuredLlmResult | null {
  // Strip markdown code fences
  let cleaned = raw.replace(/^```(?:json)?\s*|\s*```$/gm, '').trim();

  // Try direct parse first
  try {
    const parsed = JSON.parse(cleaned);
    if (parsed && typeof parsed === 'object' && parsed.lyrics) return parsed;
  } catch { /* not valid JSON as-is */ }

  // Try to find JSON object in the text
  const start = cleaned.indexOf('{');
  if (start === -1) return null;

  // Find the matching closing brace
  let depth = 0;
  for (let i = start; i < cleaned.length; i++) {
    if (cleaned[i] === '{') depth++;
    else if (cleaned[i] === '}') {
      depth--;
      if (depth === 0) {
        try {
          const parsed = JSON.parse(cleaned.slice(start, i + 1));
          if (parsed && typeof parsed === 'object' && parsed.lyrics) return parsed;
        } catch { /* not valid JSON */ }
        break;
      }
    }
  }

  return null;
}

export default router;
