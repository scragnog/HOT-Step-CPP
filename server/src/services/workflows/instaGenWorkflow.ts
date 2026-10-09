import { z } from 'zod/v4';
import { getBackend, getActiveBackendId } from '../backends/registry.js';
import { getProvider } from '../lireek/llm/registry.js';
import { stripThinkingBlocks, postprocessLyrics, fixSectionLabels, enforceLineCounts, fixAPrefix } from '../lireek/llm/postprocess.js';
import { INSTAGEN_FULL_SYSTEM_PROMPT } from '../lireek/prompts.js';
import { getSetting } from '../../db/lireekDb.js';
import type { WorkflowDocuments } from './revisions.js';
import { registerWorkflowKind, workflowDocuments } from '../../routes/workflows.js';
import { WorkflowError, type WorkflowContext, type WorkflowKind } from './workflowJobs.js';
import { instaInputSchema as base, instaApproveSchema as approve, type InstaInput, type InstaResult } from '../../contracts/studioWorkflows.js';

export type { InstaInput, InstaResult };

export function deriveInstaTitle(lyrics: string): string {
  if (!lyrics || lyrics === '[Instrumental]') return '';
  const sections: Record<string, string> = {};
  let section = '';
  for (const line of lyrics.split(/\r?\n/)) {
    const match = line.match(/^\s*\[(.+?)\]\s*$/);
    if (match) { section = match[1].trim().toLowerCase(); continue; }
    const value = line.trim();
    if (section && value && !value.startsWith('(') && !sections[section]) sections[section] = value;
  }
  const key = Object.keys(sections).find(k => k.startsWith('chorus'))
    || Object.keys(sections).find(k => k === 'verse 1')
    || Object.keys(sections).find(k => k.startsWith('verse'))
    || Object.keys(sections)[0];
  if (!key) return '';
  let title = sections[key].replace(/\s*\(.*?\)\s*/g, '').trim().replace(/[,.!?;:]+$/, '').trim();
  if (title.length > 60) title = title.substring(0, 57) + '...';
  return title;
}

export function effectiveInstaRequest(input: InstaInput, result: InstaResult, edits?: { lyrics: string; caption: string }, editableDuration = true): Record<string, unknown> {
  const lyrics = edits?.lyrics ?? result.lyrics;
  const caption = edits?.caption ?? result.caption;
  const request: Record<string, unknown> = {
    ...input.engineParams,
    caption: caption || input.caption,
    lyrics,
    instrumental: input.lyricMode === 'instrumental',
    vocalLanguage: input.lyricMode === 'instrumental' ? undefined : input.vocalLanguage,
    source: 'insta-gen', useCotCaption: input.thinking, skipLm: false,
    title: result.title || deriveInstaTitle(lyrics) || input.caption,
    coResident: input.coResident, cacheLmCodes: input.cacheLmCodes,
    expectedBackend: input.expectedBackend,
  };
  if (result.bpm) request.bpm = result.bpm;
  // A user-set duration (engineParams.duration, already spread onto request
  // above) wins; the LM's estimate is only a fallback when the user left it
  // on auto (absent, or the app's -1 "auto" sentinel).
  const userDuration = input.engineParams.duration;
  const hasUserDuration = typeof userDuration === 'number' && userDuration > 0;
  if (result.duration && editableDuration && !hasUserDuration) request.duration = result.duration;
  if (result.keyScale) request.keyScale = result.keyScale;
  if (result.timeSignature) request.timeSignature = result.timeSignature;
  return request;
}

function assertInput(input: InstaInput): void {
  if (!input.genres.length && !input.caption.trim()) throw new WorkflowError(400, 'A genre or caption is required');
  if (input.lyricMode === 'lyrics-ai' && !input.provider) throw new WorkflowError(400, 'An LLM provider is required');
  if (input.lyricMode === 'lyrics-ai' && !input.randomSubject && !input.subject.trim()) throw new WorkflowError(400, 'A subject is required');
}

async function guardBackend(input: InstaInput): Promise<boolean> {
  if (getActiveBackendId() !== input.expectedBackend) throw new WorkflowError(409, 'Active backend changed; start again');
  const backend = getBackend(input.expectedBackend);
  if (!backend) throw new WorkflowError(400, 'Unknown backend');
  const capabilities = await backend.capabilities();
  if (getActiveBackendId() !== input.expectedBackend) throw new WorkflowError(409, 'Active backend changed; start again');
  return capabilities.core.duration.editable !== false;
}

const languageNames: Record<string, string> = { en: 'English', zh: 'Chinese', ja: 'Japanese', ko: 'Korean', es: 'Spanish', fr: 'French', de: 'German', it: 'Italian', pt: 'Portuguese', ru: 'Russian', ar: 'Arabic', hi: 'Hindi', tr: 'Turkish', vi: 'Vietnamese', th: 'Thai', sv: 'Swedish', pl: 'Polish', nl: 'Dutch' };
function cleanLyrics(raw: string): string {
  return enforceLineCounts(fixAPrefix(fixSectionLabels(postprocessLyrics(raw
    .replace(/\[?(System|User|Assistant)\]?:.*/gi, '')
    .replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '')
    .replace(/ +$/gm, '')))));
}
function parseLlm(raw: string, genre: string): { lyrics: string; caption: string; title?: string; bpm?: number; duration?: number; keyScale?: string; timeSignature?: string; structured: boolean } {
  const cleaned = raw.replace(/^```(?:json)?\s*|\s*```$/gm, '').trim();
  let value: any;
  try { value = JSON.parse(cleaned); } catch { /* text path */ }
  if (!value?.lyrics) {
    const start = cleaned.indexOf('{'), end = cleaned.lastIndexOf('}');
    if (start >= 0 && end > start) try { value = JSON.parse(cleaned.slice(start, end + 1)); } catch { /* text path */ }
  }
  if (value?.lyrics) return {
    lyrics: cleanLyrics(String(value.lyrics)), caption: value.tags || genre, title: value.title || undefined,
    bpm: value.bpm || undefined, duration: value.duration || undefined,
    keyScale: value.key || undefined, timeSignature: value.time_signature || undefined, structured: true,
  };
  let text = raw.replace(/\[?(System|User|Assistant)\]?:.*/gi, '')
    .replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '').replace(/ +$/gm, '');
  let title = '';
  const endTitle = text.match(/\n\s*Title:\s*(.+?)\s*$/im);
  if (endTitle) { title = endTitle[1].replace(/^["']+|["']+$/g, '').replace(/[.!?,;:]+$/, '').trim(); text = text.replace(/\n\s*Title:\s*.+?\s*$/im, '').trimEnd(); }
  const lines = text.trim().split('\n');
  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(/^(?:Title:\s*|#\s*)(.*)/i);
    if (match) { if (!title) title = match[1].replace(/^["']+|["']+$/g, '').trim(); text = lines.slice(i + 1).join('\n').trimStart(); break; }
    if (lines[i].trim().startsWith('[') || (lines[i].trim() && i > 2)) break;
  }
  return { lyrics: cleanLyrics(text), caption: genre, title: title || undefined, structured: false };
}

export interface InstaDeps {
  documents?: WorkflowDocuments;
  prompt?: () => string;
  checkBackend?: (input: InstaInput) => Promise<boolean>;
  inspire(params: Record<string, unknown>, signal: AbortSignal, progress: (stage: string) => void): Promise<InstaResult>;
}

async function resolve(input: InstaInput, ctx: WorkflowContext<any>, deps: InstaDeps): Promise<InstaResult> {
  assertInput(input);
  let lyrics = input.lyricMode === 'instrumental' ? '[Instrumental]' : '';
  let caption = input.caption;
  let llm: ReturnType<typeof parseLlm> | undefined;
  if (input.lyricMode === 'lyrics-ai') {
    const provider = getProvider(input.provider);
    const model = input.model || provider.defaultModel;
    let subject = input.subject.trim();
    if (input.randomSubject || !subject) {
      ctx.emit('stage', { stage: 'Generating random subject...' });
      const genre = input.genres.length ? input.genres.join(', ') : 'any genre';
      subject = stripThinkingBlocks(await provider.call('You generate creative, specific song subjects for songwriters. Given a musical genre, suggest ONE vivid, concrete song subject. Output ONLY the subject, nothing else.', `Genre/Style: ${genre}\n\nSuggest a creative song subject:`, model)).trim();
      subject = subject.replace(/^["']|["']$/g, '').replace(/^(?:Subject|Topic|Concept|Idea):\s*/i, '').split(/(?<=[.!?])\s+/).slice(0, 2).join(' ').replace(/\.\s*$/, '').trim();
      ctx.throwIfCancelled();
    }
    ctx.emit('stage', { stage: 'Generating lyrics via AI...' });
    const genre = input.genres.join(', ');
    const system = input.systemPrompt || INSTAGEN_FULL_SYSTEM_PROMPT;
    const prompt = [`Genre/Style: ${genre}`, `Subject: ${subject}`, `Language: ${languageNames[input.vocalLanguage] || input.vocalLanguage || 'English'}`, '', 'Generate the complete song now:'].join('\n');
    const raw = stripThinkingBlocks(await provider.call(system, prompt, model)).replace(/<\|[a-z_]+\|>/g, '');
    ctx.throwIfCancelled();
    llm = parseLlm(raw, genre);
    lyrics = llm.lyrics;
    caption = llm.caption || caption;
    if (llm.structured && llm.bpm) return { caption, lyrics, title: llm.title, bpm: llm.bpm, duration: llm.duration || 200, keyScale: llm.keyScale || 'C major', timeSignature: llm.timeSignature || '4/4', vocalLanguage: input.vocalLanguage };
  }
  ctx.emit('stage', { stage: input.lyricMode === 'lyrics' ? 'Generating lyrics...' : 'Resolving song metadata...' });
  const params: Record<string, unknown> = {
    caption, vocalLanguage: input.vocalLanguage, useCotCaption: input.thinking,
    lmModel: input.engineParams.lmModel || undefined, lmTemperature: input.engineParams.lmTemperature,
    lmCfgScale: input.engineParams.lmCfgScale, lmTopP: input.engineParams.lmTopP,
  };
  if (lyrics) params.lyrics = lyrics;
  if (input.lyricMode === 'instrumental') params.instrumental = true;
  const meta = await deps.inspire(params, ctx.signal, stage => ctx.emit('stage', { stage }));
  ctx.throwIfCancelled();
  return {
    caption: input.lyricMode === 'lyrics-ai' ? caption : input.thinking ? (meta.caption || caption) : caption,
    lyrics: lyrics || meta.lyrics, title: llm?.title,
    bpm: meta.bpm, duration: meta.duration, keyScale: meta.keyScale,
    timeSignature: meta.timeSignature, vocalLanguage: input.vocalLanguage,
  };
}

async function render(ctx: WorkflowContext<any>, input: InstaInput, result: InstaResult, deps: InstaDeps, edits?: { lyrics: string; caption: string }): Promise<unknown> {
  const editable = await (deps.checkBackend || guardBackend)(input);
  const request = effectiveInstaRequest(input, result, edits, editable);
  ctx.throwIfCancelled();
  ctx.emit('request', { title: request.title });
  const item = ctx.audio.enqueue('render', request, { source: 'insta-gen', workflowJobId: ctx.jobId });
  const finished = await ctx.audio.wait(item.id);
  ctx.throwIfCancelled();
  if (finished.status !== 'succeeded') throw new Error(finished.error || `Audio ${finished.status}`);
  return { request, audioIntentId: item.id, audio: finished.result };
}

export function createInstaGenKinds(deps: InstaDeps): WorkflowKind<any>[] {
  const captured = base.transform(input => ({
    ...input,
    systemPrompt: input.lyricMode === 'lyrics-ai' ? (deps.prompt?.() || getSetting('instagen_system_prompt') || INSTAGEN_FULL_SYSTEM_PROMPT) : undefined,
    model: input.lyricMode === 'lyrics-ai' && input.provider && !input.model ? getProvider(input.provider).defaultModel : input.model,
  }));
  return [{ kind: 'insta-preview', input: captured, async run(ctx) {
    const input = ctx.input as InstaInput;
    await (deps.checkBackend || guardBackend)(input);
    const result = await resolve(input, ctx, deps);
    ctx.throwIfCancelled();
    const document = (deps.documents || workflowDocuments()).create(ctx.userId, 'insta-preview', { input, result, edits: { lyrics: result.lyrics, caption: result.caption } });
    return { documentId: document.id, revision: document.revision, result };
  } }, { kind: 'insta-direct', input: captured, async run(ctx) {
    const input = ctx.input as InstaInput;
    await (deps.checkBackend || guardBackend)(input);
    const result = await resolve(input, ctx, deps);
    return render(ctx, input, result, deps);
  } }, { kind: 'insta-approve', input: approve, async run(ctx) {
    const { documentId, revision } = ctx.input as z.infer<typeof approve>;
    const document = (deps.documents || workflowDocuments()).get(documentId, ctx.userId);
    if (document.kind !== 'insta-preview') throw new WorkflowError(400, 'Document is not an Insta-Gen preview');
    if (document.revision !== revision) throw new WorkflowError(409, `Stale preview revision ${revision}`);
    const data = document.data as { input: InstaInput; result: InstaResult; edits: { lyrics: string; caption: string } };
    return render(ctx, data.input, data.result, deps, data.edits);
  } }];
}

export function registerInstaGenWorkflows(deps: InstaDeps): void {
  for (const kind of createInstaGenKinds(deps)) registerWorkflowKind(kind);
}
