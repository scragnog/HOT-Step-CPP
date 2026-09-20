// yue2CaptionJob.ts — write `<stem>.yue2.txt` for a dataset track.
//
// The third caption format. The YuE2 planner is prompted with ONE sentence in
// a fixed order (language → genre → vocal → instruments → mood → production →
// BPM), which is the order its own training captions take; a dataset captioned
// this way trains an adapter whose in-distribution prompt is a sentence Lyric
// Studio can write for a NEW song, instead of a borrowed training caption.
//
// Text-only: the ACE caption already describes the audio (it was written from
// it), and the local labels hold the exact BPM, key and language. A cloud or
// local chat provider rewrites those facts into the planner's shape; MOSS is
// not an option here because it has no such prompt mode.
//
// Same sidecar discipline as the MM3 caption: written beside the audio, an
// existing file backed up to `.prev` ONCE, never re-overwritten.

import fs from 'fs';
import path from 'path';
import { getProvider } from '../lireek/llm/registry.js';
import {
  YUE2_CAPTION_SYSTEM_PROMPT, normalizeYue2Caption, validateYue2Caption,
} from '../lireek/prompts.js';
import type { TrainingSample, TrainingDatasetRow } from './types.js';

export function yue2SidecarPath(audioPath: string): string {
  return `${audioPath.replace(/\.[^.\\/]+$/, '')}.yue2.txt`;
}

const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English', it: 'Italian', es: 'Spanish', fr: 'French', de: 'German', pt: 'Portuguese',
  ja: 'Japanese', ko: 'Korean', zh: 'Chinese', ru: 'Russian', nl: 'Dutch', sv: 'Swedish',
  pl: 'Polish', tr: 'Turkish', fi: 'Finnish', no: 'Norwegian', da: 'Danish',
};

export function languageName(code: string): string {
  const c = String(code ?? '').trim().toLowerCase();
  if (!c) return '';
  return LANGUAGE_NAMES[c] ?? (c.length > 3 ? c : '');
}

export function buildYue2DatasetCaptionPrompt(sample: TrainingSample, ds: TrainingDatasetRow): string {
  const instrumental = !sample.lyrics.trim();
  const lines: string[] = [];
  lines.push(`Dataset: ${ds.name || ds.slug}`, '');
  if (sample.caption.trim()) {
    lines.push('EVIDENCE — a caption written from this recording for a different music model (source material, not a template):',
      `  "${sample.caption.trim()}"`, '');
  }
  if (sample.genre.trim()) lines.push(`Genre tags from the label: ${sample.genre.trim()}`);
  lines.push(`Language: ${instrumental ? 'instrumental (no vocal)' : (languageName(sample.language) || 'unknown — infer it from the lyrics excerpt below')}`);
  if (typeof sample.bpm === 'number' && sample.bpm > 0) {
    lines.push(`BPM (measured): ${Math.round(sample.bpm)} — end the sentence with exactly "${Math.round(sample.bpm)} BPM".`);
  }
  if (instrumental) lines.push('This track is INSTRUMENTAL: write "instrumental" as the language part and name the lead instrument in the vocal part.');
  if (!instrumental) {
    lines.push('', 'Lyrics excerpt (evidence of language and delivery only — never quote it):',
      sample.lyrics.trim().slice(0, 400));
  }
  lines.push('', 'Write the one-sentence YuE2 caption now.');
  return lines.join('\n');
}

/** Write the sidecar with the one-time `.prev` backup. */
export function writeYue2Sidecar(audioPath: string, text: string): string {
  const dst = yue2SidecarPath(audioPath);
  if (fs.existsSync(dst) && !fs.existsSync(`${dst}.prev`)) fs.renameSync(dst, `${dst}.prev`);
  fs.writeFileSync(dst, `${text.trim()}\n`, 'utf8');
  return path.basename(dst);
}

export interface Yue2CaptionCallOptions {
  provider: string;
  model?: string;
  temperature?: number;
  signal?: AbortSignal;
  log?: (level: 'info' | 'warn', message: string) => void;
}

/** One provider call (plus one validation retry). Returns the sentence, or ''
 *  when the model produced nothing usable — the caller decides what to do. */
export async function captionSampleForYue2(
  sample: TrainingSample, ds: TrainingDatasetRow, opts: Yue2CaptionCallOptions,
): Promise<string> {
  const provider = getProvider(opts.provider);
  const model = opts.model || provider.defaultModel;
  const userPrompt = buildYue2DatasetCaptionPrompt(sample, ds);
  const bpm = typeof sample.bpm === 'number' && sample.bpm > 0 ? sample.bpm : undefined;
  const call = async (prompt: string): Promise<string> => {
    let streamed = '';
    const result = await provider.call(
      YUE2_CAPTION_SYSTEM_PROMPT, prompt, model,
      (chunk: string) => { streamed += chunk; },
      { temperature: typeof opts.temperature === 'number' ? opts.temperature : 0.3, top_p: 0.9 },
    );
    return normalizeYue2Caption((result && result.trim()) ? result : streamed, { bpm });
  };
  let text = await call(userPrompt);
  let issues = validateYue2Caption(text);
  if (issues.length) {
    opts.log?.('warn', `YuE2 caption rejected (${issues.join('; ')}) — retrying once`);
    const retry = await call([userPrompt, '', `Your previous attempt was REJECTED for: ${issues.join('; ')}.`,
      'Write it again: one sentence, the fixed order, nothing else.'].join('\n'));
    const retryIssues = validateYue2Caption(retry);
    if (retryIssues.length < issues.length) { text = retry; issues = retryIssues; }
  }
  if (issues.length) opts.log?.('warn', `YuE2 caption kept with issues: ${issues.join('; ')}`);
  return text.trim();
}
