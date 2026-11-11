import fs from 'node:fs';
import type { Yue2SynthRequest } from './client.js';
import { buildYue2Guide } from './sourceGuide.js';
import { resolveSourceAudioPath } from '../../generation/sourceAudio.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

import type { Yue2Continuation } from '../../../contracts/yue2Continuation.js';
export type { Yue2Continuation } from '../../../contracts/yue2Continuation.js';

export function jobContinuation(job: GenerationJob): Yue2Continuation | undefined {
  return yue2Continuation(job.envelope.options?.yue2?.yue2Continuation);
}

/** Capture the source and section together when the job is submitted. */
export function yue2Continuation(raw: unknown): Yue2Continuation | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== 'object' || Array.isArray(raw)) throw new Error('yue2Continuation must be an object.');
  const { sourceAudioUrl, start, end } = raw as Yue2Continuation;
  if (typeof sourceAudioUrl !== 'string' || !/^\/(references|audio)\/[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+$/.test(sourceAudioUrl)) {
    throw new Error('Continuation needs an uploaded recording.');
  }
  if (typeof start !== 'number' || typeof end !== 'number' || !Number.isFinite(start) || !Number.isFinite(end)
    || start < 0 || end <= start || end - start > 30 || Math.floor(end * 25) <= Math.floor(start * 25)) {
    throw new Error('Choose a section of at least one frame (0.04 s) and at most 30 seconds.');
  }
  return { sourceAudioUrl, start, end };
}

/** Y2GD v1 stores raw semantic IDs, best first in each 25 Hz frame. */
export function continuationCodes(buf: Buffer, section: Pick<Yue2Continuation, 'start' | 'end'>): number[] {
  if (buf.length < 16 || buf.toString('ascii', 0, 4) !== 'Y2GD' || buf.readUInt32LE(4) !== 1) {
    throw new Error('Continuation tokenizer returned an invalid guide.');
  }
  const frames = buf.readUInt32LE(8), k = buf.readUInt32LE(12);
  if (!frames || !k || k > 32768 || buf.length !== 16 + frames * k * 8) throw new Error('Continuation guide is truncated.');
  const first = Math.floor(section.start * 25), last = Math.floor(section.end * 25);
  if (first < 0 || last > frames || last <= first || last - first > 750) throw new Error('The selected section is outside the recording.');
  return Array.from({ length: last - first }, (_, i) => {
    const code = buf.readInt32LE(16 + (first + i) * k * 4);
    if (code < 0 || code >= 32768) throw new Error('Continuation guide contains an invalid semantic code.');
    return code;
  });
}

export function applyContinuation(req: Yue2SynthRequest, codes: number[]): void {
  Object.assign(req, { codec_ids: codes, continue_codec_ids: true, cot: 'off', lm_batch_size: 1 });
  delete req.abc;
  delete req.lyric_schedule;
  delete req.semantic_retries;
  delete req.songs;
}

export async function buildContinuation(section: Yue2Continuation, log: (msg: string) => void, signal?: AbortSignal): Promise<number[]> {
  let file: string;
  try {
    file = await buildYue2Guide(resolveSourceAudioPath(section.sourceAudioUrl), log, signal);
  } catch (err) {
    signal?.throwIfAborted();
    throw new Error(String(err instanceof Error ? err.message : err).replaceAll('Follow source', 'Continue from recording'));
  }
  signal?.throwIfAborted();
  return continuationCodes(fs.readFileSync(file), section);
}
