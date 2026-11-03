// backends/yue2/sourceGuide.ts — "Follow source" for YuE2 covers.
//
// A cover normally reaches YuE2 only as its transcribed score. With a guide,
// the source recording also steers the composer: `ace-train yue2-guide` runs
// the real-audio tokenizer over it and writes its top-K semantic codes per
// 25 Hz frame, and the render points the engine at that file (guide_path).
// Engine side and the measurements behind the defaults: engine/src/yue2/yue2-guide.h.

import { execFile } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

import { config, getFFmpegPath } from '../../../config.js';
import { aceTrainExe } from '../../training/aceTrain.js';
import { missingYue2TokenizeModels, resolveYue2TokenizerModel, yue2TokenizerSetting, YUE2_TOKENIZERS } from '../../training/yue2Tokenize.js';
import type { Yue2SynthRequest } from './client.js';

/** Candidates kept per frame in the file. The render can use fewer (k). */
export const YUE2_GUIDE_TOPK = 64;

const SCHEDULES = ['flat', 'up', 'down', 'sin', 'inv_sin'] as const;
const LFOS = ['off', 'sine', 'triangle', 'saw', 'square'] as const;

export interface Yue2CoverGuide {
  /** 0..1: share of each frame's distribution taken from the tokenizer. */
  scale: number;
  /** Candidates per frame, 1..64. */
  k: number;
  /** Only the candidates may be drawn. */
  restrict: boolean;
  schedule: typeof SCHEDULES[number];
  lfo: typeof LFOS[number];
  lfoPeriodSec: number;
  lfoDepth: number;
}

export const YUE2_COVER_GUIDE_DEFAULTS: Yue2CoverGuide = {
  scale: 0.5, k: 32, restrict: false, schedule: 'flat', lfo: 'off', lfoPeriodSec: 8, lfoDepth: 0,
};

/** Validates `yue2Cover.guide`. Absent or null = no guide. Throws on a bad value. */
export function yue2CoverGuide(raw: unknown): Yue2CoverGuide | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== 'object' || Array.isArray(raw)) throw new Error('yue2Cover.guide must be an object.');
  const g = { ...YUE2_COVER_GUIDE_DEFAULTS, ...(raw as Partial<Yue2CoverGuide>) };
  const num = (v: unknown, lo: number, hi: number) => typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi;
  if (!num(g.scale, 0, 1)) throw new Error('yue2Cover.guide.scale must be between 0 and 1.');
  if (!num(g.k, 1, YUE2_GUIDE_TOPK) || !Number.isInteger(g.k)) throw new Error(`yue2Cover.guide.k must be an integer from 1 to ${YUE2_GUIDE_TOPK}.`);
  if (typeof g.restrict !== 'boolean') throw new Error('yue2Cover.guide.restrict must be a boolean.');
  if (!SCHEDULES.includes(g.schedule)) throw new Error(`yue2Cover.guide.schedule must be one of ${SCHEDULES.join(', ')}.`);
  if (!LFOS.includes(g.lfo)) throw new Error(`yue2Cover.guide.lfo must be one of ${LFOS.join(', ')}.`);
  if (!num(g.lfoPeriodSec, 0.5, 120)) throw new Error('yue2Cover.guide.lfoPeriodSec must be between 0.5 and 120.');
  if (!num(g.lfoDepth, 0, 1)) throw new Error('yue2Cover.guide.lfoDepth must be between 0 and 1.');
  return { scale: g.scale, k: g.k, restrict: g.restrict, schedule: g.schedule, lfo: g.lfo,
    lfoPeriodSec: g.lfoPeriodSec, lfoDepth: g.lfoDepth };
}

/** Why a guide cannot apply to this cover, or ''. The tokenizer's codes carry
 *  the source's pitch and timing, so a shifted key or a set tempo would pull
 *  the render back toward the original. */
export function yue2GuideSkipReason(cover: { keyShift?: number; key?: string; tempo?: unknown }): string {
  if ((cover.keyShift ?? 0) !== 0 || (cover.key && cover.key !== 'source')) {
    return 'Follow source is off for this render: the key is shifted, and the source guide would pull it back to the original key.';
  }
  if (typeof cover.tempo === 'number') {
    return 'Follow source is off for this render: a set BPM changes the timing the source guide is aligned to.';
  }
  return '';
}

/** The engine fields for a guide file. */
export function yue2GuideWire(g: Yue2CoverGuide, guidePath: string): Partial<Yue2SynthRequest> {
  return {
    guide_path: guidePath,
    guide_scale: g.scale,
    guide_k: g.k,
    ...(g.restrict ? { guide_outside: -100 } : {}),
    ...(g.schedule !== 'flat' ? { guide_schedule: g.schedule } : {}),
    ...(g.lfo !== 'off' && g.lfoDepth > 0
      ? { guide_lfo: g.lfo, guide_lfo_period_sec: g.lfoPeriodSec, guide_lfo_depth: g.lfoDepth } : {}),
  };
}

function guideCacheDir(): string {
  return path.join(config.data.dir, 'cache', 'yue2-guide');
}

/** The guide file for a source recording, built on first use and cached by
 *  the file's path, size and mtime. Throws with a user-facing message. */
export async function buildYue2Guide(srcPath: string, log: (msg: string) => void, signal?: AbortSignal): Promise<string> {
  if (!fs.existsSync(srcPath)) throw new Error(`Follow source: the source recording is missing (${path.basename(srcPath)}).`);
  // The cover tokenizer (preferences `yue2-tokenizer`.cover, the cover panel's picker).
  const tokenizer = yue2TokenizerSetting().cover;
  const missing = missingYue2TokenizeModels(tokenizer);
  if (missing.length) {
    throw new Error(`Follow source needs ${missing.join(', ')}. Download "${YUE2_TOKENIZERS[tokenizer].registryId}" in the Model Manager, or turn Follow source off.`);
  }
  const exe = aceTrainExe();
  if (!exe) throw new Error('Follow source needs the ace-train engine binary, which this install does not have.');

  const st = fs.statSync(srcPath);
  const key = crypto.createHash('sha1')
    .update(`${path.resolve(srcPath)}|${st.size}|${st.mtimeMs}|${YUE2_GUIDE_TOPK}|v1${tokenizer === 'v9' ? '' : `|${tokenizer}`}`)
    .digest('hex').slice(0, 20);
  const dir = guideCacheDir();
  const out = path.join(dir, `${key}.y2gd`);
  if (fs.existsSync(out)) {
    log(`[YuE2] Source guide cached: ${out}`);
    return out;
  }
  fs.mkdirSync(dir, { recursive: true });
  // Written beside the cache and renamed on success, so a killed run never
  // leaves a truncated file the cache would keep serving.
  const tmp = `${out}.${process.pid}.tmp`;
  const args = ['yue2-guide', '--audio', srcPath, '--out', tmp, '--topk', String(YUE2_GUIDE_TOPK),
    '--models', config.aceServer.models];
  const ff = getFFmpegPath();
  if (ff) args.push('--ffmpeg', ff);
  // WannabeTokenizer reads a track at its own level (its reference never
  // normalises), so the engine's -14 LUFS default is turned off for it.
  if (tokenizer !== 'v9') args.push('--tok', resolveYue2TokenizerModel(tokenizer), '--lufs', '0');
  log(`[YuE2] Reading the source for Follow source (${YUE2_TOKENIZERS[tokenizer].label}): ${path.basename(srcPath)}`);
  const started = Date.now();
  const stderr = await new Promise<string>((resolve, reject) => {
    const child = execFile(exe, args, { timeout: 10 * 60_000, maxBuffer: 16 * 1024 * 1024, windowsHide: true },
      (err, _stdout, errOut) => {
        signal?.removeEventListener('abort', onAbort);
        if (err) {
          fs.rmSync(tmp, { force: true });
          const tail = String(errOut).trim().split(/\r?\n/).filter(l => l.includes('[yue2-guide]')).slice(-2).join(' | ');
          reject(new Error(`Follow source could not read the recording: ${tail || err.message}`));
        } else resolve(String(errOut));
      });
    const onAbort = () => child.kill();
    signal?.addEventListener('abort', onAbort, { once: true });
  });
  fs.renameSync(tmp, out);
  const summary = stderr.split(/\r?\n/).find(l => l.includes('[yue2-guide] wrote')) ?? '';
  log(`[YuE2] Source guide built in ${((Date.now() - started) / 1000).toFixed(1)} s ${summary.trim()}`);
  return out;
}
