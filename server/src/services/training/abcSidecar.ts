import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { stemPathFor } from './paths.js';

export function abcSidecarPath(audioPath: string): string {
  return `${stemPathFor(audioPath)}.abc`;
}

export function readAbcSidecar(audioPath: string): string {
  try { return fs.readFileSync(abcSidecarPath(audioPath), 'utf8').trim(); }
  catch { return ''; }
}

/** A failed or empty transcription never replaces a usable score. */
export function writeAbcSidecar(audioPath: string, abc: string): boolean {
  if (!abc.trim()) return false;
  const target = abcSidecarPath(audioPath);
  const tmp = path.join(path.dirname(target), `.abc_${crypto.randomBytes(6).toString('hex')}.tmp`);
  try {
    fs.writeFileSync(tmp, `${abc.trim()}\n`, 'utf8');
    fs.renameSync(tmp, target);
    return true;
  } finally {
    try { fs.unlinkSync(tmp); } catch { /* renamed or never created */ }
  }
}
