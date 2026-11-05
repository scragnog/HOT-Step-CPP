import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { aceMidiExe, exportArgs, MUSCRIPTOR_MODELS, type MuscriptorModel, type MidiExportOptions } from './muscriptor.js';

/** Consume readiness's resolved choice instead of choosing a second model. */
export function coverMidiModel(readiness: { ready: boolean; model: string | null; message: string | null }): MuscriptorModel {
  const model = MUSCRIPTOR_MODELS.find(model => `muscriptor-${model}` === readiness.model);
  if (!readiness.ready || !model) throw new Error(readiness.message || 'MuScriptor and Beat This! must be installed');
  return model;
}

export class MidiAbcError extends Error {
  constructor(message: string, public status: number, public summary?: unknown) { super(message); }
}

export function midiAbcArgs(dir: string, opts: MidiExportOptions = {}, ins?: number, melodyOnly = false) {
  return ['--export', path.join(dir, 'events.jsonl'), '--beats-in', path.join(dir, 'beats.json'),
    '--out', path.join(dir, 'abc.export.tmp'), '--abc-out', path.join(dir, 'score.abc'), ...exportArgs(opts),
    ...(Number.isInteger(ins) && ins! >= 0 && ins! < 128 ? ['--abc-ins', String(ins)] : []),
    ...(melodyOnly ? ['--abc-melody-only'] : [])];
}

/** Same Beat This! grid and exporter as MIDI Studio; no inference or second snap. */
export async function exportMidiAbc(dir: string, opts: MidiExportOptions = {}, ins?: number,
  melodyOnly = false, signal?: AbortSignal) {
  if (signal?.aborted) throw new Error('Cancelled');
  if (!fs.existsSync(path.join(dir, 'events.jsonl')) || !fs.existsSync(path.join(dir, 'beats.json'))) {
    throw new MidiAbcError('This transcription has no beat grid. Download the tempo model and transcribe again.', 409);
  }
  const exe = aceMidiExe();
  if (!exe) throw new MidiAbcError('ace-midi engine binary not found', 503);
  let stdout = '', stderr = '';
  try {
    const code = await new Promise<number | null>((resolve, reject) => {
      const child = spawn(exe, midiAbcArgs(dir, opts, ins, melodyOnly), { windowsHide: true, signal });
      const timer = setTimeout(() => child.kill(), 120_000);
      child.stdout.on('data', (b: Buffer) => { stdout += b.toString('utf8'); });
      child.stderr.on('data', (b: Buffer) => { stderr = (stderr + b.toString('utf8')).slice(-4096); });
      child.on('error', reject);
      child.on('close', code => resolve(code));
      child.once('close', () => clearTimeout(timer));
    });
    const line = stdout.split(/\r?\n/).find(line => line.startsWith('{"type":"abc"'));
    const summary = line ? JSON.parse(line) : null;
    if (code !== 0 || !summary) throw new Error(`ace-midi failed (${code}): ${stderr.trim()}`);
    if (!summary.ok) throw new MidiAbcError(summary.error || 'The notes could not be written as ABC', 422, summary);
    return { abc: fs.readFileSync(path.join(dir, 'score.abc'), 'utf8'), summary };
  } finally { fs.rmSync(path.join(dir, 'abc.export.tmp'), { force: true }); }
}
