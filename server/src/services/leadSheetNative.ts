// Native lead-sheet source (yourmt3-chordnet): spawns `ace-midi --lead-sheet`, reads its
// JSON (raw seconds, docs/dev/engine.md "ace-midi YourMT3+") and maps it to a CoverLeadSheetResult.
// The two GGUFs are converted locally (engine/tools/convert-yourmt3.py, convert-chordnet.py);
// there is no registry entry or download, because redistribution is not cleared.
import { spawn, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { config, PORTABLE_MODE, PROJECT_ROOT } from '../config.js';
import { trackBusy } from '../dataroot/pointer.js';
import type { CoverLeadSheetResult } from '../contracts/studioWorkflows.js';
import { engineFormatPath } from './audioConvert.js';
import { nativeCoverLeadSheet, tryAbcCoverLeadSheet } from './coverLeadSheet.js';
import { aceMidiExe, BEAT_MODEL, isModelDownloaded, modelDir, MODELS_DIR } from './muscriptor.js';

export const YOURMT3_FILE = 'yourmt3-F32.gguf';
export const CHORDNET_FILE = 'chordnet-F32.gguf';
/** upstream's constant-BPM sixteenth grid; the rolls come from the JSON's BPM, never Beat This!. */
export const NATIVE_GRID_IDENTITY = 'upstream-constant-bpm-sixteenth';

const TEMPLATES = PORTABLE_MODE
  ? path.join(PROJECT_ROOT, 'server', 'data', 'lead-sheet', 'chordnet-templates.tsv')
  : path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'data', 'lead-sheet', 'chordnet-templates.tsv');

export interface NativeFiles { exe: string | null; yourmt3: string; chordnet: string; templates: string; beatDir: string; beatReady: boolean }
export const nativeFiles = (): NativeFiles => ({ exe: aceMidiExe(), yourmt3: path.join(MODELS_DIR, YOURMT3_FILE),
  chordnet: path.join(MODELS_DIR, CHORDNET_FILE), templates: TEMPLATES, beatDir: modelDir(BEAT_MODEL), beatReady: isModelDownloaded(BEAT_MODEL) });

/** Readiness text for the picker: names each missing file, or null when everything is there. */
export function nativeMissingMessage(f: NativeFiles = nativeFiles()): string | null {
  const missing: string[] = [];
  if (!f.exe) missing.push('the ace-midi executable');
  if (!fs.existsSync(f.yourmt3)) missing.push(`${YOURMT3_FILE} (YourMT3+)`);
  if (!fs.existsSync(f.chordnet)) missing.push(`${CHORDNET_FILE} (ChordNet)`);
  if (!fs.existsSync(f.templates)) missing.push('chordnet-templates.tsv');
  if (!f.beatReady) missing.push('the Beat This! model (muscriptor-beat-this)');
  return missing.length ? `YourMT3+/ChordNet is not installed: missing ${missing.join(', ')}. The two models are converted locally `
    + `with engine/tools/convert-yourmt3.py and convert-chordnet.py; put them in ${MODELS_DIR}. There is no download yet.` : null;
}

// Upstream symbolic.py _ROOTS / _CHORD_INTERVALS: exact quality first, then the first name the quality starts with.
const ROOTS: Record<string, number> = { C: 0, 'C#': 1, Db: 1, D: 2, 'D#': 3, Eb: 3, E: 4, F: 5, 'F#': 6, Gb: 6, G: 7, 'G#': 8, Ab: 8, A: 9, 'A#': 10, Bb: 10, B: 11 };
const INTERVALS: Array<[string, number[]]> = [['maj', [0, 4, 7]], ['min', [0, 3, 7]], ['dim', [0, 3, 6]], ['aug', [0, 4, 8]],
  ['7', [0, 4, 7, 10]], ['maj7', [0, 4, 7, 11]], ['min7', [0, 3, 7, 10]], ['sus2', [0, 2, 7]], ['sus4', [0, 5, 7]], ['5', [0, 7]]];
/** `N`, a label without `:`, an unknown root or an unknown quality give no notes. */
export function chordLabelPitches(label: string): number[] {
  const at = label.indexOf(':');
  if (at < 0 || !(label.slice(0, at) in ROOTS)) return [];
  const quality = label.slice(at + 1);
  const found = INTERVALS.find(([name]) => name === quality) ?? INTERVALS.find(([name]) => quality.startsWith(name));
  return (found?.[1] ?? []).map(i => (ROOTS[label.slice(0, at)] + i) % 12);
}

interface RawNote { onset: number; offset: number; pitch: number }
export interface LeadSheetJson { version: number; bpm: number; bpm_rejection: string | null; vocal: RawNote[]; vocal_rejection: string | null;
  drums: RawNote[]; chords: Array<[string, number, number]> }
const note = (n: RawNote): RawNote => ({ onset: n.onset, offset: n.offset, pitch: n.pitch });

/** JSON + the ABC written beside it -> result. A rejection (or a score the roll cannot take) returns the ABC without
 *  native timing, so rolls come from the ABC path; nothing is substituted in the raw events. */
export function nativeSheetFromJson(abc: string, json: LeadSheetJson): { sheet?: CoverLeadSheetResult; notice?: string } {
  if (json?.version !== 1 || !Array.isArray(json.vocal) || !Array.isArray(json.drums) || !Array.isArray(json.chords)) {
    throw new Error('ace-midi wrote an unreadable lead-sheet JSON');
  }
  const rejection = [json.bpm_rejection, json.vocal_rejection].filter(Boolean).join(' ');
  const abcOnly = (why: string) => ({ sheet: tryAbcCoverLeadSheet(abc, 'yourmt3-chordnet'),
    notice: `Native timing unavailable (${why}); rolls will come from the ABC score.` });
  if (rejection) return abcOnly(rejection);
  try {
    const chords = json.chords.flatMap(([label, onset, offset]) => chordLabelPitches(label).map(pitch => ({ onset, offset, pitch })));
    return { sheet: nativeCoverLeadSheet(abc, { conversionVersion: 1, path: 'native', bpm: json.bpm,
      rawEvents: { melody: json.vocal.map(note), chords, drums: json.drums.map(note) },
      grid: { identity: NATIVE_GRID_IDENTITY, data: { bpm: json.bpm, subdivision: 16 } } }) };
  } catch (error) { return abcOnly(error instanceof Error ? error.message : String(error)); }
}

export interface NativeDeps {
  files: NativeFiles; dir: string; idleMs: number; capMs: number;
  spawn: (exe: string, args: string[]) => ChildProcess;
  stall: (onTimeout: (why: 'idle' | 'cap') => void, idleMs: number, capMs: number) => { touch(): void; clear(): void };
}

/** The run itself: no GPU lane, so tests can drive it with a stub process. */
export async function runNativeLeadSheet(audioPath: string, signal: AbortSignal, deps: NativeDeps) {
  if (signal.aborted) throw new Error('Cancelled');
  const { files } = deps;
  const message = nativeMissingMessage(files);
  if (message) throw new Error(message);
  fs.mkdirSync(deps.dir, { recursive: true });
  try {
    const out = path.join(deps.dir, 'lead-sheet.json'), abcOut = path.join(deps.dir, 'score.abc');
    const args = ['--lead-sheet', engineFormatPath(audioPath, deps.dir), '--yourmt3-model', files.yourmt3, '--chord-model', files.chordnet,
      '--chord-vocab', files.templates, '--out', out, '--beat-model', files.beatDir, '--abc-out', abcOut, '--abc-chordnet'];
    const child = deps.spawn(files.exe!, args);
    const tail: string[] = [];
    const killer = deps.stall(why => { console.error(`[LeadSheetNative] ${why === 'idle' ? 'no progress' : 'timed out'} - killing ace-midi`); child.kill(); }, deps.idleMs, deps.capMs);
    const onData = (buf: Buffer, keep: boolean) => { killer.touch(); if (keep) for (const l of buf.toString('utf8').split(/[\r\n]+/)) if (l.trim()) { tail.push(l.trim()); if (tail.length > 30) tail.shift(); } };
    child.stdout?.on('data', b => onData(b, false));
    child.stderr?.on('data', b => onData(b, true));
    const abort = () => child.kill();
    signal.addEventListener('abort', abort, { once: true });
    const code = await new Promise<number | null>((resolve, reject) => {
      child.on('error', err => reject(new Error(`Failed to launch ace-midi: ${err.message}`)));
      child.on('close', (c, sig) => resolve(sig ? null : c));
    }).finally(() => { killer.clear(); signal.removeEventListener('abort', abort); });
    if (signal.aborted) throw new Error('Cancelled');
    if (code !== 0) throw new Error(`ace-midi exited with code ${code}: ${tail.join(' | ')}`);
    if (!fs.existsSync(out)) throw new Error('ace-midi finished but wrote no lead-sheet JSON');
    // Trimmed as approve-score trims, so an unedited approval still equals the sheet's ABC.
    const abc = fs.existsSync(abcOut) ? fs.readFileSync(abcOut, 'utf8').trim() : '';
    const { sheet, notice } = nativeSheetFromJson(abc, JSON.parse(fs.readFileSync(out, 'utf8')));
    return { abc, scoreSource: 'yourmt3-chordnet' as const, leadSheet: sheet, notice };
  } finally { fs.rmSync(deps.dir, { recursive: true, force: true }); }
}

/** Cover Prepare entry: MIDI Studio's GPU lane, stall timer and path rules. */
export async function transcribeNativeLeadSheet(url: string, signal: AbortSignal) {
  const midi = await import('../routes/midiStudio.js');
  const deps: NativeDeps = { files: nativeFiles(), dir: path.join(config.data.dir, 'tmp', 'lead-sheet-native', randomUUID()),
    idleMs: midi.TRANSCRIBE_IDLE_TIMEOUT_MS, capMs: midi.TRANSCRIBE_MAX_MS, stall: midi.startStallTimer,
    spawn: (exe, args) => spawn(exe, args, { windowsHide: true }) };
  const id = path.basename(deps.dir);
  let result!: Awaited<ReturnType<typeof runNativeLeadSheet>>;
  await trackBusy('MIDI', midi.runOnMidiLane(id, async () => { result = await runNativeLeadSheet(midi.resolveAudioPath(url), signal, deps); }));
  return result;
}
