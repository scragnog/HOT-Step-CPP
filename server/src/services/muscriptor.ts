// muscriptor.ts — MuScriptor (audio→MIDI) shared helpers
//
// MuScriptor is developed by Kyutai & Mirelo (Rouard, Krause, Roebel,
// Simon-Gabriel, Défossez — arXiv:2607.08168). Code MIT; model weights
// CC BY-NC 4.0 and GATED on Hugging Face (users request access + read token).
//
// NOTE (2026-07-16): the original integration ran the upstream Python CLI in
// a managed venv. That approach was removed — transcription is being ported
// to a native GGML binary (`ace-midi`). Design: docs/plans/muscriptor-cpp-port.md.
// What remains here is the part the C++ path also needs: the Hugging Face
// token store used to download the gated weights.
//
// Rob's local venv at data/muscriptor/venv is intentionally NOT deleted —
// it is the numerics-validation oracle for the port (see design doc §6).

import path from 'path';
import fs from 'fs';
import { config } from '../config.js';
import { muscriptorRoots, resolveModelDir, targetRoot } from './muscriptorPaths.js';

export const MUSCRIPTOR_DIR = path.join(config.data.dir, 'muscriptor');

export const MUSCRIPTOR_MODELS = ['small', 'medium', 'large'] as const;
export type MuscriptorModel = typeof MUSCRIPTOR_MODELS[number];

/** Beat This! (CPJKU, MIT, not gated): beats and downbeats that give the
 *  MIDI its tempo, time signature and bar lines (ace-midi --beat-model). */
export const BEAT_MODEL = 'beat-this';
/** dettato (KobiMusic, Apache-2.0, not gated): four weight folders under the same root, named as upstream's
 *  repos are. ace-midi --dettato-model takes the root, so all four sit side by side. The repairer is optional
 *  at run time (the Repair setting). */
export const DETTATO_PARTS = ['dettato', 'dettato-evidence', 'dettato-grouper', 'dettato-drums'] as const;
export type DettatoPart = typeof DETTATO_PARTS[number];
export const DETTATO_MODEL = 'dettato';
/** What MIDI Studio can transcribe with. */
export type MidiModel = MuscriptorModel | typeof DETTATO_MODEL;
export const isMidiModel = (m: unknown): m is MidiModel => m === DETTATO_MODEL || (MUSCRIPTOR_MODELS as readonly unknown[]).includes(m);
export type DownloadableModel = MuscriptorModel | typeof BEAT_MODEL | DettatoPart;

export const HF_MODEL_REPOS: Record<MuscriptorModel | typeof BEAT_MODEL, string> = {
  small: 'MuScriptor/muscriptor-small',
  medium: 'MuScriptor/muscriptor-medium',
  large: 'MuScriptor/muscriptor-large',
  // final0 as safetensors; tensors verified bit-identical to the official checkpoint
  [BEAT_MODEL]: 'safe-models/beat-this-final0',
};

// ── Hugging Face access token ────────────────────────────────────────────

const HF_TOKEN_PATH = path.join(MUSCRIPTOR_DIR, 'hf_token');

export function getHfToken(): string | null {
  try {
    const t = fs.readFileSync(HF_TOKEN_PATH, 'utf-8').trim();
    return t || null;
  } catch { return null; }
}

export function setHfToken(token: string): void {
  fs.mkdirSync(MUSCRIPTOR_DIR, { recursive: true });
  const t = (token || '').trim();
  if (!t) {
    fs.rmSync(HF_TOKEN_PATH, { force: true });
  } else {
    // A token that is not a plain hf_ string surfaces later as a gated-repo
    // 401 that looks like an access problem (#183). Refuse it here instead.
    if (!/^hf_[A-Za-z0-9]{10,}$/.test(t)) {
      const err = new Error('That does not look like a Hugging Face token: it should start with hf_ and contain only letters and digits.');
      (err as any).status = 400;
      throw err;
    }
    fs.writeFileSync(HF_TOKEN_PATH, t, { encoding: 'utf-8' });
  }
}

/** Heuristic: does this failure look like a gated-model / auth problem? */
export function looksLikeGatedError(text: string): boolean {
  return /gated|401|403|unauthorized|forbidden|restricted|access to model|awaiting a review|accept the conditions|not authenticated|invalid (user )?token|authentication/i.test(text);
}

// ── ace-midi engine binary ───────────────────────────────────────────────

/** Absolute path to ace-midi, or null if not built/shipped. Lives next to
 *  ace-server (same build output / portable layout). */
export function aceMidiExe(): string | null {
  const dir = path.dirname(config.aceServer.exe);
  const exe = path.join(dir, process.platform === 'win32' ? 'ace-midi.exe' : 'ace-midi');
  return fs.existsSync(exe) ? exe : null;
}

// ── Model weights (gated on HF — downloaded with the user's read token) ──

// Weights live in <models>/muscriptor/<variant>; a complete pre-#228 download
// under <library>/models/muscriptor is still used. MUSCRIPTOR_MODELS_DIR
// (Settings > Environment) overrides both (#86). Rules: muscriptorPaths.ts.
// Read at import, so a change needs a restart.
export const MODEL_ROOTS = muscriptorRoots(process.env.MUSCRIPTOR_MODELS_DIR, config.aceServer.models, config.data.dir);
/** Where downloads go and what Settings shows. */
export const MODELS_DIR = targetRoot(MODEL_ROOTS);

export function modelDir(m: DownloadableModel): string {
  return resolveModelDir(MODEL_ROOTS, m).dir;
}

export function isModelDownloaded(m: DownloadableModel): boolean {
  return resolveModelDir(MODEL_ROOTS, m).complete;
}

/** The folder holding dettato's four model folders (--dettato-model). Readiness, downloads and the launch all use
 *  it: resolveModelDir never reads a dettato folder from anywhere else. */
export const dettatoRoot = (): string => targetRoot(MODEL_ROOTS);

/** The folders a dettato run reads: the repairer's only when it is on. */
export const dettatoParts = (repair: boolean): DettatoPart[] => DETTATO_PARTS.filter(p => repair || p !== 'dettato-evidence');

export const isDettatoReady = (repair = true): boolean => dettatoParts(repair).every(isModelDownloaded);

/** Weights every dettato run needs, then the repairer's, by Model Manager id. */
export const dettatoMissing = (repair = true): string[] => dettatoParts(repair).filter(p => !isModelDownloaded(p));

export interface ModelDownloadState {
  downloading: boolean;
  receivedBytes: number;
  totalBytes: number;
  error?: string;
  gated?: boolean;
}

/** Live download progress for one model, supplied by the caller so this module
 *  holds no download state: the jobs live in the Model Manager's service. */
export type ModelDownloadLookup = (m: DownloadableModel) => ModelDownloadState | undefined;

export function getModelStates(lookup: ModelDownloadLookup): Record<string, ModelDownloadState & { downloaded: boolean; sizeBytes: number }> {
  const out: Record<string, any> = {};
  for (const m of [...MUSCRIPTOR_MODELS, BEAT_MODEL, ...DETTATO_PARTS] as DownloadableModel[]) {
    const dl = lookup(m);
    let sizeBytes = 0;
    const downloaded = isModelDownloaded(m);
    if (downloaded) {
      try { sizeBytes = fs.statSync(path.join(modelDir(m), 'model.safetensors')).size; } catch { /* ignore */ }
    }
    out[m] = {
      downloaded,
      sizeBytes,
      downloading: dl?.downloading ?? false,
      receivedBytes: dl?.receivedBytes ?? 0,
      totalBytes: dl?.totalBytes ?? 0,
      error: dl?.error,
      gated: dl?.gated,
    };
  }
  return withDettatoTotal(out);
}

type StateRow = ModelDownloadState & { downloaded: boolean; sizeBytes: number };
/** dettato is one choice in MIDI Studio but four downloads, so 'dettato' becomes their total. Each part keeps
 *  its own row (the transcriber's moves to 'dettato-transcriber'), so a client that needs only some parts,
 *  like the repairer being off, can still read them one by one. */
export function withDettatoTotal(out: Record<string, StateRow>): Record<string, StateRow> {
  const parts = DETTATO_PARTS.map(p => out[p]);
  out['dettato-transcriber'] = out[DETTATO_MODEL];
  out[DETTATO_MODEL] = {
    downloaded: parts.every(s => s.downloaded),
    sizeBytes: parts.reduce((n, s) => n + s.sizeBytes, 0),
    downloading: parts.some(s => s.downloading),
    receivedBytes: parts.reduce((n, s) => n + s.receivedBytes, 0),
    totalBytes: parts.reduce((n, s) => n + s.totalBytes, 0),
    error: parts.find(s => s.error)?.error,
    gated: undefined,
  };
  return out;
}

// ── Beat grid (ace-midi --beats-out) ─────────────────────────────────────

/** The grid a tempo-aware MIDI was written on, in seconds of the audio. */
export interface MidiGrid {
  bpm: number;
  /** 0 when the meter was not clear enough to write a time signature */
  beatsPerBar: number;
  /** false: the MIDI carries a tempo map that follows the performance */
  steady: boolean;
  onsetDelay: number;
  /** grid steps per beat the notes sit on (0: none found) */
  subdivision: number;
  /** add to a MIDI time (seconds from tick 0) to get the audio time */
  audioShift: number;
  beats: number[];
  bars: number[];
}

export function readGrid(beatsJsonText: string): MidiGrid | null {
  try {
    const g = JSON.parse(beatsJsonText)?.grid;
    if (!g || typeof g.bpm !== 'number') return null;
    return {
      bpm: g.bpm, beatsPerBar: g.beats_per_bar || 0, steady: !!g.steady, onsetDelay: g.onset_delay || 0,
      subdivision: g.subdivision || 0, audioShift: g.audio_shift_s || 0, beats: g.beats || [], bars: g.bars || [],
    };
  } catch { return null; }
}

/** A parsed bar-aligned MIDI moved back onto the audio's clock, with its grid,
 *  so the piano roll lines up with the source. */
export function alignToAudio<T extends { durationSec: number; notes: { start: number; duration: number }[] }>(
  parsed: T, grid: MidiGrid,
): T & { grid: MidiGrid } {
  const notes = parsed.notes.map(n => ({ ...n, start: n.start + grid.audioShift }));
  const durationSec = notes.reduce((m, n) => Math.max(m, n.start + n.duration), 0);
  return { ...parsed, notes, durationSec, grid };
}

/** ace-midi export flags from the UI's options; unknown or empty fields are skipped. */
export interface MidiExportOptions {
  bpm?: number;
  beatsPerBar?: number;
  firstDownbeat?: number;
  tempoMode?: 'auto' | 'map' | 'fixed';
  beatScale?: number;
  quantize?: boolean;
  grid?: number;
  strength?: number;
  fillGaps?: number;
  trimLeadingBars?: boolean;
}

/** ace-midi flags for speculative decoding (upstream MuScriptor PR #95): the
 *  small model drafts four tokens at a time and the large one checks them in a
 *  single pass, so the output is the large model's own. Only large gains:
 *  measured on a 4 min track, CUDA, large 122 s -> 79 s; medium 61 s -> 60 s. */
export function speculativeArgs(model: MuscriptorModel, enabled: boolean, smallReady: boolean): string[] {
  return enabled && model === 'large' && smallReady
    ? ['--draft-model', modelDir('small'), '--speculative-k', '4']
    : [];
}

export function exportArgs(o: MidiExportOptions): string[] {
  const a: string[] = [];
  const num = (flag: string, v: unknown, ok: (x: number) => boolean) => {
    if (typeof v === 'number' && Number.isFinite(v) && ok(v)) a.push(flag, String(v));
  };
  num('--bpm', o.bpm, x => x >= 20 && x <= 400);
  num('--beats-per-bar', o.beatsPerBar, x => Number.isInteger(x) && x >= 1 && x <= 16);
  num('--first-downbeat', o.firstDownbeat, x => x >= 0);
  if (o.tempoMode === 'map') a.push('--tempo-map');
  if (o.tempoMode === 'fixed') a.push('--fixed-tempo');
  num('--beat-scale', o.beatScale, x => [0.25, 0.5, 2, 4].includes(x));
  if (o.quantize) {
    a.push('--quantize');
    num('--grid', o.grid, x => Number.isInteger(x) && x >= 1 && x <= 24);
    num('--quantize-strength', o.strength, x => x > 0 && x <= 1);
  }
  num('--fill-gaps', o.fillGaps, x => x > 0 && x <= 4);
  if (o.trimLeadingBars) a.push('--trim-leading-bars');
  return a;
}

// ── Partial MIDI from the streamed events ────────────────────────────────

// ace-midi's instrument names (instrument_for_program in engine/tools/ace-midi.cpp)
// back to their GM programs; names it formats as program_<n> parse below.
const INSTRUMENT_PROGRAMS: Record<string, number> = {
  acoustic_piano: 0, electric_piano: 2, chromatic_percussion: 8, organ: 16,
  acoustic_guitar: 24, clean_electric_guitar: 26, distorted_electric_guitar: 29,
  acoustic_bass: 32, electric_bass: 33, violin: 40, viola: 41, cello: 42,
  contrabass: 43, orchestral_harp: 46, timpani: 47, string_ensemble: 48,
  synth_strings: 50, voice: 52, orchestra_hit: 55, trumpet: 56, trombone: 57,
  tuba: 58, french_horn: 60, brass_section: 61, soprano_and_alto_sax: 64,
  tenor_sax: 66, baritone_sax: 67, oboe: 68, english_horn: 69, bassoon: 70,
  clarinet: 71, flutes: 72, synth_lead: 80, synth_pad: 88,
};

function varLen(n: number): number[] {
  const out = [n & 0x7f];
  while ((n >>= 7) > 0) out.unshift((n & 0x7f) | 0x80);
  return out;
}

/**
 * Build a type-0 SMF (480 tpb, 120 bpm, velocity 100, or the event's own) from ace-midi's JSONL
 * note events, so a run that dies before writing out.mid still leaves the
 * notes it decoded. Notes still open at the cut are dropped. Null when no
 * note completed. Notes are fixed up as the engine's writer does
 * (validate_notes_fix, trim_overlapping in engine/tools/ace-midi.cpp).
 */
export function partialMidiFromEvents(events: any[]): Buffer | null {
  const open = new Map<number, { pitch: number; time: number; instrument: string; velocity?: number }>();
  const notes: { pitch: number; on: number; off: number; instrument: string; velocity: number }[] = [];
  for (const ev of events) {
    if (ev?.type === 'note_start') open.set(ev.index, ev);
    else if (ev?.type === 'note_end' && open.has(ev.index)) {
      const s = open.get(ev.index)!;
      open.delete(ev.index);
      // A too-short note gets the engine's 0.01 s minimum, so its off never sorts before its on.
      const off = ev.time - s.time < 0.01 ? s.time + 0.01 : ev.time;
      notes.push({ pitch: s.pitch, on: s.time, off, instrument: s.instrument,
        velocity: Number.isInteger(s.velocity) ? Math.min(127, Math.max(1, s.velocity!)) : 100 });
    }
  }
  // A note of the same pitch and instrument ends where the next one starts.
  notes.sort((a, b) => a.on - b.on);
  const lastOf = new Map<string, (typeof notes)[number]>();
  for (const n of notes) {
    const key = `${n.instrument}:${n.pitch}`;
    const prev = lastOf.get(key);
    if (prev && prev.off > n.on) prev.off = n.on;
    lastOf.set(key, n);
  }
  const kept = notes.filter(n => n.on < n.off);
  if (!kept.length) return null;

  // Channels by first appearance: 0-8, 10-15 (overflow shares 15), drums on 9.
  const channels = new Map<string, number>();
  const free = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
  const setup: number[] = [];
  const chanFor = (instrument: string): number => {
    let ch = channels.get(instrument);
    if (ch !== undefined) return ch;
    if (instrument === 'drums') ch = 9;
    else {
      ch = free.shift() ?? 15;
      const program = INSTRUMENT_PROGRAMS[instrument] ?? (parseInt(instrument.replace(/^program_/, ''), 10) || 0);
      setup.push(0, 0xc0 | ch, program & 0x7f);
    }
    channels.set(instrument, ch);
    return ch;
  };

  const tick = (sec: number) => Math.round(sec * 960); // 480 tpb at 120 bpm
  const evs: { tick: number; status: number; pitch: number; vel: number }[] = [];
  for (const n of kept) {
    const ch = chanFor(n.instrument);
    evs.push({ tick: tick(n.on), status: 0x90 | ch, pitch: n.pitch, vel: n.velocity });
    evs.push({ tick: tick(n.off), status: 0x80 | ch, pitch: n.pitch, vel: 0 });
  }
  // Offs before ons at the same tick, so a repeated pitch is not cut short.
  evs.sort((a, b) => a.tick - b.tick || (a.status & 0xf0) - (b.status & 0xf0));

  const body: number[] = [0, 0xff, 0x51, 3, 0x07, 0xa1, 0x20, ...setup];
  let last = 0;
  for (const e of evs) {
    body.push(...varLen(e.tick - last), e.status, e.pitch & 0x7f, e.vel);
    last = e.tick;
  }
  body.push(0, 0xff, 0x2f, 0);

  const header = Buffer.from([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, 0x01, 0xe0]);
  const trackHead = Buffer.alloc(8);
  trackHead.write('MTrk', 0, 'ascii');
  trackHead.writeUInt32BE(body.length, 4);
  return Buffer.concat([header, trackHead, Buffer.from(body)]);
}
