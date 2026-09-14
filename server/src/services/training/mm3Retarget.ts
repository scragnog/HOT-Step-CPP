// mm3Retarget.ts — the server half of `ace-train mm3-retarget`.
//
// MM3's checkpoint caps audio at 9000 frames (6:00) and MM3 LM training drops every longer track. This rescues them
// by cutting one repeated section out of the middle, so the intro and the real ending both survive. The engine does
// the analysis and the splice; this file does the one thing the engine deliberately cannot.
//
// WHY THE SEPARATION LIVES HERE. The excision must never cut through singing — a dataset lyric sheet is untimed
// plain text, and removing sung audio without editing it teaches the model a lyric/audio mismatch. So the engine
// needs to know where the vocals are, and that means SuperSep. But `ace-train` is deliberately standalone
// (header-only, no acestep-core, no ONNX Runtime), and supersep.h states its VRAM policy is sequential with the
// GGML model store, which a second process cannot honour while ace-server is alive. Running it here, through the
// engine's existing endpoint, keeps one owner of the GPU and one copy of the models.
//
// Output per track: <vocalDir>/<id>.json  { "hop": 0.1, "runs": [[12.4, 31.8], ...] }  — vocal-ACTIVE intervals.
// The engine treats a missing sidecar as a hard failure for that track, never as "assume instrumental".
//
// Note for callers: a SuperSep pass evicts MM3 from the engine, so anything holding an MM3 model needs to expect a
// reload afterwards. Same cost tools/vocal-end has always paid.

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

import { config } from '../../config.js';
import { ensureEngineFormat } from '../audioConvert.js';
import { parseWavHeader } from '../audioCrop.js';
import { aceTrainExe } from './aceTrain.js';
import { mm3RetargetDir } from './paths.js';

const ACE_URL = config.aceServer.url;

/** Vocal-activity detection, matching tools/vocal-end and the engine's own rt_vocal_mask. */
const VOCAL_HOP_S = 0.1;
const VOCAL_THRESH_DB = -30;     // relative to the stem's loudest window; stable anywhere from -20 to -40
const VOCAL_MIN_RUN_S = 0.6;     // shorter "runs" are breaths and bleed, not singing

export interface RetargetSample {
  id: string;
  filename: string;
  audioPath: string;
  duration: number;
}

export interface RetargetResult {
  manifest: string;              // the derived dataset.json
  edited: number;
  refused: number;
  log: string[];
}

// ── vocal sidecars ──────────────────────────────────────────────────────────────────────────────────────────────

/** Decode a mono/stereo PCM WAV buffer to a mono float array. Handles the 16-bit and 32-bit float the engine serves. */
function wavToMono(buf: Buffer): { samples: Float32Array; sampleRate: number } {
  const info = parseWavHeader(buf);
  const { sampleRate, numChannels, bitsPerSample, dataOffset, dataSize } = info;
  const bytes = bitsPerSample / 8;
  const frames = Math.floor(dataSize / (bytes * numChannels));
  const out = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let acc = 0;
    for (let c = 0; c < numChannels; c++) {
      const off = dataOffset + (i * numChannels + c) * bytes;
      if (off + bytes > buf.length) break;
      acc += bitsPerSample === 32 ? buf.readFloatLE(off)
           : bitsPerSample === 24 ? ((buf.readUInt8(off) | (buf.readUInt8(off + 1) << 8)
                                      | (buf.readInt8(off + 2) << 16)) / 8388608)
           : buf.readInt16LE(off) / 32768;
    }
    out[i] = acc / numChannels;
  }
  return { samples: out, sampleRate };
}

/** RMS envelope -> active windows above a threshold relative to the loudest window -> merged runs in seconds. */
function vocalRuns(samples: Float32Array, sampleRate: number): Array<[number, number]> {
  const n = Math.max(1, Math.round(VOCAL_HOP_S * sampleRate));
  const windows = Math.floor(samples.length / n);
  if (windows <= 0) return [];
  const rms = new Float64Array(windows);
  let peak = 0;
  for (let w = 0; w < windows; w++) {
    let sum = 0;
    for (let i = w * n; i < (w + 1) * n; i++) sum += samples[i] * samples[i];
    rms[w] = Math.sqrt(sum / n) + 1e-12;
    if (rms[w] > peak) peak = rms[w];
  }
  const active: boolean[] = [];
  for (let w = 0; w < windows; w++) active.push(20 * Math.log10(rms[w] / peak) > VOCAL_THRESH_DB);

  const runs: Array<[number, number]> = [];
  let w = 0;
  while (w < windows) {
    if (!active[w]) { w++; continue; }
    let e = w;
    while (e < windows && active[e]) e++;
    if ((e - w) * VOCAL_HOP_S >= VOCAL_MIN_RUN_S) runs.push([w * VOCAL_HOP_S, e * VOCAL_HOP_S]);
    w = e;
  }
  return runs;
}

async function separateVocalStem(audioPath: string): Promise<Buffer> {
  const body = ensureEngineFormat(audioPath);
  const sepRes = await fetch(`${ACE_URL}/supersep/separate?level=4`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: body as unknown as BodyInit,
  });
  if (!sepRes.ok) throw new Error(`SuperSep rejected the job: ${await sepRes.text()}`);
  const { id } = await sepRes.json() as { id: string };

  // Same poll shape as Stem Studio. Separation of a 7-minute track is seconds to a couple of minutes.
  for (let i = 0; i < 14400; i++) {
    const progRes = await fetch(`${ACE_URL}/supersep/progress?id=${id}`);
    const prog = await progRes.json() as { status: string; error?: string };
    if (prog.status === 'done') break;
    if (prog.status === 'failed' || prog.status === 'cancelled') {
      throw new Error(prog.error || `separation ${prog.status}`);
    }
    await new Promise(r => setTimeout(r, 500));
  }
  const resultRes = await fetch(`${ACE_URL}/supersep/result?id=${id}`);
  if (!resultRes.ok) throw new Error('SuperSep result unavailable');
  const { stems } = await resultRes.json() as
    { stems: Array<{ name: string; category: string; index: number }> };
  const vocal = stems.find(s => /vocal/i.test(s.category) || /vocal/i.test(s.name));
  if (!vocal) throw new Error(`no vocal stem among ${stems.map(s => s.name).join(', ')}`);

  const stemRes = await fetch(`${ACE_URL}/supersep/serve?id=${id}&stem=${vocal.index}`);
  if (!stemRes.ok) throw new Error('cannot fetch the vocal stem');
  const wav = Buffer.from(await stemRes.arrayBuffer());
  // The engine's job pool has no eviction, so an unreleased job costs about a gigabyte until restart (#133).
  try { await fetch(`${ACE_URL}/supersep/release?id=${id}`, { method: 'POST' }); } catch { /* best effort */ }
  return wav;
}

/**
 * Write one <id>.json vocal-activity sidecar per sample. Existing sidecars are reused: separation is the expensive
 * half of this and a re-run after a refused cut should not pay for it twice.
 */
export async function writeVocalSidecars(
  samples: RetargetSample[],
  vocalDir: string,
  onProgress?: (done: number, total: number, name: string) => void,
): Promise<{ ok: number; failed: Array<{ id: string; error: string }> }> {
  fs.mkdirSync(vocalDir, { recursive: true });
  const failed: Array<{ id: string; error: string }> = [];
  let ok = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    onProgress?.(i, samples.length, s.filename);
    const dst = path.join(vocalDir, `${s.id}.json`);
    const stemPath = path.join(vocalDir, `${s.id}.vocals.wav`);
    if (fs.existsSync(dst) && fs.existsSync(stemPath)) { ok++; continue; }
    try {
      const wav = await separateVocalStem(s.audioPath);
      // Keep the stem. Separation is by far the expensive half of this, and the stem is what the lyric-alignment
      // work needs next; throwing it away means paying for a second GPU pass to get back something we already had.
      fs.writeFileSync(stemPath, wav);
      const { samples: mono, sampleRate } = wavToMono(wav);
      const runs = vocalRuns(mono, sampleRate);
      fs.writeFileSync(dst, JSON.stringify({ hop: VOCAL_HOP_S, runs }), 'utf-8');
      console.log(`[mm3-retarget] ${s.filename}: ${runs.length} vocal run(s)`);
      ok++;
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      console.warn(`[mm3-retarget] ${s.filename}: vocal detection failed — ${error}`);
      failed.push({ id: s.id, error });
    }
  }
  onProgress?.(samples.length, samples.length, '');
  return { ok, failed };
}

// ── the engine pass ─────────────────────────────────────────────────────────────────────────────────────────────

export interface RetargetOptions {
  datasetJson: string;
  outDir: string;
  vocalDir: string;
  targetSeconds?: number;
  maxCost?: number;
  protectHead?: number;
  protectTail?: number;
  signal?: AbortSignal;
}

/**
 * Run `ace-train mm3-retarget` and return the derived manifest. Tracks it refuses are copied through untouched, so
 * they stay exactly as they are today — dropped by --drop-over-frames — and a refusal never makes a run worse.
 */
export function runRetarget(o: RetargetOptions): Promise<RetargetResult> {
  const exe = aceTrainExe();
  if (!exe) return Promise.reject(new Error('ace-train binary not found; build the engine first'));
  if (!config.essentia.bin || !fs.existsSync(config.essentia.bin)) {
    return Promise.reject(new Error(
      `Essentia not found at ${config.essentia.bin}. mm3-retarget needs it for the beat grid.`));
  }
  const args = [
    'mm3-retarget',
    '--dataset', o.datasetJson,
    '--out', o.outDir,
    '--vocal-dir', o.vocalDir,
    '--essentia', config.essentia.bin,
    '--target', String(o.targetSeconds ?? 360),
  ];
  if (o.maxCost !== undefined) args.push('--max-cost', String(o.maxCost));
  if (o.protectHead !== undefined) args.push('--protect-head', String(o.protectHead));
  if (o.protectTail !== undefined) args.push('--protect-tail', String(o.protectTail));
  const ffmpeg = findFfmpeg();
  if (ffmpeg) args.push('--ffmpeg', ffmpeg);

  return new Promise((resolve, reject) => {
    const log: string[] = [];
    const child = spawn(exe, args, { signal: o.signal });
    const take = (chunk: Buffer) => {
      for (const line of chunk.toString().split(/\r?\n/)) {
        if (line.trim()) { log.push(line); console.log(`[mm3-retarget] ${line}`); }
      }
    };
    child.stdout.on('data', take);
    child.stderr.on('data', take);
    child.on('error', reject);
    child.on('close', code => {
      if (code !== 0) { reject(new Error(`ace-train mm3-retarget exited ${code}`)); return; }
      const manifest = path.join(o.outDir, 'dataset.json');
      if (!fs.existsSync(manifest)) { reject(new Error('mm3-retarget wrote no derived manifest')); return; }
      const edited = log.filter(l => l.includes('  CUT ')).length;
      const refused = log.filter(l => l.includes('KEPT AS-IS')).length;
      resolve({ manifest, edited, refused, log });
    });
  });
}

/**
 * Both halves in order: separate what needs separating, then run the engine pass.
 *
 * ORDER MATTERS AND IS NOT NEGOTIABLE. Step 1 talks to the running ace-server, step 2 does not touch the GPU at all
 * (Essentia, ffmpeg and plain DSP). So this must complete BEFORE anything that stops the engine for a codes or
 * training run, and it can never be folded into one of those jobs.
 */
export async function retargetDataset(
  slug: string,
  samples: RetargetSample[],
  opts: Omit<RetargetOptions, 'datasetJson' | 'outDir' | 'vocalDir'> & { datasetJson: string },
  onProgress?: (done: number, total: number, name: string) => void,
): Promise<RetargetResult & { vocalFailed: Array<{ id: string; error: string }> }> {
  const target = opts.targetSeconds ?? 360;
  const over = samples.filter(s => s.duration > target);
  const outDir = mm3RetargetDir(slug);
  const vocalDir = path.join(outDir, 'vocal');
  if (!over.length) {
    return { manifest: opts.datasetJson, edited: 0, refused: 0, log: [], vocalFailed: [] };
  }
  console.log(`[mm3-retarget] ${slug}: ${over.length} track(s) over ${target}s`);
  const { failed } = await writeVocalSidecars(over, vocalDir, onProgress);
  // A track whose separation failed simply has no sidecar, and the engine refuses it by name. That is the right
  // outcome — it stays excluded, exactly as it is today — so a failure here does not fail the whole pass.
  const result = await runRetarget({ ...opts, outDir, vocalDir });
  return { ...result, vocalFailed: failed };
}

function findFfmpeg(): string | null {
  const candidates = [
    path.join(process.cwd(), 'node_modules', 'ffmpeg-static', 'ffmpeg.exe'),
    path.join(process.cwd(), 'node_modules', 'ffmpeg-static', 'ffmpeg'),
    path.join(process.cwd(), 'server', 'node_modules', 'ffmpeg-static', 'ffmpeg.exe'),
    path.join(process.cwd(), 'server', 'node_modules', 'ffmpeg-static', 'ffmpeg'),
  ];
  return candidates.find(p => fs.existsSync(p)) ?? null;
}
