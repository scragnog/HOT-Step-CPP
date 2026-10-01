// datasetProfile.ts — measurements of a training dataset, for Dataset-Calibrated
// Training: the album is profiled, the profile is joined with the ear scores of
// the adapters trained on it, and whatever predicts those scores becomes the
// rule that sizes the next run's recipe.
//
// Read-only: nothing here changes training. The profile is built from what
// survives a cache cleanup (the audio and its sidecars), plus the lead sheets
// when the YuE2 cache is still on disk, and saved as
// <trainingBaseDir>/datasets/<slug>/dataset-profile.json.
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { getFFmpegPath } from '../../config.js';
import { readSidecar } from './sidecarIO.js';
import { yue2StemSources } from './yue2Stems.js';
import { datasetDir } from './paths.js';
import { classifyYue2Score } from '../backends/yue2/scoreHealth.js';

export const DATASET_PROFILE_VERSION = 1;

export interface SongAudioStats {
  durationSec: number;
  lufs: number | null;
  lra: number | null;
  /** Spectral centroid in Hz over ~46 ms frames: mean is brightness, std how much it moves. */
  centroidMean: number;
  centroidStd: number;
  fluxMean: number;
  flatnessMean: number;
  /** Std of frame RMS in dB (frames above -70 dB): how much the loudness moves inside the song. */
  rmsStdDb: number;
}

export interface SongProfile extends SongAudioStats {
  file: string;
  bpm: number | null;
  key: string;
  genre: string;
  instrumental: boolean;
  captionWords: number;
  lyricWords: number;
  lyricSections: number;
  wordsPerSec: number | null;
  /** From the YuE2 cache's lead sheet; absent when the cache was cleaned up. */
  sheet?: { vocalShare: number; vocalPitches: number; chords: number; insLoopBars: number; vocalPhraseVariety: number; failedChecks: number };
}

export interface DatasetProfile {
  version: number;
  slug: string;
  sourceDir: string;
  builtAt: string;
  songs: SongProfile[];
  /** One number per measure, named so the report can print it as-is. */
  album: Record<string, number | null>;
  /** The item-0 caption steers every preview; kept here so a report can show it. */
  firstCaption: string;
}

export function datasetProfilePath(slug: string): string {
  return path.join(datasetDir(slug), 'dataset-profile.json');
}

export function readDatasetProfile(slug: string): DatasetProfile | null {
  try {
    const p = JSON.parse(fs.readFileSync(datasetProfilePath(slug), 'utf8')) as DatasetProfile;
    return p.version === DATASET_PROFILE_VERSION ? p : null;
  } catch { return null; }
}

const mean = (xs: number[]) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
const std = (xs: number[]) => { const m = mean(xs); return xs.length > 1 ? Math.sqrt(mean(xs.map(x => (x - m) ** 2))) : 0; };
const median = (xs: number[]) => { if (!xs.length) return NaN; const s = [...xs].sort((a, b) => a - b); const h = s.length >> 1; return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2; };
const fin = (x: number) => Number.isFinite(x) ? Math.round(x * 1e4) / 1e4 : null;

/** One ffmpeg pass: EBU R128 summary on stderr, per-frame spectral stats and RMS on stdout. */
export function measureAudio(file: string): Promise<SongAudioStats> {
  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) return Promise.reject(new Error('ffmpeg not found'));
  const af = 'ebur128=framelog=quiet,aformat=channel_layouts=mono,aresample=22050,'
    + 'aspectralstats=measure=centroid+flux+flatness,'
    + 'astats=metadata=1:reset=1:measure_perchannel=RMS_level:measure_overall=none,ametadata=print:file=-';
  return new Promise((resolve, reject) => {
    const p = spawn(ffmpeg, ['-hide_banner', '-nostats', '-i', file, '-af', af, '-f', 'null', '-'], { windowsHide: true });
    const centroid: number[] = [], flux: number[] = [], flat: number[] = [], rms: number[] = [];
    let lastT = 0, err = '', buf = '';
    const line = (l: string) => {
      const eq = l.indexOf('=');
      if (l.startsWith('frame:')) { const m = /pts_time:([\d.]+)/.exec(l); if (m) lastT = +m[1]; return; }
      if (eq < 0) return;
      const v = +l.slice(eq + 1);
      if (!Number.isFinite(v)) return;
      if (l.includes('.centroid=')) centroid.push(v);
      else if (l.includes('.flux=')) flux.push(v);
      else if (l.includes('.flatness=')) flat.push(v);
      else if (l.includes('RMS_level=') && v > -70) rms.push(v);
    };
    p.stdout.on('data', (d: Buffer) => { buf += d.toString(); const ls = buf.split('\n'); buf = ls.pop()!; ls.forEach(line); });
    p.stderr.on('data', (d: Buffer) => { err += d.toString(); if (err.length > 1 << 20) err = err.slice(-65536); });
    p.on('error', reject);
    p.on('close', code => {
      if (code !== 0) return reject(new Error(`ffmpeg exited ${code} on ${path.basename(file)}: ${err.slice(-300)}`));
      const summary = err.slice(err.lastIndexOf('Summary:'));
      const num = (re: RegExp) => { const m = re.exec(summary); return m ? +m[1] : null; };
      resolve({
        durationSec: lastT,
        lufs: num(/I:\s+(-?[\d.]+) LUFS/), lra: num(/LRA:\s+([\d.]+) LU/),
        centroidMean: mean(centroid), centroidStd: std(centroid),
        fluxMean: mean(flux), flatnessMean: mean(flat), rmsStdDb: std(rms),
      });
    });
  });
}

const STOP = new Set('a an and the with of in on to for by at is its it as or from into over under very slightly'.split(' '));
function words(s: string): string[] { return s.toLowerCase().match(/[a-z][a-z'-]+/g)?.filter(w => !STOP.has(w)) ?? []; }

/** Mean pairwise Jaccard of the caption word sets: 1 = every song captioned alike, 0 = nothing shared. */
function captionAgreement(captions: string[]): number {
  const sets = captions.filter(Boolean).map(c => new Set(words(c)));
  let sum = 0, n = 0;
  for (let i = 0; i < sets.length; i++) for (let j = i + 1; j < sets.length; j++) {
    let inter = 0; for (const w of sets[i]) if (sets[j].has(w)) inter++;
    const union = sets[i].size + sets[j].size - inter;
    if (union) { sum += inter / union; n++; }
  }
  return n ? sum / n : NaN;
}

/** The style caption YuE2 trains on, by the route's rule: .yue2.txt, then the sidecar, then the MM3 block. */
function yue2Caption(audio: string, sidecar: Record<string, string>): string {
  const stem = audio.slice(0, audio.length - path.extname(audio).length);
  const read = (f: string) => { try { return fs.readFileSync(f, 'utf8').trim(); } catch { return ''; } };
  return read(`${stem}.yue2.txt`) || (sidecar.caption ?? '').trim() || read(`${stem}.mm3.txt`).replace(/\s+/g, ' ').trim();
}

const sheetSnapshotPath = (slug: string) => path.join(datasetDir(slug), 'yue2-sheets.json');

/** Lead sheets by source file name: the YuE2 cache while it is on disk, else
 *  the snapshot clearPreparedCaches left behind. */
function cachedSheets(slug: string): Map<string, { abc: string; lyrics: string }> {
  const out = new Map<string, { abc: string; lyrics: string }>();
  for (const file of [path.join(datasetDir(slug), 'yue2-latents', 'yue2_preprocess.json'), sheetSnapshotPath(slug)]) {
    try {
      const m = JSON.parse(fs.readFileSync(file, 'utf8'));
      for (const s of m.sources ?? []) if (s?.abc && s?.name) out.set(s.name, { abc: s.abc, lyrics: s.lyrics ?? '' });
      if (out.size) return out;
    } catch { /* try the next source */ }
  }
  return out;
}

/** Keep the lead sheets (a few KB of text) before the YuE2 cache is deleted:
 *  re-running SheetSage on an album costs GPU minutes. No cache, no-op. */
export function snapshotYue2Sheets(slug: string): void {
  let m: { sources?: Array<{ name?: string; abc?: string; lyrics?: string }> };
  try { m = JSON.parse(fs.readFileSync(path.join(datasetDir(slug), 'yue2-latents', 'yue2_preprocess.json'), 'utf8')); }
  catch { return; }
  const sources = (m.sources ?? []).filter(s => s?.name && s?.abc).map(s => ({ name: s.name, abc: s.abc, lyrics: s.lyrics ?? '' }));
  if (sources.length) fs.writeFileSync(sheetSnapshotPath(slug), JSON.stringify({ savedAt: new Date().toISOString(), sources }));
}

export const trainLogArchiveDir = (slug: string) => path.join(datasetDir(slug), 'train-logs');

/** A run's train.jsonl files as they sit under its output folder: one per
 *  segment (`segments/<seg>/train.jsonl`), or a single root log
 *  (`train.jsonl`, written by runs with no previews — see
 *  yue2JointTrainRunner.ts's segment loop) keyed by the run folder's own name. */
export function listYue2TrainLogs(output: string): Array<{ seg: string; file: string }> {
  const out: Array<{ seg: string; file: string }> = [];
  const root = path.join(output, 'train.jsonl');
  if (fs.existsSync(root)) out.push({ seg: path.basename(output), file: root });
  let segs: string[] = [];
  try { segs = fs.readdirSync(path.join(output, 'segments')); }
  catch (err: any) { if (err?.code !== 'ENOENT') throw err; segs = []; }
  for (const seg of segs.sort()) {
    const file = path.join(output, 'segments', seg, 'train.jsonl');
    if (fs.existsSync(file)) out.push({ seg, file });
  }
  return out;
}

/** Copy a joint run's train.jsonl file(s) into the dataset's train-logs/ as
 *  <jobId>-<seg>.jsonl, so the loss curve outlives the run folder. Throws if a
 *  copy fails (caller decides whether that should block further cleanup);
 *  returns the segment names actually saved. */
export function archiveYue2TrainLogs(slug: string, jobId: string, output: string): string[] {
  if (!slug) return [];
  const logs = listYue2TrainLogs(output);
  if (!logs.length) return [];
  fs.mkdirSync(trainLogArchiveDir(slug), { recursive: true });
  const saved: string[] = [];
  for (const { seg, file } of logs) {
    const dest = path.join(trainLogArchiveDir(slug), `${jobId}-${seg}.jsonl`);
    const tmp = `${dest}.${process.pid}.tmp`;
    fs.copyFileSync(file, tmp);
    fs.renameSync(tmp, dest);
    saved.push(seg);
  }
  return saved;
}

export interface Yue2TrainLogNote {
  output?: string;
  status?: 'running' | 'done' | 'failed' | 'cancelled' | 'finished';
  keptStep?: number;
  segments?: string[];
  /** Set when the log arrived through a worker pull rather than locally. */
  pulledFrom?: string;
  pulledAt?: number;
}

/** Durable sidecar beside the archived log(s): which rung was kept, and the
 *  run's end state, so that survives after the run folder (and its
 *  meters.json/kept-checkpoint marker) is gone. A patch that omits keptStep
 *  never clears a previously-noted one — fields not present in `patch` are
 *  left as they were. */
export function noteYue2TrainLog(slug: string, jobId: string, patch: Yue2TrainLogNote): void {
  if (!slug || !jobId) return;
  const file = path.join(trainLogArchiveDir(slug), `${jobId}.json`);
  let existing: Record<string, unknown> = {};
  // Missing sidecar (first note for this run) starts empty; any other read or
  // parse failure must not silently discard whatever was already noted.
  try { existing = JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch (err: any) { if (err?.code !== 'ENOENT') throw err; }
  const next = { ...existing, ...patch, jobId, datasetSlug: slug, updatedAt: Date.now() };
  fs.mkdirSync(trainLogArchiveDir(slug), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(next, null, 2));
  fs.renameSync(tmp, file);
}

async function pool<T, R>(items: T[], n: number, fn: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length); let next = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i]); }
  }));
  return out;
}

export async function buildDatasetProfile(slug: string, sourceDir: string, opts: { concurrency?: number; log?: (s: string) => void } = {}): Promise<DatasetProfile> {
  const files = yue2StemSources(sourceDir);
  if (!files.length) throw new Error(`No audio in ${sourceDir}`);
  const sheets = cachedSheets(slug);
  const captions: string[] = [];
  const songs = await pool(files, opts.concurrency ?? 4, async (audio): Promise<SongProfile> => {
    const stem = audio.slice(0, audio.length - path.extname(audio).length);
    const sc = readSidecar(`${stem}.txt`);
    const a = await measureAudio(audio);
    opts.log?.(`  ${path.basename(audio)}  ${a.durationSec.toFixed(0)} s  ${a.lufs ?? '?'} LUFS`);
    const lyrics = sc.lyrics ?? '';
    const lyricWords = lyrics.replace(/\[[^\]]*\]/g, ' ').split(/\s+/).filter(Boolean).length;
    const instrumental = /^true$/i.test(sc.is_instrumental ?? '') || lyricWords === 0;
    const caption = yue2Caption(audio, sc);
    const bpm = parseFloat(sc.bpm ?? '');
    const song: SongProfile = {
      file: path.basename(audio), ...a,
      bpm: Number.isFinite(bpm) && bpm > 0 ? bpm : null,
      key: (sc.keyscale ?? sc.key ?? '').trim(), genre: (sc.genre ?? '').trim().toLowerCase(), instrumental,
      captionWords: words(caption).length, lyricWords, lyricSections: (lyrics.match(/^\s*\[[^\]]+\]/gm) ?? []).length,
      wordsPerSec: !instrumental && a.durationSec > 0 ? lyricWords / a.durationSec : null,
    };
    const sheet = sheets.get(path.basename(audio));
    if (sheet) {
      const h = classifyYue2Score(sheet.abc, 'completed', sheet.lyrics);
      song.sheet = { vocalShare: h.vocalShare, vocalPitches: h.legibility.vocalPitches, chords: h.legibility.chords,
        insLoopBars: h.legibility.insLoop.bars, vocalPhraseVariety: h.legibility.vocalPhraseVariety, failedChecks: h.failed.length };
    }
    captions[files.indexOf(audio)] = caption;
    return song;
  });

  const col = (f: (s: SongProfile) => number | null | undefined) => songs.map(f).filter((x): x is number => typeof x === 'number' && Number.isFinite(x));
  // Folded into 70-140: a half-time label (91 for a 182 song) is the same tempo, not album variety.
  const fold = (b: number) => { while (b >= 140) b /= 2; while (b < 70) b *= 2; return b; };
  const dur = col(s => s.durationSec), bpms = col(s => s.bpm).map(fold), genres = songs.map(s => s.genre).filter(Boolean);
  const topGenre = genres.length ? Math.max(...[...new Set(genres)].map(g => genres.filter(x => x === g).length)) / genres.length : NaN;
  const keys = songs.map(s => s.key).filter(Boolean);
  const sheeted = songs.filter(s => s.sheet);
  const album: Record<string, number | null> = {
    songs: songs.length,
    totalMin: fin(dur.reduce((a, b) => a + b, 0) / 60),
    durMeanSec: fin(mean(dur)), durCv: fin(std(dur) / mean(dur)),
    longShare: fin(dur.filter(d => d > 300).length / dur.length),
    instrumentalShare: fin(songs.filter(s => s.instrumental).length / songs.length),
    bpmMedian: fin(median(bpms)), bpmCv: fin(std(bpms) / mean(bpms)),
    keyCount: new Set(keys).size || null, majorShare: fin(keys.filter(k => /major/i.test(k)).length / keys.length),
    genreCount: new Set(genres).size || null, topGenreShare: fin(topGenre),
    captionWordsMean: fin(mean(col(s => s.captionWords))), captionAgreement: fin(captionAgreement(captions)),
    wordsPerSecMedian: fin(median(col(s => s.wordsPerSec))), lyricSectionsMean: fin(mean(col(s => s.lyricSections))),
    lufsMean: fin(mean(col(s => s.lufs))), lufsSpread: fin(std(col(s => s.lufs))), lraMean: fin(mean(col(s => s.lra))),
    centroidMean: fin(mean(col(s => s.centroidMean))), centroidSpread: fin(std(col(s => s.centroidMean))),
    centroidStdMean: fin(mean(col(s => s.centroidStd))), fluxMean: fin(mean(col(s => s.fluxMean))),
    flatnessMean: fin(mean(col(s => s.flatnessMean))), rmsStdDbMean: fin(mean(col(s => s.rmsStdDb))),
    sheetCoverage: fin(sheeted.length / songs.length),
    sheetVocalShare: sheeted.length ? fin(mean(sheeted.map(s => s.sheet!.vocalShare))) : null,
    sheetVocalPitches: sheeted.length ? fin(mean(sheeted.map(s => s.sheet!.vocalPitches))) : null,
    sheetChords: sheeted.length ? fin(mean(sheeted.map(s => s.sheet!.chords))) : null,
    sheetInsLoopBars: sheeted.length ? fin(mean(sheeted.map(s => s.sheet!.insLoopBars))) : null,
    sheetFailedShare: sheeted.length ? fin(sheeted.filter(s => s.sheet!.failedChecks > 0).length / sheeted.length) : null,
  };
  return { version: DATASET_PROFILE_VERSION, slug, sourceDir, builtAt: new Date().toISOString(), songs, album, firstCaption: captions[0] ?? '' };
}

export function saveDatasetProfile(p: DatasetProfile): string {
  const file = datasetProfilePath(p.slug);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(p, null, 2));
  fs.renameSync(tmp, file);
  return file;
}

// ── Dataset-Calibrated Training (opt-in) ────────────────────────────────────
//
// Rule 1, 2026-09-29: size the run by the album's minutes of audio. Across 23
// ear-scored ladders of the tuned recipe, the step where likeness first reached
// 4 rose with total minutes (Spearman 0.55). The scale is set so a 45-minute
// album (the middle of the scored set) keeps the preset's count.
//
// ponytail: one measure, fitted on the tuned recipe. Its ceiling is that the
// relation is unconfirmed under base-matched; the opt-in runs are the test.
// Replace with a rule refitted on base-matched album scores once there are
// enough (dataset-profile-report.ts).
export const CALIBRATION_REFERENCE_MINUTES = 45;
const CALIBRATION_MIN_FACTOR = 0.6;
const CALIBRATION_MAX_FACTOR = 2;

export interface Yue2Calibration {
  rule: 'minutes-v1';
  minutes: number;
  factor: number;
  requestedSteps: number;
  requestedSaveEvery: number;
  steps: number;
  saveEvery: number;
}

/** Scale steps and saveEvery together, so the ladder keeps its rung count. */
export function calibrateYue2Length(minutes: number, steps: number, saveEvery: number): Yue2Calibration {
  const factor = Math.min(CALIBRATION_MAX_FACTOR, Math.max(CALIBRATION_MIN_FACTOR, minutes / CALIBRATION_REFERENCE_MINUTES));
  const every = Math.max(1, Math.round(saveEvery * factor));
  const scaled = Math.max(every, Math.round(steps * factor / every) * every);
  return { rule: 'minutes-v1', minutes: Math.round(minutes * 10) / 10, factor: Math.round(factor * 1000) / 1000,
    requestedSteps: steps, requestedSaveEvery: saveEvery, steps: scaled, saveEvery: every };
}

/** The saved profile when it covers the same audio files, else a fresh one (saved). */
export async function ensureDatasetProfile(slug: string, sourceDir: string): Promise<DatasetProfile> {
  const names = yue2StemSources(sourceDir).map(f => path.basename(f)).join('|');
  const saved = readDatasetProfile(slug);
  if (saved && saved.songs.map(s => s.file).join('|') === names) return saved;
  const fresh = await buildDatasetProfile(slug, sourceDir);
  saveDatasetProfile(fresh);
  return fresh;
}
