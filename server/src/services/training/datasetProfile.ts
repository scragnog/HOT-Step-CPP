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

/** Lead sheets by source file name, from the YuE2 cache when it is still on disk. */
function cachedSheets(slug: string): Map<string, { abc: string; lyrics: string }> {
  const out = new Map<string, { abc: string; lyrics: string }>();
  try {
    const m = JSON.parse(fs.readFileSync(path.join(datasetDir(slug), 'yue2-latents', 'yue2_preprocess.json'), 'utf8'));
    for (const s of m.sources ?? []) if (s?.abc && s?.name) out.set(s.name, { abc: s.abc, lyrics: s.lyrics ?? '' });
  } catch { /* no cache: the sheet measures are left out */ }
  return out;
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
