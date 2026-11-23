/**
 * whisperTranscribe.ts — Whisper CLI transcription service
 *
 * Wraps whisper.cpp's whisper-cli to transcribe audio files to word-level
 * timestamped text. Used for lyrics synchronisation in the player.
 *
 * Workflow:
 *   1. Locate whisper-cli via config.whisper.exe or PATH
 *   2. Find the best available GGML model in config.whisper.modelsDir
 *   3. Run whisper-cli with -oj (JSON output) and --max-len 1 (word-level)
 *   4. Parse the sidecar JSON file whisper writes, clean up, return result
 */
import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import https from 'https';
import { pipeline } from 'stream/promises';
import { config } from '../config.js';
import { planWhisperWindows } from './audioActivity.js';
import { removeSlice, sliceToWav } from './audioSlice.js';

const execFileAsync = promisify(execFile);
let resolvedWhisperExe: string | null = null;

// ── Auto-download constants ─────────────────────────────────────────
// whisper.cpp v1.8.4 CUDA 12.4 build (works with CUDA 12.x and 13.x)
const WHISPER_RELEASE_URL = 'https://github.com/ggerganov/whisper.cpp/releases/download/v1.8.4/whisper-cublas-12.4.0-bin-x64.zip';
const WHISPER_FALLBACK_URL = 'https://github.com/ggerganov/whisper.cpp/releases/download/v1.8.4/whisper-bin-x64.zip';

/** Names the CLI has shipped under: `whisper-cli` now, `main` in older builds. */
const WHISPER_BIN_NAMES = ['whisper-cli', 'whisper-cpp', 'main'];

/** Install directories a GUI-launched server often leaves out of PATH. */
const COMMON_BIN_DIRS = ['/usr/local/bin', '/usr/bin', '/opt/homebrew/bin', '/opt/local/bin'];

/** The result of a CLI lookup, with a reason a user can act on. */
export type WhisperCliCheck =
  | { ok: true; exe: string }
  | { ok: false; reason: string };

type CandidateState = 'ok' | 'missing' | 'not-a-file' | 'not-executable';

/** Why a candidate cannot be run. A 0644 copy and a missing file look identical otherwise. */
/** `statSync` that answers null instead of throwing, so the caller can branch. */
function statOrNull(file: string): fs.Stats | null {
  try { return fs.statSync(file); } catch { return null; }
}

function candidateState(file: string): CandidateState {
  const stat = statOrNull(file);
  if (!stat) return 'missing';
  if (!stat.isFile()) return 'not-a-file';
  if (process.platform !== 'win32') {
    try { fs.accessSync(file, fs.constants.X_OK); } catch { return 'not-executable'; }
  }
  return 'ok';
}

function isExecutableFile(file: string): boolean {
  return candidateState(file) === 'ok';
}

function describeState(state: CandidateState): string {
  if (state === 'not-executable') return 'exists but is not executable (chmod +x it)';
  if (state === 'not-a-file') return 'exists but is not a file';
  return 'missing';
}

interface WhisperLookup {
  exe: string | null;
  /** Candidates that exist but cannot be run, with the reason. */
  problems: string[];
}

/** Look a binary up on PATH, without shelling out to `which`. */
function findOnPath(name: string, problems: string[]): string | null {
  const suffixes = process.platform === 'win32' ? ['.exe', ''] : [''];
  for (const dir of (process.env.PATH || '').split(path.delimiter)) {
    if (!dir) continue;
    for (const suffix of suffixes) {
      const candidate = path.join(dir, name + suffix);
      const state = candidateState(candidate);
      if (state === 'ok') return candidate;
      if (state !== 'missing') problems.push(`${candidate}: ${describeState(state)}`);
    }
  }
  return null;
}

/** The first usable whisper binary, and every near-miss that could not be run. */
function locateWhisperCli(): WhisperLookup {
  const problems: string[] = [];

  const configured = candidateState(config.whisper.exe);
  if (configured === 'ok') return { exe: config.whisper.exe, problems };
  if (configured !== 'missing') problems.push(`${config.whisper.exe}: ${describeState(configured)}`);

  for (const name of WHISPER_BIN_NAMES) {
    const found = findOnPath(name, problems);
    if (found) return { exe: found, problems };
  }
  if (process.platform !== 'win32') {
    for (const dir of COMMON_BIN_DIRS) {
      for (const name of WHISPER_BIN_NAMES) {
        const candidate = path.join(dir, name);
        const state = candidateState(candidate);
        if (state === 'ok') return { exe: candidate, problems };
        if (state !== 'missing') problems.push(`${candidate}: ${describeState(state)}`);
      }
    }
  }
  return { exe: null, problems };
}


// ── Types ───────────────────────────────────────────────────────────

export interface WhisperWord {
  word: string;
  start: number;
  end: number;
  /** Absent when the decoder did not report one. It is carried through as
   *  absent rather than replaced with a 1, so nothing downstream mistakes it
   *  for a confident word. */
  probability?: number;
}

export interface WhisperSegment {
  start: number;
  end: number;
  text: string;
  words?: WhisperWord[];
}

export interface WhisperResult {
  segments: WhisperSegment[];
  /** What produced this transcript: the model file, whether token DTW ran, and
   *  how many activity-gated windows were joined. */
  meta?: { model: string; dtw: boolean; windows?: number };
}

export interface WhisperOptions {
  /** Whisper model name override (e.g. 'ggml-large-v3-turbo.bin') */
  model?: string;
  /** Language code (e.g. 'en', 'ja') or 'auto' for auto-detect. Default: 'auto' */
  language?: string;
  /** Beam size for decoding. Default: 5 */
  beamSize?: number;
  /** Aborts the CLI. The process is killed and no output is kept. */
  signal?: AbortSignal;
  /**
   * Ask for whisper.cpp token DTW. Off unless a caller opts in: the lyric
   * aligner asks for it because token times are what it reconciles, while
   * generation-time transcription does not, and there the flag can only cost a
   * second full run when a build does not know the preset name. A caller that
   * does ask is retried once without it rather than failing the whole run.
   */
  dtw?: boolean;
  /** CLI budget for one call. Each activity-gated window gets the full budget. */
  timeoutMs?: number;
}

// ── Model priority for fallback selection ───────────────────────────

const MODEL_PRIORITY = [
  'ggml-large-v3-turbo.bin',
  'ggml-large-v3.bin',
  'ggml-medium.bin',
  'ggml-base.bin',
];

/**
 * whisper.cpp's `--dtw` preset names, keyed by the model family in the file
 * name. The names are whisper.cpp's own; a build that predates one of them
 * exits non-zero and the caller retries without the flag.
 */
const DTW_PRESETS: Array<[RegExp, string]> = [
  [/^ggml-large-v3-turbo/i, 'large.v3.turbo'],
  [/^ggml-large-v3/i, 'large.v3'],
  [/^ggml-large-v2/i, 'large.v2'],
  [/^ggml-large-v1/i, 'large.v1'],
  [/^ggml-medium/i, 'medium'],
  [/^ggml-small/i, 'small'],
  [/^ggml-base/i, 'base'],
  [/^ggml-tiny/i, 'tiny'],
];

/** The preset for a GGML model file, or null when its family has none. */
export function dtwPresetFor(modelFile: string): string | null {
  const name = path.basename(modelFile);
  const family = DTW_PRESETS.find(([pattern]) => pattern.test(name));
  if (!family) return null;
  // `ggml-base.en.bin` is the English-only build of the same architecture.
  return /\.en\.bin$/i.test(name) ? `${family[1]}.en` : family[1];
}

// ── Helpers ─────────────────────────────────────────────────────────

/**
 * Strip section markers like [Verse 1], [Chorus], etc. from lyrics text.
 * Collapses multiple newlines, joins into a single line, and trims to 800 chars.
 * Used to build a vocabulary-priming prompt for whisper.
 */
export function stripSectionMarkers(lyrics: string): string {
  return lyrics
    .replace(/\[.*?\]/g, '')           // remove [Verse 1], [Chorus], etc.
    .replace(/\r\n/g, '\n')            // normalise line endings
    .replace(/\n{2,}/g, '\n')          // collapse multiple newlines
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .join(' ')
    .slice(0, 800);
}

/**
 * Find the best available whisper GGML model file.
 *
 * If `preferredModel` is given and exists in the models directory, use it.
 * Otherwise falls back through MODEL_PRIORITY in order.
 *
 * @returns Absolute path to the model file, or null if none found.
 */
/** Filenames the model pickers know; anything else shows its own name. */
const MODEL_LABELS: Record<string, string> = {
  'ggml-large-v3-turbo.bin': 'Large v3 Turbo',
  'ggml-large-v3.bin': 'Large v3',
  'ggml-medium.bin': 'Medium',
  'ggml-base.bin': 'Base',
};

export function whisperModelLabel(file: string): string {
  return MODEL_LABELS[file] || file;
}

/** Position in the priority list; unknown files sort last. */
function modelRank(file: string): number {
  const index = MODEL_PRIORITY.indexOf(file);
  return index === -1 ? MODEL_PRIORITY.length : index;
}

/** The Whisper models actually installed, most-preferred first. */
export function listWhisperModels(): { file: string; label: string }[] {
  try {
    return fs.readdirSync(config.whisper.modelsDir)
      .filter(f => f.toLowerCase().endsWith('.bin'))
      .sort((a, b) => modelRank(a) - modelRank(b) || a.localeCompare(b))
      .map(file => ({ file, label: whisperModelLabel(file) }));
  } catch {
    return [];
  }
}

/**
 * Find the best available whisper GGML model file.
 *
 * If `preferredModel` is given and exists in the models directory, use it.
 * Otherwise fall back through MODEL_PRIORITY in order, then to any `.bin`.
 * `quiet` suppresses the choice log, for callers that only report the pick.
 */
export function findWhisperModel(preferredModel?: string, opts: { quiet?: boolean } = {}): string | null {
  const modelsDir = config.whisper.modelsDir;
  const say = opts.quiet ? () => {} : console.log;
  const warn = opts.quiet ? () => {} : console.warn;

  if (!fs.existsSync(modelsDir)) {
    warn(`[Whisper] Models directory does not exist: ${modelsDir}`);
    return null;
  }

  // Try preferred model first
  if (preferredModel) {
    const preferredPath = path.join(modelsDir, preferredModel);
    if (fs.existsSync(preferredPath)) {
      return preferredPath;
    }
    warn(`[Whisper] Preferred model not found: ${preferredModel}`);
  }

  // Fall through priority list
  for (const modelName of MODEL_PRIORITY) {
    const modelPath = path.join(modelsDir, modelName);
    if (fs.existsSync(modelPath)) {
      say(`[Whisper] Using model: ${modelName}`);
      return modelPath;
    }
  }

  // Last resort: any .bin file in the directory
  try {
    const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.bin')).sort((a, b) => modelRank(a) - modelRank(b) || a.localeCompare(b));
    if (files.length > 0) {
      const fallback = path.join(modelsDir, files[0]);
      say(`[Whisper] Fallback model: ${files[0]}`);
      return fallback;
    }
  } catch {
    // Can't read directory
  }

  warn('[Whisper] No model files found in models directory');
  return null;
}

// ── Auto-download ───────────────────────────────────────────────────

/** Download a file from a URL, following redirects. */
function httpsDownload(url: string, dest: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const doRequest = (reqUrl: string, redirectsLeft: number) => {
      https.get(reqUrl, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          if (redirectsLeft <= 0) return reject(new Error('Too many redirects'));
          return doRequest(res.headers.location, redirectsLeft - 1);
        }
        if (res.statusCode !== 200) {
          return reject(new Error(`HTTP ${res.statusCode} for ${reqUrl}`));
        }
        const fileStream = fs.createWriteStream(dest);
        res.pipe(fileStream);
        fileStream.on('finish', () => { fileStream.close(); resolve(); });
        fileStream.on('error', reject);
      }).on('error', reject);
    };
    doRequest(url, 5);
  });
}

/**
 * Ensure whisper-cli is available.
 *
 * Windows fetches the official release once. macOS and Linux have no official
 * binary to download, so they use WHISPER_EXE, the data folder's `tools/whisper`,
 * PATH, or a common install directory. Failures carry a reason: on a platform
 * that never downloads anything, "auto-download failed" is a dead end.
 */
export async function ensureWhisperCli(): Promise<WhisperCliCheck> {
  const found = locateWhisperCli();
  if (found.exe) {
    if (found.exe !== config.whisper.exe) console.log(`[Whisper] Using whisper-cli at ${found.exe}`);
    resolvedWhisperExe = found.exe;
    return { ok: true, exe: found.exe };
  }

  if (process.platform !== 'win32') {
    const unusable = found.problems.length ? ` Found but unusable: ${found.problems.join('; ')}.` : '';
    const reason =
      `whisper-cli not found: looked for ${WHISPER_BIN_NAMES.join(', ')} at ${config.whisper.exe}, `
      + `on PATH, and in ${COMMON_BIN_DIRS.join(', ')}.${unusable} `
      + 'It is not bundled with the app and is not auto-downloaded on this platform. '
      + `Build whisper.cpp and put the binary at ${config.whisper.exe}, or set WHISPER_EXE to its path.`;
    console.error(`[Whisper] ${reason}`);
    return { ok: false, reason };
  }

  const whisperExe = config.whisper.exe;
  const whisperDir = path.dirname(whisperExe);
  fs.mkdirSync(whisperDir, { recursive: true });

  const zipPath = path.join(whisperDir, 'whisper-download.zip');
  const failures: string[] = [];

  // Try CUDA build first, fall back to CPU
  for (const url of [WHISPER_RELEASE_URL, WHISPER_FALLBACK_URL]) {
    try {
      console.log(`[Whisper] Downloading whisper-cli from: ${url}`);
      console.log('[Whisper] One-time download: about 460 MB for the CUDA build, 4 MB for the CPU build...');

      await httpsDownload(url, zipPath);

      if (!fs.existsSync(zipPath) || fs.statSync(zipPath).size < 1_000_000) {
        console.warn('[Whisper] Downloaded file too small, trying next URL...');
        try { fs.unlinkSync(zipPath); } catch {}
        failures.push(`${path.basename(url)}: download too small`);
        continue;
      }

      // Extract with PowerShell
      console.log('[Whisper] Extracting...');
      await execFileAsync('powershell', [
        '-NoProfile', '-Command',
        `Expand-Archive -Path '${zipPath}' -DestinationPath '${whisperDir}' -Force`,
      ], { timeout: 60_000 });

      // Clean up zip
      try { fs.unlinkSync(zipPath); } catch {}

      // Check if whisper-cli.exe landed (might be in a subdirectory)
      if (fs.existsSync(whisperExe)) {
        resolvedWhisperExe = whisperExe;
        console.log(`[Whisper] ✓ whisper-cli ready at: ${whisperExe}`);
        return { ok: true, exe: whisperExe };
      }

      // Search for it in subdirectories (some releases nest in a folder)
      const nested = findFileRecursive(whisperDir, 'whisper-cli.exe');
      if (nested && nested !== whisperExe) {
        // Move all files from nested dir up to whisperDir
        const nestedDir = path.dirname(nested);
        if (nestedDir !== whisperDir) {
          for (const f of fs.readdirSync(nestedDir)) {
            const src = path.join(nestedDir, f);
            const dest = path.join(whisperDir, f);
            if (!fs.existsSync(dest)) fs.renameSync(src, dest);
          }
          // Clean up empty nested dir
          try { fs.rmdirSync(nestedDir); } catch {}
        }
        if (fs.existsSync(whisperExe)) {
          resolvedWhisperExe = whisperExe;
          console.log(`[Whisper] ✓ whisper-cli ready at: ${whisperExe}`);
          return { ok: true, exe: whisperExe };
        }
      }

      console.warn('[Whisper] Extracted but whisper-cli.exe not found in expected location');
      failures.push(`${path.basename(url)}: archive contained no whisper-cli.exe`);
    } catch (err: any) {
      console.error(`[Whisper] Download failed: ${err.message}`);
      failures.push(`${path.basename(url)}: ${err.message}`);
      try { fs.unlinkSync(zipPath); } catch {}
    }
  }

  const reason =
    `could not download whisper-cli (${failures.join('; ') || 'no download attempted'}). `
    + 'Download whisper-cublas-12.4.0-bin-x64.zip or whisper-bin-x64.zip from '
    + `https://github.com/ggml-org/whisper.cpp/releases, extract it into ${whisperDir}, `
    + 'or set WHISPER_EXE to an existing whisper-cli.exe.';
  console.error(`[Whisper] ${reason}`);
  return { ok: false, reason };
}

/** Recursively find a file by name in a directory. */
function findFileRecursive(dir: string, filename: string): string | null {
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isFile() && entry.name.toLowerCase() === filename.toLowerCase()) return fullPath;
      if (entry.isDirectory()) {
        const found = findFileRecursive(fullPath, filename);
        if (found) return found;
      }
    }
  } catch {}
  return null;
}

/**
 * Check whether whisper-cli is available at the configured path.
 */
export function isWhisperAvailable(): boolean {
  try {
    return (!!resolvedWhisperExe && isExecutableFile(resolvedWhisperExe)) || !!locateWhisperCli().exe;
  } catch {
    return false;
  }
}

/**
 * Transcribe an audio file using whisper-cli.
 *
 * Runs whisper.cpp with JSON output (-oj) and word-level timestamps (--max-len 1).
 * Source lyrics are passed as a vocabulary-priming --prompt to improve accuracy.
 *
 * On Windows, downloads whisper-cli if needed. On other platforms, checks PATH.
 *
 * @param audioPath    Absolute path to the audio file (WAV/MP3)
 * @param sourceLyrics Original lyrics text for vocabulary priming
 * @param options      Optional overrides for model, language, beam size
 * @returns            Parsed WhisperResult, or null on failure
 */
export async function transcribeWithWhisper(
  audioPath: string,
  sourceLyrics: string,
  options: WhisperOptions = {},
): Promise<WhisperResult | null> {
  // Auto-download if needed (Windows only); ensureWhisperCli logs why it failed.
  const ready = await ensureWhisperCli();
  if (!ready.ok) {
    return null;
  }

  const whisperExe = ready.exe;

  // Find model
  const modelPath = findWhisperModel(options.model);
  if (!modelPath) {
    console.error('[Whisper] No model available — cannot transcribe');
    return null;
  }

  // Validate audio file
  if (!fs.existsSync(audioPath)) {
    console.error(`[Whisper] Audio file not found: ${audioPath}`);
    return null;
  }

  const beamSize = options.beamSize ?? 5;
  const language = options.language ?? 'auto';
  const modelName = path.basename(modelPath);
  const preset = options.dtw === true ? dtwPresetFor(modelName) : null;
  const timeoutMs = options.timeoutMs ?? config.whisper.timeoutMs;

  // Build CLI args
  //   --split-on-word:  split segments at word boundaries, not BPE token boundaries
  //   --max-len 1:      with --split-on-word, each segment is one whole word with
  //                     its own timestamp from whisper (real word-level timing).
  //   NOTE: --prompt removed — vocabulary priming seeds hallucinations during instrumentals.
  //         The reconciliation service handles word matching post-hoc instead.
  //   NOTE: --no-fallback removed — causes repetition loops when decoder gets stuck.
  const args: string[] = [
    '-m', modelPath,
    '-f', audioPath,
    // Full JSON (`-ojf`), not the plain one. `-oj` carries no token array at
    // all, so every word time had to be interpolated evenly across its segment,
    // which is what puts lines early or late on sung audio. Tokens carry the
    // per-word offsets, and normaliseWhisperJson already merges them into words.
    '-ojf',
    '--split-on-word',        // split at word boundaries, not BPE tokens
    '--max-len', '1',         // 1 word per segment = real per-word timestamps
    '--beam-size', String(beamSize),
    // Token DTW over the model's alignment heads. `normaliseWhisperJson`
    // already prefers the `t_dtw` it produces; nothing ever asked for it.
    ...(preset ? ['--dtw', preset] : []),
    '--no-prints',            // suppress progress to stderr
  ];

  // Language (skip if auto-detect)
  if (language !== 'auto') {
    args.push('--language', language);
  }

  // NOTE: We intentionally do NOT pass --prompt with source lyrics.
  // While it helps whisper recognise vocabulary, it also causes hallucinations
  // during instrumental sections (whisper fills silence with prompted text).
  // The Needleman-Wunsch reconciliation handles vocabulary alignment post-hoc.

  // whisper.cpp -oj writes JSON to <audioPath>.json
  const jsonOutputPath = audioPath + '.json';

  console.log(`[Whisper] Transcribing: ${path.basename(audioPath)}`);
  console.log(`[Whisper] Model: ${modelName}, language: ${language}, beam: ${beamSize}` +
    `${preset ? `, dtw: ${preset}` : ''}`);
  const t0 = Date.now();

  const withoutDtw = (): string[] => {
    const copy = [...args];
    const at = copy.indexOf('--dtw');
    if (at >= 0) copy.splice(at, 2);
    return copy;
  };
  const run = (argv: string[]) => execFileAsync(whisperExe, argv, {
    timeout: timeoutMs,
    maxBuffer: 10 * 1024 * 1024,  // 10 MB stdout/stderr buffer
    ...(options.signal ? { signal: options.signal } : {}),
  });

  let dtwUsed = !!preset;
  try {
    try {
      await run(args);
    } catch (err: any) {
      // The preset names track whisper.cpp's model list, which a bundled build
      // can predate. An unknown flag is a bad name here, not a bad render: the
      // decoder offsets are worse than token times but better than no lyrics.
      const stderr = err?.stderr?.toString() || '';
      if (!preset || options.signal?.aborted || !/dtw|unknown argument|invalid argument/i.test(stderr)) throw err;
      console.warn(`[Whisper] --dtw ${preset} rejected, retrying without it: ${stderr.trim().split('\n').pop()}`);
      dtwUsed = false;
      await run(withoutDtw());
    }
  } catch (err: any) {
    // A cancelled run is not a failure: the caller aborted on purpose.
    if (options.signal?.aborted) {
      console.log(`[Whisper] Cancelled: ${path.basename(audioPath)}`);
      try { fs.unlinkSync(jsonOutputPath); } catch {}
      return null;
    }
    const stderr = err.stderr?.toString()?.slice(-500) || '';
    const code = err.code || 'unknown';
    console.error(`[Whisper] Process failed (code: ${code}): ${stderr || err.message}`);
    // Clean up any partial JSON output
    try { fs.unlinkSync(jsonOutputPath); } catch {}
    return null;
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

  // Read and parse the JSON output
  if (!fs.existsSync(jsonOutputPath)) {
    console.error('[Whisper] Expected JSON output file not found — whisper may have failed silently');
    return null;
  }

  let result: WhisperResult;
  try {
    const raw = fs.readFileSync(jsonOutputPath, 'utf-8');
    const parsed = JSON.parse(raw);

    // whisper.cpp JSON format: { transcription: [{ timestamps: {from, to}, text, ... }] }
    // Normalise into our WhisperResult format
    result = normaliseWhisperJson(parsed);
    result.meta = { model: modelName, dtw: dtwUsed };
  } catch (err: any) {
    console.error(`[Whisper] Failed to parse JSON output: ${err.message}`);
    try { fs.unlinkSync(jsonOutputPath); } catch {}
    return null;
  }

  // Clean up the sidecar JSON file
  try { fs.unlinkSync(jsonOutputPath); } catch {}

  const segCount = result.segments.length;
  const wordCount = result.segments.reduce((n, s) => n + (s.words?.length ?? 0), 0);
  console.log(`[Whisper] Done in ${elapsed}s — ${segCount} segments, ${wordCount} words`);

  return result;
}

// ── Internal: normalise whisper.cpp JSON ────────────────────────────

/**
 * whisper.cpp -oj produces JSON in this shape:
 * {
 *   "transcription": [
 *     {
 *       "timestamps": { "from": "00:00:00,000", "to": "00:00:02,500" },
 *       "offsets": { "from": 0, "to": 2500 },
 *       "text": " Hello world",
 *       "tokens": [
 *         { "text": " Hello", "timestamps": { "from": "...", "to": "..." },
 *           "offsets": { "from": 0, "to": 1200 }, "p": 0.95 },
 *         ...
 *       ]
 *     }
 *   ]
 * }
 *
 * We normalise this into our simpler WhisperResult format.
 */
export function normaliseWhisperJson(raw: any): WhisperResult {
  const segments: WhisperSegment[] = [];

  const transcription = raw?.transcription;
  if (!Array.isArray(transcription)) {
    console.warn('[Whisper] Unexpected JSON structure — no transcription array');
    return { segments };
  }

  // Debug: log first non-empty segment structure to help diagnose parsing issues
  const firstNonEmpty = transcription.find((s: any) => (s?.text ?? '').trim().length > 0);
  if (firstNonEmpty) {
    console.log(`[Whisper] JSON sample — first segment: text="${firstNonEmpty?.text}", ` +
      `offsets=${JSON.stringify(firstNonEmpty?.offsets)}, ` +
      `tokens=${Array.isArray(firstNonEmpty?.tokens) ? firstNonEmpty.tokens.length + ' items' : 'none'}`);
    if (Array.isArray(firstNonEmpty?.tokens) && firstNonEmpty.tokens.length > 0) {
      const t = firstNonEmpty.tokens[0];
      console.log(`[Whisper] JSON sample — first token: ${JSON.stringify(t)}`);
    }
  }

  for (const seg of transcription) {
    const startMs = seg?.offsets?.from ?? 0;
    const endMs = seg?.offsets?.to ?? 0;
    const text = (seg?.text ?? '').trim();

    // Skip empty segments
    if (text.length === 0) continue;
    // Skip punctuation-only segments
    if (/^[\s.,!?;:'"()\-–—…]+$/.test(text)) continue;

    const words: WhisperWord[] = [];

    if (Array.isArray(seg?.tokens) && seg.tokens.length > 0) {
      // With --dtw, tokens are BPE sub-word pieces with timestamps.
      // Word-starting tokens have a leading space (e.g. " Hello").
      // Merge consecutive sub-tokens into whole words.
      let currentWord = '';
      let wordStart = 0;
      let wordEnd = 0;
      let wordProb = 0;
      let tokenCount = 0;

      for (const tok of seg.tokens) {
        const rawText = tok?.text ?? '';

        // Skip special tokens ([_BEG_], [_TT_xxx], [_SOT_], etc.)
        if (/^\[.*\]$/.test(rawText.trim())) continue;
        // Skip empty tokens
        if (rawText.trim().length === 0 && currentWord.length === 0) continue;

        // Token timing: t_dtw (ms) when DTW ran, else the decoder offsets.
        // t_dtw is -1, not null, when it did not run, so `??` would keep the
        // -1 and produce negative word times.
        const dtw = tok?.t_dtw;
        const tokStartMs = typeof dtw === 'number' && dtw >= 0 ? dtw : (tok?.offsets?.from ?? 0);
        const tokEndMs = tok?.offsets?.to ?? tokStartMs;

        // Leading space = start of new word
        const isNewWord = rawText.startsWith(' ') || rawText.startsWith('\u00a0');

        if (isNewWord && currentWord.length > 0) {
          // Flush previous word
          const trimmed = currentWord.trim();
          if (trimmed.length > 0 && !/^[.,!?;:'"()\-–—…]+$/.test(trimmed)) {
            words.push({
              word: trimmed,
              start: wordStart / 1000,
              end: wordEnd / 1000,
              probability: tokenCount > 0 ? wordProb / tokenCount : 0,
            });
          }
          currentWord = rawText;
          wordStart = tokStartMs;
          wordEnd = tokEndMs;
          wordProb = tok?.p ?? 0;
          tokenCount = 1;
        } else {
          // Continue building current word
          if (currentWord.length === 0) {
            wordStart = tokStartMs;
          }
          currentWord += rawText;
          wordEnd = tokEndMs;
          wordProb += tok?.p ?? 0;
          tokenCount++;
        }
      }

      // Flush last word
      if (currentWord.trim().length > 0) {
        const trimmed = currentWord.trim();
        if (!/^[.,!?;:'"()\-–—…]+$/.test(trimmed)) {
          words.push({
            word: trimmed,
            start: wordStart / 1000,
            end: wordEnd / 1000,
            probability: tokenCount > 0 ? wordProb / tokenCount : 0,
          });
        }
      }
    }

    // Fallback: if no tokens or token parsing yielded nothing,
    // split segment text into words with interpolated timestamps
    if (words.length === 0 && text.length > 0) {
      const textWords = text.split(/\s+/).filter((w: string) => w.length > 0 && !/^[.,!?;:'"()\-–—…]+$/.test(w));
      if (textWords.length > 0) {
        const segDuration = (endMs - startMs) / 1000;
        const wordDuration = segDuration / textWords.length;
        for (let i = 0; i < textWords.length; i++) {
          words.push({
            word: textWords[i],
            start: startMs / 1000 + i * wordDuration,
            end: startMs / 1000 + (i + 1) * wordDuration,
            probability: 1.0,
          });
        }
      }
    }

    if (words.length > 0) {
      segments.push({
        start: startMs / 1000,
        end: endMs / 1000,
        text,
        words,
      });
    }
  }

  return { segments };
}

// ── Whole-track transcription ───────────────────────────────────────

export interface TranscribeTrackOptions extends WhisperOptions {
  /** Longest activity-gated window. */
  maxWindowSec?: number;
}

/**
 * Transcribe a whole render in activity-gated windows and stitch the results.
 *
 * Each window is cut to its own WAV, so the times Whisper reports are relative
 * to that cut and adding the window's start is exact — no dependence on how a
 * particular build treats `--offset-t`. A file that cannot be analysed is one
 * window, which is the old behaviour with a longer budget.
 */
export async function transcribeTrack(
  audioPath: string,
  sourceLyrics: string,
  options: TranscribeTrackOptions = {},
): Promise<WhisperResult | null> {
  const plan = planWhisperWindows(audioPath, {
    maxWindowSec: options.maxWindowSec ?? config.whisper.windowSec,
  });
  const windows = plan.windows.length ? plan.windows : [{ startSec: 0, endSec: 0 }];

  const segments: WhisperSegment[] = [];
  let meta: NonNullable<WhisperResult['meta']> | null = null;
  const slices: string[] = [];

  try {
    for (const window of windows) {
      if (options.signal?.aborted) return null;
      const whole = window.startSec <= 0 && window.endSec <= 0;
      let file = audioPath;
      if (!whole) {
        file = await sliceToWav(audioPath, window.startSec, window.endSec, { signal: options.signal });
        slices.push(file);
      }
      const part = await transcribeWithWhisper(file, sourceLyrics, options);
      if (!part) continue;
      if (!meta) meta = { ...(part.meta ?? { model: '', dtw: false }) };
      const shift = whole ? 0 : window.startSec;
      for (const segment of part.segments) {
        segments.push({
          ...segment,
          start: segment.start + shift,
          end: segment.end + shift,
          words: segment.words?.map(w => ({ ...w, start: w.start + shift, end: w.end + shift })),
        });
      }
    }
  } finally {
    for (const slice of slices) removeSlice(slice);
  }

  if (segments.length === 0) return null;
  return { segments, ...(meta ? { meta: { ...meta, windows: windows.length } } : {}) };
}
