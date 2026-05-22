/**
 * whisperTranscribe.ts — Whisper CLI transcription service
 *
 * Wraps whisper.cpp's whisper-cli to transcribe audio files to word-level
 * timestamped text. Used for lyrics synchronisation in the player.
 *
 * Workflow:
 *   1. Locate whisper-cli.exe via config.whisper.exe
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

const execFileAsync = promisify(execFile);

// ── Auto-download constants ─────────────────────────────────────────
// whisper.cpp v1.8.4 CUDA 12.4 build (works with CUDA 12.x and 13.x)
const WHISPER_RELEASE_URL = 'https://github.com/ggerganov/whisper.cpp/releases/download/v1.8.4/whisper-cublas-12.4.0-bin-x64.zip';
const WHISPER_FALLBACK_URL = 'https://github.com/ggerganov/whisper.cpp/releases/download/v1.8.4/whisper-bin-x64.zip';


// ── Types ───────────────────────────────────────────────────────────

export interface WhisperWord {
  word: string;
  start: number;
  end: number;
  probability: number;
}

export interface WhisperSegment {
  start: number;
  end: number;
  text: string;
  words?: WhisperWord[];
}

export interface WhisperResult {
  segments: WhisperSegment[];
}

export interface WhisperOptions {
  /** Whisper model name override (e.g. 'ggml-large-v3-turbo.bin') */
  model?: string;
  /** Language code (e.g. 'en', 'ja') or 'auto' for auto-detect. Default: 'auto' */
  language?: string;
  /** Beam size for decoding. Default: 5 */
  beamSize?: number;
}

// ── Model priority for fallback selection ───────────────────────────

const MODEL_PRIORITY = [
  'ggml-large-v3-turbo.bin',
  'ggml-large-v3.bin',
  'ggml-medium.bin',
  'ggml-base.bin',
];

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
export function findWhisperModel(preferredModel?: string): string | null {
  const modelsDir = config.whisper.modelsDir;

  if (!fs.existsSync(modelsDir)) {
    console.warn(`[Whisper] Models directory does not exist: ${modelsDir}`);
    return null;
  }

  // Try preferred model first
  if (preferredModel) {
    const preferredPath = path.join(modelsDir, preferredModel);
    if (fs.existsSync(preferredPath)) {
      return preferredPath;
    }
    console.warn(`[Whisper] Preferred model not found: ${preferredModel}`);
  }

  // Fall through priority list
  for (const modelName of MODEL_PRIORITY) {
    const modelPath = path.join(modelsDir, modelName);
    if (fs.existsSync(modelPath)) {
      console.log(`[Whisper] Using model: ${modelName}`);
      return modelPath;
    }
  }

  // Last resort: any .bin file in the directory
  try {
    const files = fs.readdirSync(modelsDir).filter(f => f.endsWith('.bin'));
    if (files.length > 0) {
      const fallback = path.join(modelsDir, files[0]);
      console.log(`[Whisper] Fallback model: ${files[0]}`);
      return fallback;
    }
  } catch {
    // Can't read directory
  }

  console.warn('[Whisper] No model files found in models directory');
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
 * Ensure whisper-cli.exe is available, downloading it if needed.
 * Downloads CUDA build first, falls back to CPU-only.
 * Returns true if whisper-cli is ready, false if download failed.
 */
export async function ensureWhisperCli(): Promise<boolean> {
  const whisperExe = config.whisper.exe;
  if (fs.existsSync(whisperExe)) return true;

  const whisperDir = path.dirname(whisperExe);
  fs.mkdirSync(whisperDir, { recursive: true });

  const zipPath = path.join(whisperDir, 'whisper-download.zip');

  // Try CUDA build first, fall back to CPU
  for (const url of [WHISPER_RELEASE_URL, WHISPER_FALLBACK_URL]) {
    try {
      console.log(`[Whisper] Downloading whisper-cli from: ${url}`);
      console.log(`[Whisper] This is a one-time download (~30 MB)...`);

      await httpsDownload(url, zipPath);

      if (!fs.existsSync(zipPath) || fs.statSync(zipPath).size < 1_000_000) {
        console.warn('[Whisper] Downloaded file too small, trying next URL...');
        try { fs.unlinkSync(zipPath); } catch {}
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
        console.log(`[Whisper] ✓ whisper-cli ready at: ${whisperExe}`);
        return true;
      }

      // Search for it in subdirectories (some releases nest in a folder)
      const found = findFileRecursive(whisperDir, 'whisper-cli.exe');
      if (found && found !== whisperExe) {
        // Move all files from nested dir up to whisperDir
        const nestedDir = path.dirname(found);
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
          console.log(`[Whisper] ✓ whisper-cli ready at: ${whisperExe}`);
          return true;
        }
      }

      console.warn('[Whisper] Extracted but whisper-cli.exe not found in expected location');
    } catch (err: any) {
      console.error(`[Whisper] Download failed: ${err.message}`);
      try { fs.unlinkSync(zipPath); } catch {}
    }
  }

  console.error('[Whisper] Could not download whisper-cli.exe — feature unavailable');
  return false;
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
    return fs.existsSync(config.whisper.exe);
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
 * If whisper-cli.exe is missing, attempts to auto-download it first.
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
  // Auto-download if needed
  const ready = await ensureWhisperCli();
  if (!ready) {
    console.error('[Whisper] whisper-cli not available and auto-download failed');
    return null;
  }

  const whisperExe = config.whisper.exe;

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

  // Build CLI args
  //   --split-on-word: split segments at word boundaries, not BPE token boundaries
  //   --max-len 1:     with --split-on-word, this gives exactly 1 word per segment
  //   --suppress-nst:  suppress non-speech tokens (reduces hallucination in silence)
  //   --no-fallback:   don't retry with higher temperature (reduces hallucination)
  const args: string[] = [
    '-m', modelPath,
    '-f', audioPath,
    '-oj',                    // output JSON (writes <input>.json sidecar)
    '--split-on-word',        // split at word boundaries, not BPE tokens
    '--max-len', '1',         // with --split-on-word: 1 word per segment
    '--beam-size', String(beamSize),
    '--no-prints',            // suppress progress to stderr
    '--suppress-nst',         // suppress non-speech tokens
    '--no-fallback',          // don't retry with higher temperature
  ];

  // Language (skip if auto-detect)
  if (language !== 'auto') {
    args.push('--language', language);
  }

  // Vocabulary priming prompt from source lyrics
  const prompt = stripSectionMarkers(sourceLyrics);
  if (prompt.length > 0) {
    args.push('--prompt', prompt);
  }

  // whisper.cpp -oj writes JSON to <audioPath>.json
  const jsonOutputPath = audioPath + '.json';

  console.log(`[Whisper] Transcribing: ${path.basename(audioPath)}`);
  console.log(`[Whisper] Model: ${path.basename(modelPath)}, language: ${language}, beam: ${beamSize}`);
  const t0 = Date.now();

  try {
    await execFileAsync(whisperExe, args, {
      timeout: 120_000,       // 2 minute timeout
      maxBuffer: 10 * 1024 * 1024,  // 10 MB stdout/stderr buffer
    });
  } catch (err: any) {
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
function normaliseWhisperJson(raw: any): WhisperResult {
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

        // DTW timestamps can be in t_dtw (ms) or offsets.from/to
        const tokStartMs = tok?.t_dtw ?? tok?.offsets?.from ?? 0;
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
