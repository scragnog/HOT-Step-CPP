// training/enhanceService.ts — the optional cloud steps
//
// Nothing here runs unless the user presses a button: Genius fetches real
// lyrics for a matched artist/title, and an LLM rewrites the local caption into
// Side-Step's structured 9-sentence form.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.9

import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { config, getFFmpegPath } from '../../config.js';
import { searchSongLyrics } from '../lireek/geniusService.js';
import { getProvider } from '../lireek/llm/registry.js';
import { sanitizeHeaders } from './lyricsSanitizer.js';
import { buildUserPrompt, parseStructuredResponse, CAPTION_INSTRUCTIONS, CAPTION_TOP_P } from './captionPrompt.js';
import {
  applyFactSubstitution, captionWithMoss, correctFactsInProse, MM3_INSTRUCTIONS,
} from './mossCaption.js';
import type { TrainingDatasetRow, TrainingSample } from './types.js';

const MAX_CAPTION_CHARS = 4000;
const MAX_LYRICS_CHARS = 20000;

/** Zero-width and control chars we never want inside a sidecar (§7.6). */
const ZERO_WIDTH = new RegExp('[\\u200B-\\u200D\\uFEFF]', 'g');
const CONTROL_CHARS = new RegExp('[\\u0000-\\u0008\\u000B\\u000C\\u000E-\\u001F\\u007F]', 'g');

function cleanText(raw: string, cap: number): string {
  return String(raw ?? '')
    .replace(/\r\n?/g, '\n')
    .replace(ZERO_WIDTH, '')
    .replace(CONTROL_CHARS, '')
    .trim()
    .slice(0, cap);
}

// ── Genius ───────────────────────────────────────────────────────────────

/**
 * Pull artist/title out of a filename. Handles `NN-artist-title`,
 * `artist - title` and `NN. title`, converting underscores to spaces and
 * stripping leading track numbers.
 */
export function parseArtistTitleFromFilename(filename: string): { artist: string; title: string } {
  const ext = path.extname(filename);
  let stem = (ext ? filename.slice(0, filename.length - ext.length) : filename).trim();
  stem = stem.replace(/_/g, ' ').trim();

  // `NN. title` / `NN - title` / `NN title`
  const numbered = /^\d{1,3}\s*[.\-–]?\s+(.*)$/.exec(stem);
  const numberedDash = /^\d{1,3}\s*-\s*(.*)$/.exec(stem);
  let body = stem;
  if (numberedDash) body = numberedDash[1].trim();
  else if (numbered) body = numbered[1].trim();

  // `artist - title`
  const dashed = /^(.+?)\s+-\s+(.+)$/.exec(body);
  if (dashed) return { artist: dashed[1].trim(), title: dashed[2].trim() };

  // `artist-title` (no spaces around the dash) — only when it splits cleanly in two
  const tight = body.split('-');
  if (tight.length === 2 && tight[0].trim() && tight[1].trim()) {
    return { artist: tight[0].trim(), title: tight[1].trim() };
  }

  return { artist: '', title: body };
}

/** Artist/title resolution order: embedded tags → filename → dataset defaults. */
export function resolveArtistTitle(
  sample: TrainingSample,
  ds: TrainingDatasetRow,
  opts: { artist?: string },
): { artist: string; title: string } {
  const fromName = parseArtistTitleFromFilename(sample.filename);
  const artist = sample.tagArtist || fromName.artist || opts.artist || ds.defaultArtist || '';
  const title = sample.tagTitle || fromName.title || '';
  return { artist: artist.trim(), title: title.trim() };
}

/** Strip "(feat. X)" / "[ft. X]" parentheticals and trailing feat clauses from a title. */
export function cleanTitleForSearch(title: string): string {
  return title
    .replace(/[([]\s*(?:feat\.?|ft\.?|featuring|with)\b[^)\]]*[)\]]/gi, ' ')
    .replace(/\s+(?:feat\.?|ft\.?|featuring)\s+.*$/i, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Strip a trailing "feat./ft. X" clause from an artist string. */
export function cleanArtistForSearch(artist: string): string {
  return artist.replace(/\s+(?:feat\.?|ft\.?|featuring)\s+.*$/i, '').trim();
}

/** Words that mark a parenthetical as an EDITION, not part of the song title. */
const EDITION_WORDS =
  /\b(?:re-?master(?:ed)?|remix(?:es|ed)?|mono|stereo|deluxe|bonus|reissue|expanded|anniversary|edition|version|mix|edit|live|acoustic|demo|instrumental|unplugged|session|produced\s+by|feat\.?|ft\.?|featuring|explicit|clean|radio|single|take\s+\d+|\d{4})\b/i;

/**
 * Drop edition/version parentheticals from a title before it is SEARCHED.
 *
 * cleanTitleForSearch only ever removed feat-clauses, so a tag like
 * "Rio (2009 Remaster)" went into the Genius query verbatim — and the query
 * `Rio (2009 Remaster) Duran Duran` returns five unrelated user playlists, with
 * the actual song nowhere in the candidate set. No amount of match-side
 * normalisation can recover that; the search itself has to be clean. This one
 * pattern accounted for whole albums: duranduran_rio, madness_wonderful,
 * panicatthedisco_fever, snoopdogg_randg ("(Produced By The Neptunes)"),
 * joydivision_bestof ("(12\" Single Version)") — verified 2026-08-05.
 *
 * Keyword-gated on purpose: a blanket parenthetical strip would maim titles
 * where the brackets are the name — "That's The Way (I Like It)",
 * "(Don't Fear) The Reaper", "Let It Go (Part One)".
 */
export function stripEditionSuffixes(title: string): string {
  return title
    .replace(/[([][^)\]]*[)\]]/g, m => (EDITION_WORDS.test(m) ? ' ' : m))
    .replace(/\s+/g, ' ')
    .replace(/[\s\-–—]+$/, '')
    .trim();
}

/**
 * Fetch lyrics from Genius for one sample.
 * `null` means "no confident match" — the caller logs a warning and moves on.
 *
 * Retry ladder (collabs and feats break exact primary-artist matching —
 * "Electric Callboy & BABYMETAL" is the primary artist of RATATATA):
 *   1. exact match, artist/title as resolved (tags first)
 *   2. exact match, feat-clauses stripped from both
 *   3. exact match, edition/version parentheticals also dropped
 *   4. relaxed match (artist containment + title agreement), barest title
 *
 * Steps run widest-query-first so an exact tag match is still preferred; the
 * stripped forms are only tried once the verbatim tag has failed.
 *
 * Throws `GeniusNoArtistError` when there is nothing to search WITH, so the
 * caller can tell "Genius has no lyrics for this" apart from "we never asked".
 */
export class GeniusNoArtistError extends Error {}

export async function enhanceGenius(
  sample: TrainingSample,
  ds: TrainingDatasetRow,
  opts: { artist?: string; album?: string; sanitizeHeaders: boolean },
): Promise<{ lyrics: string; source: 'genius'; searched: string } | null> {
  const { artist, title } = resolveArtistTitle(sample, ds, opts);
  const searched = `${artist} — ${title}`;
  if (!artist || !title) {
    // pod_satellite lost all 12 tracks here silently: no embedded tags, and
    // "01. Set It Off.mp3" parses to an empty artist, so Genius was never even
    // called. Surfacing it tells the user to set the dataset's default artist
    // instead of leaving a blank-lyrics album that looks like a Genius miss.
    throw new GeniusNoArtistError(
      !artist
        ? 'no artist — set the dataset default artist, or tag the files'
        : 'no title — tag the files',
    );
  }

  const cleanArtist = cleanArtistForSearch(artist) || artist;
  const cleanTitle = cleanTitleForSearch(title) || title;
  const bareTitle = stripEditionSuffixes(cleanTitle) || cleanTitle;

  let hit = await searchSongLyrics(artist, title);
  if (!hit && (cleanArtist !== artist || cleanTitle !== title)) {
    hit = await searchSongLyrics(cleanArtist, cleanTitle);
  }
  if (!hit && bareTitle !== cleanTitle) {
    hit = await searchSongLyrics(cleanArtist, bareTitle);
  }
  if (!hit) {
    hit = await searchSongLyrics(cleanArtist, bareTitle, { relaxed: true });
  }
  if (!hit || !hit.lyrics?.trim()) return null;

  let lyrics = cleanText(hit.lyrics, MAX_LYRICS_CHARS);
  if (opts.sanitizeHeaders) lyrics = sanitizeHeaders(lyrics);
  if (!lyrics.trim()) return null;

  return { lyrics, source: 'genius', searched };
}

// ── LLM caption ──────────────────────────────────────────────────────────

/** Inline-data request cap (Gemini inline limit is 20 MB; leave headroom). */
const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;

/**
 * Transcode any input to a 96 kbps stereo MP3 for upload — small enough to
 * inline (≈0.7 MB/min) and lossless-enough for captioning. `null` = no ffmpeg
 * or transcode failure; the caller falls back to text-only.
 */
async function transcodeForUpload(audioPath: string, signal?: AbortSignal): Promise<Buffer | null> {
  const ffmpeg = getFFmpegPath();
  if (!ffmpeg) return null;
  const tmp = path.join(os.tmpdir(), `hs_training_caption_${crypto.randomBytes(6).toString('hex')}.mp3`);
  try {
    await new Promise<void>((resolve, reject) => {
      execFile(
        ffmpeg,
        ['-y', '-i', audioPath, '-ar', '44100', '-ac', '2', '-codec:a', 'libmp3lame', '-b:a', '96k', tmp],
        { timeout: 180_000, maxBuffer: 10 * 1024 * 1024, signal },
        (error) => {
          if (error && (error as NodeJS.ErrnoException).name === 'AbortError') { reject(error); return; }
          resolve();
        },
      );
    });
    if (!fs.existsSync(tmp)) return null;
    const buf = fs.readFileSync(tmp);
    return buf.length > 0 && buf.length <= MAX_UPLOAD_BYTES ? buf : null;
  } finally {
    try { fs.unlinkSync(tmp); } catch { /* never existed */ }
  }
}

/** One-shot Gemini generateContent with the MP3 inlined next to the prompt. */
async function callGeminiWithAudio(
  model: string,
  prompt: string,
  mp3: Buffer,
  gen: { temperature: number; topP: number },
  signal?: AbortSignal,
): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${config.lireek.geminiApiKey}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      contents: [{
        role: 'user',
        parts: [
          { text: prompt },
          { inline_data: { mime_type: 'audio/mpeg', data: mp3.toString('base64') } },
        ],
      }],
      generationConfig: { temperature: gen.temperature, topP: gen.topP },
    }),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`Gemini audio request failed (${res.status}): ${body.slice(0, 300)}`);
  }
  const data = await res.json() as any;
  const parts = data?.candidates?.[0]?.content?.parts;
  return Array.isArray(parts) ? parts.map((p: any) => p?.text ?? '').join('') : '';
}

function isNetworkError(err: any): boolean {
  const msg = String(err?.message || err || '');
  return /fetch failed|ECONNRESET|ECONNREFUSED|ETIMEDOUT|ENOTFOUND|socket hang up|network|timed out/i.test(msg);
}

/**
 * Caption one sample locally with MOSS-Music-8B.
 *
 * Unlike the LLM path this does not "rewrite" anything — the local caption, BPM
 * and key are not shown to the model at all, because the whole value here is a
 * description written from the audio rather than anchored on a weak text prior.
 * Local analysis re-enters afterwards, on the MM3 side only, where it replaces
 * MOSS's unreliable tempo/key (see mossCaption.ts).
 *
 * The MM3 caption is written to `<stem>.mm3.txt` beside the audio, which is the
 * file `ace-train mm3-condition` already reads. An existing one is backed up to
 * `.mm3.txt.prev` ONCE — re-running must never overwrite the backup, or the
 * second run destroys the original rather than the previous hybrid output.
 */
async function captionWithMossProvider(
  sample: TrainingSample,
  opts: {
    wantMm3?: boolean;
    signal?: AbortSignal;
    log?: (level: 'info' | 'warn', message: string) => void;
  },
): Promise<Record<string, string>> {
  // One fact set drives both corrections: the MM3 `Basic Attributes:` line and
  // the tempo/key welded into the AS1.5 caption sentence.
  const facts = {
    bpm: sample.bpm,
    keyscale: sample.key,
    signature: sample.signature,
    genre: sample.genre,
    // The pre-existing caption counts as an observed source for genre picking
    // only — it was written from audio too. It is never shown to MOSS.
    observedCaption: sample.caption,
  };

  const result = await captionWithMoss(
    sample.audioPath, facts,
    { wantMm3: opts.wantMm3, signal: opts.signal, log: opts.log },
  );

  const out: Record<string, string> = {};
  if (result.prose?.trim()) {
    const parsed = parseStructuredResponse(result.prose);
    if (parsed.caption) {
      // MOSS welds tempo/key into the sentence ("...in B minor at 120 BPM..."),
      // where the MM3 line-replacement cannot reach. Left alone, the caption
      // would contradict the `bpm` field written beside it in the same sidecar.
      out.caption = cleanText(correctFactsInProse(parsed.caption, facts), MAX_CAPTION_CHARS);
    }
    if (parsed.genre) out.genre = cleanText(parsed.genre, 300);
    // BPM and key are deliberately NOT taken from MOSS — local analysis already
    // has them exactly, and MOSS is measurably wrong on both (burn: ~102 / C#
    // minor vs Essentia's 90 / E major). Signature likewise.
    opts.log?.('info', 'captioned locally from audio (MOSS-Music-8B)');
  }

  if (result.mm3?.trim()) {
    writeMm3Sidecar(sample, result.mm3, opts.log);
  }

  return out;
}

/** Write `<stem>.mm3.txt` with the one-time `.prev` backup. Shared by the MOSS
 *  and cloud caption paths so both leave identical files behind. */
function writeMm3Sidecar(
  sample: TrainingSample,
  mm3: string,
  log?: (level: 'info' | 'warn', message: string) => void,
): void {
  const stem = sample.audioPath.replace(/\.[^.\\/]+$/, '');
  const dst = `${stem}.mm3.txt`;
  try {
    if (fs.existsSync(dst) && !fs.existsSync(`${dst}.prev`)) fs.renameSync(dst, `${dst}.prev`);
    fs.writeFileSync(dst, mm3, 'utf8');
    log?.('info', `wrote MM3 structured caption (${path.basename(dst)})`);
  } catch (err: any) {
    log?.('warn', `MM3 caption not written: ${String(err?.message || err).slice(0, 160)}`);
  }
}

/**
 * Rewrite one sample's caption with an LLM. Returns the subset of sidecar
 * fields the model produced — `{}` when it produced nothing usable.
 */
export async function enhanceCaption(
  sample: TrainingSample,
  ds: TrainingDatasetRow,
  opts: {
    provider: string; model?: string; includeLyricsExcerpt: boolean; temperature: number;
    /** MOSS only: also write the MM3 Structured Caption sidecar. */
    wantMm3?: boolean;
    signal?: AbortSignal;
    log?: (level: 'info' | 'warn', message: string) => void;
  },
): Promise<Record<string, string>> {
  // Local, audio-grounded path. Checked BEFORE getProvider because 'moss' is not
  // an LLM provider — it has no API key, no model list, and it listens.
  if (opts.provider === 'moss') {
    return captionWithMossProvider(sample, opts);
  }

  const provider = getProvider(opts.provider);
  const { artist, title } = resolveArtistTitle(sample, ds, {});

  const promptArgs = {
    title: sample.tagTitle || title,
    artist,
    lyricsExcerpt: opts.includeLyricsExcerpt ? sample.lyrics.slice(0, 500) : undefined,
    localCaption: sample.caption,
    bpm: sample.bpm,
    key: sample.key,
    signature: sample.signature,
  };
  const model = opts.model || provider.defaultModel;

  let text = '';

  // Audio-grounded path (Gemini only): a text-only rewrite anchors on the local
  // caption, which is exactly what the user is trying to escape. Any failure
  // here falls through to the text-only path rather than failing the sample.
  if (opts.provider === 'gemini' && config.lireek.geminiApiKey) {
    try {
      const mp3 = await transcodeForUpload(sample.audioPath, opts.signal);
      if (mp3) {
        const audioPrompt = buildUserPrompt({ ...promptArgs, audioAttached: true });
        text = await callGeminiWithAudio(model, audioPrompt, mp3,
          { temperature: opts.temperature, topP: CAPTION_TOP_P }, opts.signal);
        if (text.trim()) {
          opts.log?.('info', `captioned from audio (${(mp3.length / 1048576).toFixed(1)} MB mp3)`);
        }
        // ── the MM3 Structured Caption, from the same audio ──
        //
        // MM3 emission was a MOSS-only feature by accident of history (the MM3
        // tooling shipped while MOSS was the captioner). Same proven prompt,
        // second call on the same mp3 buffer — the transcode is paid once and
        // the audio upload is the only real per-call cost. Low temperature:
        // captions are facts, not prose style (MOSS runs greedy for the same
        // reason). The result gets the exact post-processing MOSS output gets:
        // fact substitution rewrites Basic Attributes from LOCAL analysis, so
        // the model's bpm/key guesses never reach the trainer.
        if (opts.wantMm3 !== false && text.trim()) {
          try {
            const mm3raw = await callGeminiWithAudio(model, MM3_INSTRUCTIONS, mp3,
              { temperature: 0.2, topP: CAPTION_TOP_P }, opts.signal);
            if (mm3raw.trim()) {
              const mm3 = applyFactSubstitution(mm3raw, {
                bpm: sample.bpm,
                keyscale: sample.key,
                signature: sample.signature,
                genre: sample.genre,
                observedCaption: sample.caption,
              });
              writeMm3Sidecar(sample, mm3, opts.log);
            } else {
              opts.log?.('warn', 'MM3 caption call returned nothing — .mm3.txt not written');
            }
          } catch (err: any) {
            if ((err as NodeJS.ErrnoException)?.name === 'AbortError') throw err;
            opts.log?.('warn',
              `MM3 caption failed (${String(err?.message || err).slice(0, 160)}) — AS1.5 caption kept`);
          }
        }
      } else {
        opts.log?.('warn', 'audio attach unavailable (no ffmpeg or file too large) — captioning from text only');
      }
    } catch (err: any) {
      if ((err as NodeJS.ErrnoException)?.name === 'AbortError') throw err;
      opts.log?.('warn', `audio caption failed (${String(err?.message || err).slice(0, 160)}) — falling back to text`);
      text = '';
    }
  }

  if (!text.trim()) {
    const userPrompt = buildUserPrompt({ ...promptArgs, audioAttached: false });
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        let streamed = '';
        const result = await provider.call(
          CAPTION_INSTRUCTIONS,
          userPrompt,
          model,
          (chunk: string) => { streamed += chunk; },
          { temperature: opts.temperature, top_p: CAPTION_TOP_P },
        );
        // Defensive fallback (assistant.ts): some providers return '' and only stream.
        text = (result && result.trim()) ? result : streamed;
        break;
      } catch (err: any) {
        if (attempt === 0 && isNetworkError(err)) {
          await new Promise(r => setTimeout(r, 2000));
          continue;
        }
        throw err;
      }
    }
  }

  if (!text.trim()) return {};

  const parsed = parseStructuredResponse(text);
  const out: Record<string, string> = {};
  if (parsed.caption) out.caption = cleanText(parsed.caption, MAX_CAPTION_CHARS);
  if (parsed.genre) out.genre = cleanText(parsed.genre, 300);
  if (parsed.bpm) out.bpm = cleanText(parsed.bpm, 16);
  if (parsed.key) out.key = cleanText(parsed.key, 64);
  if (parsed.signature) out.signature = cleanText(parsed.signature, 16);
  return out;
}
