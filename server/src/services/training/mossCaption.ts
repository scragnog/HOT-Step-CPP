// training/mossCaption.ts — the first caption provider that actually LISTENS
//
// Every other provider in the registry rewrites text. MOSS-Music-8B runs locally
// through `ace-caption` (engine/tools/ace-caption.cpp) and writes captions from
// the audio itself, which is a different kind of output, not a cheaper one:
//
//   Rob's blind-ish A/B, 2026-08-15, same track / lyrics / 5 seeds, Basic
//   Attributes forced identical in both arms so prose was the only variable:
//     restructured-from-Gemini -> "more plain rock, not particularly punk/emo"
//     MOSS-from-audio          -> "sounds like the album-A artist already, that's
//                                  crazy ... MOSS is WAY better"
//   Two of five seeds hit the target artist with NO ADAPTER LOADED.
//
// So this is the preferred path when the binary and weights are present, and the
// cloud rewrite in enhanceService.ts is the fallback, not the other way round.
//
// ── What this module does NOT trust MOSS with ────────────────────────────────
//
// Tempo and key. MOSS is a documented blind spot on both — measured on
// `03-alkaline_trio-burn.flac` it said ~102 BPM / C# minor where Essentia says
// 90 / E major, and Essentia is right. Those two numbers are already known
// exactly from local analysis, so `applyFactSubstitution` overwrites MOSS's
// `Basic Attributes:` line with the sidecar's values and keeps every other
// section MOSS wrote. This supersedes engine/tools/mm3-caption-hybrid.py, which
// did the same job as a post-pass over dataset folders.

import fs from 'fs';
import os from 'os';
import path from 'path';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { config, getFFmpegPath } from '../../config.js';
import { CAPTION_INSTRUCTIONS } from './captionPrompt.js';
import { normalizeYue2Caption } from '../lireek/prompts.js';
import { normaliseLyricTags } from './lyricTags.js';

/** Modes `ace-caption` can decode off a single encode. */
export type MossMode = 'prose' | 'mm3' | 'lyrics' | 'yue2';

/**
 * The local audio models `ace-caption` runs. Both take the same modes, prompts and
 * batching; the binary picks the family from the models folder. Music Flamingo is
 * the stronger lyric transcriber by a wide margin (median WER 24% vs MOSS's 59% on
 * a 17-track album test, with section tags on every track).
 */
export type LocalCaptioner = 'moss' | 'flamingo';

export const LOCAL_CAPTIONER_NAMES: Record<LocalCaptioner, string> = {
  moss: 'MOSS-Music-8B',
  flamingo: 'Music Flamingo',
};

export function isLocalCaptioner(provider: string | undefined | null): provider is LocalCaptioner {
  return provider === 'moss' || provider === 'flamingo';
}

/**
 * Lyric transcription prompt for the local models. The exact wording the lyric
 * bake-off measured (Music Flamingo, median WER 24% with section tags on every
 * track); passed via `--prompt-file lyrics=` like the other modes.
 */
export const LYRICS_INSTRUCTIONS: string =
  'Transcribe the complete lyrics of this song.\n\n'
  + 'Format the output exactly like this:\n'
  + '- Mark every section with a bracketed tag on its own line: [Intro], [Verse], [Pre-Chorus], '
  + '[Chorus], [Bridge], [Breakdown], [Instrumental], [Outro].\n'
  + '- Put the sung words underneath each tag, one line per sung line.\n'
  + '- Use [Instrumental] for passages with no vocals.\n'
  + '- Transcribe only what is actually sung. Do not invent lines, do not translate, and do not '
  + 'add commentary, timestamps or explanations.\n\n'
  + 'If the track has no vocals at all, reply with exactly: [Instrumental]';

/**
 * The YuE2 planner caption, from the local model's ears and no chat provider.
 *
 * Neither local model can hold the one-sentence, fixed-order format: given the
 * chat prompt (YUE2_CAPTION_SYSTEM_PROMPT) both wrote 150-300 word paragraphs.
 * Both fill short labelled fields reliably, so they answer six fields and
 * `assembleYue2FromFields` builds the sentence in that order; BPM comes from
 * local analysis via normalizeYue2Caption, as it does on the chat path.
 */
export const YUE2_LOCAL_INSTRUCTIONS: string =
  'Listen to this track and fill in these six fields, one per line, each a short phrase of a few words:\n\n'
  + 'language: <the language the vocal is sung in, or "instrumental">\n'
  + 'genre: <the specific style you hear, with era words if they help; never a bare umbrella like "rock" or "pop">\n'
  + 'vocal: <register, gender and delivery of the lead voice, or what carries the lead line if instrumental>\n'
  + 'instruments: <the instruments actually present, named concretely>\n'
  + 'mood: <two to four plain words>\n'
  + 'production: <the mix and era character>\n\n'
  + 'Write only those six lines. Do not name the artist, the band or the song, and do not quote the lyrics.';

const YUE2_FIELDS = ['language', 'genre', 'vocal', 'instruments', 'mood', 'production'] as const;

/**
 * Six field answers -> the YuE2 sentence body (no BPM; normalizeYue2Caption adds
 * it). Accepts `key: value` lines (Flamingo) or six bare lines in order (MOSS).
 * Each answer is cut to its first clause and parentheticals are dropped — that is
 * where a model names the original song ("(as a cover of ...)").
 */
export function assembleYue2FromFields(raw: string): string {
  const lines = String(raw ?? '').split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const labelled = new Map<string, string>();
  for (const l of lines) {
    const m = /^(language|genre|vocal|instruments|mood|production)\s*:\s*(.*)$/i.exec(l);
    if (m) labelled.set(m[1].toLowerCase(), m[2]);
  }
  const values = labelled.size >= 4
    ? YUE2_FIELDS.map(f => labelled.get(f) ?? '')
    : lines.slice(0, YUE2_FIELDS.length);
  const clean = (v: string, maxWords: number) => {
    let s = v.replace(/\([^)]*\)/g, ' ').replace(/[“”"]/g, '')
      .split(/\.\s|\s[–—-]\s|\.$/)[0]
      .replace(/;/g, ' /').replace(/\s+/g, ' ').replace(/[\s,.;:]+$/, '').trim();
    const words = s.split(' ');
    if (words.length > maxWords) {
      // Over the cap: back off to the last comma or slash in range, else to the
      // cap, and never end on a dangling "with" / "for a".
      s = words.slice(0, maxWords).join(' ');
      const cut = Math.max(s.lastIndexOf(','), s.lastIndexOf(' /'));
      if (cut > s.length / 3) s = s.slice(0, cut);
      s = s.replace(/(?:\s+(?:a|an|the|and|or|with|for|of|to|in|on|at|by|that|which|is|are))+\s*$/i, '')
        .replace(/[\s,.;:/]+$/, '');
    }
    return s;
  };
  const caps = [3, 10, 16, 18, 6, 14];
  return values.map((v, i) => clean(v, caps[i])).filter(Boolean).join(', ');
}

/**
 * The per-track hint for the local models: the file's own genre tag. Without it
 * both models recognise a cover from its lyrics and describe the ORIGINAL song's
 * genre — on a dataset of lounge covers of rock songs they wrote "garage punk",
 * "synth-pop" and "hip-hop"; with it they describe the piano, brushed drums and
 * crooned vocal actually on the recording (measured 2026-10-24, both models).
 */
export function genreHint(genre: string | undefined | null): string {
  const g = (genre ?? '').replace(/[\t\r\n]+/g, ' ').trim();
  return g
    ? `The file is tagged with the genre "${g}". Describe the arrangement you actually hear, `
      + "which may be a cover in a different style from the song's original."
    : '';
}

/**
 * The MM3 Structured Caption prompt.
 *
 * Starts as a copy of `PROMPT_MM3` in engine/tools/ace-caption.cpp and is
 * passed via `--prompt-file mm3=` so prompt iteration never needs a C++
 * rebuild — the app's captions are driven from HERE; the CLI built-in is the
 * fallback for bare `ace-caption` runs. Two rules were added over the C++
 * copy after auditing real output against the 1000 official templates:
 * MOSS wrote the Instrument Lifecycle Description as one flowing line
 * instead of Primary:/Secondary: sub-lines on 5/9 americanfootball tracks
 * (999/1000 templates split them), and once appended a 12-line run of bare
 * "Chorus:"/"Verse N:" labels after the final field.
 *
 * A third rule (copy labels exactly) and the `<Specific Genre>` placeholder
 * came from Music Flamingo on a lounge-covers dataset: it echoed the old
 * `<Genre / Subgenre>` back as a literal label, and with the genre hint
 * appended it dropped whole fields on 56/130 tracks. The rule brought that to
 * 11/56 on a greedy re-run; moving the hint before the template made it worse.
 */
export const MM3_INSTRUCTIONS: string =
  'Listen to this track and write a Structured Caption in exactly the format below.\n\n'
  + "Use these three section labels on their own lines, with no markdown, no '#', and no "
  + 'extra sections:\n\n'
  + 'Global Metadata\n'
  + 'Basic Attributes: bpm is <N>. key is <X>, and scale is <major|minor>. <Specific Genre>.\n'
  + 'Global Emotional Progression: <how the emotional arc moves across the song>\n'
  + 'Application Scenarios & Imagery: <where this music would be used; concrete visual imagery>\n'
  + 'Sonics & Production Profile: <soundstage, density, frequency balance, dynamics, mix character>\n'
  + 'Vocal Details\n'
  + 'Vocal Gender & Timbre: <Singer A (Male/Female). timbre, register, texture>\n'
  + 'Vocal Style: <delivery and how it evolves across sections>\n'
  + 'Harmony/Backing Vocals: <layering, doubling, gang vocals, call-and-response, or state none>\n'
  + 'Vocal FX: <reverb, delay, distortion, pitch correction, or state minimal>\n'
  + 'Arrangement\n'
  + 'Instrument Lifecycle Description (Primary/Secondary Layering):\n'
  + 'Primary: <the instruments carrying the track and how they change>\n'
  + 'Secondary: <supporting instruments and when they enter or drop>\n'
  + 'Groove & Foundation Progression: <rhythm section, feel changes, section by section>\n'
  + 'Embellishments, Textures & Spatial FX: <risers, sweeps, reverse reverb, glitches, ambience>\n\n'
  + 'Rules:\n'
  + '- Copy every section and field label exactly as written above, each one once; do not '
  + 'rename, merge or skip any.\n'
  + '- Describe only what you actually hear. Do not invent an exact BPM or key if unsure.\n'
  + '- If the track is instrumental, say so under Vocal Details and name the lead instrument.\n'
  + '- Do NOT quote or summarise the lyrics anywhere.\n'
  + '- Write flowing prose inside each field.\n'
  + "- Under 'Instrument Lifecycle Description (Primary/Secondary Layering):', write "
  + "'Primary:' and 'Secondary:' each at the start of its own line, exactly as the "
  + 'template shows. Never merge them into one paragraph.\n'
  + "- The caption ENDS after 'Embellishments, Textures & Spatial FX'. Do not append "
  + 'section names, chorus or verse labels, timestamps, or anything else after it.';

export interface MossPaths {
  exe: string;
  modelsDir: string;
  lm: string;
  audio: string;
}

const BIN = process.platform === 'win32' ? '.exe' : '';

/**
 * Resolve the binary + weights, or explain what is missing.
 *
 * The LM is looked up q8_0-first to match `ace-caption`'s own probe order: q8_0
 * is half the download and measurably no worse for captioning, so a user who has
 * both should get the small one, and the resolution order here must not disagree
 * with the engine's or the UI would advertise a model the engine won't load.
 */
export function resolveMossPaths(model: LocalCaptioner = 'moss'): { paths: MossPaths } | { missing: string } {
  const exe = path.join(path.dirname(config.aceServer.exe), `ace-caption${BIN}`);
  if (!fs.existsSync(exe)) return { missing: `ace-caption${BIN} not built` };

  // models/moss holds moss-*.gguf, models/flamingo holds flamingo-*.gguf.
  const modelsDir = path.join(config.aceServer.models, model);
  const lm = [`${model}-lm-q8_0.gguf`, `${model}-lm-f16.gguf`]
    .map(n => path.join(modelsDir, n))
    .find(p => fs.existsSync(p));
  const audio = path.join(modelsDir, `${model}-aud-f16.gguf`);

  const get = 'download it in Model Manager > Shared > Captioning';
  if (!lm) return { missing: `no ${model}-lm-*.gguf in ${modelsDir}; ${get}` };
  if (!fs.existsSync(audio)) return { missing: `${model}-aud-f16.gguf missing from ${modelsDir}; ${get}` };
  return { paths: { exe, modelsDir, lm, audio } };
}

export function mossCaptionAvailable(model: LocalCaptioner = 'moss'): boolean {
  return 'paths' in resolveMossPaths(model);
}

// ── Running the model ────────────────────────────────────────────────────

/**
 * Repetition penalty for redoing a lyric transcript that looped. Decoding runs with
 * none (MOSS_REP_PENALTY, 1.0): on the 17-track album test that scored median WER
 * 19.6% against 24.0% at 1.05, but looped on 3 tracks (one at 136% WER). ace-caption
 * redoes just those, in-process and off the same prefill (--retry-loops), giving
 * median 20.0% / mean 22.5% against 19.6% / 29.4% for one pass. Quantisation (q8_0
 * vs f16), sampling (temperature 0.4/0.7) and penalty 1.02 were measured and did
 * not help.
 *
 * In-process on purpose: a redo as a second ace-caption run queued behind the
 * dataset's batch (one Flamingo process at a time, for VRAM), so four looped tracks
 * stalled a label job's whole worker pool until the batch finished.
 */
const LYRICS_RETRY_REP_PENALTY = 1.05;

function retryLoopArgs(model: LocalCaptioner, modes: MossMode[]): string[] {
  return model === 'flamingo' && modes.includes('lyrics')
    ? ['--retry-loops', String(LYRICS_RETRY_REP_PENALTY)] : [];
}

export interface MossRunOptions {
  /** Which local model; default MOSS. */
  model?: LocalCaptioner;
  modes: MossMode[];
  /** Single-track runs: appended to every non-lyrics prompt (`--hint`). */
  hint?: string;
  /** Batch runs: audio path -> hint (the list file's third column). */
  hints?: Map<string, string>;
  /** Per-mode prompt overrides. `prose` normally carries CAPTION_INSTRUCTIONS. */
  prompts?: Partial<Record<MossMode, string>>;
  maxTokens?: number;
  /** Hard audio cap in seconds. Cost is roughly linear past the encode. */
  maxSeconds?: number;
  temperature?: number;
  repPenalty?: number;
  freqPenalty?: number;
  signal?: AbortSignal;
  log?: (level: 'info' | 'warn', message: string) => void;
}

/**
 * Sampling defaults, measured rather than guessed.
 *
 * rep 1.05 + freq 0.3 was the optimum of a sweep over the repetition failure
 * mode that made MOSS's early lyric output look far worse than it is: bare
 * `repetition_penalty` fires on PRESENCE, so at the strength needed to break a
 * loop it also suppresses words a lyric legitimately repeats — a chorus. Adding
 * `frequency_penalty`, which scales with COUNT, breaks loops without flattening
 * choruses. Do not raise rep to fix a loop; raise freq.
 */
export const MOSS_TEMPERATURE = 0;      // greedy — captions are facts, not prose style
// 1.0 = off, and that is parity, not a guess: the "rep 1.05 + freq 0.3" sweep
// ran through SGLang, which parses repetition_penalty and NEVER applies it (no
// repetition_penalty.py in its penaltylib, nothing in the sampling path reads
// it). So the sweep's optimum was frequency_penalty alone, and applying rep
// here is a divergence from the runtime the settings were tuned on. Verified
// 2026-08-18: rep 1.0 vs 1.05 changes none of the 9 americanfootball captions.
export const MOSS_REP_PENALTY = 1.0;
export const MOSS_FREQ_PENALTY = 0.3;

/**
 * MOSS runs strictly one at a time, enforced HERE rather than at the call sites.
 *
 * The caption path is governed by `captionLimiter`, whose concurrency is
 * user-configurable (`config.labeling.captionConcurrency`) and which several
 * people will reasonably raise — for a cloud provider, N parallel API calls is
 * exactly the right thing to do. For MOSS it is fatal: each run is a separate
 * process loading ~8.1 GB of LM plus 1.6 GB of audio tower, so concurrency 2
 * exceeds a 5090 before either run reaches the first decode. Serialising inside
 * the module means raising the limiter stays safe for the cloud providers it was
 * meant for, and no future caller has to know this.
 */
let mossQueue: Promise<unknown> = Promise.resolve();

function serialised<T>(body: () => Promise<T>): Promise<T> {
  const run = mossQueue.then(body, body);
  // Keep the chain alive after a rejection, and never leak the rejection into
  // the next waiter — it must see a settled predecessor, not a failed one.
  mossQueue = run.then(() => undefined, () => undefined);
  return run;
}

/**
 * One encode, N decodes. Returns the text per requested mode.
 *
 * The encoder (mel + 32 Whisper layers) is prompt-independent, so asking for
 * prose+mm3 together costs one audio pass and two short decodes rather than two
 * full runs. That is the whole reason `--mode` takes a list.
 */
export function runMossCaption(
  audioPath: string,
  opts: MossRunOptions,
): Promise<Partial<Record<MossMode, string>>> {
  return serialised(() => runMossCaptionInner(audioPath, opts));
}

async function runMossCaptionInner(
  audioPath: string,
  opts: MossRunOptions,
): Promise<Partial<Record<MossMode, string>>> {
  const model = opts.model ?? 'moss';
  const resolved = resolveMossPaths(model);
  if ('missing' in resolved) throw new Error(`${LOCAL_CAPTIONER_NAMES[model]} unavailable: ${resolved.missing}`);
  const { exe, modelsDir } = resolved.paths;

  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'hs_moss_'));
  const outBase = path.join(work, 'out.txt');
  const args: string[] = [
    '--models', modelsDir,
    '--family', model,
    '--src-audio', audioPath,
    ...(opts.hint ? ['--hint', opts.hint] : []),
    '--mode', opts.modes.join(','),
    '-o', outBase,
    '--temperature', String(opts.temperature ?? MOSS_TEMPERATURE),
    '--rep-penalty', String(opts.repPenalty ?? MOSS_REP_PENALTY),
    '--freq-penalty', String(opts.freqPenalty ?? MOSS_FREQ_PENALTY),
  ];
  if (opts.maxTokens) args.push('--max-tokens', String(opts.maxTokens));
  if (opts.maxSeconds) args.push('--max-seconds', String(opts.maxSeconds));
  args.push(...retryLoopArgs(model, opts.modes));

  // ffmpeg is only consulted for non-WAV/MP3 input, but passing our resolved
  // path means a portable release doesn't depend on ffmpeg being on PATH.
  const ffmpeg = getFFmpegPath();
  if (ffmpeg) args.push('--ffmpeg', ffmpeg);

  // Per-mode prompt files. The `<mode>=` prefix is what keeps the AS1.5 block and
  // the MM3 Structured Caption on ONE encode — a bare --prompt-file would force
  // both modes onto the same prompt and defeat the point.
  for (const [mode, text] of Object.entries(opts.prompts ?? {})) {
    if (!text?.trim()) continue;
    const pf = path.join(work, `prompt.${mode}.txt`);
    fs.writeFileSync(pf, text, 'utf8');
    args.push('--prompt-file', `${mode}=${pf}`);
  }

  try {
    const stderr = await new Promise<string>((resolve, reject) => {
      execFile(exe, args,
        { timeout: 30 * 60_000, maxBuffer: 32 * 1024 * 1024, signal: opts.signal },
        (error, _stdout, errText) => {
          if (error) { reject(new Error(`ace-caption failed: ${String(errText || error).slice(0, 400)}`)); return; }
          resolve(errText || '');
        });
    });
    const timing = /encode[^\n]*|\b\d+\.\d+\s*s\b/.exec(stderr);
    if (timing) opts.log?.('info', `${LOCAL_CAPTIONER_NAMES[model]}: ${timing[0].trim()}`);

    const out: Partial<Record<MossMode, string>> = {};
    for (const mode of opts.modes) {
      // `-o out.txt --mode a,b` writes out.a.txt / out.b.txt — but with a SINGLE
      // mode ace-caption writes the bare `-o` path with no infix. Reading only
      // the suffixed name silently yields no caption for single-mode calls: the
      // run costs a full encode, exits 0, and returns nothing. Try both.
      const text = [`out.${mode}.txt`, 'out.txt']
        .map(n => path.join(work, n))
        .filter(p => fs.existsSync(p))
        .map(p => fs.readFileSync(p, 'utf8').trim())
        .find(t => t.length > 0);
      if (text) out[mode] = text;
    }
    if (Object.keys(out).length === 0) {
      // Never fail silently. A zero-output run means the CLI contract moved, and
      // the caller must not mistake it for "the model had nothing to say".
      opts.log?.('warn', `${LOCAL_CAPTIONER_NAMES[model]} produced no output for modes: ${opts.modes.join(',')}`);
    }
    return out;
  } finally {
    try { fs.rmSync(work, { recursive: true, force: true }); } catch { /* tmp */ }
  }
}

// ── Fact substitution (supersedes mm3-caption-hybrid.py) ─────────────────

/**
 * Genre labels, MOST SPECIFIC FIRST — the ordering is load-bearing.
 *
 * `pickGenre` scans this list against every source at once and returns the first
 * hit, so a broad label placed too early swallows the specific one. Barry's
 * first version searched MOSS's own Basic Attributes line first and returned on
 * the first match there, which let a generic "rock" beat "Emo" in MOSS's body
 * and "punk rock" in the Gemini caption — collapsing every track to `Rock`.
 * That is not cosmetic: plain "Rock" is precisely the failure mode Rob
 * identified by ear in the losing arm of the A/B ("more plain rock, not
 * particularly punk/emo"). Genre is the single biggest lever on MM3 adherence.
 *
 * The rock family below is Barry's validated list, order preserved verbatim.
 * The other families follow the same most-specific-first rule but have not been
 * ear-tested; add to them freely, but never insert an umbrella term above a
 * subgenre it contains.
 *
 * ── The second trap: specific-sounding words that are WEAK evidence ──────────
 *
 * Ordering by specificity is necessary but not sufficient, because this scans
 * MOSS's whole body — and that body is production prose, where several genre
 * words appear as ordinary adjectives. A real capture of Daft Punk's "One More
 * Time" contained "blends vocal into ambient space", and a bare /\bambient\b/
 * placed above /\bhouse\b/ (correct by specificity!) labelled the track
 * `Ambient`. The negative lookaheads below exist for exactly that: "ambient
 * space", "classical guitar" on a folk track, "blues scale" on a rock one. A new
 * single common word needs the same treatment — check it against a production
 * description before adding it bare.
 */
export const GENRE_HINTS: ReadonlyArray<readonly [RegExp, string]> = [
  // — lounge / swing (2026-10-24) — PREPENDED above everything: a lounge cover
  // of a rock song is described as "a lounge cover ... of a punk anthem", and
  // the arrangement, not the source song, is the genre. Before these existed a
  // whole dataset of lounge covers labelled as House, Pop and Hip-Hop.
  [/\bbig[-\s]?band\b/i, 'Big Band'],
  [/\bbossa[-\s]?nova\b/i, 'Bossa Nova'],
  [/\beasy[-\s]?listening\b/i, 'Easy Listening'],
  [/\bexotica\b/i, 'Exotica'],
  [/\blounge\b/i, 'Lounge'],
  [/\bswing\b(?!\s+(?:feel|rhythm|groove|eighths?|note)s?\b)/i, 'Swing'],
  // — rock family (validated ordering, do not reorder) —
  // The 2026-08-18 block below is PREPENDED, not reordered: each term contains
  // an umbrella lower in the family (post-rock ⊃ rock, midwest emo ⊃ emo …),
  // so before they were listed, MOSS's genre-first output collapsed to the
  // umbrella — "post-rock, ambient rock" scanned as `Rock`, Barry's bug in
  // new clothes. Every addition must stay above the umbrella it contains.
  [/midwest emo/i, 'Midwest Emo'],
  [/math rock/i, 'Math Rock'],
  [/post[-\s]?rock/i, 'Post-Rock'],
  [/post[-\s]?punk/i, 'Post-Punk'],
  [/garage rock/i, 'Garage Rock'],
  [/surf[-\s]?rock/i, 'Surf Rock'],
  [/shoegaze/i, 'Shoegaze'],
  [/pop[-\s]?punk/i, 'Pop-Punk'],
  [/post[-\s]?hardcore/i, 'Post-Hardcore'],
  [/punk rock/i, 'Punk Rock'],
  [/alternative rock/i, 'Alternative Rock'],
  [/indie rock/i, 'Indie Rock'],
  [/\bemo\b/i, 'Emo'],
  [/\bhardcore\b/i, 'Hardcore'],
  [/\bmetal\b/i, 'Metal'],
  [/\brock\b/i, 'Rock'],
  // — electronic —
  [/drum\s*(?:and|'?n'?|&)\s*bass|\bdnb\b/i, 'Drum and Bass'],
  [/\bbass house\b/i, 'Bass House'],
  [/\belectro house\b/i, 'Electro House'],
  [/\bprogressive house\b/i, 'Progressive House'],
  [/\bdeep house\b/i, 'Deep House'],
  [/\btech house\b/i, 'Tech House'],
  [/\bdubstep\b/i, 'Dubstep'],
  [/\bhardstyle\b/i, 'Hardstyle'],
  [/\beurodance\b/i, 'Eurodance'],
  [/\bsynth[-\s]?pop\b/i, 'Synth-Pop'],
  [/\bsynthwave\b/i, 'Synthwave'],
  [/\bfrench house\b/i, 'French House'],
  [/\btrance\b/i, 'Trance'],
  [/\btechno\b/i, 'Techno'],
  // Not "coffee-house atmosphere" or "in-house"; Flamingo writes U+2011 hyphens.
  [/(?<!coffee[\s\-‐‑]?|in[\-‐‑])\bhouse\b/i, 'House'],
  // "ambient <production noun>" is mix vocabulary, not a genre claim.
  [/\bambient\b(?!\s+(?:space|texture|pad|tail|wash|noise|bed|layer|reverb|atmosphere)s?\b)/i, 'Ambient'],
  // — hip-hop / r&b —
  [/\bboom bap\b/i, 'Boom Bap'],
  [/\btrap\b/i, 'Trap'],
  [/\bhip[-\s]?hop\b|\brap\b/i, 'Hip-Hop'],
  [/\br\s*&\s*b\b|\brnb\b/i, 'R&B'],
  // The hyphen is a word boundary, so \bsoul\b matches inside "neo-soul".
  [/neo[-\s]?soul/i, 'Neo-Soul'],
  [/\bsoul\b/i, 'Soul'],
  [/\bfunk\b/i, 'Funk'],
  // — acoustic / traditional —
  [/\bindie folk\b/i, 'Indie Folk'],
  [/singer[-\s]?songwriter/i, 'Singer-Songwriter'],
  [/\bbluegrass\b/i, 'Bluegrass'],
  [/\bcountry\b/i, 'Country'],
  [/\bfolk\b/i, 'Folk'],
  [/\bjazz\b/i, 'Jazz'],
  [/\breggae\b/i, 'Reggae'],
  // "blues scale"/"blues note" is theory vocabulary; "classical guitar" and
  // "orchestral pad" are instrument names. All three appear constantly in
  // descriptions of tracks that are not remotely those genres.
  [/\bblues\b(?!\s+(?:scale|note|progression|lick|inflection)s?\b)/i, 'Blues'],
  [/\bclassical\b(?!\s+(?:guitar|piano|influence|training)s?\b)|\borchestral\b(?!\s+(?:pad|swell|hit|stab|sample|texture)s?\b)/i, 'Classical'],
  // indie pop above dream pop: providers list the primary genre first, and
  // "indie pop, dream pop" should resolve to the primary.
  [/\bindie pop\b/i, 'Indie Pop'],
  [/\bdream pop\b/i, 'Dream Pop'],
  [/\bpop\b/i, 'Pop'],
];

/** First hint the text matches, or undefined. The hint ORDER is the ranking. */
export function scanGenreHints(text: string | undefined): string | undefined {
  const hay = text ?? '';
  for (const [re, label] of GENRE_HINTS) {
    if (re.test(hay)) return label;
  }
  return undefined;
}

/**
 * The most specific genre named by any source that actually OBSERVED the track.
 *
 * MOSS heard the audio; an existing audio-written caption (e.g. a Gemini
 * sidecar) also did. The dataset's `genre` field is the weak fallback because it
 * is usually a hand-typed umbrella like "Rock".
 */
export function pickGenre(
  mossText: string | undefined,
  observedCaption: string | undefined,
  fallback: string | undefined,
): string {
  return scanGenreHints(`${mossText ?? ''}\n${observedCaption ?? ''}`)
    ?? ((fallback ?? '').trim() || 'Alternative Rock');
}

/**
 * The genre that should land in the AS1.5 sidecar, weighing the caption
 * provider's claim against what the sidecar already holds — which is usually
 * the audio file's embedded rip tag, seeded by labeling Pass A before any
 * captioning runs (labelingQueue), and therefore "occupied" by the time the
 * provider answers.
 *
 * Cross-source specificity, not source priority: every claim goes into ONE
 * scan, so hint order arbitrates. A hand-typed "Midwest Emo" tag beats MOSS's
 * "indie rock"; MOSS's "indie pop" beats a rip tag's bare "Indie". When no
 * hint recognises anything, the provider's explicit genre line wins over the
 * tag — it listened to the track; the tag is a 2014 rip habit. Unlike
 * `pickGenre` this NEVER invents a default: `undefined` means "write
 * nothing", and the existing value survives untouched.
 */
export function arbitrateSidecarGenre(opts: {
  providerGenre?: string | null;
  providerCaption?: string | null;
  existing?: string | null;
}): string | undefined {
  const providerGenre = (opts.providerGenre ?? '').trim();
  const providerCaption = (opts.providerCaption ?? '').trim();
  if (!providerGenre && !providerCaption) return undefined;
  const existing = (opts.existing ?? '').trim();
  const hinted = scanGenreHints([providerGenre, providerCaption, existing].filter(Boolean).join('\n'));
  return hinted ?? (providerGenre || undefined);
}

/** `C# minor` / `F major` -> parts. Anything else -> nulls, and the caller omits the clause. */
export function parseKeyScale(keyscale: string | undefined | null): { key: string | null; scale: string | null } {
  const m = /^\s*([A-G][#b]?)\s+(major|minor)\s*$/i.exec(String(keyscale ?? '').trim());
  return m ? { key: m[1], scale: m[2].toLowerCase() } : { key: null, scale: null };
}

export interface LocalFacts {
  bpm?: number | null;
  keyscale?: string | null;
  signature?: string | null;
  /** Dataset genre field — weak fallback only. */
  genre?: string | null;
  /** A caption previously written FROM AUDIO, if one exists. */
  observedCaption?: string | null;
}

/** MM3's `Basic Attributes:` line, built from facts we know exactly. */
export function buildBasicAttributes(facts: LocalFacts, mossText?: string): string {
  const bpm = Math.round(Number(facts.bpm ?? 0));
  const { key, scale } = parseKeyScale(facts.keyscale);
  const genre = pickGenre(mossText, facts.observedCaption ?? undefined, facts.genre ?? undefined);

  // Upstream shape, measured over all 1000 official templates: 998 are
  // `bpm is N. key is K, and scale is S. GENRE.` and NONE carry a time
  // signature — the previous `, in 4/4` clause appeared in 0/1000 and was
  // therefore off-distribution for the conditioner. Signature stays in the
  // AS1.5 sidecar only.
  const parts: string[] = [];
  if (bpm > 0) parts.push(`bpm is ${bpm}.`);
  if (key) parts.push(`key is ${key}, and scale is ${scale}.`);
  parts.push(`${genre}.`);
  return `Basic Attributes: ${parts.join(' ')}`;
}

/**
 * Replace ONLY the `Basic Attributes:` line of an MM3 Structured Caption with
 * one built from local analysis. Everything MOSS heard — Global Emotional
 * Progression, Application Scenarios & Imagery, Sonics & Production Profile,
 * Vocal Details, Arrangement — is kept verbatim.
 *
 * If MOSS omitted the line entirely, it is inserted directly under
 * `Global Metadata` rather than dropping known-good facts on the floor.
 */
/** The three bare section headers, in template order. */
const MM3_SECTIONS: ReadonlySet<string> = new Set(['Global Metadata', 'Vocal Details', 'Arrangement']);

/** Every field label an official template line may start with. */
const MM3_LABELS: readonly string[] = [
  'Basic Attributes:',
  'Global Emotional Progression:',
  'Application Scenarios & Imagery:',
  'Sonics & Production Profile:',
  'Vocal Gender & Timbre:',
  'Vocal Style:',
  'Harmony/Backing Vocals:',
  'Vocal FX:',
  'Instrument Lifecycle Description (Primary/Secondary Layering):',
  'Primary:',
  'Secondary:',
  'Groove & Foundation Progression:',
  'Embellishments, Textures & Spatial FX:',
];

/** Near-miss label -> official label, applied anywhere in a line. */
const MM3_LABEL_ALIASES: ReadonlyArray<readonly [RegExp, string]> = [
  [/\bBasic (?:Elements|Metadata):/g, 'Basic Attributes:'],
  [/\b(?:Instrumentation & Production )?Instrument(?:al)? Scenarios & Imagery:/g, 'Application Scenarios & Imagery:'],
  [/\bInstrument(?:al|ation)? (?:Description|Overview|Arrangement) \(Primary\/Secondary Layering\):/g,
    'Instrument Lifecycle Description (Primary/Secondary Layering):'],
  [/\bEmbellishments & Textures & Spatial FX:/g, 'Embellishments, Textures & Spatial FX:'],
  [/\bSonics & Production Profile [A-Z][A-Za-z &]*:/g, 'Sonics & Production Profile:'],
];

const reEscape = (s: string): string => s.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&');
const MM3_SECTION_ALT = [...MM3_SECTIONS].join('|');
/** The space before a label (optionally led by its section header) that ends a sentence or a bare header. */
const MM3_INLINE_LABEL_SPLIT = new RegExp(
  `(?<=[.!?"”)]|^(?:${MM3_SECTION_ALT})) (?=(?:(?:${MM3_SECTION_ALT}) )?(?:${MM3_LABELS.map(reEscape).join('|')}))`);

/**
 * Reshape MOSS's raw MM3 output into the official template shape — measured
 * over all 1000 upstream files, not assumed: zero blank lines, zero
 * leading/trailing whitespace, label + content on ONE line per field, the
 * Instrument Lifecycle Description header bare with Primary:/Secondary: on
 * their own lines, and the file ENDING at the Embellishments field.
 *
 * MOSS deviates in each of these ways (measured on americanfootball +
 * chasestatus): blank lines between paragraphs (18/24 files), `Primary:`
 * welded inline onto the ILD header, bare labels with their prose on the NEXT
 * line, off-template trailing sections ("Chorus: <prose>"), and decode-loop
 * runs of bare labels ("Verse 1:\nVerse 1:\n…"). Everything here reshapes or
 * drops off-template trailing matter — nothing is ever invented.
 */
export function normalizeMm3Shape(mm3: string): string {
  let lines = mm3.split('\n').map(l => l.trim()).filter(l => l !== '');

  // Canonicalise near-miss labels, then give every label welded mid-line its
  // own line. Music Flamingo, with the genre hint after the template, wrote
  // "Basic Elements:", "Instrumental Description (…)" and an inline
  // ". Instrumental Scenarios & Imagery:"; left alone, the first gained a
  // second Basic Attributes line above it and the rest folded into the
  // previous field.
  lines = lines.map(l => MM3_LABEL_ALIASES.reduce((s, [re, to]) => s.replace(re, to), l));
  lines = lines.flatMap(l => l.split(MM3_INLINE_LABEL_SPLIT));

  // Split anything welded onto the ILD header — upstream keeps it bare.
  const ild = 'Instrument Lifecycle Description (Primary/Secondary Layering):';
  lines = lines.flatMap(l => {
    if (!l.startsWith(ild) || l === ild) return [l];
    const rest = l.slice(ild.length).trim();
    return [ild, rest];
  });
  // Give inline Primary/Secondary their own lines wherever they appear.
  lines = lines.flatMap(l => {
    const out: string[] = [];
    let cur = l;
    for (const marker of [' Secondary:', ' Primary:']) {
      const at = cur.indexOf(marker);
      if (at > 0) {
        out.unshift(cur.slice(at + 1));
        cur = cur.slice(0, at).trim();
      }
    }
    out.unshift(cur);
    return out.filter(x => x !== '');
  });

  // Merge continuation lines: upstream is one line per field, so a line that
  // is neither a section header nor label-led belongs to the previous line.
  // Inside the Instrument Lifecycle block, upstream also uses free-form
  // instrument-group labels (Tertiary:, Bass:, Synthesizers:, …) — those are
  // structural too, but only there; elsewhere "Chorus: …" is off-template.
  const GROUP_LABEL_RE = /^[A-Z][A-Za-z /()&'-]{0,34}: \S/;
  const merged: string[] = [];
  let inIld = false;
  for (const line of lines) {
    if (line.startsWith('Instrument Lifecycle Description')) inIld = true;
    else if (line.startsWith('Embellishments, Textures & Spatial FX:')) inIld = false;
    const isStructural = MM3_SECTIONS.has(line)
      || MM3_LABELS.some(lb => line.startsWith(lb))
      || (inIld && GROUP_LABEL_RE.test(line));
    if (!isStructural && merged.length > 0 && !MM3_SECTIONS.has(merged[merged.length - 1])) {
      merged[merged.length - 1] += ` ${line}`;
    } else {
      merged.push(line);
    }
  }

  // Upstream files end at the Embellishments field (1000/1000). Anything after
  // it — "Chorus: <prose>" sections, decode-loop label runs — is off-template.
  const ei = merged.findIndex(l => l.startsWith('Embellishments, Textures & Spatial FX:'));
  const bounded = ei >= 0 ? merged.slice(0, ei + 1) : merged;
  // Defensive tail strip for the no-Embellishments case.
  while (bounded.length > 0 && /^[^:]{1,40}:$/.test(bounded[bounded.length - 1])) {
    bounded.pop();
  }
  return bounded.join('\n');
}

export function applyFactSubstitution(mm3: string, facts: LocalFacts): string {
  const lines = normalizeMm3Shape(mm3).split('\n');
  const bi = lines.findIndex(l => l.startsWith('Basic Attributes:'));
  const built = buildBasicAttributes(facts, mm3);
  if (bi >= 0) {
    lines[bi] = built;
    // A drifted duplicate ("Basic Elements:", canonicalised) would contradict it.
    for (let i = lines.length - 1; i > bi; i--) if (lines[i].startsWith('Basic Attributes:')) lines.splice(i, 1);
  } else {
    const gi = lines.findIndex(l => l === 'Global Metadata');
    lines.splice(gi + 1, 0, built);
  }
  // MOSS welds tempo/key claims into the BODY prose too — ABR's Groove line
  // said "at 120 BPM" two lines under a Basic Attributes we had just set to
  // "bpm is 105", a self-contradiction inside one caption. The same rewrite
  // that keeps the AS1.5 caption honest keeps every MM3 line honest; the
  // rebuilt Basic Attributes line is already true and is left alone.
  return lines
    .map(l => (l.startsWith('Basic Attributes:') ? l : correctFactsInProse(l, facts)))
    .join('\n');
}

/**
 * Correct tempo/key claims that MOSS baked into CAPTION PROSE.
 *
 * `applyFactSubstitution` fixes the MM3 side by replacing a whole labelled line.
 * The AS1.5 caption has no such line — the numbers are welded into the sentence
 * ("A driving Electro House track in B minor at 120 BPM, this piece pulses…") —
 * so a line replacement cannot reach them, and a caption that asserts the wrong
 * tempo is worse than one that omits it: it is training data that disagrees with
 * the `bpm` field sitting next to it in the same sidecar.
 *
 * Measured on the Daft Punk capture: caption prose said "120 BPM", the MM3
 * section of the SAME encode said "approximately 128", truth is 123.
 *
 * Values are REWRITTEN rather than deleted. Deleting leaves "…track  , this
 * piece…" and needs fragile comma surgery; rewriting keeps MOSS's sentence
 * intact and makes it true. Only substitutes what local analysis actually knows,
 * and only where the value differs — so a caption that happens to be right, or a
 * dataset with no Essentia pass, is left completely alone.
 *
 * Chord spellings are untouched: the key pattern requires a space and a literal
 * "major"/"minor", so "Bm7", "Gmaj7" and "A6" never match.
 */
export function correctFactsInProse(caption: string, facts: LocalFacts): string {
  let out = caption;

  const bpm = Math.round(Number(facts.bpm ?? 0));
  if (bpm > 0) {
    out = out.replace(/\b(\d{2,3})(\s*)(BPM|bpm)\b/g, (m, n, sp, unit) =>
      Number(n) === bpm ? m : `${bpm}${sp}${unit}`);
  }

  const { key, scale } = parseKeyScale(facts.keyscale);
  if (key && scale) {
    const truth = `${key} ${scale}`;
    // Replace only the FIRST key statement. Later ones are usually modulations
    // ("shifts to D minor"), and rewriting those would produce "from F# minor to
    // F# minor" — nonsense that reads as a transcription error in the dataset.
    let done = false;
    out = out.replace(/\b([A-G][#b]?)\s+(major|minor)\b/g, (m) => {
      if (done) return m;
      done = true;
      return m.toLowerCase() === truth.toLowerCase() ? m : truth;
    });
  }

  return out;
}

// ── The Training Studio entry point ──────────────────────────────────────

// ── Batch prefetch ───────────────────────────────────────────────────────

/**
 * Results from a `--src-list` run, keyed by audio path, consumed once.
 *
 * The alternative was restructuring `labelingQueue` to caption in bulk, which
 * would have meant duplicating its per-sample merge policy, error marking,
 * progress and cancellation. Prefetching instead keeps every one of those
 * per-sample behaviours exactly as they are: the queue still walks samples one
 * at a time, but the expensive part is already done.
 *
 * Entries are PROMISES, not values: the batch resolves each track's entry the
 * moment `ace-caption` reports its files written, so a consumer awaiting track
 * 1 gets its caption while track 7 is still decoding. Before 5c75591 each
 * caption landed as its track completed; the first batch implementation
 * regressed that to all-at-the-end (the review grid sat empty for minutes on a
 * big dataset), and this restores it without giving up the single model load.
 */
const batchPending = new Map<string, Promise<Partial<Record<MossMode, string>> | undefined>>();

/** Drop anything left over. Call when a job ends, however it ends. */
export function clearMossBatch(): void {
  batchPending.clear();
  lyricsPending.clear();
}

/**
 * Caption every track in one `ace-caption` process, holding the model across
 * them. Resolves once the batch is UNDERWAY — consumers pick up per-track
 * results through `captionWithMoss` as each track completes. Failure is
 * non-fatal: a track whose entry resolves `undefined` just falls back to its
 * own run (queued behind the batch by `serialised`, so VRAM is never doubled),
 * which is slower but produces identical output.
 */
export async function prepareMossBatch(
  audioPaths: string[],
  opts: {
    model?: LocalCaptioner;
    wantMm3?: boolean; wantLyrics?: boolean; wantYue2?: boolean; maxSeconds?: number;
    /** audio path -> per-track hint (genreHint), appended to non-lyrics prompts. */
    hints?: Map<string, string>;
    signal?: AbortSignal; log?: (level: 'info' | 'warn', message: string) => void;
  } = {},
): Promise<void> {
  clearMossBatch();
  if (audioPaths.length < 2) return;   // one track cannot amortise anything

  const modes: MossMode[] = ['prose'];
  if (opts.wantMm3) modes.push('mm3');
  if (opts.wantYue2) modes.push('yue2');
  if (opts.wantLyrics) modes.push('lyrics');

  void runMossBatch(audioPaths, {
    model: opts.model, modes, hints: opts.hints,
    prompts: {
      prose: CAPTION_INSTRUCTIONS, mm3: MM3_INSTRUCTIONS,
      lyrics: LYRICS_INSTRUCTIONS, yue2: YUE2_LOCAL_INSTRUCTIONS,
    },
    maxSeconds: opts.maxSeconds, signal: opts.signal, log: opts.log,
  });
  // Lyrics that ride the caption batch are claimed separately (takeLocalLyrics),
  // so the label job can use them per track whether or not it captions.
  if (opts.wantLyrics) {
    for (const p of audioPaths) {
      const entry = batchPending.get(p);
      if (entry) lyricsPending.set(p, entry.then(r => r?.lyrics?.trim() || undefined));
    }
  }
}

// ── Local lyrics, per track ─────────────────────────────────────────────

/** audio path -> lyrics promise, from prepareMossBatch(wantLyrics) or
 *  prepareLocalLyrics. Consumed once by takeLocalLyrics. */
const lyricsPending = new Map<string, Promise<string | undefined>>();

/**
 * Start Music Flamingo lyric transcription for every path that is not already
 * covered (by a Flamingo caption batch that includes lyrics), in ONE process.
 * Returns at once; each track's lyrics resolve as that track finishes, so a label
 * job writes them per track instead of after the whole dataset.
 */
export function prepareLocalLyrics(
  audioPaths: string[],
  opts: { maxSeconds?: number; signal?: AbortSignal; log?: (level: 'info' | 'warn', message: string) => void } = {},
): void {
  const todo = audioPaths.filter(p => !lyricsPending.has(p));
  if (todo.length === 0) return;
  const pending = new Map<string, Promise<Partial<Record<MossMode, string>> | undefined>>();
  void runMossBatch(todo, {
    model: 'flamingo', modes: ['lyrics'], prompts: { lyrics: LYRICS_INSTRUCTIONS },
    maxSeconds: opts.maxSeconds, signal: opts.signal, log: opts.log,
  }, pending);
  for (const p of todo) {
    lyricsPending.set(p, pending.get(p)!.then(r => r?.lyrics?.trim() || undefined));
  }
}

/** This track's local lyrics: from the prepared batch when there is one, else a
 *  single-track run. `undefined` when the model returned nothing. A looped
 *  transcript is already redone inside ace-caption (--retry-loops). */
export async function takeLocalLyrics(
  audioPath: string,
  opts: { signal?: AbortSignal; log?: (level: 'info' | 'warn', message: string) => void } = {},
): Promise<string | undefined> {
  const pending = lyricsPending.get(audioPath);
  if (pending) lyricsPending.delete(audioPath);
  const raw = (pending ? await pending : undefined) ?? (await runMossCaption(audioPath, {
    model: 'flamingo', modes: ['lyrics'], prompts: { lyrics: LYRICS_INSTRUCTIONS },
    signal: opts.signal, log: opts.log,
  })).lyrics?.trim();
  return raw ? normaliseLyricTags(raw) : undefined;
}

/** A track that no longer needs local lyrics (Genius found them). The batch
 *  still produces them; this just stops the entry being kept. */
export function dropLocalLyrics(audioPath: string): void {
  lyricsPending.delete(audioPath);
}

/**
 * One process, N tracks, each track's `batchPending` entry resolved as its
 * files land. Registers every entry SYNCHRONOUSLY (before the first await) so
 * a consumer that runs immediately after `prepareMossBatch` returns finds a
 * promise, never a gap.
 *
 * Per-track completion is read from the CLI's stderr: it prints
 * `[MOSS] <mode>   -> <base> (N.Ns)` AFTER fclose()ing each output file
 * (ace-caption.cpp), so a track is done when every requested mode has
 * reported. Tracks the process skipped (unreadable audio) or that were still
 * unreported at exit resolve from whatever files exist, then `undefined`.
 *
 * Output bases are numbered (`t0`, `t1`, …) rather than derived from filenames:
 * dataset filenames collide across albums, and two tracks resolving to the same
 * base would silently overwrite one caption with another's.
 */
function runMossBatch(
  audioPaths: string[],
  opts: MossRunOptions,
  pending: Map<string, Promise<Partial<Record<MossMode, string>> | undefined>> = batchPending,
): Promise<void> {
  const settlers = new Map<string, (v: Partial<Record<MossMode, string>> | undefined) => void>();
  for (const audioPath of audioPaths) {
    pending.set(audioPath, new Promise(resolve => { settlers.set(audioPath, resolve); }));
  }
  const model = opts.model ?? 'moss';
  const name = LOCAL_CAPTIONER_NAMES[model];

  const done = serialised(async () => {
    const resolved = resolveMossPaths(model);
    if ('missing' in resolved) throw new Error(`${name} unavailable: ${resolved.missing}`);
    const { exe, modelsDir } = resolved.paths;

    const work = fs.mkdtempSync(path.join(os.tmpdir(), 'hs_mossb_'));
    const bases = audioPaths.map((_, i) => path.join(work, `t${i}`));
    try {
      fs.writeFileSync(
        path.join(work, 'list.txt'),
        audioPaths.map((p, i) => {
          const hint = (opts.hints?.get(p) ?? '').replace(/[\t\r\n]+/g, ' ').trim();
          return hint ? `${p}\t${bases[i]}\t${hint}` : `${p}\t${bases[i]}`;
        }).join('\n'),
        'utf8',
      );

      const args: string[] = [
        '--models', modelsDir,
        '--family', model,
        '--src-list', path.join(work, 'list.txt'),
        '--mode', opts.modes.join(','),
        '--temperature', String(opts.temperature ?? MOSS_TEMPERATURE),
        '--rep-penalty', String(opts.repPenalty ?? MOSS_REP_PENALTY),
        '--freq-penalty', String(opts.freqPenalty ?? MOSS_FREQ_PENALTY),
      ];
      if (opts.maxTokens) args.push('--max-tokens', String(opts.maxTokens));
      if (opts.maxSeconds) args.push('--max-seconds', String(opts.maxSeconds));
      args.push(...retryLoopArgs(model, opts.modes));
      const ffmpeg = getFFmpegPath();
      if (ffmpeg) args.push('--ffmpeg', ffmpeg);
      for (const [mode, text] of Object.entries(opts.prompts ?? {})) {
        if (!text?.trim()) continue;
        const pf = path.join(work, `prompt.${mode}.txt`);
        fs.writeFileSync(pf, text, 'utf8');
        args.push('--prompt-file', `${mode}=${pf}`);
      }

      // Reads whatever landed for track i and settles its entry.
      const settle = (i: number) => {
        const audioPath = audioPaths[i];
        const resolve = settlers.get(audioPath);
        if (!resolve) return;
        settlers.delete(audioPath);
        const modeText: Partial<Record<MossMode, string>> = {};
        for (const mode of opts.modes) {
          // Multi-mode writes <base>.<mode>.txt; single-mode writes -o as
          // given, which for our extensionless bases is the bare base path.
          const text = [`${bases[i]}.${mode}.txt`, `${bases[i]}.txt`, bases[i]]
            .filter(p => fs.existsSync(p))
            .map(p => fs.readFileSync(p, 'utf8').trim())
            .find(t => t.length > 0);
          if (text) modeText[mode] = text;
        }
        resolve(Object.keys(modeText).length > 0 ? modeText : undefined);
      };

      await new Promise<void>((resolve, reject) => {
        // A non-zero exit means SOME file failed, not that all did — the CLI
        // captions every readable track first. So resolve either way; per-file
        // settle() decides what actually landed.
        const child = execFile(exe, args,
          { timeout: 6 * 60 * 60_000, maxBuffer: 64 * 1024 * 1024, signal: opts.signal },
          (error) => {
            if (error && (error as NodeJS.ErrnoException).name === 'AbortError') { reject(error); return; }
            resolve();
          });

        // `[MOSS] <mode>   -> <base> (N.Ns)` (or `[Flamingo] ...`), one line
        // per written file. A track is complete when all its requested modes
        // have reported.
        const reported = new Map<number, number>();
        let buffer = '';
        child.stderr?.setEncoding('utf8');
        child.stderr?.on('data', (chunk: string) => {
          buffer += chunk;
          let nl: number;
          while ((nl = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, nl); buffer = buffer.slice(nl + 1);
            const m = /^\[(?:MOSS|Flamingo)\] \S+\s+-> (.+?) \(\d+(?:\.\d+)?s\)\s*$/.exec(line);
            if (!m) continue;
            const base = m[1].replace(/\.(?:prose|mm3|lyrics|yue2)\.txt$/, '').replace(/\.txt$/, '');
            const i = bases.indexOf(base);
            if (i < 0) continue;
            const n = (reported.get(i) ?? 0) + 1;
            reported.set(i, n);
            if (n >= opts.modes.length) settle(i);
          }
        });
      });

      audioPaths.forEach((_, i) => settle(i));   // stragglers: read what exists
      opts.log?.('info', `${name}: batch of ${audioPaths.length} tracks complete`);
    } finally {
      // Everything unsettled — spawn failure, abort, missing weights — falls
      // back to a per-track run rather than hanging its consumer.
      for (const resolve of settlers.values()) resolve(undefined);
      settlers.clear();
      try { fs.rmSync(work, { recursive: true, force: true }); } catch { /* tmp */ }
    }
  });
  return done.catch((err: any) => {
    if ((err as NodeJS.ErrnoException)?.name === 'AbortError') return;
    opts.log?.('warn', `${name} batch failed (${String(err?.message || err).slice(0, 160)}) — falling back to per-track runs`);
  });
}

export interface MossCaptionResult {
  /** Side-Step's 5-line block, unparsed — the caller runs parseStructuredResponse. */
  prose?: string;
  /** MM3 Structured Caption, already fact-substituted. */
  mm3?: string;
  lyrics?: string;
  /** The YuE2 planner caption, normalised (BPM from local facts). */
  yue2?: string;
}

/**
 * Caption one track locally, in every format the app consumes.
 *
 * `wantMm3` and `wantLyrics` are separate because each adds a decode (cheap)
 * but lyrics in particular adds a LOT of tokens on a long song. The AS1.5 block
 * is always produced — it is what the sidecar needs.
 */
export async function captionWithMoss(
  audioPath: string,
  facts: LocalFacts,
  opts: {
    model?: LocalCaptioner;
    wantMm3?: boolean;
    wantLyrics?: boolean;
    wantYue2?: boolean;
    /** genreHint() of the file's tag; only used when no batch result is waiting. */
    hint?: string;
    maxSeconds?: number;
    signal?: AbortSignal;
    log?: (level: 'info' | 'warn', message: string) => void;
  } = {},
): Promise<MossCaptionResult> {
  const modes: MossMode[] = ['prose'];
  if (opts.wantMm3) modes.push('mm3');
  if (opts.wantYue2) modes.push('yue2');
  if (opts.wantLyrics) modes.push('lyrics');

  // Prefetched by prepareMossBatch, if the caller ran one. The entry is a
  // promise that resolves the moment the batch finishes THIS track — awaiting
  // it here is what lets each caption land in the UI as its track completes
  // instead of when the whole batch exits. Deleted on read so a retry of the
  // same sample re-runs the model rather than replaying a stale result — and
  // so the map cannot grow across a long job.
  const pending = batchPending.get(audioPath);
  if (pending) batchPending.delete(audioPath);
  const prefetched = pending ? await pending : undefined;

  const raw = prefetched ?? await runMossCaption(audioPath, {
    model: opts.model,
    modes,
    hint: opts.hint,
    // The AS1.5 prompt is passed by file rather than restated in C++ so the two
    // can't drift; it is the same string the cloud providers get.
    prompts: {
      prose: CAPTION_INSTRUCTIONS, mm3: MM3_INSTRUCTIONS,
      lyrics: LYRICS_INSTRUCTIONS, yue2: YUE2_LOCAL_INSTRUCTIONS,
    },
    maxSeconds: opts.maxSeconds,
    signal: opts.signal,
    log: opts.log,
  });

  const out: MossCaptionResult = { prose: raw.prose, lyrics: raw.lyrics };
  if (raw.yue2?.trim()) {
    const bpm = Number(facts.bpm);
    const body = assembleYue2FromFields(raw.yue2);
    if (body) out.yue2 = normalizeYue2Caption(body, { bpm: bpm > 0 ? bpm : undefined });
  }
  if (raw.mm3) {
    // The freshest audio-observed evidence for the Basic Attributes genre is
    // the prose MOSS wrote seconds ago FROM THIS SAME ENCODE — its line-1
    // genre and caption both. facts.observedCaption alone is the sidecar's
    // PREVIOUS caption, so the pick was scanning stale text while the right
    // answer sat in raw.prose: on ABR's The Blinding Light the new prose said
    // "thrash and groove metal" (sidecar genre: Metal) while the MM3 line
    // scanned only its own body ("rock beat") and landed Rock (2026-08-19).
    const observed = [raw.prose, facts.observedCaption]
      .filter(v => (v ?? '').trim()).join('\n');
    out.mm3 = applyFactSubstitution(raw.mm3, { ...facts, observedCaption: observed });
  }
  return out;
}
