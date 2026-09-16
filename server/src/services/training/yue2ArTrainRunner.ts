// training/yue2ArTrainRunner.ts — runs the three jobs the AR half needs:
// `ace-train yue2-tokenize`, `ace-train yue2-align` and `ace-train
// yue2-ar-train`.
//
// The spawn machinery is NOT here. yue2TrainRunner.ts's runYue2AceTrain does
// the engine stop, the two-stream reader, the timeout killer and the engine
// restore for every YuE2 subcommand; this file supplies three relays and three
// budgets. That is the whole difference between the NAR half and the AR half at
// this layer — a third copy of the same 100 lines would be a third place for
// the "restore the engine in `finally`" rule to rot.
//
// THE REGEXES ARE NOT SHARED, and cannot be. The AR trainer's step line carries
// `cursor %s` and `S %lld` where the NAR's carries `t %.3f`, and the tag is
// `[yue2-ar-train]` rather than `[yue2-train]`, so a relay that accepted both
// would have to be loose enough to match the wrong field. Each regex below
// names the printf it mirrors in engine/src/train/, and every one fails SOFT:
// an unmatched line still reaches the console log and the stderr tail.
//
// ONE FIELD IS NOT A NUMBER. `cursor %s` prints "nan" when the cursor head is
// live but no song in the step bound one, and "off" when --cursor-weight is 0
// (yue2-ar-train-run.h:3103-3108). Matching it as a float would drop the whole
// step line — and with it the loss, the progress bar and the chart — on every
// step of a run with no alignment cache.
//
// WHAT TO WATCH: `minted_val`. The engine labels it in the log itself ("<- THE
// NUMBER TO WATCH: it should stay FLAT"), because it is the only series that
// answers "has YuE2's token grammar been damaged" rather than "have these songs
// been memorised". It is the one eval promoted to a `metric` event; the artist
// hold-out is logged as text, so the chart draws a single unambiguous line.
//
// All three kinds are GPU-lane and all three stop the engine. Nothing has been
// measured for the AR trainer's per-step cost — see AR_MS_PER_STEP_BUDGET.

import fs from 'fs';
import path from 'path';

import { YUE2_LICENSE_NOTICE } from '../backends/yue2/index.js';
import { getDataset } from './datasetsRepo.js';
import { refreshYue2PresetsForNewRun } from './lyricStudioExport.js';
import { missingYue2TrainModels, readYue2PreprocessSummary } from './yue2Train.js';
import {
  YUE2_AR_ADAPTER_STEM, YUE2_AR_DEFAULTS, type ResolvedYue2ArTrainOptions,
  buildYue2ArTrainArgs,
} from './yue2ArTrain.js';
import { writeYue2ArRunManifest } from './yue2ArRuns.js';
import {
  buildYue2TokenizeArgs, missingYue2TokenizeModels, readYue2CodecIdsStatus,
  type ResolvedYue2TokenizeOptions,
} from './yue2Tokenize.js';
import {
  buildYue2SheetArgs, missingYue2SheetModels, readYue2AbcStatus,
  type ResolvedYue2SheetOptions,
} from './yue2Sheet.js';
import {
  buildYue2AlignArgs, missingYue2AlignModels, readYue2CursorWordsStatus,
  type ResolvedYue2AlignOptions,
} from './yue2Align.js';
import { log, numOr, runYue2AceTrain, type RelayState } from './yue2TrainRunner.js';
import { yue2SeparateDataset } from './yue2Stems.js';
import { emitProgress, finishJob, isCancelled, pushEvent, type TrainingJob } from './labelingQueue.js';

// ── yue2-ar-train: stderr -> events ─────────────────────────────────────────

interface ArState extends RelayState {
  /** The first minted_val this run saw, and the newest. The pair is the whole
   *  read: the number's LEVEL means nothing on its own, its DRIFT is the
   *  damage. */
  mintedVal0?: number;
  mintedVal?: number;
  artistEval?: number;
  /** True once the engine has said the pack is absent, so the finish line can
   *  say the run had no token-grammar check rather than silently having none. */
  noMintedCheck?: boolean;
}

/** `[yue2-ar-train] step %5lld/%lld  loss %.5f (win %.5f, run %.5f)  cursor %s
 *   |g| %.4f  lr %.2e  S %lld  %.2fs/it  vram %zu/%zu MB (%s)  [%s %s]`
 *  — yue2-ar-train-run.h:3109. `%5lld` pads with spaces, hence the `\s+` before
 *  the step number, and `cursor` is matched as `\S+` because it is a string
 *  field ("nan" / "off" / a float). */
const RE_STEP = /^\[yue2-ar-train\]\s+step\s+(\d+)\/(\d+)\s+loss\s+(\S+)\s+\(win\s+(\S+?),\s+run\s+([^)]+)\)\s+cursor\s+(\S+)\s+\|g\|\s+(\S+)\s+lr\s+(\S+)\s+S\s+(\d+)\s+([\d.]+)s\/it\s+vram\s+(\d+)\/(\d+)\s+MB/;
/** `[yue2-ar-train] EVAL step %lld  minted_val %.4f (full-vocab %.4f) <- THE
 *   NUMBER TO WATCH: it should stay FLAT` — :3008. */
const RE_EVAL_MINTED = /^\[yue2-ar-train\]\s+EVAL step\s+(\d+)\s+minted_val\s+(\S+)\s+\(full-vocab\s+([^)]+)\)/;
/** `[yue2-ar-train] EVAL step %lld  minted_val ABSENT — no minted pack, so this
 *   run has NO token-grammar check` — :3015. */
const RE_EVAL_ABSENT = /^\[yue2-ar-train\]\s+EVAL step\s+(\d+)\s+minted_val ABSENT\b/;
/** `[yue2-ar-train] EVAL step %lld  artist %.4f (full-vocab %.4f)` — :3004. */
const RE_EVAL_ARTIST = /^\[yue2-ar-train\]\s+EVAL step\s+(\d+)\s+artist\s+(\S+)\s+\(full-vocab\s+([^)]+)\)/;
/** `[yue2-ar-train] checkpoint at step %lld -> %s (+ snapshot)` — :3144. Like
 *  the NAR trainer's, the path it prints is the RESUME STATE, not the snapshot,
 *  so the snapshot path is rebuilt from the run directory and the step. */
const RE_CKPT = /^\[yue2-ar-train\]\s+checkpoint at step\s+(\d+)\s+->/;
/** `[yue2-ar-train] exported %zu tensors -> %s` — :1575. Fires for snapshots
 *  too; the FINAL export is the one with no `_step<N>` suffix. */
const RE_EXPORT = /^\[yue2-ar-train\]\s+exported\s+(\d+)\s+tensors\s+->\s+(.+)$/;
/** `[yue2-ar-train] done: %lld steps, mean loss %.5f` — :3159. */
const RE_DONE = /^\[yue2-ar-train\]\s+done:\s+(\d+)\s+steps,\s+mean loss\s+([\d.eE+-]+)/;
/** `[%s] %zu LoRA tensors over %d AR layers, rank %lld, alpha %.1f, target %s
 *   (%.1fM trainable parameters%s)` — :1074, with the tag "yue2-ar-train". */
const RE_ADAPTERS = /^\[yue2-ar-train\]\s+(\d+)\s+LoRA tensors over\s+(\d+)\s+AR layers,\s+rank\s+(\d+),\s+alpha\s+(\S+?),\s+target\s+(\S+?)\s+\(([\d.]+)M/;
/** `[yue2-ar-train] artist: %zu song(s), mean %lld codec frames (%.0f s at
 *   25 Hz), longest sequence %lld tokens` — :2711. */
const RE_ARTIST_SET = /^\[yue2-ar-train\]\s+artist:\s+(\d+)\s+song\(s\),\s+mean\s+(\d+)\s+codec frames\s+\(([\d.]+)\s+s/;
/** `[yue2-ar-train] minted: %zu song(s), %lld held out as minted_val` — :2724. */
const RE_MINTED_SET = /^\[yue2-ar-train\]\s+minted:\s+(\d+)\s+song\(s\),\s+(\d+)\s+held out as minted_val/;
/** `[yue2-ar-train] cursor: weight %.3f, %lld of %zu artist songs bound,
 *   longest lyric span %lld tokens` — :2692. */
const RE_CURSOR = /^\[yue2-ar-train\]\s+cursor:\s+weight\s+(\S+?),\s+(\d+)\s+of\s+(\d+)\s+artist songs bound/;
/** `[yue2-ar-train] resumed from %s at step %lld/%lld (AdamW iter %d)` — :2842. */
const RE_RESUMED = /^\[yue2-ar-train\]\s+resumed from\s+.*?\s+at step\s+(\d+)\/(\d+)/;
/** `[yue2-ar-train] SKIPPING "%s": %s` — :2954. One song dropped, the run
 *  continues; the user has to know which, because a corpus that silently loses
 *  half its songs still trains and still exports. */
const RE_SKIPPING = /^\[yue2-ar-train\]\s+SKIPPING\s+(".+")$/;

/** A line the engine prints on its way out, used ONLY to replace "exited with
 *  code 2" with the sentence it actually printed. THE EXIT CODE IS STILL THE
 *  AUTHORITY — same rule, and the same reasoning, as yue2TrainRunner's fatalish:
 *  these are regexes over prose, and failing a run that exited 0 on a guess
 *  would be worse than losing the sentence.
 *
 *  Two shapes, because these three subcommands use two. The usage refusals are
 *  printed as `ace-train yue2-<cmd>: …` with no bracket (yue2-ar-train-run.h
 *  :2476-2552, ace-train.cpp:4652), everything later as `[yue2-<cmd>] …`. The
 *  bracketed alternatives are anchored on openings the engine actually uses, not
 *  on "contains the word failed", which would fire on the ordinary `… 1 failed),
 *  240 clips …` summary and on every per-file `3/12 FAIL <name>: …` line. */
function fatalish(line: string): string {
  const usage = /^ace-train yue2-(?:ar-train|tokenize|align):\s+(.+)$/.exec(line);
  if (usage) return usage[1];
  const m = /^\[yue2-(?:ar-train|tokenize|align)\]\s+(REFUSING.*|cannot .+|no yue2-.+|no source in .+|nothing (?:tokenized|aligned).+|manifest not found:.+|MERT load failed:.+|tokenizer head load failed:.+|MMS_FA load failed:.+|--\S+ .*(?:is required|must be).+)$/.exec(line);
  return m ? m[1] : '';
}

function relayArLine(job: TrainingJob, line: string, st: ArState, outDir: string): void {
  let m: RegExpExecArray | null;

  if ((m = RE_STEP.exec(line))) {
    const step = Number(m[1]);
    const total = Number(m[2]);
    const loss = numOr(m[3]);
    const runMean = numOr(m[5]);
    // undefined for "nan" and "off", which is what they mean: no cursor term
    // was scored this step.
    const cursor = numOr(m[6]);
    const gradNorm = numOr(m[7]);
    const lr = numOr(m[8]);
    const seqLen = numOr(m[9]);
    const stepMs = (numOr(m[10]) ?? 0) * 1000;
    const usedMb = numOr(m[11]);
    const totalMb = numOr(m[12]);
    st.lastStep = Math.max(st.lastStep, step);
    st.totalSteps = total || st.totalSteps;
    st.lastLoss = loss ?? st.lastLoss;
    pushEvent(job, {
      type: 'metric', metric: 'step', ts: Date.now(),
      step, totalSteps: total, loss, lr, gradNorm, stepMs,
      ma5: runMean,
      // `S` is the token length of the song this step drew — an AR row is a
      // whole song, so it moves every step and it is what the [H,S] buffers are
      // sized by. Carried on `maxLen`, the existing field for a sequence
      // length, rather than as a sixth YuE2-only name on the metric event.
      maxLen: seqLen,
      usedMb, totalMb,
    });
    st.onJsonl?.({
      type: 'step', ts: Date.now(), step, totalSteps: total, loss, lr, gradNorm, stepMs,
      cursor, seqLen,
    });
    job.done = step;
    job.total = total || job.total;
    job.phase = 'training';
    emitProgress(job);
    return;
  }

  if ((m = RE_EVAL_MINTED.exec(line))) {
    const step = Number(m[1]);
    const val = numOr(m[2]);
    const full = numOr(m[3]);
    if (val !== undefined) {
      if (st.mintedVal0 === undefined) st.mintedVal0 = val;
      st.mintedVal = val;
    }
    // The ONLY eval promoted to a metric event. The artist hold-out below is
    // logged as text: two series on one `eval` metric would interleave into a
    // single chart line that means neither.
    pushEvent(job, { type: 'metric', metric: 'eval', ts: Date.now(), step, loss: val });
    const drift = (val !== undefined && st.mintedVal0 !== undefined)
      ? val - st.mintedVal0 : undefined;
    log(job, 'info',
      `minted_val ${m[2]} at step ${step} (full-vocab ${m[3]})`
      + (drift !== undefined && step > 0 ? ` — ${drift >= 0 ? '+' : ''}${drift.toFixed(4)} since the first eval` : '')
      + '. This is the number to watch: it should stay FLAT.');
    st.onJsonl?.({ type: 'eval', ts: Date.now(), step, mintedVal: val, mintedValFullVocab: full });
    return;
  }

  if ((m = RE_EVAL_ABSENT.exec(line))) {
    // Said once. The banner the engine prints alongside it explains why at
    // length, and it is already in the console log.
    if (!st.noMintedCheck) {
      st.noMintedCheck = true;
      log(job, 'warn',
        'No minted pack, so this run has NO token-grammar check — nothing will say whether the '
        + 'adapter has damaged YuE2\'s own token statistics.');
    }
    return;
  }

  if ((m = RE_EVAL_ARTIST.exec(line))) {
    st.artistEval = numOr(m[2]);
    log(job, 'info', `Artist hold-out ${m[2]} at step ${m[1]} (full-vocab ${m[3]}).`);
    st.onJsonl?.({ type: 'eval-artist', ts: Date.now(), step: Number(m[1]), loss: st.artistEval });
    return;
  }

  if ((m = RE_CKPT.exec(line))) {
    const step = Number(m[1]);
    const snap = path.join(outDir, `${YUE2_AR_ADAPTER_STEM}_step${step}.safetensors`);
    st.lastCkpt = snap;
    pushEvent(job, {
      type: 'metric', metric: 'milestone', ts: Date.now(),
      step, loss: st.lastLoss, path: snap,
    });
    st.onJsonl?.({ type: 'milestone', ts: Date.now(), step, loss: st.lastLoss, path: snap });
    log(job, 'info', `Snapshot at step ${step}`
      + (st.lastLoss !== undefined ? ` (loss ${st.lastLoss.toFixed(4)})` : ''));
    return;
  }

  if ((m = RE_EXPORT.exec(line))) {
    const file = m[2].trim();
    const isFinal = path.basename(file) === `${YUE2_AR_ADAPTER_STEM}.safetensors`;
    if (isFinal) st.exported = file;
    log(job, 'info', `Exported ${m[1]} tensors → ${path.basename(file)}`);
    st.onJsonl?.({ type: 'export', ts: Date.now(), tensors: Number(m[1]), path: file, final: isFinal });
    return;
  }

  if ((m = RE_DONE.exec(line))) {
    st.doneSeen = true;
    st.lastStep = Math.max(st.lastStep, Number(m[1]));
    log(job, 'info', `Training finished: ${m[1]} steps, mean loss ${m[2]}`);
    st.onJsonl?.({ type: 'done', ts: Date.now(), steps: Number(m[1]), meanLoss: numOr(m[2]) });
    return;
  }

  if ((m = RE_ADAPTERS.exec(line))) {
    pushEvent(job, {
      type: 'metric', metric: 'data', ts: Date.now(),
      loraParams: Number(m[1]), layers: Number(m[2]),
    });
    log(job, 'info',
      `${m[1]} LoRA tensors over ${m[2]} AR layers, rank ${m[3]}, alpha ${m[4]}, target ${m[5]} `
      + `(${m[6]}M trainable parameters).`);
    st.onJsonl?.({
      type: 'adapters', ts: Date.now(), tensors: Number(m[1]), layers: Number(m[2]),
      rank: Number(m[3]), alpha: Number(m[4]), target: m[5],
    });
    return;
  }

  if ((m = RE_ARTIST_SET.exec(line))) {
    const songs = Number(m[1]);
    st.clips = songs;   // songs, not clips: an AR row is a whole song
    pushEvent(job, {
      type: 'metric', metric: 'data', ts: Date.now(),
      samples: songs, totalSteps: st.totalSteps || undefined,
    });
    log(job, 'info', `${songs} artist song(s), mean ${m[2]} codec frames (${m[3]} s each).`);
    st.onJsonl?.({ type: 'init', ts: Date.now(), samples: songs, songFrames: Number(m[2]) });
    return;
  }

  if ((m = RE_MINTED_SET.exec(line))) {
    log(job, 'info', `Regulariser pack: ${m[1]} song(s), ${m[2]} held out as minted_val.`);
    return;
  }

  if ((m = RE_CURSOR.exec(line))) {
    const bound = Number(m[2]);
    const of = Number(m[3]);
    // Not fatal — the engine already refused the zero case — but a run where
    // most songs are unbound trains the cursor term on a handful of songs and
    // says so nowhere else.
    log(job, bound < of ? 'warn' : 'info',
      `Lyric cursor at weight ${m[1]}: ${bound} of ${of} artist songs bound.`
      + (bound < of ? ' The unbound ones train with the cursor loss off — re-run yue2-align for them.' : ''));
    return;
  }

  if ((m = RE_RESUMED.exec(line))) {
    st.lastStep = Math.max(st.lastStep, Number(m[1]));
    st.totalSteps = Number(m[2]) || st.totalSteps;
    log(job, 'info', `Resumed at step ${m[1]}/${m[2]} with optimizer state intact.`);
    return;
  }

  if ((m = RE_SKIPPING.exec(line))) {
    log(job, 'warn', `Song skipped: ${m[1]}`);
    return;
  }

  // The recipe, the conditioning and the warnings are written for a human
  // already, so they go through verbatim. Anchored on the openings rather than
  // passed wholesale: the REGULARIZER ABSENT banner alone is eleven lines.
  if (/^\[yue2-ar-train\]\s+(lr |WARNING|NOTE|AdamW betas|caption dropout|persistent training buffers|cursor: OFF|\d+ song\(s\); lyrics from)/.test(line)) {
    log(job, line.includes('WARNING') ? 'warn' : 'info', line.replace(/^\[yue2-ar-train\]\s+/, ''));
    return;
  }
  const bad = fatalish(line);
  if (bad) st.fatalMessage = st.fatalMessage || bad;
}

// ── yue2-tokenize: stderr -> events ─────────────────────────────────────────

/** `[yue2-tokenize] %zu/%zu <encoded|cached|SKIP|FAIL> …` — the per-source lines
 *  at yue2-tokenize-run.h:485, :492, :507, :516, :543, :557, :563, :575, :591
 *  and :597. One regex for all four verbs: they share the `i/n` prefix, which is
 *  what the progress bar needs. */
const RE_TOK_FILE = /^\[yue2-tokenize\]\s+(\d+)\/(\d+)\s+(encoded|cached|SKIP|FAIL)\b\s*(.*)$/;
/** `[yue2-tokenize] %zu of %zu source(s) from %s | …` — :454. The FIRST count is
 *  the picked set (what --only/--limit left), which is what `i/n` counts. */
const RE_TOK_HEAD = /^\[yue2-tokenize\]\s+(\d+)\s+of\s+(\d+)\s+source\(s\)\s+from\s+/;
/** `[yue2-tokenize] done: %zu source(s) tokenized (%zu from cache, %zu failed),
 *   %lld of %zu clip(s) now carry codec_ids, …` — :850. */
const RE_TOK_DONE = /^\[yue2-tokenize\]\s+done:\s+(\d+)\s+source\(s\)\s+tokenized\s+\((\d+)\s+from cache,\s+(\d+)\s+failed\),\s+(\d+)\s+of\s+(\d+)\s+clip\(s\)/;
/** `[yue2-tokenize]   %lld source(s) checked, %lld clip code file(s), %lld
 *   mismatch(es)` — :707, the tail of the engine's own alignment re-check. */
const RE_TOK_ALIGN = /^\[yue2-tokenize\]\s+(\d+)\s+source\(s\)\s+checked,\s+(\d+)\s+clip code file\(s\),\s+(\d+)\s+mismatch/;

function relayTokenizeLine(job: TrainingJob, line: string, st: RelayState): void {
  let m: RegExpExecArray | null;

  if ((m = RE_TOK_HEAD.exec(line))) {
    job.total = Number(m[1]) || job.total;
    job.phase = 'tokenizing';
    emitProgress(job);
    log(job, 'info', line.replace(/^\[yue2-tokenize\]\s+/, ''));
    return;
  }

  if ((m = RE_TOK_FILE.exec(line))) {
    const verb = m[3];
    job.done = Number(m[1]);
    job.total = Number(m[2]) || job.total;
    job.phase = 'tokenizing';
    if (verb === 'FAIL') job.failed = (job.failed || 0) + 1;
    emitProgress(job);
    // SKIP and FAIL are the two the user has to act on: a source without codes
    // is a source the AR trainer cannot train. The successes are left to the
    // console log — one line per track is noise on a 15-track album.
    if (verb === 'SKIP' || verb === 'FAIL') {
      log(job, verb === 'FAIL' ? 'error' : 'warn', `${verb}: ${m[4].trim()}`);
    }
    return;
  }

  if ((m = RE_TOK_ALIGN.exec(line))) {
    const mismatches = Number(m[3]);
    log(job, mismatches ? 'error' : 'info',
      `Alignment check: ${m[1]} source(s), ${m[2]} clip code file(s), ${mismatches} mismatch(es).`);
    return;
  }

  if ((m = RE_TOK_DONE.exec(line))) {
    st.doneSeen = true;
    st.encoded = Number(m[1]);
    st.cached = Number(m[2]);
    st.failed = Number(m[3]);
    st.clips = Number(m[4]);
    log(job, 'info',
      `${st.encoded} source(s) tokenized (${st.cached} from cache, ${st.failed} failed) — `
      + `${st.clips} of ${m[5]} clip(s) now carry codec_ids.`);
    return;
  }

  // The first codes of the first source. Worth showing: it is the one line that
  // proves the raw range (codes are [0, 32768) and the +151853 codec_offset is
  // added inside the trainer, never here).
  if (/^\[yue2-tokenize\]\s+(first 16 codes|NOTE|WARNING)/.test(line)) {
    log(job, line.includes('WARNING') ? 'warn' : 'info', line.replace(/^\[yue2-tokenize\]\s+/, ''));
    return;
  }
  const bad = fatalish(line);
  if (bad) st.fatalMessage = st.fatalMessage || bad;
}

// ── yue2-align: stderr -> events ────────────────────────────────────────────

interface AlignState extends RelayState {
  aligned?: number;
  withCursor?: number;
  /** Sources skipped for want of a vocal stem, specifically. Separation is not
   *  this stage's job, so this is the count that tells the user to go and run
   *  it — as opposed to a source skipped for having no lyrics. */
  noStem?: number;
}

/** `[yue2-align] %zu/%zu <ok|SKIP|FAIL> %-46s …` — yue2-align-run.h:411, :418,
 *  :427, :438, :453, :463, :472 and :488. */
const RE_AL_FILE = /^\[yue2-align\]\s+(\d+)\/(\d+)\s+(ok|SKIP|FAIL)\b\s*(.*)$/;
/** `[yue2-align] %zu of %zu source(s) from %s` — :390. */
const RE_AL_HEAD = /^\[yue2-align\]\s+(\d+)\s+of\s+(\d+)\s+source\(s\)\s+from\s+/;
/** `[yue2-align] done: %zu aligned, %zu skipped, %zu failed, of %zu source(s);
 *   %lld now carry cursor_words` — :611. */
const RE_AL_DONE = /^\[yue2-align\]\s+done:\s+(\d+)\s+aligned,\s+(\d+)\s+skipped,\s+(\d+)\s+failed,\s+of\s+(\d+)\s+source\(s\);\s+(\d+)\s+now carry/;

function relayAlignLine(job: TrainingJob, line: string, st: AlignState): void {
  let m: RegExpExecArray | null;

  if ((m = RE_AL_HEAD.exec(line))) {
    job.total = Number(m[1]) || job.total;
    job.phase = 'aligning';
    emitProgress(job);
    log(job, 'info', line.replace(/^\[yue2-align\]\s+/, ''));
    return;
  }

  if ((m = RE_AL_FILE.exec(line))) {
    const verb = m[3];
    const detail = m[4].trim();
    job.done = Number(m[1]);
    job.total = Number(m[2]) || job.total;
    job.phase = 'aligning';
    if (verb === 'FAIL') job.failed = (job.failed || 0) + 1;
    if (verb === 'SKIP' && /no vocal stem at/.test(detail)) st.noStem = (st.noStem || 0) + 1;
    emitProgress(job);
    if (verb === 'SKIP' || verb === 'FAIL') {
      log(job, verb === 'FAIL' ? 'error' : 'warn', `${verb}: ${detail}`);
    }
    return;
  }

  if ((m = RE_AL_DONE.exec(line))) {
    st.doneSeen = true;
    st.aligned = Number(m[1]);
    st.skipped = Number(m[2]);
    st.failed = Number(m[3]);
    st.withCursor = Number(m[5]);
    log(job, 'info',
      `${st.aligned} aligned, ${st.skipped} skipped, ${st.failed} failed of ${m[4]} source(s) — `
      + `${st.withCursor} now carry cursor_words.`);
    return;
  }

  if (/^\[yue2-align\]\s+(model |--cpu|NOTE|WARNING)/.test(line)) {
    log(job, line.includes('WARNING') ? 'warn' : 'info', line.replace(/^\[yue2-align\]\s+/, ''));
    return;
  }
  const bad = fatalish(line);
  if (bad) st.fatalMessage = st.fatalMessage || bad;
}

// ── yue2-tokenize ───────────────────────────────────────────────────────────

/**
 * YuE2 cache stage 2a — vocal stems, so stage 3 has something to align against.
 *
 * This one does NOT spawn ace-train: separation is an engine HTTP call, the same
 * one Stem Studio makes, so there is no process to relay and the progress comes
 * from the engine's own per-track fraction. The stage exists at all because
 * yue2-align skips by name — a dataset with no stems aligns nothing and exits 0,
 * so without this the honest options were "refuse" or "lie".
 */
export async function runYue2StemsJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as { audioDir?: string; slug?: string; level?: number; force?: boolean } | undefined;
  if (!opts?.audioDir || !opts.slug) {
    finishJob(job, 'failed', 'yue2-stems job is missing its audio directory');
    return;
  }
  if (!fs.existsSync(opts.audioDir)) {
    finishJob(job, 'failed', `The dataset's audio folder is gone (${opts.audioDir})`);
    return;
  }

  job.phase = 'separating';
  try {
    const res = await yue2SeparateDataset({
      audioDir: opts.audioDir,
      slug: opts.slug,
      level: opts.level,
      force: opts.force,
      isCancelled: () => isCancelled(job),
      onProgress: (p) => {
        job.done = p.index - (p.fraction === null ? 1 : 0);
        job.total = p.total;
        job.currentSampleId = p.name;
        emitProgress(job);
      },
    });

    // A partial result is a real result here: the align stage reports its own
    // coverage per source, so eleven stems out of thirteen is a run worth
    // having and the two failures are named rather than folded into a count.
    for (const f of res.failed) {
      log(job, 'warn', `${f.name}: ${f.error}`);
    }
    log(job, 'info', `[yue2-stems] ${res.written} written, ${res.skipped} already present, `
      + `${res.failed.length} failed -> ${res.stemsDir}`);

    if (!res.written && !res.skipped) {
      finishJob(job, 'failed', 'No vocal stems could be separated — see the warnings above');
      return;
    }
    finishJob(job, isCancelled(job) ? 'cancelled' : 'done');
  } catch (err) {
    finishJob(job, 'failed', err instanceof Error ? err.message : String(err));
  }
}

export async function runYue2TokenizeJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2TokenizeOptions | undefined;
  if (!opts?.manifest) {
    finishJob(job, 'failed', 'yue2-tokenize job is missing its manifest path');
    return;
  }
  // Re-checked here as well as at the route: the route's answer can be stale by
  // the time the job reaches the head of the GPU lane.
  const missing = missingYue2TokenizeModels();
  if (missing.length) {
    finishJob(job, 'failed', `The YuE2 tokenizer is missing: ${missing.join(', ')}`);
    return;
  }
  if (!fs.existsSync(opts.manifest)) {
    finishJob(job, 'failed',
      `The latent cache manifest is gone (${opts.manifest}) — run the YuE2 preprocess stage again`);
    return;
  }

  const summary = readYue2PreprocessSummary(opts.manifest);
  const sources = Math.max(1, summary?.sources ?? getDataset(job.datasetId)?.sampleCount ?? 1);
  // Nothing measured for MERT + the 8-layer head on this corpus; 5 min a track
  // with a 30 min floor is a hang guard, not an estimate. The done line reports
  // the real rate, and it is in the console log for whoever tightens this.
  const timeoutMs = Math.max(30 * 60 * 1000, sources * 5 * 60 * 1000);

  const st: RelayState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: 0 };
  try {
    log(job, 'info',
      'Tokenizing every source in the latent cache to semantic codes. The manifest is rewritten in '
      + 'place, so this stage and yue2-align both add to what yue2-preprocess wrote.');

    await runYue2AceTrain(job, 'yue2-tokenize', buildYue2TokenizeArgs(opts), timeoutMs, () => {
      const s = readYue2CodecIdsStatus(opts.manifest);
      if (!s) return 'yue2-tokenize finished but the manifest could not be read back';
      return s.sourcesWithCodes > 0
        ? null
        : 'yue2-tokenize exited cleanly but no source carries codec_ids';
    }, (line, state) => relayTokenizeLine(job, line, state), st);

    if (!isCancelled(job)) {
      const s = readYue2CodecIdsStatus(opts.manifest);
      if (s) {
        log(job, 'info',
          `Codes cached for ${s.sourcesWithCodes} of ${s.sources} source(s) (${s.clipsWithCodes} of `
          + `${s.clips} clips) via ${s.tokenizer}. The AR trainer reads the SOURCE codes; a source `
          + 'without them is not trainable however many of its clips have slices.');
      }
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}

// ── yue2-sheet: stderr -> events ─────────────────────────────────────────────

/** `[yue2-sheet] %zu/%zu <ok|error|FAIL> %-46s …` — yue2-sheet-run.h's
 *  per-source lines (decode FAIL, resample FAIL, transcribe ok/error). One
 *  regex for all three verbs: they share the `i/n` prefix, which is what the
 *  progress bar needs. `error` (lower-case, unlike the other stages' `SKIP`/
 *  `FAIL`) is the soft-failure case — abc_error written, not a run problem. */
const RE_SHEET_FILE = /^\[yue2-sheet\]\s+(\d+)\/(\d+)\s+(ok|error|FAIL)\b\s*(.*)$/;
/** `[yue2-sheet] %zu of %zu source(s) match --only '%s' (%zu already have
 *   abc/abc_error) | %zu selected for this run%s` — the run's head line. */
const RE_SHEET_HEAD = /^\[yue2-sheet\]\s+(\d+)\s+of\s+(\d+)\s+source\(s\)\s+match\s+/;
/** `[yue2-sheet] done: %zu ok, %zu soft-failed (abc_error), %zu infra-failed
 *   (decode), of %zu selected (%zu already cached); manifest now shows
 *   abc_sources_ok=%lld abc_sources_failed=%lld`. */
const RE_SHEET_DONE = /^\[yue2-sheet\]\s+done:\s+(\d+)\s+ok,\s+(\d+)\s+soft-failed\s+\(abc_error\),\s+(\d+)\s+infra-failed\s+\(decode\),\s+of\s+(\d+)\s+selected\s+\((\d+)\s+already cached\)/;

interface SheetState extends RelayState {
  ok?: number;
  softFailed?: number;
  infraFailed?: number;
}

function relaySheetLine(job: TrainingJob, line: string, st: SheetState): void {
  let m: RegExpExecArray | null;

  if ((m = RE_SHEET_HEAD.exec(line))) {
    job.total = Number(m[2]) || job.total;
    job.phase = 'transcribing';
    emitProgress(job);
    log(job, 'info', line.replace(/^\[yue2-sheet\]\s+/, ''));
    return;
  }

  if ((m = RE_SHEET_FILE.exec(line))) {
    const verb = m[3];
    job.done = Number(m[1]);
    job.total = Number(m[2]) || job.total;
    job.phase = 'transcribing';
    if (verb === 'FAIL') job.failed = (job.failed || 0) + 1;
    emitProgress(job);
    // FAIL (could not even decode) is the one the user has to act on; `error`
    // (abc_error — the reference's own soft failure, ~4% of real tracks per
    // doc 23's survey) is expected and goes to the console log only, same
    // reasoning as yue2-tokenize's SKIP/FAIL split.
    if (verb === 'FAIL') {
      log(job, 'error', `FAIL: ${m[4].trim()}`);
    }
    return;
  }

  if ((m = RE_SHEET_DONE.exec(line))) {
    st.doneSeen = true;
    st.ok = Number(m[1]);
    st.softFailed = Number(m[2]);
    st.infraFailed = Number(m[3]);
    log(job, 'info',
      `${st.ok} transcribed, ${st.softFailed} soft-failed (abc_error), ${st.infraFailed} could not be `
      + `decoded, of ${m[4]} selected (${m[5]} already cached).`);
    return;
  }

  if (/^\[yue2-sheet\]\s+(model |NOTE|WARNING|--model not given)/.test(line)) {
    log(job, line.includes('WARNING') ? 'warn' : 'info', line.replace(/^\[yue2-sheet\]\s+/, ''));
    return;
  }
  const bad = fatalishSheet(line);
  if (bad) st.fatalMessage = st.fatalMessage || bad;
}

/** Same reasoning as fatalish() above, but for yue2-sheet's own openings
 *  (ace-train.cpp:4786-4813, yue2-sheet-run.h) — NOT folded into the shared
 *  helper, which is anchored on `yue2-(?:ar-train|tokenize|align)` and would
 *  otherwise have to loosen in a way that risks matching the wrong tool's
 *  prose. THE EXIT CODE IS STILL THE AUTHORITY; this only replaces "exited
 *  with code N" with the sentence the engine actually printed. */
function fatalishSheet(line: string): string {
  const usage = /^ace-train yue2-sheet:\s+(.+)$/.exec(line);
  if (usage) return usage[1];
  const m = /^\[yue2-sheet\]\s+(cannot .+|no sheetsage2-.+|sources\[\d+\] is missing.+|FATAL .+|manifest not found:.+|SheetSage2 load failed.+)$/.exec(line);
  return m ? m[1] : '';
}

export async function runYue2SheetJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2SheetOptions | undefined;
  if (!opts?.manifest) {
    finishJob(job, 'failed', 'yue2-sheet job is missing its manifest path');
    return;
  }
  // Re-checked here as well as at the route: the route's answer can be stale by
  // the time the job reaches the head of the GPU lane.
  const missing = missingYue2SheetModels();
  if (missing.length) {
    finishJob(job, 'failed', `The SheetSage2 transcriber is missing: ${missing.join(', ')}`);
    return;
  }
  if (!fs.existsSync(opts.manifest)) {
    finishJob(job, 'failed',
      `The latent cache manifest is gone (${opts.manifest}) — run the YuE2 preprocess stage again`);
    return;
  }

  const summary = readYue2PreprocessSummary(opts.manifest);
  const sources = Math.max(1, summary?.sources ?? getDataset(job.datasetId)?.sampleCount ?? 1);
  // Doc 23's G6/phase-3c tables: 3-13 minutes per track on the CPU backend
  // (the exact-load, offline-cache path this stage runs), a few seconds on
  // CUDA. 10 min a track with a 30 min floor covers the slow end with margin.
  const timeoutMs = Math.max(30 * 60 * 1000, sources * 10 * 60 * 1000);

  const st: SheetState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: 0 };
  try {
    log(job, 'info',
      'Transcribing every source in the latent cache to a SheetSage2 lead sheet. Roughly 4% of real '
      + 'tracks decode fine but fail to render (abc_error) — that is expected, not a job failure; those '
      + 'sources train cot=off on every draw. The manifest is rewritten in place after every source.');

    await runYue2AceTrain(job, 'yue2-sheet', buildYue2SheetArgs(opts), timeoutMs, () => {
      const s = readYue2AbcStatus(opts.manifest);
      if (!s) return 'yue2-sheet finished but the manifest could not be read back';
      return (s.sourcesWithAbc + s.sourcesWithError) > 0
        ? null
        : 'yue2-sheet exited cleanly but no source carries abc or abc_error';
    }, (line, state) => relaySheetLine(job, line, state), st);

    if (!isCancelled(job)) {
      const s = readYue2AbcStatus(opts.manifest);
      if (s) {
        log(job, 'info',
          `Lead sheets cached for ${s.sourcesWithAbc} of ${s.sources} source(s) (${s.sourcesWithError} `
          + `soft-failed as abc_error). --abc-dropout reads this to draw cot=full for a source that has `
          + 'one; a source with neither trains cot=off, same as before this stage existed.');
      }
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}

// ── yue2-align ──────────────────────────────────────────────────────────────

export async function runYue2AlignJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2AlignOptions | undefined;
  if (!opts?.manifest || !opts.stemsDir) {
    finishJob(job, 'failed', 'yue2-align job is missing its manifest or stems folder');
    return;
  }
  const missing = missingYue2AlignModels();
  if (missing.length) {
    finishJob(job, 'failed', `The forced aligner is missing: ${missing.join(', ')}`);
    return;
  }
  if (!fs.existsSync(opts.manifest)) {
    finishJob(job, 'failed',
      `The latent cache manifest is gone (${opts.manifest}) — run the YuE2 preprocess stage again`);
    return;
  }

  const summary = readYue2PreprocessSummary(opts.manifest);
  const sources = Math.max(1, summary?.sources ?? getDataset(job.datasetId)?.sampleCount ?? 1);
  // ~100 s per four minutes of audio on 16 CPU threads, and several times
  // faster on the GPU default (yue2Align.ts). 5 min a track with a 30 min floor
  // covers the CPU path on long tracks.
  const timeoutMs = Math.max(30 * 60 * 1000, sources * 5 * 60 * 1000);

  const st: AlignState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: 0 };
  try {
    // The one thing the engine cannot say, because separation is not its job:
    // it looks for `<stems>/<source stem>/vocals.wav` and SKIPS BY NAME when
    // there is none. A whole run of skips is the expected result of never
    // having separated the corpus, and it looks identical to a broken aligner.
    log(job, 'info',
      `Aligning each source's lyrics against its vocal stem under ${opts.stemsDir}. Sources with no `
      + '`<name>/vocals.wav` there are skipped by name — separate them first if the SKIP count is high.');

    await runYue2AceTrain(job, 'yue2-align', buildYue2AlignArgs(opts), timeoutMs, () => {
      const s = readYue2CursorWordsStatus(opts.manifest);
      if (!s) return 'yue2-align finished but the manifest could not be read back';
      return s.sourcesWithCursor > 0
        ? null
        : 'yue2-align exited cleanly but no source carries cursor_words';
    }, (line, state) => relayAlignLine(job, line, state), st);

    if (!isCancelled(job)) {
      const s = readYue2CursorWordsStatus(opts.manifest);
      if (s) {
        log(job, 'info',
          `Cursor spans cached for ${s.sourcesWithCursor} of ${s.sources} source(s) via ${s.model}. `
          + 'The rest train with the cursor loss off, which is legal and silent.');
      }
      if (st.noStem) {
        log(job, 'warn',
          `${st.noStem} source(s) had no vocal stem. Separate them into `
          + `${opts.stemsDir}/<name>/vocals.wav and re-run this stage with --only, or train with a `
          + 'lower --cursor-weight.');
      }
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}

// ── yue2-ar-train ───────────────────────────────────────────────────────────

/** Per-step wall budget for the timeout killer.
 *
 *  NOT AN ESTIMATE, and deliberately not dressed as one: nothing has been
 *  measured for this trainer, for the reason yue2ArTrain.ts gives about
 *  YUE2_VRAM_MODEL — the NAR half's 0.065 s/step belongs to fixed 250-frame
 *  clips, and an AR step is one whole song of up to 12,288 tokens through a
 *  chunked CE head. 30 s is a ceiling chosen to be far above anything plausible
 *  on a card that is thermally throttled or shared, so the killer only ever
 *  fires on a genuine hang. The 2 h floor covers the model load plus a short
 *  run; the eval and export passes ride inside the same budget. */
const AR_MS_PER_STEP_BUDGET = 30_000;

export async function runYue2ArTrainJob(job: TrainingJob): Promise<void> {
  const opts = job.opts as ResolvedYue2ArTrainOptions | undefined;
  if (!opts?.manifest || !opts.outDir) {
    finishJob(job, 'failed', 'yue2-ar-train job is missing its manifest or output path');
    return;
  }
  const missing = missingYue2TrainModels('train', { lmType: opts.lmType });
  if (missing.length) {
    finishJob(job, 'failed', `YuE2 training models are missing: ${missing.join(', ')}`);
    return;
  }
  if (!fs.existsSync(opts.manifest)) {
    finishJob(job, 'failed',
      `The latent cache manifest is gone (${opts.manifest}) — run the YuE2 preprocess stage again`);
    return;
  }
  // THE STAGE-2 GATE. A manifest with latents but no codes is the ordinary
  // state of a cache built for the NAR half, it looks complete from the outside,
  // and the AR trainer would spend the engine stop and the model load before
  // saying so. Checked here rather than trusted from the route for the reason
  // every other check in this file is: the route's answer can be stale.
  const codes = readYue2CodecIdsStatus(opts.manifest);
  if (!codes || codes.sourcesWithCodes === 0) {
    finishJob(job, 'failed',
      'This latent cache carries no codec_ids, so the AR trainer has nothing to predict. '
      + 'Run the YuE2 tokenize stage against the same manifest first.');
    return;
  }
  if (opts.cursorWeight > 0) {
    const cursor = readYue2CursorWordsStatus(opts.manifest);
    if (!cursor || cursor.sourcesWithCursor === 0) {
      // The engine refuses this pair itself (exit 2 before any GPU work); this
      // only names the stage that fixes it.
      finishJob(job, 'failed',
        `--cursor-weight ${opts.cursorWeight} needs cursor_words and this manifest carries none. `
        + 'Run the YuE2 align stage, or set the cursor weight to 0 to train without the term.');
      return;
    }
  }

  // Record the recipe BEFORE the first step, not after the last: a run that
  // dies at step 3 is exactly the one someone will want to continue, and a
  // manifest written on success would not be there.
  const ds = getDataset(job.datasetId);
  writeYue2ArRunManifest(opts, {
    datasetId: job.datasetId,
    datasetSlug: ds?.slug || opts.datasetSlug || '',
    datasetName: opts.datasetName || ds?.name || ds?.slug || '',
    clips: readYue2PreprocessSummary(opts.manifest) ?? undefined,
  });

  const st: ArState = { fatalMessage: '', doneSeen: false, lastStep: 0, totalSteps: opts.steps };
  job.total = opts.steps;
  job.done = 0;

  try {
    // The derived-weight half of the licence, as the NAR runner says it: a
    // trained adapter is a derivative of CC BY-NC weights and carries the
    // same terms, so the run that produces one says so.
    log(job, 'warn', YUE2_LICENSE_NOTICE
      + ' An adapter trained from them is a derivative work and carries the same terms.');
    if (!opts.trigger) {
      log(job, 'warn',
        'No trigger word. The adapter will have nothing to address it by at generation time, which is '
        + 'the whole mechanism for reaching the trained style.');
    }
    if (!opts.minted) {
      // Not a refusal here — buildYue2ArTrainArgs deliberately does not infer
      // --allow-no-minted, so the engine's own guard is what decides. This is
      // the sentence that says where the pack comes from.
      log(job, 'warn',
        'No regulariser pack. It is a Model Manager download (yue2-minted-manifest + '
        + 'yue2-minted-codes, both into <models>/yue2), and without it the run has no minted_val '
        + 'series and trains 100% on encoder-predicted codes, which is out of distribution in the '
        + 'direction that produces LOOPING.');
    }
    log(job, 'info',
      `${opts.steps} steps at rank ${opts.rank}/alpha ${opts.alpha}, target ${opts.target}, `
      + `style template "${opts.styleTemplate}", caption dropout ${opts.captionDropout}, `
      + `cursor weight ${opts.cursorWeight}. Whole songs to --max-len ${opts.maxLen}.`);
    log(job, 'info',
      `Codes present for ${codes.sourcesWithCodes} of ${codes.sources} source(s) via ${codes.tokenizer}.`);

    const timeoutMs = Math.max(2 * 60 * 60 * 1000, opts.steps * AR_MS_PER_STEP_BUDGET);
    const args = buildYue2ArTrainArgs(opts);

    await runYue2AceTrain(job, 'yue2-ar-train', args, timeoutMs, () => {
      const final = path.join(opts.outDir, `${YUE2_AR_ADAPTER_STEM}.safetensors`);
      if (fs.existsSync(final)) return null;
      // Reaching here means exit 0, so the final export should be on disk; a
      // run that only got as far as snapshots is worth distinguishing, because
      // those snapshots are still auditionable.
      const anySnap = fs.existsSync(opts.outDir)
        && fs.readdirSync(opts.outDir).some(f => f.startsWith(`${YUE2_AR_ADAPTER_STEM}_step`));
      return anySnap
        ? 'yue2-ar-train exited cleanly but never wrote its final adapter (snapshots are on disk)'
        : 'yue2-ar-train finished but wrote no adapter';
    }, (line, state) => relayArLine(job, line, state, opts.outDir), st);

    if (!isCancelled(job)) {
      if (st.mintedVal0 !== undefined && st.mintedVal !== undefined) {
        const drift = st.mintedVal - st.mintedVal0;
        log(job, Math.abs(drift) > 0.05 ? 'warn' : 'info',
          `minted_val moved ${drift >= 0 ? '+' : ''}${drift.toFixed(4)} over the run `
          + `(${st.mintedVal0.toFixed(4)} → ${st.mintedVal.toFixed(4)}). It should stay FLAT: a rise `
          + 'means the adapter has damaged YuE2\'s own token grammar, whatever the artist loss did.');
      }
      // Which rung to reach for, and why the engine's own closing line says
      // something else: it suggests 600-1600 from upstream's 1600-step recipe,
      // and the settled HOT-Step recipe is a 400-step run whose ear-picked rung
      // was YUE2_AR_DEFAULTS.ckptPickStep.
      log(job, 'info',
        `Adapter written to ${path.join(opts.outDir, `${YUE2_AR_ADAPTER_STEM}.safetensors`)}, with a `
        + `snapshot every ${opts.saveEvery} steps from ${opts.ckptFrom}. The ladder is what the ear `
        + `picks from — step ${YUE2_AR_DEFAULTS.ckptPickStep} is the settled starting point, not a `
        + 'verdict on this dataset.');
      log(job, 'info',
        'The resume state was removed on the clean finish, so this run cannot be extended — start a new '
        + 'one with a higher step count instead. Its snapshots stay where they are.');

      // THE PRESET GETS THE PICK RUNG, NOT THE FINAL ADAPTER — the one place
      // this differs from MM3, and deliberately. The AR's likeness curve is
      // compressed and it is possible to train past the top of it: on Green Day
      // the ear chose 250-300 off a 400-step ladder, and the 400-step export
      // rendered six minutes without ever reaching an ending. Pointing an album
      // at the last file written would hand the user the one rung the campaign
      // knows is wrong. The final export is the fallback only when the rung is
      // missing (a run shorter than ckptPickStep, or saveEvery that skipped it).
      // ckptPickStep is a DEFAULT, not a per-run option — the form uses it to
      // preselect and the run itself never carries it.
      const want = YUE2_AR_DEFAULTS.ckptPickStep;
      const final = path.join(opts.outDir, `${YUE2_AR_ADAPTER_STEM}.safetensors`);
      const rungs = fs.readdirSync(opts.outDir)
        .map(f => /^.*_step(\d+)\.safetensors$/.exec(f))
        .filter((m): m is RegExpExecArray => !!m)
        .map(m => ({ step: Number(m[1]), file: path.join(opts.outDir, m[0]) }))
        .sort((a, b) => a.step - b.step);
      // Nearest rung AT OR BELOW the pick, not the final: a run whose ladder
      // skipped the exact step should fall back DOWN the curve, since the far
      // end is the end we know is wrong.
      const below = rungs.filter(r => r.step <= want).pop();
      const chosen = below?.file || (fs.existsSync(final) ? final : (rungs[0]?.file ?? ''));
      if (chosen) {
        const dsRow = getDataset(job.datasetId);
        const touched = refreshYue2PresetsForNewRun(
          { slug: dsRow?.slug || opts.datasetSlug || '', lyricsSetId: dsRow?.lyricsSetId }, 'ar', chosen);
        if (touched) {
          log(job, 'info',
            `${touched} Lyric Studio album preset(s) now load this run's AR adapter `
            + `(${path.basename(chosen)}${below ? '' : ' — no rung at or below the pick was on disk'}). `
            + 'Change it there if the ear picks a different rung.');
        }
      }
      finishJob(job, 'done');
    }
  } catch (err: any) {
    if (!isCancelled(job)) finishJob(job, 'failed', err?.message || String(err));
  }
}
