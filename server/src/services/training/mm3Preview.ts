// training/mm3Preview.ts — mid-run audio previews for MM3 LM training.
//
// WHY IT LOOKS LIKE THIS
//
// A training run is 50+ minutes and its only live signal is a loss curve, and a
// loss curve cannot answer the question that actually decides whether to let
// the run finish: "is this becoming the artist, or is it just eating the base
// model?" Both look like a falling loss. One of them sounds like the artist
// over a backing track that has quietly gone simple and cheap.
//
// So we render. The trainer cannot do it while resident — it holds 22.6 GB of a
// 32.6 GB card at the q8_0 recipe and a warm MM3 stack is another 11.8 GB — so
// the runner pauses it (mm3-lm-resume.h), renders on the freed card, and
// resumes. This module owns only the render half.
//
// WHAT GETS RENDERED, AND WHY TWO OF THEM
//
//   artist   the held-out song's own caption, trigger prepended, adapter on
//   control  a neutral off-genre caption, adapter still on
//
// The artist take answers "is identity arriving". The control take answers "is
// the base still intact", which is the failure the loss curve is blindest to
// and the one that matches what we hear: a LoRA at rank 256 over ~13 songs has
// ~700 M trainable parameters and can simply overwrite the planner's general
// competence. Rendering only the artist caption would let that happen
// invisibly, because on the artist caption a damaged planner and a working one
// both sound broadly like the artist.
//
// A `baseline` pass renders BOTH captions with no adapter before step 1. That
// is the reference the ear needs — "worse than base" is only a judgement you
// can make against base, and by the time the run finishes nobody remembers what
// base sounded like on this exact caption and seed.
//
// Everything is rendered at ONE FIXED SEED. Trap 10 in the mm3-backend skill
// says single-seed spectral judgments are meaningless, and it is right about
// judging a model. This is not that: it is an A/B of the SAME prompt and the
// same seed across steps, where the only variable is the adapter. Holding the
// seed is what makes the comparison legible; it is not a claim about the
// checkpoint's average behaviour.

import fs from 'fs';
import path from 'path';

import { aceClient } from '../aceClient.js';
import { mm3JobDetail, mm3SelectModel, mm3Synth } from '../backends/minimax/client.js';
import { MM3_LM_ADAPTER_DEFAULT_SCALES } from '../backends/minimax/lmAdapter.js';
import type { Mm3PreviewOptions, TrainingPreview } from './types.js';

/** A neutral caption, deliberately nothing like any artist dataset we train on.
 *  Its job is to be a canary: if the adapter has damaged the planner's general
 *  competence, THIS is where it shows, because nothing in the adapter's
 *  training data pulls toward it.
 *
 *  Format follows the Structured Caption contract (mm3-captioning skill) —
 *  Basic Attributes on the same line as the Global Metadata header, no blank
 *  lines, no signature. A malformed control caption would produce a bad render
 *  that reads as adapter damage. */
export const MM3_CONTROL_CAPTION =
  'Global Metadata\nBasic Attributes: instrumental, 96 BPM, C major, 4/4\n'
  + 'Genre: downtempo electronic\nMood: calm, spacious\n'
  + 'Arrangement\nIntro: soft synth pad, light percussion entering\n'
  + 'Main: steady kick, warm bass, sparse piano motif, wide reverb';

export const MM3_PREVIEW_DEFAULTS = {
  everySteps: 0,
  everyMinutes: 0,
  seconds: 40,
  seed: 424242,
  /** Both OFF by default (Rob, 2026-08-25): at a preview per checkpoint the
   *  baseline and control takes double the pause cost for renders that rarely
   *  get listened to. The form re-enables either with a checkbox. */
  baseline: false,
  control: false,
  /** 1.0, matching the generation default: a preview should predict what the
   *  user will hear, and with the acoustic loss holding timbre there is no
   *  longer a reason to derate MLP anywhere. */
  scaleMlp: 1.0,
  scaleAttn: 1.0,
} as const;

/** Wall-clock bound on one preview render. A 24 s sample measures ~16 s of GPU
 *  plus a 4.8 s warm; 6 minutes is a "the engine is wedged" tripwire, not a
 *  budget. A preview that hangs must never hold a training run hostage. */
const PREVIEW_DEADLINE_MS = 6 * 60_000;
const POLL_MS = 1_000;

export interface Mm3PreviewSpec {
  kind: 'artist' | 'control';
  caption: string;
  lyrics: string;
}

export interface Mm3PreviewPlan {
  specs: Mm3PreviewSpec[];
  seconds: number;
  seed: number;
  everySteps: number;
  everyMinutes: number;
  baseline: boolean;
  /** Adapter scales every preview in this run renders at. Fixed for the run so
   *  step-to-step comparison stays legible. */
  scaleMlp: number;
  scaleAttn: number;
  /** Where the WAVs go: <run dir>/previews. */
  dir: string;
}

// ── the artist caption ──────────────────────────────────────────────────────

interface ManifestRow { id: string; filename: string; lyrics: string }

/** The trainer's own view of the dataset, in the trainer's own order.
 *
 *  Mirrors mm3_lm_load_samples: a row is usable only when BOTH its
 *  `<stem>.mm3.txt` caption and its `<id>.codes` exist, and the skipped ones do
 *  not shift the order. That matters because the held-out split is "the LAST
 *  ceil(holdout * n) rows" — computed over the surviving rows, not the manifest
 *  rows. Picking the preview caption off the raw manifest would, on any dataset
 *  with a skip near the end, quietly preview a song the model was TRAINED on,
 *  which is the one thing a held-out preview must not do. */
function usableRows(manifest: string, captionsDir: string, codesDir: string,
                    requireCaption = true): ManifestRow[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(fs.readFileSync(manifest, 'utf-8'));
  } catch {
    return [];
  }
  const arr: unknown[] = Array.isArray(parsed)
    ? parsed
    : Array.isArray((parsed as any)?.samples) ? (parsed as any).samples : [];
  const out: ManifestRow[] = [];
  for (const raw of arr) {
    const r = raw as Record<string, unknown>;
    const id = typeof r?.id === 'string' ? r.id : '';
    const filename = typeof r?.filename === 'string' ? r.filename : '';
    if (!id || !filename) continue;
    const stem = filename.replace(/\.[^.]*$/, '');
    // A per-song caption is only REQUIRED when there is no shared one. Demanding
    // it unconditionally returned an empty list for every dataset trained from
    // _shared-caption.txt, which silently disabled previews for exactly the
    // workflow that is now the default.
    if (requireCaption && !fs.existsSync(path.join(captionsDir, `${stem}.mm3.txt`))) continue;
    if (!fs.existsSync(path.join(codesDir, `${id}.codes`))) continue;
    out.push({ id, filename, lyrics: typeof r?.lyrics === 'string' ? r.lyrics : '' });
  }
  return out;
}

/** The trainer's held-out tail, by the trainer's own rule
 *  (mm3-lm-train-run.h): the last ceil(holdout * n) rows, never more than a
 *  quarter, and disabled below 6 songs. Returns [] when the trainer would not
 *  have held anything out — in which case the preview falls back to a training
 *  song and SAYS so, rather than pretending to be held out. */
function holdoutRows(rows: ManifestRow[], holdout: number): ManifestRow[] {
  if (!(holdout > 0) || rows.length < 6) return [];
  let n = Math.ceil(holdout * rows.length);
  n = Math.min(n, Math.floor(rows.length / 4));
  n = Math.max(n, 1);
  return rows.slice(rows.length - n);
}

/** Prepend the trigger the way the training rows carry it, and the way
 *  lm_apply_tag / applyTriggerTag write it: `trigger, ` at the very front of
 *  the caption's FIRST line, on the same line as `Global Metadata`.
 *
 *  This is not cosmetic. Nothing auto-injects a trigger on the MM3 path, and a
 *  trigger on its own line above the caption is a different token sequence from
 *  the one that was trained — it dilutes to nothing. A preview that got this
 *  wrong would report "the adapter does very little" for every checkpoint of a
 *  perfectly good run. */
export function applyMm3Trigger(caption: string, trigger: string): string {
  const t = trigger.trim().replace(/,+$/, '');
  if (!t) return caption;
  const body = caption.replace(/^﻿/, '').trimStart();
  if (body.toLowerCase().startsWith(`${t.toLowerCase()},`)) return body;
  return `${t}, ${body}`;
}

/** Resolve the whole preview plan from the job's options and the dataset.
 *  Returns null when previews are switched off. */
export function planMm3Previews(o: {
  preview?: Mm3PreviewOptions;
  manifest: string;
  captionsDir: string;
  codesDir: string;
  outDir: string;
  holdout: number;
  trigger: string;
  /** The dataset-wide caption, when one is in force. Used for the 'artist'
   *  preview so a dataset with no per-song .mm3.txt still gets one. */
  captionFile?: string;
}): Mm3PreviewPlan | null {
  const p = o.preview;
  const everySteps = Math.max(0, Math.trunc(p?.everySteps ?? MM3_PREVIEW_DEFAULTS.everySteps));
  const everyMinutes = Math.max(0, Math.trunc(p?.everyMinutes ?? MM3_PREVIEW_DEFAULTS.everyMinutes));
  if (everySteps <= 0 && everyMinutes <= 0) return null;

  const seconds = Math.min(120, Math.max(8, Math.trunc(p?.seconds ?? MM3_PREVIEW_DEFAULTS.seconds)));
  const seed = Number.isFinite(p?.seed) ? Math.trunc(p!.seed as number) : MM3_PREVIEW_DEFAULTS.seed;

  const specs: Mm3PreviewSpec[] = [];
  let caption = (p?.caption ?? '').trim();
  let lyrics = (p?.lyrics ?? '').trim();
  if (!caption) {
    // The shared caption IS the training prompt when one is in force, so the
    // preview should be rendered with it rather than with a per-song caption the
    // adapter never saw.
    let shared = '';
    if (o.captionFile) {
      try {
        shared = fs.readFileSync(o.captionFile, 'utf-8').trim();
      } catch { /* unreadable — fall back to per-song below */ }
    }
    // Lyrics still come from a real song, and when a shared caption is in force
    // a row does not need its own .mm3.txt to be usable.
    const rows = usableRows(o.manifest, o.captionsDir, o.codesDir, !shared);
    const held = holdoutRows(rows, o.holdout);
    const pick = held.length ? held[0] : rows.length ? rows[rows.length - 1] : null;
    if (shared) {
      caption = shared;
      if (!lyrics && pick) lyrics = pick.lyrics;
    } else if (pick) {
      const stem = pick.filename.replace(/\.[^.]*$/, '');
      try {
        caption = fs.readFileSync(path.join(o.captionsDir, `${stem}.mm3.txt`), 'utf-8').trim();
      } catch { /* fall through to the control-only plan below */ }
      if (!lyrics) lyrics = pick.lyrics;
    }
  }
  if (caption) {
    specs.push({ kind: 'artist', caption: applyMm3Trigger(caption, o.trigger), lyrics });
  }

  const wantControl = (p?.controlCaption ?? '').trim() !== '-'
    && (p?.control ?? MM3_PREVIEW_DEFAULTS.control);
  if (wantControl) {
    const ctrl = (p?.controlCaption ?? '').trim() || MM3_CONTROL_CAPTION;
    // The control is rendered WITHOUT the trigger on purpose: it asks what the
    // adapter does to a prompt that never asked for it, which is exactly the
    // collateral-damage question.
    specs.push({ kind: 'control', caption: ctrl, lyrics: '' });
  }
  if (!specs.length) return null;

  return {
    specs,
    seconds,
    seed,
    everySteps,
    everyMinutes,
    baseline: p?.baseline ?? MM3_PREVIEW_DEFAULTS.baseline,
    scaleMlp: clampScale(p?.scaleMlp, MM3_PREVIEW_DEFAULTS.scaleMlp),
    scaleAttn: clampScale(p?.scaleAttn, MM3_PREVIEW_DEFAULTS.scaleAttn),
    dir: path.join(o.outDir, 'previews'),
  };
}

/** The base a preview renders its adapter on.
 *
 *  NOT the engine's own choice, which is best-first and therefore f16 — and an
 *  adapter on f16 is garbled. This is the one render setting a preview is not
 *  allowed to inherit from whatever the user last selected. */
const PREVIEW_ADAPTER_BASE = 'q8_0';

/** Pin the LM before an adapter render. Idempotent — posting the current
 *  selection returns changed:false and touches nothing — so the cost is one
 *  cheap POST per preview and a re-warm only on the first.
 *
 *  Deliberately NOT restored afterwards. The training runner restarts the engine
 *  when the run ends, which resets the selection anyway, and q8_0 is the base
 *  adapters are supposed to be rendered on regardless. Restoring it would mean
 *  putting the engine back on a base we know is broken for the very next
 *  preview. */
async function pinPreviewBase(): Promise<string> {
  try {
    const r = await mm3SelectModel({ lm: PREVIEW_ADAPTER_BASE });
    return r?.lm || PREVIEW_ADAPTER_BASE;
  } catch (err: any) {
    // Soft-fail: a preview on the wrong base is worth having with a warning
    // attached. A preview that did not happen tells the user nothing at all.
    return `SELECTION FAILED (${err?.message || String(err)}) — this preview may be garbled`;
  }
}

/** A scale off the request body. 0 is MEANINGFUL — that component of the
 *  adapter off — so it is not treated as "unset" the way a falsy check would. */
function clampScale(v: number | undefined, dflt: number): number {
  return Number.isFinite(v) ? Math.min(2, Math.max(0, v as number)) : dflt;
}

// ── rendering ───────────────────────────────────────────────────────────────

export interface Mm3RenderRequest {
  spec: Mm3PreviewSpec;
  seconds: number;
  seed: number;
  /** Absolute path to a checkpoint directory, or '' for a base-model render. */
  adapterPath: string;
  step: number;
  totalSteps: number;
  dir: string;
  /** Training loss at this checkpoint, for the label. */
  loss?: number;
  scaleAttn?: number;
  scaleMlp?: number;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Render one preview and write it under `dir`. Throws on any failure — the
 * caller decides whether a failed preview should stop the training run (it
 * should not).
 */
export async function renderMm3Preview(r: Mm3RenderRequest): Promise<TrainingPreview> {
  const t0 = Date.now();
  fs.mkdirSync(r.dir, { recursive: true });

  const d = MM3_LM_ADAPTER_DEFAULT_SCALES;
  const body: Record<string, unknown> = {
    caption: r.spec.caption,
    lyrics: r.spec.lyrics,
    duration: r.seconds,
    seed: r.seed,
    get_wav_bits: 16,
  };
  // The engine's loader opens the path as a SAFETENSORS FILE (st_open in
  // mm3-lm-adapter.h) — handing it the checkpoint directory fails with "cannot
  // open adapter". The trainer's export writes a PEFT directory, so resolve to
  // the weights file inside it.
  let adapter = r.adapterPath;
  if (adapter) {
    try {
      if (fs.statSync(adapter).isDirectory()) {
        // LoKr writes lokr_weights.safetensors; LoRA writes a PEFT directory.
        // Same ckpt-<step>/ folder, different file name.
        const peft = path.join(adapter, 'adapter_model.safetensors');
        const lokr = path.join(adapter, 'lokr_weights.safetensors');
        adapter = fs.existsSync(peft) ? peft : lokr;
      }
    } catch { /* left as given; the engine reports it */ }
  }
  let renderBase = '';
  if (adapter) {
    renderBase = await pinPreviewBase();
    // "runtime", not "merge". Merge is cheaper per render in the f16 case, but
    // it is not universally available: on an IQ-quantized base it refuses
    // outright (mm3-lm-merge.h needs an importance matrix), and on any other
    // quantized base it REQUANTIZES 8.6 B parameters on the host per merge.
    // Previews must work on whatever base the user happens to have selected,
    // and ~+28% on a 16 s render is a cheaper price than a preview that fails
    // for a whole class of users. Runtime also never touches the resident base
    // weights, which matters when the same warm engine renders two previews
    // back to back.
    body.lm_adapter = adapter;
    body.lm_adapter_mode = 'runtime';
    body.lm_adapter_scale = d.scale;
    body.lm_adapter_scale_attn = r.scaleAttn ?? d.scaleAttn;
    body.lm_adapter_scale_mlp = r.scaleMlp ?? d.scaleMlp;
    body.lm_adapter_scale_early = d.scaleEarly;
    body.lm_adapter_scale_mid = d.scaleMid;
    body.lm_adapter_scale_late = d.scaleLate;
  }

  const sub = await mm3Synth(body as any);
  const jobId = sub.job_id || sub.id;
  if (!jobId) throw new Error('the engine accepted the preview but returned no job id');

  const deadline = Date.now() + PREVIEW_DEADLINE_MS;
  for (;;) {
    if (Date.now() > deadline) {
      throw new Error(`preview render did not finish within ${Math.round(PREVIEW_DEADLINE_MS / 60000)} min`);
    }
    await sleep(POLL_MS);
    const detail = await mm3JobDetail(jobId);
    if (!detail) continue;
    if (detail.status === 'done') break;
    if (detail.status === 'failed' || detail.status === 'cancelled') {
      throw new Error(detail.error || `preview render ${detail.status}`);
    }
  }

  const res = await aceClient.getJobResult(jobId);
  if (!res.ok) throw new Error(`preview result fetch returned HTTP ${res.status}`);
  const audio = Buffer.from(await res.arrayBuffer());
  if (audio.length < 1024) throw new Error(`preview result was ${audio.length} bytes`);

  const tag = r.adapterPath ? `step-${String(r.step).padStart(6, '0')}` : 'base';
  const file = `${tag}-${r.spec.kind}.wav`;
  fs.writeFileSync(path.join(r.dir, file), audio);

  return {
    id: `${tag}-${r.spec.kind}`,
    step: r.step,
    totalSteps: r.totalSteps,
    kind: r.spec.kind,
    base: !r.adapterPath,
    file,
    seconds: r.seconds,
    seed: r.seed,
    caption: r.spec.caption,
    renderBase: renderBase || undefined,
    loss: r.loss,
    bytes: audio.length,
    ms: Date.now() - t0,
    ts: Date.now(),
  };
}
