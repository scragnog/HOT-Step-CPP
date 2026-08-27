// minimax/trigger.ts — MM3 LM adapter trigger words, one implementation.
//
// A trigger has to reach the model as the SAME token sequence it was trained
// on, which for the MM3 LM trainer is `<trigger>, ` at the very front of the
// caption's first line (engine/src/train/mm3-lm-train-run.h, the
// `trigger_prefix` block). On a Structured Caption that first line is
// `Global Metadata`, so a correctly triggered caption opens
// `alk3_damnesia, Global Metadata`. A trigger on its own line above the caption
// is a different sequence and dilutes to nothing.
//
// Two consumers, so they can never drift apart: the mid-run training previews
// (training/mm3Preview.ts) and the generation path (minimax/generate.ts).
//
// The ACE equivalent is generation/triggerWords.ts. They are deliberately not
// shared: ACE resolves a whole stack of adapters with per-adapter placement out
// of safetensors metadata and re-injects after the planner CoT rewrites the
// caption. MM3 has one LM adapter, one placement, a sidecar instead of embedded
// metadata, and no CoT stage to rewrite anything.

import fs from 'fs';

/** Prepend the trigger the way the training rows carry it, and the way
 *  lm_apply_tag / applyTriggerTag write it: `trigger, ` at the very front of
 *  the caption's FIRST line, on the same line as `Global Metadata`.
 *
 *  IDEMPOTENT, case-insensitively, matching the trainer: a caption that already
 *  opens with the trigger is returned unchanged rather than growing a second
 *  copy. That is what makes it safe to apply to a caption the user has already
 *  triggered by hand. */
export function applyMm3Trigger(caption: string, trigger: string): string {
  const t = trigger.trim().replace(/,+$/, '');
  if (!t) return caption;
  const body = caption.replace(/^﻿/, '').trimStart();
  if (body.toLowerCase().startsWith(`${t.toLowerCase()},`)) return body;
  return `${t}, ${body}`;
}

export interface Mm3TriggerInfo {
  /** '' when the adapter has no sidecar, or its sidecar names no trigger. */
  trigger: string;
  /**
   * Whether the trigger was actually injected into the training captions, and
   * therefore learned at all.
   *
   * The sidecar records `trigger` either way — `--trigger-prepend` is a
   * separate flag, and an early SOAD run wrote `soad_toxicity` into its sidecar
   * without ever training it (mm3-lm-train-run.h:239). Pasting an untrained
   * trigger in front of a caption is an unseen token sequence, so a sidecar
   * that says `triggerPrepend: false` must NOT be auto-applied.
   *
   * Missing means trained. Every sidecar written before the field existed came
   * from a run with the flag on (it defaults on, and all eight local runs log
   * `<trigger>, Global Metadata` as their first training caption), and treating
   * absence as "untrained" would silently disable the feature for every adapter
   * that already exists.
   */
  prepend: boolean;
}

const NONE: Mm3TriggerInfo = { trigger: '', prepend: false };

/** `path -> {size, mtimeMs, value}`, mirroring adapters/stMetadata.ts: a
 *  sidecar is read once per file version, not once per generation. */
const cache = new Map<string, { size: number; mtimeMs: number; value: Mm3TriggerInfo }>();

/**
 * Read an MM3 LM adapter's trigger from its `<file>.safetensors.json` sidecar.
 *
 * Never throws. A missing or malformed sidecar is a normal state (an adapter
 * without a trigger is a perfectly good adapter), so it degrades to "no
 * trigger" rather than failing the generation.
 */
export function readMm3AdapterTrigger(adapterFile: string): Mm3TriggerInfo {
  try {
    const side = `${adapterFile}.json`;
    const st = fs.statSync(side);
    if (!st.isFile()) return NONE;

    const hit = cache.get(side);
    if (hit && hit.size === st.size && hit.mtimeMs === st.mtimeMs) return hit.value;

    const parsed: unknown = JSON.parse(fs.readFileSync(side, 'utf-8'));
    const md = (parsed && typeof parsed === 'object') ? parsed as Record<string, unknown> : {};
    const trigger = typeof md.trigger === 'string' ? md.trigger.trim() : '';
    const value: Mm3TriggerInfo = trigger
      ? { trigger, prepend: md.triggerPrepend !== false }
      : NONE;

    cache.set(side, { size: st.size, mtimeMs: st.mtimeMs, value });
    return value;
  } catch {
    return NONE;
  }
}
