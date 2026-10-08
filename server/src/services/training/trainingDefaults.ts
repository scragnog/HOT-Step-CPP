// training/trainingDefaults.ts — persisted per-stage option defaults
//
// The batch pipeline runs stages with no user in the loop, so the option bodies
// it POSTs come from here. Storage is the main db's key/value `settings` table
// (db/database.ts owns the CREATE TABLE IF NOT EXISTS) under one JSON key.
//
// There is deliberately NO deep validation: a section is an opaque bag of the
// stage route's own option fields, and that route re-validates every one of them
// at execution time — which is the honest gate.
//
// Spec: docs/plans/2026-07-28-training-batch-pipeline.md §2.3

import { getDb } from '../../db/database.js';
import type { TrainingDefaults } from './types.js';

const SETTINGS_KEY = 'training_defaults_v1';

const SECTIONS = ['label', 'preprocess', 'trainLm', 'trainDit'] as const;

function emptyDefaults(): TrainingDefaults {
  return { label: {}, preprocess: {}, trainLm: {}, trainDit: {} };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

/** Stored partials; `{}` for every section that was never set. */
export function getTrainingDefaults(): TrainingDefaults {
  const out = emptyDefaults();
  let raw = '';
  try {
    const row = getDb()
      .prepare('SELECT value FROM settings WHERE key = ?')
      .get(SETTINGS_KEY) as { value?: string } | undefined;
    raw = row?.value ?? '';
  } catch (err: any) {
    console.warn(`[Training] Could not read training defaults: ${err.message}`);
    return out;
  }
  if (!raw) return out;

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!isPlainObject(parsed)) return out;
    for (const section of SECTIONS) {
      const value = parsed[section];
      if (isPlainObject(value)) out[section] = value;
    }
  } catch (err: any) {
    console.warn(`[Training] Training defaults are corrupt, using empty: ${err.message}`);
    return emptyDefaults();
  }
  return out;
}

/**
 * Shallow-merge per section: a section present in the patch REPLACES the stored
 * one, an absent section is left alone. Replacing rather than key-merging is
 * what lets the (future) configuration page clear a field — a key-merge would
 * make every stored option permanent.
 */
export function setTrainingDefaults(patch: Partial<TrainingDefaults>): TrainingDefaults {
  const merged = getTrainingDefaults();
  for (const section of SECTIONS) {
    const value = patch[section];
    if (isPlainObject(value)) merged[section] = value;
  }
  getDb().prepare(
    'INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
  ).run(SETTINGS_KEY, JSON.stringify(merged));
  return merged;
}

/** Keep the three layers visible while resolving a recipe. A stored section
 *  is partial, and an explicit zero, false or empty string is still an edit. */
export function resolveTrainingDefaultLayers<T extends Record<string, unknown>>(
  builtin: T, stored: Record<string, unknown>, overrides: Record<string, unknown>,
): { builtin: T; stored: Record<string, unknown>; resolved: Record<string, unknown>;
  provenance: Record<string, 'builtin' | 'stored' | 'override'> } {
  const resolved = { ...builtin, ...stored, ...overrides };
  const provenance: Record<string, 'builtin' | 'stored' | 'override'> = {};
  for (const key of Object.keys(resolved)) {
    provenance[key] = Object.prototype.hasOwnProperty.call(overrides, key) ? 'override'
      : Object.prototype.hasOwnProperty.call(stored, key) ? 'stored' : 'builtin';
  }
  return { builtin, stored, resolved, provenance };
}
