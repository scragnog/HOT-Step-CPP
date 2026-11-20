import type Database from 'better-sqlite3';
import type { z } from 'zod/v4';
import { appSchema, backupSchema, backupTables } from '../contracts/settingsBackup.js';
import type { SettingsBackup, BackupCounts, BackupPreview } from '../contracts/settingsBackup.js';
export type { SettingsBackup, BackupCounts, BackupPreview } from '../contracts/settingsBackup.js';
export { backupTables } from '../contracts/settingsBackup.js';

const LYRIC_PROMPT_PREFIX = 'prompt_';
const INSTAGEN_PROMPT_KEY = 'instagen_system_prompt';
const LYRIC_SETTINGS_FILTER = 'substr(key, 1, ?) = ? OR key = ?';
const LYRIC_SETTINGS_PARAMS = [LYRIC_PROMPT_PREFIX.length, LYRIC_PROMPT_PREFIX, INSTAGEN_PROMPT_KEY] as const;
// Mirrors DEFAULT_SETTINGS in ui/src/components/settings/SettingsPanel.tsx.
const appDefaults: z.infer<typeof appSchema> = {
  coResident: false, cacheLmCodes: true, speculativeDecoding: true, downloadFormat: 'flac',
  downloadMp3Bitrate: 192, downloadOpusBitrate: 192, downloadVersion: 'mastered', downloadIncludeLatent: false,
  triggerUseFilename: false, triggerPlacement: 'prepend', discoKickExtract: false, parallelWhisper: false,
  parallelQualityEval: false, parallelCoverArt: false, generationTimeoutMinutes: 30,
};
// Browser storage can hold missing keys (older installs) or stale values; export must not fail on them.
function sanitizeApp(app: unknown) {
  const stored = app && typeof app === 'object' ? app as Record<string, unknown> : {};
  return Object.fromEntries(Object.entries(appSchema.shape).map(([key, field]) => {
    const parsed = field.safeParse(stored[key]);
    return [key, parsed.success ? parsed.data : appDefaults[key as keyof typeof appDefaults]];
  }));
}

export class BackupValidationError extends Error {}

export function validateBackup(input: unknown): SettingsBackup {
  if (input && typeof input === 'object' && 'version' in input && input.version !== 1) {
    throw new BackupValidationError(`Unsupported backup version ${String(input.version)}. This app supports version 1.`);
  }
  const result = backupSchema.safeParse(input);
  if (!result.success) {
    const issue = result.error.issues[0];
    throw new BackupValidationError(`Invalid backup at ${issue.path.join('.') || 'file'}: ${issue.message}`);
  }
  const data = result.data;
  const ids = Object.fromEntries(Object.entries(data.lyricStudio).map(([table, rows]) => {
    const values = new Set(rows.map(row => row.id));
    if (values.size !== rows.length) throw new BackupValidationError(`Duplicate IDs in ${table}.`);
    return [table, values];
  }));
  const requireRef = (table: string, value: number | null) => {
    if (value !== null && !ids[table].has(value)) throw new BackupValidationError(`Missing ${table} reference ${value}.`);
  };
  for (const row of data.lyricStudio.lyrics_sets) requireRef('artists', row.artist_id);
  for (const row of data.lyricStudio.profiles) requireRef('lyrics_sets', row.lyrics_set_id);
  for (const row of data.lyricStudio.generations) {
    requireRef('profiles', row.profile_id);
    requireRef('generations', row.parent_generation_id);
  }
  for (const row of data.lyricStudio.album_presets) requireRef('lyrics_sets', row.lyrics_set_id);
  for (const row of data.lyricStudio.audio_generations) {
    requireRef('generations', row.generation_id);
    requireRef('lyrics_sets', row.lyrics_set_id);
  }
  return data;
}

export function exportSettingsBackup(db: Database.Database, app: unknown): SettingsBackup {
  return db.transaction(() => {
    const lyricStudio = Object.fromEntries(Object.entries(backupTables).map(([table, schema]) => [
      table, db.prepare(`SELECT ${Object.keys(schema.shape).map(key => `"${key}"`).join(',')} FROM ${table} ORDER BY id`).all(),
    ]));
    const settings = Object.fromEntries((db.prepare(`SELECT key, value FROM settings WHERE ${LYRIC_SETTINGS_FILTER} ORDER BY key`)
      .all(...LYRIC_SETTINGS_PARAMS) as { key: string; value: string }[])
      .map(row => [row.key, row.value]));
    return validateBackup({ version: 1, exportedAt: new Date().toISOString(), settings: { app: sanitizeApp(app), lyricStudio: settings }, lyricStudio });
  })();
}

export function previewSettingsBackup(db: Database.Database, input: unknown): BackupPreview {
  const data = validateBackup(input);
  const current = Object.fromEntries([...Object.keys(backupTables), 'settings'].map(table => [
    table, ((table === 'settings'
      ? db.prepare(`SELECT COUNT(*) AS count FROM settings WHERE ${LYRIC_SETTINGS_FILTER}`).get(...LYRIC_SETTINGS_PARAMS)
      : db.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get()) as { count: number }).count,
  ])) as BackupCounts;
  const incoming = { ...Object.fromEntries(Object.entries(data.lyricStudio).map(([table, rows]) => [table, rows.length])),
    settings: Object.keys(data.settings.lyricStudio).length } as BackupCounts;
  return { current, incoming, exportedAt: data.exportedAt };
}

export function importSettingsBackup(db: Database.Database, input: unknown): SettingsBackup['settings']['app'] {
  const data = validateBackup(input);
  db.transaction(() => {
    // Parent generations may occur after their children in an exported file.
    db.pragma('defer_foreign_keys = ON');
    for (const table of Object.keys(backupTables).reverse()) db.prepare(`DELETE FROM ${table}`).run();
    db.prepare(`DELETE FROM settings WHERE ${LYRIC_SETTINGS_FILTER}`).run(...LYRIC_SETTINGS_PARAMS);
    for (const [table, schema] of Object.entries(backupTables)) {
      const columns = Object.keys(schema.shape);
      const insert = db.prepare(`INSERT INTO ${table} (${columns.map(key => `"${key}"`).join(',')}) VALUES (${columns.map(() => '?').join(',')})`);
      for (const row of data.lyricStudio[table as keyof typeof backupTables]) {
        const values = row as Record<string, string | number | null>;
        insert.run(...columns.map(key => values[key]));
      }
    }
    const insert = db.prepare('INSERT INTO settings (key, value) VALUES (?, ?)');
    for (const [key, value] of Object.entries(data.settings.lyricStudio)) insert.run(key, value);
  })();
  return data.settings.app;
}
