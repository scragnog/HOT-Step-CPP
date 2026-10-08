/**
 * ScaleOverridePresets.tsx — Reusable preset manager for adapter scale overrides.
 *
 * Presets live on the server (preferences.scale-override-preset, installation
 * scope). `hs-scaleOverridePresets` is imported once as a migration source
 * and kept, but no longer written to.
 * Used in both the Create page and Lyric Studio sidebar.
 */

import React, { useState, useEffect, useCallback, useSyncExternalStore } from 'react';
import { Save, Trash2, X, Check } from 'lucide-react';
import { StyledSelect } from './StyledSelect';
import { ParamLabel } from './ParamLabel';
import { hashImportValue, type ScaleOverridePresetBody } from '../../services/preferencesApi';
import { PresetCollection } from '../../services/presetCollection';
import { PresetSyncNotices } from './PresetSyncNotices';

// ── Types ────────────────────────────────────────────────────────────────────

export interface GroupScales {
  self_attn: number;
  cross_attn: number;
  mlp: number;
  cond_embed: number;
}

export interface ScalePreset {
  name: string;
  overallScale: number;
  groupScales: GroupScales;
}

export const STORAGE_KEY = 'hs-scaleOverridePresets';
export const MIGRATED_FLAG = 'hs-scaleOverridePresets:server-migrated';

// ── Helpers ──────────────────────────────────────────────────────────────────

function loadLegacyPresets(): ScalePreset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/** Server presets as a PresetCollection keyed by document id. The legacy
 *  key is imported once; a same-name collision waits for the user's
 *  keep-both or replace. */
export function createScalePresetCollection(): PresetCollection<ScaleOverridePresetBody> {
  return new PresetCollection<ScaleOverridePresetBody>({
    family: 'scale-override',
    migratedFlag: MIGRATED_FLAG,
    legacyItems: () => Promise.all(loadLegacyPresets().map(async p => {
      const body = p as unknown as ScaleOverridePresetBody;
      return { storageKey: `${STORAGE_KEY}:${p.name}`, sourceHash: await hashImportValue(JSON.stringify(body)), name: p.name, body };
    })),
  });
}

let collection: PresetCollection<ScaleOverridePresetBody> | null = null;
/** Shared by every mount (Create and Lyric Studio). */
export function scalePresetCollection(): PresetCollection<ScaleOverridePresetBody> {
  return collection ??= createScalePresetCollection();
}
/** Test-only: start from a fresh browser state. */
export function _resetScalePresetsForTests(): void { collection = null; }

// ── Component ────────────────────────────────────────────────────────────────

interface ScaleOverridePresetsProps {
  currentOverallScale: number;
  currentGroupScales: GroupScales;
  onLoad: (overallScale: number, groupScales: GroupScales) => void;
  /** Optional compact mode for tighter layouts */
  compact?: boolean;
}

export const ScaleOverridePresets: React.FC<ScaleOverridePresetsProps> = ({
  currentOverallScale,
  currentGroupScales,
  onLoad,
  compact = false,
}) => {
  const presetsCollection = scalePresetCollection();
  const snapshot = useSyncExternalStore(presetsCollection.subscribe, presetsCollection.getSnapshot);
  const entries = snapshot.entries;
  const presets: ScalePreset[] = entries.map(e => e.body as unknown as ScalePreset);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const selectedIdx = entries.findIndex(e => e.key === selectedKey);
  const [saving, setSaving] = useState(false);
  const [newName, setNewName] = useState('');

  useEffect(() => { void presetsCollection.load(); }, [presetsCollection]);

  // ── Select & load ──
  const handleSelect = useCallback((idx: number) => {
    const entry = entries[idx];
    setSelectedKey(entry ? entry.key : null);
    if (entry) {
      const p = entry.body;
      onLoad(p.overallScale, { ...(p.groupScales as GroupScales) });
    }
  }, [entries, onLoad]);

  // ── Save current as preset ──
  const handleSave = useCallback(() => {
    const name = newName.trim();
    if (!name) return;
    const body: ScaleOverridePresetBody = { name, overallScale: currentOverallScale, groupScales: { ...currentGroupScales } };
    setNewName('');
    setSaving(false);
    // Overwrite the one preset with this name; with none (or several), add one.
    const same = entries.filter(e => e.body.name.toLowerCase() === name.toLowerCase());
    if (same.length === 1) { presetsCollection.update(same[0]!.key, body); setSelectedKey(same[0]!.key); }
    else void presetsCollection.create(body).then(key => setSelectedKey(key));
  }, [newName, currentOverallScale, currentGroupScales, entries, presetsCollection]);

  // ── Delete selected preset ──
  const handleDelete = useCallback(() => {
    if (!selectedKey) return;
    void presetsCollection.remove(selectedKey);
    setSelectedKey(null);
  }, [selectedKey, presetsCollection]);

  const textSize = compact ? 'text-[10px]' : 'text-xs';

  return (
    <div className="space-y-1.5">
      <ParamLabel
        label="Adapter scale presets"
        info="Save the overall scale and the four per-group scale sliders above (Self-Attn, Cross-Attn, MLP, Conditioning) as a named preset, then load them back later. Presets save to this installation — they are not saved with the song or the adapter file. An older browser-only preset set is imported once and kept."
        className={`${textSize} font-semibold text-zinc-600 dark:text-zinc-400`}
      />
      <div className="flex items-center gap-1.5">
        {/* Preset selector */}
        <StyledSelect
          value={selectedIdx}
          onChange={handleSelect}
          accent="pink"
          size={compact ? 'sm' : 'md'}
          className="flex-1"
          aria-label="Preset"
          options={[
            { value: -1, label: presets.length === 0 ? 'No presets saved' : '— Select preset —' },
            ...presets.map((p, i) => ({ value: i, label: p.name })),
          ]}
        />

        {/* Save button */}
        {!saving ? (
          <button
            onClick={() => setSaving(true)}
            className="p-1 rounded-md text-zinc-500 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
            title="Save current scales as preset"
          >
            <Save className="w-3.5 h-3.5" />
          </button>
        ) : null}

        {/* Delete button */}
        {selectedIdx >= 0 && !saving && (
          <button
            onClick={handleDelete}
            className="p-1 rounded-md text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
            title="Delete selected preset"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Save name input */}
      {saving && (
        <div className="flex items-center gap-1.5 animate-in slide-in-from-top-1 duration-150">
          <input
            type="text"
            value={newName}
            onChange={e => setNewName(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') handleSave();
              if (e.key === 'Escape') { setSaving(false); setNewName(''); }
            }}
            placeholder="Preset name..."
            autoFocus
            className={`flex-1 bg-black/30 border border-emerald-500/30 rounded-md px-2 py-1 ${textSize} text-white placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-emerald-500 transition-colors`}
          />
          <button
            onClick={handleSave}
            disabled={!newName.trim()}
            className="p-1 rounded-md text-emerald-400 hover:bg-emerald-500/10 transition-colors disabled:opacity-30"
            title="Confirm save"
          >
            <Check className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => { setSaving(false); setNewName(''); }}
            className="p-1 rounded-md text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 hover:bg-white/5 transition-colors"
            title="Cancel"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <PresetSyncNotices collection={presetsCollection} snapshot={snapshot} labelOf={b => b.name} compact />

      {/* Preview of selected preset values */}
      {selectedIdx >= 0 && selectedIdx < presets.length && !saving && (
        <div className="flex items-center gap-2 text-[9px] text-zinc-500">
          <span>Scale: {presets[selectedIdx].overallScale.toFixed(2)}</span>
          <span className="text-zinc-700">|</span>
          <span>SA: {presets[selectedIdx].groupScales.self_attn.toFixed(2)}</span>
          <span>CA: {presets[selectedIdx].groupScales.cross_attn.toFixed(2)}</span>
          <span>MLP: {presets[selectedIdx].groupScales.mlp.toFixed(2)}</span>
          <span>C: {presets[selectedIdx].groupScales.cond_embed.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
};
