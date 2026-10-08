/**
 * ScaleOverridePresets.tsx — Reusable preset manager for adapter scale overrides.
 *
 * Presets live on the server (preferences.scale-override-preset, installation
 * scope). `hs-scaleOverridePresets` is imported once as a migration source
 * and kept, but no longer written to.
 * Used in both the Create page and Lyric Studio sidebar.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Save, Trash2, X, Check } from 'lucide-react';
import { StyledSelect } from './StyledSelect';
import { ParamLabel } from './ParamLabel';
import { hashImportValue, preferencesApi, type ScaleOverridePresetBody } from '../../services/preferencesApi';

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

const STORAGE_KEY = 'hs-scaleOverridePresets';
const MIGRATED_FLAG = 'hs-scaleOverridePresets:server-migrated';

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

interface PresetDoc { id: string; revision: number; preset: ScalePreset }

async function loadServerPresets(): Promise<PresetDoc[]> {
  if (!localStorage.getItem(MIGRATED_FLAG)) {
    const legacy = loadLegacyPresets();
    if (legacy.length > 0) {
      const items = await Promise.all(legacy.map(async p => {
        const body = p as ScaleOverridePresetBody;
        const raw = JSON.stringify(body);
        return { storageKey: `${STORAGE_KEY}:${p.name}`, sourceHash: await hashImportValue(raw), name: p.name, body, resolution: 'keep-both' as const };
      }));
      await preferencesApi.presets.import('scale-override', items);
    }
    try { localStorage.setItem(MIGRATED_FLAG, '1'); } catch {}
  }
  const { documents } = await preferencesApi.presets.list<ScaleOverridePresetBody>('scale-override');
  return documents.map(d => ({ id: d.id, revision: d.revision, preset: d.body }));
}

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
  const [docs, setDocs] = useState<PresetDoc[]>([]);
  const presets = docs.map(d => d.preset);
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);
  const [saving, setSaving] = useState(false);
  const [newName, setNewName] = useState('');

  useEffect(() => {
    loadServerPresets().then(setDocs).catch(err => console.error('[ScaleOverridePresets] Failed to load:', err));
  }, []);

  // ── Select & load ──
  const handleSelect = useCallback((idx: number) => {
    setSelectedIdx(idx);
    if (idx >= 0 && idx < presets.length) {
      const p = presets[idx];
      onLoad(p.overallScale, { ...p.groupScales });
    }
  }, [presets, onLoad]);

  // ── Save current as preset ──
  const handleSave = useCallback(() => {
    const name = newName.trim();
    if (!name) return;

    const preset: ScalePreset = {
      name,
      overallScale: currentOverallScale,
      groupScales: { ...currentGroupScales },
    };

    // Overwrite if name already exists
    const existingIdx = docs.findIndex(d => d.preset.name.toLowerCase() === name.toLowerCase());
    setNewName('');
    setSaving(false);

    (existingIdx >= 0
      ? preferencesApi.presets.update('scale-override', docs[existingIdx]!.id, docs[existingIdx]!.revision, preset)
      : preferencesApi.presets.create('scale-override', preset)
    ).then(({ document }) => {
      setDocs(prev => {
        const next = [...prev];
        const doc = { id: document.id, revision: document.revision, preset };
        if (existingIdx >= 0) { next[existingIdx] = doc; setSelectedIdx(existingIdx); }
        else { next.push(doc); setSelectedIdx(next.length - 1); }
        return next;
      });
    }).catch(err => console.error('[ScaleOverridePresets] Failed to save:', err));
  }, [newName, currentOverallScale, currentGroupScales, docs]);

  // ── Delete selected preset ──
  const handleDelete = useCallback(() => {
    if (selectedIdx < 0 || selectedIdx >= docs.length) return;
    const doc = docs[selectedIdx]!;
    setDocs(docs.filter((_, i) => i !== selectedIdx));
    setSelectedIdx(-1);
    preferencesApi.presets.remove('scale-override', doc.id, doc.revision)
      .catch(err => console.error('[ScaleOverridePresets] Failed to delete:', err));
  }, [selectedIdx, docs]);

  const textSize = compact ? 'text-[10px]' : 'text-xs';

  return (
    <div className="space-y-1.5">
      <ParamLabel
        label="Adapter scale presets"
        info="Save the overall scale and the four per-group scale sliders above (Self-Attn, Cross-Attn, MLP, Conditioning) as a named preset, then load them back later. Presets are stored in this browser only — they are not saved with the song or the adapter file, and are not visible on another device or after clearing site data."
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
