/**
 * ScaleOverridePresets.tsx — Reusable preset manager for adapter scale overrides.
 *
 * Stores presets in localStorage (`hs-scaleOverridePresets`).
 * Used in both the Create page and Lyric Studio sidebar.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Save, Trash2, X, Check } from 'lucide-react';

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

// ── Helpers ──────────────────────────────────────────────────────────────────

function loadPresets(): ScalePreset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function savePresets(presets: ScalePreset[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
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
  const [presets, setPresets] = useState<ScalePreset[]>(loadPresets);
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);
  const [saving, setSaving] = useState(false);
  const [newName, setNewName] = useState('');

  // Sync from localStorage (for cross-tab / cross-component updates)
  useEffect(() => {
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setPresets(loadPresets());
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
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
    const existingIdx = presets.findIndex(
      p => p.name.toLowerCase() === name.toLowerCase()
    );

    let updated: ScalePreset[];
    if (existingIdx >= 0) {
      updated = [...presets];
      updated[existingIdx] = preset;
      setSelectedIdx(existingIdx);
    } else {
      updated = [...presets, preset];
      setSelectedIdx(updated.length - 1);
    }

    setPresets(updated);
    savePresets(updated);
    setNewName('');
    setSaving(false);
  }, [newName, currentOverallScale, currentGroupScales, presets]);

  // ── Delete selected preset ──
  const handleDelete = useCallback(() => {
    if (selectedIdx < 0 || selectedIdx >= presets.length) return;
    const updated = presets.filter((_, i) => i !== selectedIdx);
    setPresets(updated);
    savePresets(updated);
    setSelectedIdx(-1);
  }, [selectedIdx, presets]);

  const textSize = compact ? 'text-[10px]' : 'text-xs';

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        {/* Preset selector */}
        <select
          value={selectedIdx}
          onChange={e => handleSelect(parseInt(e.target.value, 10))}
          className={`flex-1 bg-black/30 border border-white/10 rounded-md px-2 py-1 ${textSize} text-zinc-300 focus:outline-none focus:border-pink-500 transition-colors cursor-pointer appearance-none`}
          style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'12\' height=\'12\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'%23888\' stroke-width=\'2\'%3E%3Cpath d=\'M6 9l6 6 6-6\'/%3E%3C/svg%3E")', backgroundRepeat: 'no-repeat', backgroundPosition: 'right 6px center' }}
        >
          <option value={-1} className="bg-zinc-900">
            {presets.length === 0 ? 'No presets saved' : '— Select preset —'}
          </option>
          {presets.map((p, i) => (
            <option key={i} value={i} className="bg-zinc-900">
              {p.name}
            </option>
          ))}
        </select>

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
            className={`flex-1 bg-black/30 border border-emerald-500/30 rounded-md px-2 py-1 ${textSize} text-white placeholder-zinc-600 focus:outline-none focus:border-emerald-500 transition-colors`}
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
            className="p-1 rounded-md text-zinc-500 hover:text-zinc-300 hover:bg-white/5 transition-colors"
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
