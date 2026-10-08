// AiContinuePresetModal.tsx — Preset manager for AI lyric/style continuation
// MDMAchine / A&E Concepts 2026 — STORM Streaming

import React from 'react';
import { X, Plus, Trash2 } from 'lucide-react';
import { hashImportValue, preferencesApi, type AiContinuePresetBody, type AiContinueTemplateBody } from '../../services/preferencesApi';

// ── Preset types ──────────────────────────────────────────────────────────────
export interface AiPreset {
  id: string;
  label: string;
  value: string;   // the direction string sent to the LLM
}

export type PresetCategory = 'style' | 'lyric';

// ── Built-in presets ──────────────────────────────────────────────────────────
export const BUILTIN_STYLE_PRESETS: AiPreset[] = [
  { id: 'bi-darker',      label: '→ Darker',       value: 'shift toward a darker, moodier tone' },
  { id: 'bi-brighter',    label: '→ Brighter',      value: 'shift toward a brighter, more uplifting feel' },
  { id: 'bi-heavier',     label: '→ Heavier',       value: 'make it heavier, more aggressive, more energy' },
  { id: 'bi-softer',      label: '→ Softer',        value: 'soften the sound, more intimate and quiet' },
  { id: 'bi-electronic',  label: '→ Electronic',    value: 'push toward electronic, synthetic textures' },
  { id: 'bi-organic',     label: '→ Organic',       value: 'push toward organic, acoustic, live-sounding' },
  { id: 'bi-stripped',    label: '→ Stripped',      value: 'strip it back, minimal arrangement' },
  { id: 'bi-dense',       label: '→ Dense',         value: 'make it denser, more layered and complex' },
  { id: 'bi-flip',        label: 'Flip Genre',      value: 'completely flip the genre to something unexpected but complementary' },
  { id: 'bi-timeskip',    label: 'Era Shift',       value: 'shift the sonic palette to a different era or decade' },
  { id: 'bi-mash',        label: 'Genre Mash',      value: 'blend two contrasting genres together in an unexpected way' },
  { id: 'bi-plottwist',   label: 'Plot Twist',      value: 'dramatic tonal shift — surprise the listener' },
];

export const BUILTIN_LYRIC_PRESETS: AiPreset[] = [
  { id: 'bl-chorus',      label: '→ Chorus',        value: 'write a chorus section that captures the emotional peak' },
  { id: 'bl-verse',       label: '→ Verse',         value: 'write the next verse, advancing the narrative' },
  { id: 'bl-bridge',      label: '→ Bridge',        value: 'write a bridge that shifts perspective or breaks the pattern' },
  { id: 'bl-outro',       label: '→ Outro',         value: 'write an outro that brings the song to a close' },
  { id: 'bl-darker',      label: 'Darker theme',    value: 'shift the lyrical theme toward something darker and more complex' },
  { id: 'bl-resolve',     label: 'Resolve',         value: 'resolve the tension — bring it to a satisfying conclusion' },
  { id: 'bl-escalate',    label: 'Escalate',        value: 'escalate the emotional intensity, raise the stakes' },
  { id: 'bl-timejump',    label: 'Time Jump',       value: 'jump forward in time — write from a future perspective' },
  { id: 'bl-pov',         label: 'Flip POV',        value: 'switch to the opposite point of view' },
  { id: 'bl-abstract',    label: 'Abstract',        value: 'get more abstract and metaphorical, less literal' },
];

// ── Default prompt template ───────────────────────────────────────────────────
export const DEFAULT_TEMPLATE = `Continue these song lyrics naturally. Keep the same voice, rhyme scheme, and emotional theme. Write only the next section (verse, chorus, or bridge as appropriate).{direction}

Existing lyrics:
{lyrics}`;

export const TEMPLATE_KEY    = 'hs-ai-continue-template';
export const USER_STYLE_KEY  = 'hs-ai-continue-user-style';
export const USER_LYRIC_KEY  = 'hs-ai-continue-user-lyric';
export const PRESET_FAMILY: Record<PresetCategory, 'ai-continue-style' | 'ai-continue-lyric'> = {
  style: 'ai-continue-style', lyric: 'ai-continue-lyric',
};
export const MIGRATED_FLAG: Record<PresetCategory, string> = {
  style: `${USER_STYLE_KEY}:server-migrated`, lyric: `${USER_LYRIC_KEY}:server-migrated`,
};

/** Test-only: module-level template state (`templateDoc`, the save chain)
 *  otherwise persists across a test file's test cases. */
export function _resetTemplateStateForTests(): void {
  templateDoc = null;
  templateSaveChain = Promise.resolve();
}

function loadLegacyUserPresets(key: string): AiPreset[] {
  try { return JSON.parse(localStorage.getItem(key) || '[]'); } catch { return []; }
}

interface PresetDoc { id: string; revision: number; preset: AiPreset }

/** Named presets now live server-side (preferences.ai-continue-{style,lyric}-preset,
 *  installation scope). The legacy browser keys are imported once and kept,
 *  but no longer written to. A same-name, different-content collision
 *  against an already-imported preset is left unresolved — reported, never
 *  duplicated with a guessed keep-both/replace — so the flag stays unset
 *  and that preset retries on the next load until it is resolved
 *  explicitly. */
export async function loadServerPresets(category: PresetCategory): Promise<PresetDoc[]> {
  const family = PRESET_FAMILY[category];
  const flag = MIGRATED_FLAG[category];
  if (!localStorage.getItem(flag)) {
    const legacy = loadLegacyUserPresets(category === 'style' ? USER_STYLE_KEY : USER_LYRIC_KEY);
    if (legacy.length > 0) {
      const items = await Promise.all(legacy.map(async p => {
        const body: AiContinuePresetBody = { label: p.label, value: p.value };
        return { storageKey: `${category === 'style' ? USER_STYLE_KEY : USER_LYRIC_KEY}:${p.id}`, sourceHash: await hashImportValue(JSON.stringify(body)), name: p.label, body };
      }));
      const { results } = await preferencesApi.presets.import(family, items);
      const conflicts = results.filter(r => r.outcome === 'name-conflict');
      if (conflicts.length > 0) console.warn(`[AiContinuePresetModal] ${category} presets need an explicit import choice:`, conflicts.map(c => c.storedName));
      else try { localStorage.setItem(flag, '1'); } catch {}
    } else {
      try { localStorage.setItem(flag, '1'); } catch {}
    }
  }
  const { documents } = await preferencesApi.presets.list<AiContinuePresetBody>(family);
  return documents.map(d => ({ id: d.id, revision: d.revision, preset: { id: d.id, label: d.body.label, value: d.body.value } }));
}

/** The continuation prompt template. `loadTemplate`/synchronous localStorage
 *  stays the fast initial-render path for other readers (StormLiveControls.tsx);
 *  the server document (preferences.ai-continue-template) is the durable copy.
 *  `hydrateTemplateFromServer` pulls it down (and mirrors it into localStorage)
 *  so a second browser sees the saved value instead of falling back to
 *  DEFAULT_TEMPLATE; `queueTemplateSave` is this modal's write path. */
export function loadTemplate(): string {
  try { return localStorage.getItem(TEMPLATE_KEY) || DEFAULT_TEMPLATE; } catch { return DEFAULT_TEMPLATE; }
}
function saveTemplateLocal(t: string) {
  try { localStorage.setItem(TEMPLATE_KEY, t); } catch {}
}
let templateDoc: { id: string; revision: number } | null = null;

/** Returns the server's saved template, or null if none exists yet (in
 *  which case the legacy browser value, if non-default, is imported once). */
export async function hydrateTemplateFromServer(): Promise<string | null> {
  try {
    const { document } = await preferencesApi.settings.get<AiContinueTemplateBody>('ai-continue-template');
    if (document) {
      templateDoc = { id: document.id, revision: document.revision };
      saveTemplateLocal(document.body.template);
      return document.body.template;
    }
    const flag = `${TEMPLATE_KEY}:server-migrated`;
    const local = loadTemplate();
    if (!localStorage.getItem(flag) && local !== DEFAULT_TEMPLATE) {
      try {
        const result = await preferencesApi.settings.import('ai-continue-template', {
          storageKey: TEMPLATE_KEY, sourceHash: await hashImportValue(local), body: { template: local },
        });
        try { localStorage.setItem(flag, '1'); } catch {}
        if (result.documentId) templateDoc = { id: result.documentId, revision: 1 };
      } catch (err) { console.error('[AiContinuePresetModal] Failed to import legacy template:', err); }
    }
    return null;
  } catch (err) {
    console.error('[AiContinuePresetModal] Failed to load template:', err);
    return null;
  }
}

// Saves are chained through this promise so two edits in quick succession
// never race: the second always waits for the first's revision update,
// instead of both firing at the same stale revision and one 409ing silently.
let templateSaveChain: Promise<void> = Promise.resolve();
export function queueTemplateSave(t: string, onError: (message: string) => void): void {
  templateSaveChain = templateSaveChain.then(async () => {
    try {
      if (!templateDoc) {
        const { document: existing } = await preferencesApi.settings.get<AiContinueTemplateBody>('ai-continue-template');
        templateDoc = existing ? { id: existing.id, revision: existing.revision } : null;
      }
      const { document } = await preferencesApi.settings.upsert<AiContinueTemplateBody>(
        'ai-continue-template', templateDoc?.revision, { template: t });
      templateDoc = { id: document.id, revision: document.revision };
    } catch (err) {
      console.error('[AiContinuePresetModal] Failed to sync template:', err);
      templateDoc = null; // re-fetch the current revision next time rather than retry with a stale one
      onError(err instanceof Error ? err.message : String(err));
    }
  });
}

/** A preset create and a delete of the same (still-unsaved) row can race:
 *  the user deletes it before the create resolves. `wasDeletedMeanwhile`
 *  is checked only after `create` settles, so this always sees the final
 *  decision; if it returns true, the document that just landed is deleted
 *  instead of being kept — the create's result never silently reappears
 *  on reload after the user already deleted it. */
export async function resolvePresetCreate(
  create: () => Promise<{ document: { id: string; revision: number } }>,
  remove: (id: string, revision: number) => Promise<unknown>,
  wasDeletedMeanwhile: () => boolean,
): Promise<{ id: string; revision: number } | 'deleted'> {
  const { document } = await create();
  if (wasDeletedMeanwhile()) {
    await remove(document.id, document.revision).catch(() => {});
    return 'deleted';
  }
  return { id: document.id, revision: document.revision };
}

// ── Props ────────────────────────────────────────────────────────────────────
interface AiContinuePresetModalProps {
  isOpen: boolean;
  onClose: () => void;
  /** called when user clicks a preset — fires immediately as one-shot direction */
  onPresetFire: (preset: AiPreset, category: PresetCategory) => void;
  /** called when template changes */
  onTemplateChange: (template: string) => void;
}

// ── Component ─────────────────────────────────────────────────────────────────
export const AiContinuePresetModal: React.FC<AiContinuePresetModalProps> = ({
  isOpen, onClose, onPresetFire, onTemplateChange,
}) => {
  const [tab, setTab]                     = React.useState<'style' | 'lyric' | 'template'>('style');
  const [userStyleDocs, setUserStyleDocs] = React.useState<PresetDoc[]>([]);
  const [userLyricDocs, setUserLyricDocs] = React.useState<PresetDoc[]>([]);
  const userStylePresets = userStyleDocs.map(d => d.preset);
  const userLyricPresets = userLyricDocs.map(d => d.preset);
  const [template, setTemplate]         = React.useState(() => loadTemplate());
  const [templateError, setTemplateError] = React.useState<string | null>(null);
  const templateDirtyRef                = React.useRef(false);
  const [newLabel, setNewLabel]         = React.useState('');
  const [newValue, setNewValue]         = React.useState('');
  const [firedId,  setFiredId]          = React.useState<string | null>(null);
  const [presetError, setPresetError]   = React.useState<string | null>(null);
  // tempIds whose create is still in flight when the user deletes them —
  // the delete is queued against the eventual server id/revision instead
  // of being dropped (which would let the create's document reappear later).
  const pendingDeletesRef               = React.useRef<Set<string>>(new Set());

  React.useEffect(() => { if (!isOpen) { setNewLabel(''); setNewValue(''); } }, [isOpen]);

  React.useEffect(() => {
    loadServerPresets('style').then(setUserStyleDocs).catch(err => console.error('[AiContinuePresetModal] Failed to load style presets:', err));
    loadServerPresets('lyric').then(setUserLyricDocs).catch(err => console.error('[AiContinuePresetModal] Failed to load lyric presets:', err));
    // Hydrate from the server so a second browser sees the saved template
    // instead of DEFAULT_TEMPLATE. Skipped if the user already started typing.
    hydrateTemplateFromServer().then(t => {
      if (t === null || templateDirtyRef.current) return;
      setTemplate(t);
      onTemplateChange(t); // keep StormLiveControls' own loadTemplate()-seeded state in step
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const addPreset = (category: PresetCategory) => {
    const label = newLabel.trim(), value = newValue.trim();
    if (!label || !value) return;
    const tempId = `user-${Date.now()}`;
    const setDocs = category === 'style' ? setUserStyleDocs : setUserLyricDocs;
    setDocs(prev => [...prev, { id: tempId, revision: 0, preset: { id: tempId, label, value } }]);
    resolvePresetCreate(
      () => preferencesApi.presets.create<AiContinuePresetBody>(PRESET_FAMILY[category], { label, value }),
      (id, revision) => preferencesApi.presets.remove(PRESET_FAMILY[category], id, revision),
      () => pendingDeletesRef.current.delete(tempId),
    ).then(result => {
      if (result === 'deleted') return;
      setDocs(prev => prev.map(d => d.id === tempId
        ? { id: result.id, revision: result.revision, preset: { id: result.id, label, value } } : d));
    }).catch(err => {
      console.error('[AiContinuePresetModal] Failed to save preset:', err);
      pendingDeletesRef.current.delete(tempId);
      setDocs(prev => prev.filter(d => d.id !== tempId));
      setPresetError(`Failed to save "${label}": ${err instanceof Error ? err.message : String(err)}`);
    });
    setNewLabel(''); setNewValue('');
  };

  const deletePreset = (category: PresetCategory, id: string) => {
    const docs = category === 'style' ? userStyleDocs : userLyricDocs;
    const setDocs = category === 'style' ? setUserStyleDocs : setUserLyricDocs;
    const doc = docs.find(d => d.id === id);
    setDocs(prev => prev.filter(d => d.id !== id));
    if (!doc) return;
    if (doc.revision === 0) { pendingDeletesRef.current.add(id); return; } // create still in flight
    preferencesApi.presets.remove(PRESET_FAMILY[category], doc.id, doc.revision)
      .catch(err => {
        console.error('[AiContinuePresetModal] Failed to delete preset:', err);
        setPresetError(`Failed to delete "${doc.preset.label}": ${err instanceof Error ? err.message : String(err)}`);
        // A failed delete (stale revision, offline) must not disappear from the list.
        setDocs(prev => prev.some(d => d.id === doc.id) ? prev : [...prev, doc]);
      });
  };

  const firePreset = (preset: AiPreset, category: PresetCategory) => {
    setFiredId(preset.id);
    setTimeout(() => setFiredId(null), 1200);
    onPresetFire(preset, category);
  };

  const handleTemplateChange = (val: string) => {
    templateDirtyRef.current = true;
    setTemplate(val);
    saveTemplateLocal(val);
    setTemplateError(null);
    queueTemplateSave(val, setTemplateError);
    onTemplateChange(val);
  };

  if (!isOpen) return null;

  const tabBtn = (id: typeof tab, label: string) => (
    <button onClick={() => setTab(id)}
      className={`flex-1 py-1.5 text-xs font-medium rounded-lg transition-colors ${
        tab === id ? 'bg-violet-600 text-white' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800'}`}>
      {label}
    </button>
  );

  const PresetPill = ({ preset, category, isUser }: { preset: AiPreset; category: PresetCategory; isUser?: boolean }) => (
    <div className="flex items-center gap-0.5 group">
      <button
        onClick={() => firePreset(preset, category)}
        className={`text-[10px] px-2 py-0.5 rounded-l-md font-medium transition-all ${
          firedId === preset.id
            ? 'bg-green-600 text-white scale-95'
            : 'bg-zinc-800 text-zinc-300 hover:bg-violet-700 hover:text-white'}`}>
        {firedId === preset.id ? '✓ fired' : preset.label}
      </button>
      {/* delete button for user presets */}
      {isUser ? (
        <button onClick={() => deletePreset(category, preset.id)}
          className="text-[9px] px-1 py-0.5 rounded-r-md bg-zinc-800 text-zinc-700 hover:bg-red-900/40 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100">
          <Trash2 size={8} />
        </button>
      ) : (
        <span className="w-0 rounded-r-md bg-zinc-800" /> // keep pill shape consistent
      )}
    </div>
  );

  const PresetSection = ({ category, builtins, user }: {
    category: PresetCategory; builtins: AiPreset[]; user: AiPreset[];
  }) => (
    <div className="space-y-3">
      {/* Built-ins */}
      <div>
        <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-1.5">Built-in</div>
        <div className="flex flex-wrap gap-1">
          {builtins.map(p => <PresetPill key={p.id} preset={p} category={category} />)}
        </div>
      </div>
      {/* User presets */}
      <div>
        <div className="text-[9px] text-zinc-600 uppercase tracking-wider mb-1.5">My Presets</div>
        {user.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {user.map(p => <PresetPill key={p.id} preset={p} category={category} isUser />)}
          </div>
        )}
        {/* Add new */}
        <div className="space-y-1.5 p-2 rounded-lg bg-zinc-900/50 border border-zinc-800">
          <input type="text" value={newLabel} onChange={e => setNewLabel(e.target.value)}
            placeholder="Button label (e.g. My vibe shift)"
            className="w-full px-2 py-1 rounded bg-zinc-800 border border-zinc-700 text-[10px] text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-violet-500/40 transition-colors" />
          <input type="text" value={newValue} onChange={e => setNewValue(e.target.value)}
            placeholder="Direction sent to AI (e.g. shift toward jazz influences)"
            onKeyDown={e => { if (e.key === 'Enter') addPreset(category); }}
            className="w-full px-2 py-1 rounded bg-zinc-800 border border-zinc-700 text-[10px] text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-violet-500/40 transition-colors" />
          <button onClick={() => addPreset(category)}
            disabled={!newLabel.trim() || !newValue.trim()}
            className="flex items-center gap-1 text-[9px] px-2 py-0.5 rounded bg-violet-600 hover:bg-violet-500 text-white font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            <Plus size={9} /> Add preset
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 pointer-events-none">
        <div
          className="w-full max-w-lg bg-zinc-900/98 rounded-2xl border border-zinc-700 shadow-2xl pointer-events-auto overflow-hidden"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-white">🤖 AI Continue Presets</span>
            </div>
            <button onClick={onClose}
              className="p-1 rounded text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800 transition-colors">
              <X size={15} />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 px-4 pt-3">
            {tabBtn('style',    '🎨 Style Presets')}
            {tabBtn('lyric',    '🎤 Lyric Presets')}
            {tabBtn('template', '⚙ Template')}
          </div>

          {/* Body */}
          <div className="px-4 py-3 max-h-[60vh] overflow-y-auto">
            {presetError && tab !== 'template' && (
              <div className="mb-2 flex items-center justify-between gap-2 px-2 py-1.5 rounded-lg bg-red-950/40 border border-red-900 text-[10px] text-red-300" role="alert">
                <span className="flex-1">{presetError}</span>
                <button onClick={() => setPresetError(null)} className="hover:text-red-100"><X size={10} /></button>
              </div>
            )}
            {tab === 'style' && (
              <PresetSection
                category="style"
                builtins={BUILTIN_STYLE_PRESETS}
                user={userStylePresets}
              />
            )}
            {tab === 'lyric' && (
              <PresetSection
                category="lyric"
                builtins={BUILTIN_LYRIC_PRESETS}
                user={userLyricPresets}
              />
            )}
            {tab === 'template' && (
              <div className="space-y-2">
                {templateError && (
                  <div className="flex items-center justify-between gap-2 px-2 py-1.5 rounded-lg bg-red-950/40 border border-red-900 text-[10px] text-red-300" role="alert">
                    <span className="flex-1">Template save failed: {templateError}. Your edit is still shown above.</span>
                    <button onClick={() => { setTemplateError(null); queueTemplateSave(template, setTemplateError); }} className="underline flex-shrink-0">Retry</button>
                  </div>
                )}
                <p className="text-[10px] text-zinc-500 leading-relaxed">
                  Raw prompt sent to the LLM. Tokens: <code className="text-violet-400">{"\\{direction\\}"}</code> (replaced with direction hint if set), <code className="text-violet-400">{"\\{lyrics\\}"}</code> (current lyrics), <code className="text-violet-400">{"\\{style\\}"}</code> (current caption).
                </p>
                <textarea
                  value={template}
                  onChange={e => handleTemplateChange(e.target.value)}
                  rows={10}
                  className="w-full px-2.5 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-[10px] text-zinc-200 font-mono outline-none resize-none focus:border-violet-500/50 transition-colors"
                />
                <button
                  onClick={() => handleTemplateChange(DEFAULT_TEMPLATE)}
                  className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-700 transition-colors"
                >
                  Reset to default
                </button>
              </div>
            )}
          </div>

          {/* Footer note */}
          <div className="px-4 py-2 border-t border-zinc-800">
            <p className="text-[9px] text-zinc-600">
              Clicking a preset fires immediately — doesn't wait for the interval. Direction field sets the persistent hint picked up each interval.
            </p>
          </div>
        </div>
      </div>
    </>
  );
};