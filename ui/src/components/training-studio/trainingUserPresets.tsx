// trainingUserPresets.tsx — named presets and "changed from defaults" chips, shared by the
// YuE2 joint and MiniMax-Music3 train cards. Presets live in localStorage under a key the
// caller owns, so each family keeps its own list. Lifted out of Yue2AitkTrainCard unchanged.

import { useEffect, useRef, useState } from 'react';
import { Download, RotateCcw, Save, Upload, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '../shared/ParamLabel';

export function readStored<T>(key: string, fallback: T): T {
  if (typeof window === 'undefined') return fallback;
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) as T : fallback;
  } catch { return fallback; }
}

export function writeStored(key: string, value: unknown): void {
  if (typeof window === 'undefined') return;
  try { window.localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage may be full or unavailable */ }
}

export interface NamedPreset<S> { version?: 2; name: string; settings: Partial<S> }

export interface NamedPresetsOptions<S> {
  /** localStorage key; one list per family. */
  storageKey: string;
  /** Exported file kind, so a file from another family is still accepted but labelled. */
  fileKind: string;
  /** The current form settings, per-run paths already left out. */
  snapshot: () => Partial<S>;
  /** Keys a shared file must not carry into this machine's run. */
  excluded: ReadonlySet<string>;
  /** Migrations on first read. */
  initial?: () => NamedPreset<S>[];
}

export function useNamedPresets<S>({ storageKey, fileKind, snapshot, excluded, initial }: NamedPresetsOptions<S>) {
  const { t } = useTranslation();
  const [presets, setPresets] = useState<NamedPreset<S>[]>(() => initial ? initial() : readStored<NamedPreset<S>[]>(storageKey, []));
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  useEffect(() => { writeStored(storageKey, presets); }, [storageKey, presets]);

  const save = () => {
    const trimmed = name.trim();
    if (!trimmed) { setError(t('trainingStudio.yue2.method.presetNameRequired', 'Give the preset a name first.')); return; }
    // Saving under an existing name updates that preset, after asking (#203).
    const existing = presets.find(preset => preset.name.toLowerCase() === trimmed.toLowerCase());
    if (existing && !window.confirm(t('trainingStudio.yue2.method.presetOverwrite', 'Replace the preset "{{name}}" with the current settings?', { name: existing.name }))) return;
    setError('');
    const saved: NamedPreset<S> = { version: 2, name: existing?.name ?? trimmed, settings: snapshot() };
    setPresets(previous => existing ? previous.map(preset => preset === existing ? saved : preset) : [...previous, saved]);
    setName('');
  };
  const update = (preset: NamedPreset<S>) => {
    if (!window.confirm(t('trainingStudio.yue2.method.presetOverwrite', 'Replace the preset "{{name}}" with the current settings?', { name: preset.name }))) return;
    const saved: NamedPreset<S> = { version: 2, name: preset.name, settings: snapshot() };
    setPresets(previous => previous.map(p => p === preset ? saved : p));
  };
  const remove = (presetName: string) => {
    if (!window.confirm(t('trainingStudio.yue2.method.presetDeleteConfirm', 'Delete the preset "{{name}}"?', { name: presetName }))) return;
    setPresets(previous => previous.filter(preset => preset.name !== presetName));
  };
  const exportOne = (preset: NamedPreset<S>) => {
    const url = URL.createObjectURL(new Blob([JSON.stringify({ kind: fileKind, ...preset }, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${preset.name.replace(/[^a-zA-Z0-9 _-]/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const importFile = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text()) as unknown;
      const incoming = (Array.isArray(parsed) ? parsed : [parsed]).filter((item): item is NamedPreset<S> =>
        !!item && typeof item === 'object' && typeof (item as NamedPreset<S>).name === 'string'
        && !!(item as NamedPreset<S>).settings && typeof (item as NamedPreset<S>).settings === 'object');
      if (!incoming.length) throw new Error('no preset in that file');
      setPresets(previous => {
        const next = [...previous];
        for (const item of incoming) {
          // A shared file must not carry someone else's paths into this machine's run.
          const settings = Object.fromEntries(Object.entries(item.settings).filter(([key]) => !excluded.has(key))) as Partial<S>;
          let presetName = item.name.trim() || 'Imported preset';
          for (let n = 2; next.some(p => p.name.toLowerCase() === presetName.toLowerCase()); n++) presetName = `${item.name.trim()} (${n})`;
          next.push({ version: 2, name: presetName, settings });
        }
        return next;
      });
      setError('');
    } catch (err) {
      setError(t('trainingStudio.yue2.method.presetImportFailed', 'Could not import that file: {{error}}', { error: err instanceof Error ? err.message : String(err) }));
    }
  };
  return { presets, setPresets, name, setName, error, setError, save, update, remove, exportOne, importFile };
}

type NamedPresets<S> = ReturnType<typeof useNamedPresets<S>>;

const INPUT = 'px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20 disabled:opacity-50';

/** "Your presets": a chip per saved preset (load, overwrite, export, delete), a name box that saves on Enter, and Import. */
export function UserPresetRow<S>({ state, isActive, onLoad, busy, loading = false, info }: {
  state: NamedPresets<S>;
  isActive: (preset: NamedPreset<S>) => boolean;
  onLoad: (preset: NamedPreset<S>) => void;
  busy: boolean;
  loading?: boolean;
  info: string;
}) {
  const { t } = useTranslation();
  const fileRef = useRef<HTMLInputElement>(null);
  return <>
    <div className="mt-3 flex items-center gap-2 flex-wrap">
      <ParamLabel label={t('trainingStudio.yue2.method.presets', 'Your presets')}
        className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" info={info} />
      {state.presets.map(preset => {
        const on = isActive(preset);
        return <span key={preset.name} className={`inline-flex items-center gap-1 rounded-lg border-2 pl-3 pr-1 py-1 text-xs ${on
          ? 'border-blue-500 bg-blue-500/15 text-blue-700 dark:text-blue-300'
          : 'border-zinc-300 dark:border-white/15 text-zinc-800 dark:text-zinc-100'}`}>
          <button type="button" onClick={() => onLoad(preset)} disabled={busy || loading}
            title={t('trainingStudio.yue2.method.presetLoad', 'Load this preset into the form')}
            className="font-semibold hover:underline disabled:no-underline disabled:opacity-40">{preset.name}</button>
          <button type="button" onClick={() => state.update(preset)} disabled={busy}
            title={t('trainingStudio.yue2.method.presetUpdate', 'Overwrite this preset with the current settings')}
            className="rounded p-0.5 text-zinc-400 hover:text-amber-600 dark:hover:text-amber-400 disabled:opacity-40"><Save size={12} /></button>
          <button type="button" onClick={() => state.exportOne(preset)}
            title={t('trainingStudio.yue2.method.presetExport', 'Download this preset as a .json file to share or back up')}
            className="rounded p-0.5 text-zinc-400 hover:text-amber-600 dark:hover:text-amber-400"><Download size={12} /></button>
          <button type="button" onClick={() => state.remove(preset.name)}
            title={t('trainingStudio.yue2.method.presetRemove', 'Delete this preset')}
            className="rounded p-0.5 text-zinc-400 hover:text-red-600 dark:hover:text-red-400"><X size={12} /></button>
        </span>;
      })}
      <input className={`${INPUT} w-48 !py-1 !text-xs`} placeholder={t('trainingStudio.yue2.method.presetNamePlaceholder', 'New preset name')}
        value={state.name} disabled={busy}
        onChange={event => { state.setName(event.target.value); state.setError(''); }}
        onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); state.save(); } }} />
      <button type="button" onClick={state.save} disabled={busy}
        className="px-2.5 py-1 rounded-lg text-[11px] font-semibold border border-amber-500/50 text-amber-700 dark:text-amber-300 hover:bg-amber-500/10 disabled:opacity-40">
        {t('trainingStudio.yue2.method.presetSave', 'Save current settings')}
      </button>
      <button type="button" onClick={() => fileRef.current?.click()} disabled={busy}
        title={t('trainingStudio.yue2.method.presetImportTitle', 'Add presets from a .json file exported here or by someone else')}
        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-[11px] font-semibold border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10 disabled:opacity-40">
        <Upload size={12} />{t('trainingStudio.yue2.method.presetImport', 'Import')}
      </button>
      <input ref={fileRef} type="file" accept=".json,application/json" className="hidden"
        onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void state.importFile(file); }} />
    </div>
    {state.error && <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">{state.error}</p>}
  </>;
}

/** Chips for every setting that differs from the defaults, each with its own reset, plus Reset all. */
export function ChangedSettingsRow({ changed, onReset, onResetAll, disabled }: {
  changed: Array<[string, unknown]>;
  onReset: (key: string) => void;
  onResetAll: () => void;
  disabled: boolean;
}) {
  const { t } = useTranslation();
  return <div className="mt-3 flex items-center gap-2 flex-wrap">
    <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">
      {t('trainingStudio.yue2.method.changedSettings', 'Changed from defaults')}
    </span>
    {changed.length === 0
      ? <span className="text-[11px] text-zinc-500">{t('trainingStudio.yue2.method.noChangedSettings', 'none: this form is the recipe defaults')}</span>
      : <>
        {changed.map(([key, value]) => (
          <span key={key} className="inline-flex items-center gap-1 rounded-lg border border-sky-500/40 bg-sky-500/10 pl-2 pr-1 py-0.5 text-[11px] text-sky-700 dark:text-sky-300">
            {key}{typeof value === 'object' ? '' : `: ${String(value)}`}
            <button type="button" onClick={() => onReset(key)} disabled={disabled}
              title={t('trainingStudio.yue2.method.resetSetting', 'Reset this setting to its default')}
              className="rounded p-0.5 hover:text-sky-900 dark:hover:text-white disabled:opacity-40">
              <RotateCcw size={11} />
            </button>
          </span>
        ))}
        <button type="button" onClick={onResetAll} disabled={disabled}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg text-[11px] font-semibold border border-zinc-300 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-500/10 disabled:opacity-40">
          <RotateCcw size={11} />{t('trainingStudio.yue2.method.resetAll', 'Reset all to defaults')}
        </button>
      </>}
  </div>;
}
