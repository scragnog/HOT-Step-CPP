// NewDatasetWizard.tsx — "New dataset" import step
//
// Pick a folder, name it, choose a trigger word. The scan preview is a
// read-only look at what the server found on disk before anything is created.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, FolderOpen, Info, Loader2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { scanPreview, type ScanPreview } from '../../services/trainingApi';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { useTrainingStore } from '../../stores/trainingStore';
import { FolderPicker } from './FolderPicker';

/** Same shape as the server's slugify: lowercase, [a-z0-9-_], collapsed. */
function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9\-_]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^[-_]+|[-_]+$/g, '')
    .slice(0, 64);
}

interface NewDatasetWizardProps {
  open: boolean;
  onClose: () => void;
}

/** The path field is free text; a scan per keystroke would recursive-readdir the
 *  server once per character and flash "Directory not found" for every prefix. */
const SCAN_DEBOUNCE_MS = 400;

export const NewDatasetWizard: React.FC<NewDatasetWizardProps> = ({ open, onClose }) => {
  const { t } = useTranslation();
  const createDataset = useTrainingStore(s => s.createDataset);

  const [sourceDir, setSourceDir] = useState('');
  const [recursive, setRecursive] = useState(true);
  const [name, setName] = useState('');
  const [nameTouched, setNameTouched] = useState(false);
  const [triggerTouched, setTriggerTouched] = useState(false);
  const [customTag, setCustomTag] = useState('');
  const [language, setLanguage] = useState('english');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [preview, setPreview] = useState<ScanPreview | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  // Reset each time the modal is opened fresh.
  useEffect(() => {
    if (!open) return;
    setSourceDir(''); setRecursive(true); setName(''); setNameTouched(false); setTriggerTouched(false);
    setCustomTag(''); setLanguage('english'); setPreview(null); setScanError(null); setCreateError(null);
  }, [open]);

  // Monotonic request id — a slow scan of an earlier prefix must never overwrite
  // the preview for the path the user actually settled on.
  const scanSeq = useRef(0);

  const runScan = useCallback(async (dir: string, rec: boolean) => {
    if (!dir) return;
    const seq = ++scanSeq.current;
    setScanning(true);
    setScanError(null);
    try {
      const result = await scanPreview(dir, rec);
      if (seq !== scanSeq.current) return;
      setPreview(result);
    } catch (err) {
      if (seq !== scanSeq.current) return;
      setPreview(null);
      setScanError(err instanceof Error ? err.message : 'Scan failed');
    } finally {
      if (seq === scanSeq.current) setScanning(false);
    }
  }, []);

  useEffect(() => {
    if (!sourceDir) return;
    const handle = window.setTimeout(() => void runScan(sourceDir, recursive), SCAN_DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [sourceDir, recursive, runScan]);

  if (!open) return null;

  // Name defaults to the selected folder's own name; trigger word follows the
  // name. Each stops auto-following the moment the user edits it.
  const folderName = sourceDir.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? '';
  const effectiveName = nameTouched ? name : folderName;
  const effectiveTag = triggerTouched ? customTag : slugify(effectiveName);
  const canCreate = !!sourceDir && !!effectiveName.trim() && !creating && !!preview && preview.audioFiles > 0;

  const handleCreate = async () => {
    if (!canCreate) return;
    setCreating(true);
    setCreateError(null);
    try {
      await createDataset({
        name: effectiveName.trim(),
        sourceDir,
        recursive,
        customTag: effectiveTag,
        defaultLanguage: language.trim().toLowerCase() || 'english',
      });
      onClose();
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : 'Failed to create dataset');
    } finally {
      setCreating(false);
    }
  };

  const field = 'w-full rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500';

  return createPortal(
    <>
      <div
        className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
        onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      >
        <div className="w-[620px] max-h-[85vh] flex flex-col overflow-hidden rounded-xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-card shadow-2xl">
          <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-200 dark:border-white/5">
            <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.wizard.title')}</h3>
            <button onClick={onClose} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-white transition-colors">
              <X size={16} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
            {/* Folder */}
            <div className="flex flex-col gap-1.5">
              <ParamLabel
                label={t('trainingStudio.wizard.folder')}
                info={t('trainingStudio.wizard.folderInfo')}
              />
              <div className="flex gap-2">
                <input
                  type="text"
                  value={sourceDir}
                  onChange={(e) => setSourceDir(e.target.value)}
                  placeholder={t('trainingStudio.wizard.pathPlaceholder')}
                  className={`${field} font-mono text-xs`}
                />
                <button
                  onClick={() => setPickerOpen(true)}
                  className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors whitespace-nowrap"
                >
                  <FolderOpen size={14} /> {t('trainingStudio.wizard.browse')}
                </button>
              </div>
              <Toggle
                accent="amber"
                checked={recursive}
                onChange={setRecursive}
                label={t('trainingStudio.wizard.recursive')}
                info={t('trainingStudio.wizard.recursiveInfo')}
                className="mt-1"
              />
            </div>

            {/* Scan preview */}
            {scanning && (
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <Loader2 size={14} className="animate-spin" /> {t('trainingStudio.wizard.scanning')}
              </div>
            )}
            {scanError && (
              <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{scanError}</div>
            )}
            {preview && !scanning && (
              <div className="rounded-xl border border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20 p-3 flex flex-col gap-2">
                <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">
                  {preview.audioFiles > 0
                    ? t('trainingStudio.wizard.found', { count: preview.audioFiles })
                    : t('trainingStudio.wizard.foundNone')}
                </div>
                {preview.withSidecar > 0 && (
                  <div className="text-xs text-zinc-600 dark:text-zinc-400">
                    {t('trainingStudio.wizard.withSidecar', { count: preview.withSidecar })}
                  </div>
                )}
                {Object.keys(preview.extensions).length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(preview.extensions).map(([ext, n]) => (
                      <span key={ext} className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-zinc-200/70 dark:bg-white/5 text-zinc-600 dark:text-zinc-400">
                        {ext} {n}
                      </span>
                    ))}
                  </div>
                )}
                {preview.sampleNames.length > 0 && (
                  <div className="text-[11px] text-zinc-500 font-mono truncate" title={preview.sampleNames.join('\n')}>
                    {preview.sampleNames.slice(0, 10).join(' · ')}
                  </div>
                )}
                {preview.hasDatasetJson && (
                  <div className="flex items-start gap-2 text-xs text-sky-600 dark:text-sky-400">
                    <Info size={13} className="mt-0.5 flex-shrink-0" />
                    {t('trainingStudio.wizard.hasDatasetJson')}
                  </div>
                )}
                {preview.audioFiles > 0 && preview.audioFiles < 10 && (
                  <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
                    <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
                    {t('trainingStudio.wizard.smallWarning', { count: preview.audioFiles })}
                  </div>
                )}
              </div>
            )}

            {/* Name + trigger */}
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-1.5">
                <ParamLabel
                  label={t('trainingStudio.wizard.name')}
                  info={t('trainingStudio.wizard.nameInfo')}
                />
                <input
                  type="text"
                  value={effectiveName}
                  onChange={(e) => { setNameTouched(true); setName(e.target.value); }}
                  placeholder={t('trainingStudio.wizard.namePlaceholder')}
                  maxLength={100}
                  className={field}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <ParamLabel
                  label={t('trainingStudio.wizard.trigger')}
                  info={t('trainingStudio.wizard.triggerInfo')}
                />
                <input
                  type="text"
                  value={effectiveTag}
                  onChange={(e) => { setTriggerTouched(true); setCustomTag(e.target.value); }}
                  placeholder={t('trainingStudio.wizard.namePlaceholder')}
                  className={field}
                />
              </div>
            </div>

            {/* Lyric language — declared, never guessed by the local AI */}
            <div className="flex flex-col gap-1.5">
              <ParamLabel
                label={t('trainingStudio.wizard.language')}
                info={t('trainingStudio.wizard.languageInfo')}
              />
              <StyledSelect
                accent="amber"
                className="max-w-56"
                value={language}
                onChange={setLanguage}
                options={['english', 'german', 'spanish', 'french', 'italian', 'portuguese', 'russian', 'japanese', 'korean', 'chinese']
                  .map(l => ({ value: l, label: l.charAt(0).toUpperCase() + l.slice(1) }))}
                searchPlaceholder="Filter languages…"
              />
            </div>

            {createError && (
              <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{createError}</div>
            )}
          </div>

          <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
            >
              {t('trainingStudio.wizard.cancel')}
            </button>
            <button
              onClick={handleCreate}
              disabled={!canCreate}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {creating && <Loader2 size={13} className="animate-spin" />}
              {t('trainingStudio.wizard.create')}
            </button>
          </div>
        </div>
      </div>

      <FolderPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onSelect={(p) => { setSourceDir(p); setPickerOpen(false); }}
      />
    </>,
    document.body,
  );
};

export default NewDatasetWizard;
