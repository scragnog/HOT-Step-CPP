import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { profilesOptions } from '@/queries/shell-queue/settings';
import { queryManager } from '@/queries/invalidation';
// ProfilesModal.tsx — named parameter-profile manager
//
// Lists server-stored profiles (every generation parameter as a raw
// snapshot — see utils/training/paramProfiles.ts) and lets the user save the
// current config under a name, apply, overwrite, or delete. Applying is
// live: no page reload, CreatePanel content included.

import React, { useCallback, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Bookmark, Check, ChevronDown, ChevronRight, Download, Pencil, RefreshCw, Save, Trash2, Upload, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { profileApi, type ParamProfile } from '@/services/api';
import { validateProfileImport } from '@/services/exportImportApi';
import { useAuth } from '@/context/AuthContext';
import { applyProfileData, collectProfileData, describeProfileGroups, summarizeProfile } from '@/utils/training/paramProfiles';
import { safeFileStem, saveJson } from '@/utils/storage/saveFile';
import { ConfirmDialog } from '@/components/shared/overlays/ConfirmDialog';
import { Toggle } from '@/components/shared/controls/Toggle';
import { useBackendStore } from '@/stores/backendStore';
import { formatDateTime } from '@/utils/format/dateUtils';

interface ProfilesModalProps {
  onClose: () => void;
}

export const ProfilesModal: React.FC<ProfilesModalProps> = ({ onClose }) => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const client = useQueryClient();
  // Profiles belong to the engine they were saved on; this lists, saves and
  // edits the active engine's only.
  const backend = useBackendStore(s => s.activeBackendId) || 'ace';
  const engineName = useBackendStore(s => s.backends.find(b => b.id === backend)?.displayName) || backend;
  const options = profilesOptions(backend);
  const query = useQuery(options);
  const profiles = query.data?.profiles ?? [];
  const loading = query.isPending;
  const [newName, setNewName] = useState('');
  const [includeContent, setIncludeContent] = useState(true);
  const [appliedName, setAppliedName] = useState('');
  const [confirmAction, setConfirmAction] = useState<{ kind: 'delete' | 'overwrite' | 'import'; name: string; data?: Record<string, unknown> } | null>(null);
  const [expandedName, setExpandedName] = useState('');
  const [renamingName, setRenamingName] = useState('');
  const [renameValue, setRenameValue] = useState('');
  const [importError, setImportError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const invalidateProfiles = useCallback(() => void queryManager.invalidate(client, options.queryKey), [client, backend]);

  // ── Mutations ───────────────────────────────────────────────────────────────

  const saveMut = useMutation({
    mutationFn: ({ name, data }: { name: string; data?: Record<string, unknown> }) =>
      profileApi.save(backend, name, data ?? collectProfileData(includeContent, backend === 'ace')),
    onSuccess: () => { setNewName(''); invalidateProfiles(); },
  });

  const deleteMut = useMutation({
    mutationFn: (name: string) => profileApi.remove(backend, name),
    onSuccess: invalidateProfiles,
  });

  const renameMut = useMutation({
    mutationFn: ({ from, to }: { from: string; to: string }) => profileApi.rename(backend, from, to),
    onSuccess: (_, { from, to }) => {
      if (appliedName === from) setAppliedName(to);
      if (expandedName === from) setExpandedName(to);
      setRenamingName('');
      invalidateProfiles();
    },
  });

  // ── Handlers ────────────────────────────────────────────────────────────────

  const handleSaveNew = useCallback(() => {
    const name = newName.trim();
    if (!name) return;
    if (profiles.some(p => p.name.toLowerCase() === name.toLowerCase())) {
      setConfirmAction({ kind: 'overwrite', name });
    } else {
      saveMut.mutate({ name });
    }
  }, [newName, profiles, saveMut]);

  const handleApply = useCallback((p: ParamProfile) => {
    applyProfileData(p.data);
    setAppliedName(p.name);
  }, []);

  const startRename = useCallback((p: ParamProfile) => {
    setRenamingName(p.name);
    setRenameValue(p.name);
  }, []);

  const commitRename = useCallback(() => {
    const from = renamingName;
    const to = renameValue.trim();
    if (!to || to === from) { setRenamingName(''); return; }
    if (profiles.some(p => p.name.toLowerCase() === to.toLowerCase() && p.name !== from)) {
      renameMut.mutate({ from, to }); // collision handled by server error
      return;
    }
    renameMut.mutate({ from, to });
  }, [renamingName, renameValue, profiles, renameMut]);

  // ── JSON export/import (same preset format as saved profiles) ──

  const handleExportProfile = useCallback((p: ParamProfile) => {
    saveJson(p.data, `${safeFileStem(p.name, 'profile')}.json`);
  }, []);

  const handleImportFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !token) return;
    setImportError('');
    const reader = new FileReader();
    reader.onload = async () => {
      try {
        const parsed = JSON.parse(reader.result as string);
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
          setImportError(t('profiles.importInvalid'));
          return;
        }
        const { name, data } = await validateProfileImport(file.name, parsed);
        if (profiles.some(p => p.name.toLowerCase() === name.toLowerCase())) {
          setConfirmAction({ kind: 'import', name, data });
        } else {
          saveMut.mutate({ name, data });
        }
      } catch {
        setImportError(t('profiles.importInvalid'));
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }, [profiles, saveMut, t, token]);

  // Combined error for the list: query errors + mutation errors
  const error = query.error?.message
    ?? saveMut.error?.message
    ?? deleteMut.error?.message
    ?? renameMut.error?.message
    ?? importError;

  // Portal to body: the global bar's backdrop-filter makes it the containing
  // block for fixed descendants, which would pin this overlay to the bar.
  return createPortal(
    <div data-portal-layer className="fixed inset-0 z-[150] flex items-center justify-center bg-black/60 backdrop-blur-sm"
         onClick={onClose}>
      <div className="w-full max-w-xl max-h-[80vh] flex flex-col bg-zinc-50 dark:bg-zinc-900/95 rounded-2xl border border-zinc-200 dark:border-white/10 shadow-2xl overflow-hidden"
           onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-2">
            <Bookmark size={16} className="text-pink-500" />
            <h2 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">{t('profiles.title')} · {engineName}</h2>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={() => fileInputRef.current?.click()} title={t('profiles.import')}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-sky-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
              <Download size={14} />
            </button>
            <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImportFile} />
            <button onClick={onClose}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Save current */}
        <div className="px-5 py-3 border-b border-zinc-200 dark:border-white/5 space-y-2">
          <div className="flex items-center gap-2">
            <input
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleSaveNew(); }}
              placeholder={t('profiles.namePlaceholder')}
              className="flex-1 px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors"
            />
            <button onClick={handleSaveNew} disabled={!newName.trim()}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-pink-600 hover:bg-pink-500 text-white text-sm font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
              <Save size={13} />
              {t('profiles.saveCurrent')}
            </button>
          </div>
          <Toggle
            size="sm"
            accent="pink"
            checked={includeContent}
            onChange={setIncludeContent}
            label={t('profiles.includeContent')}
            info={t('profiles.includeContentInfo')}
          />
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto px-3 py-2">
          {loading && (
            <div className="py-8 text-center text-sm text-zinc-500">{t('profiles.loading')}</div>
          )}
          {!loading && profiles.length === 0 && (
            <div className="py-8 text-center text-sm text-zinc-500">{t('profiles.empty')}</div>
          )}
          {profiles.map(p => {
            const isExpanded = expandedName === p.name;
            const isRenaming = renamingName === p.name;
            return (
            <div key={p.name} className="rounded-xl overflow-hidden">
              <div className="group flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors">
                {isRenaming ? (
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={e => setRenameValue(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') commitRename();
                      else if (e.key === 'Escape') setRenamingName('');
                    }}
                    onBlur={commitRename}
                    className="flex-1 min-w-0 px-2 py-1 rounded-lg bg-white dark:bg-zinc-900 border border-pink-500/50 text-sm text-zinc-800 dark:text-zinc-200 focus:ring-1 focus:ring-pink-500/20 outline-none"
                  />
                ) : (
                  <button onClick={() => setExpandedName(isExpanded ? '' : p.name)}
                    title={t('profiles.inspect')}
                    className="flex-1 min-w-0 flex items-center gap-1.5 text-left">
                    <span className="flex-shrink-0 text-zinc-400">
                      {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                    </span>
                    <span className="min-w-0">
                      <span className="flex items-center gap-2">
                        <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate">{p.name}</span>
                        {appliedName === p.name && (
                          <span className="flex items-center gap-1 text-[10px] text-emerald-500 flex-shrink-0">
                            <Check size={11} /> {t('profiles.applied')}
                          </span>
                        )}
                      </span>
                      <span className="block text-[10px] text-zinc-500 truncate">
                        {summarizeProfile(p.data)}
                        {p.saved_at && <> · {formatDateTime(p.saved_at)}</>}
                      </span>
                    </span>
                  </button>
                )}
                <button onClick={() => handleApply(p)} title={t('profiles.apply')}
                  className="flex-shrink-0 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-pink-600 hover:text-white dark:hover:bg-pink-600 transition-colors">
                  {t('profiles.apply')}
                </button>
                <button onClick={() => startRename(p)} title={t('profiles.rename')}
                  className="p-1.5 rounded-lg text-zinc-400 hover:text-amber-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
                  <Pencil size={13} />
                </button>
                <button onClick={() => setConfirmAction({ kind: 'overwrite', name: p.name })} title={t('profiles.update')}
                  className="p-1.5 rounded-lg text-zinc-400 hover:text-sky-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
                  <RefreshCw size={13} />
                </button>
                <button onClick={() => handleExportProfile(p)} title={t('profiles.export')}
                  className="p-1.5 rounded-lg text-zinc-400 hover:text-emerald-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
                  <Upload size={13} />
                </button>
                <button onClick={() => setConfirmAction({ kind: 'delete', name: p.name })} title={t('profiles.delete')}
                  className="p-1.5 rounded-lg text-zinc-400 hover:text-red-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
                  <Trash2 size={13} />
                </button>
              </div>

              {/* Inspector — grouped parameter list */}
              {isExpanded && (
                <div className="mx-2 mb-2 mt-0.5 rounded-xl bg-zinc-100/70 dark:bg-black/20 border border-zinc-200 dark:border-white/5 px-3 py-3 space-y-3">
                  {describeProfileGroups(p.data).map(g => (
                    <div key={g.title}>
                      <div className="text-[10px] font-semibold uppercase tracking-wider text-pink-500/80 mb-1">{g.title}</div>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                        {g.rows.map(r => (
                          <div key={r.key} className="flex items-baseline justify-between gap-2 min-w-0">
                            <span className="text-[11px] text-zinc-500 truncate">{r.label}</span>
                            <span className="text-[11px] font-mono text-zinc-700 dark:text-zinc-300 truncate text-right">{r.value}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            );
          })}
          {error && (
            <div className="px-2 py-2 text-xs text-red-400">{error}</div>
          )}
        </div>

        {/* Nested inside the stopPropagation panel so backdrop clicks don't
            bubble to the overlay's onClose; renders fixed/fullscreen anyway. */}
        <ConfirmDialog
        isOpen={confirmAction !== null}
        title={confirmAction?.kind === 'delete' ? t('profiles.deleteTitle') : t('profiles.overwriteTitle')}
        message={confirmAction?.kind === 'delete'
          ? t('profiles.deleteMessage', { name: confirmAction?.name })
          : confirmAction?.kind === 'import'
            ? t('profiles.importOverwriteMessage', { name: confirmAction?.name })
            : t('profiles.overwriteMessage', { name: confirmAction?.name })}
        danger={confirmAction?.kind === 'delete'}
        onConfirm={() => {
          if (!confirmAction) return;
          if (confirmAction.kind === 'delete') deleteMut.mutate(confirmAction.name);
          else if (confirmAction.kind === 'import' && confirmAction.data) saveMut.mutate({ name: confirmAction.name, data: confirmAction.data });
          else saveMut.mutate({ name: confirmAction.name });
          setConfirmAction(null);
        }}
        onCancel={() => setConfirmAction(null)}
        />
      </div>
    </div>,
    document.body
  );
};

export default ProfilesModal;
