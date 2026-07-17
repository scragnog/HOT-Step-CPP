// ProfilesModal.tsx — named parameter-profile manager
//
// Lists server-stored profiles (every generation parameter as a raw
// snapshot — see utils/paramProfiles.ts) and lets the user save the
// current config under a name, apply, overwrite, or delete. Applying is
// live: no page reload, CreatePanel content included.

import React, { useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Bookmark, Check, RefreshCw, Save, Trash2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { profileApi, type ParamProfile } from '../../services/api';
import { applyProfileData, collectProfileData, summarizeProfile } from '../../utils/paramProfiles';
import { ConfirmDialog } from '../shared/ConfirmDialog';

interface ProfilesModalProps {
  onClose: () => void;
}

export const ProfilesModal: React.FC<ProfilesModalProps> = ({ onClose }) => {
  const { t } = useTranslation();
  const [profiles, setProfiles] = useState<ParamProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [newName, setNewName] = useState('');
  const [appliedName, setAppliedName] = useState('');
  const [confirmAction, setConfirmAction] = useState<{ kind: 'delete' | 'overwrite'; name: string } | null>(null);

  const refresh = useCallback(() => {
    profileApi.list()
      .then(r => { setProfiles(r.profiles); setError(''); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const saveAs = useCallback((name: string) => {
    profileApi.save(name, collectProfileData())
      .then(() => { setNewName(''); refresh(); })
      .catch(e => setError(e.message));
  }, [refresh]);

  const handleSaveNew = useCallback(() => {
    const name = newName.trim();
    if (!name) return;
    if (profiles.some(p => p.name.toLowerCase() === name.toLowerCase())) {
      setConfirmAction({ kind: 'overwrite', name });
    } else {
      saveAs(name);
    }
  }, [newName, profiles, saveAs]);

  const handleApply = useCallback((p: ParamProfile) => {
    applyProfileData(p.data);
    setAppliedName(p.name);
  }, []);

  const handleDelete = useCallback((name: string) => {
    profileApi.remove(name)
      .then(() => refresh())
      .catch(e => setError(e.message));
  }, [refresh]);

  // Portal to body: the global bar's backdrop-filter makes it the containing
  // block for fixed descendants, which would pin this overlay to the bar.
  return createPortal(
    <div className="fixed inset-0 z-[150] flex items-center justify-center bg-black/60 backdrop-blur-sm"
         onClick={onClose}>
      <div className="w-full max-w-xl max-h-[80vh] flex flex-col bg-zinc-50 dark:bg-zinc-900/95 rounded-2xl border border-zinc-200 dark:border-white/10 shadow-2xl overflow-hidden"
           onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-2">
            <Bookmark size={16} className="text-pink-500" />
            <h2 className="text-sm font-semibold text-zinc-800 dark:text-zinc-200">{t('profiles.title')}</h2>
          </div>
          <button onClick={onClose}
            className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Save current */}
        <div className="px-5 py-3 border-b border-zinc-200 dark:border-white/5 flex items-center gap-2">
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

        {/* List */}
        <div className="flex-1 overflow-y-auto px-3 py-2">
          {loading && (
            <div className="py-8 text-center text-sm text-zinc-500">{t('profiles.loading')}</div>
          )}
          {!loading && profiles.length === 0 && (
            <div className="py-8 text-center text-sm text-zinc-500">{t('profiles.empty')}</div>
          )}
          {profiles.map(p => (
            <div key={p.name}
              className="group flex items-center gap-3 px-2 py-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate">{p.name}</span>
                  {appliedName === p.name && (
                    <span className="flex items-center gap-1 text-[10px] text-emerald-500 flex-shrink-0">
                      <Check size={11} /> {t('profiles.applied')}
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-zinc-500 truncate">
                  {summarizeProfile(p.data)}
                  {p.saved_at && <> · {new Date(p.saved_at).toLocaleString()}</>}
                </div>
              </div>
              <button onClick={() => handleApply(p)} title={t('profiles.apply')}
                className="px-2.5 py-1.5 rounded-lg text-xs font-medium bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:bg-pink-600 hover:text-white dark:hover:bg-pink-600 transition-colors">
                {t('profiles.apply')}
              </button>
              <button onClick={() => setConfirmAction({ kind: 'overwrite', name: p.name })} title={t('profiles.update')}
                className="p-1.5 rounded-lg text-zinc-400 hover:text-sky-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
                <RefreshCw size={13} />
              </button>
              <button onClick={() => setConfirmAction({ kind: 'delete', name: p.name })} title={t('profiles.delete')}
                className="p-1.5 rounded-lg text-zinc-400 hover:text-red-400 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors">
                <Trash2 size={13} />
              </button>
            </div>
          ))}
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
          : t('profiles.overwriteMessage', { name: confirmAction?.name })}
        danger={confirmAction?.kind === 'delete'}
        onConfirm={() => {
          if (!confirmAction) return;
          if (confirmAction.kind === 'delete') handleDelete(confirmAction.name);
          else saveAs(confirmAction.name);
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
