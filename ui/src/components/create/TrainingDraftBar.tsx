// TrainingDraftBar.tsx — offer a Training Studio audition draft to Create.
//
// Shown while the address carries ?trainingDraft=<id> (trainingCreateDraft.ts).
// Nothing changes until the user presses Apply; Dismiss leaves the form alone.
// Either way the query is cleared, so a reload does not offer it again.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore, scopedKey } from '../../stores/globalParamsStore';
import { writePersistedState } from '../../hooks/usePersistedState';
import {
  applyTrainingCreateDraft, clearTrainingDraftQuery, loadTrainingCreateDraft, trainingDraftIdFromUrl,
} from './trainingCreateDraft';

export const TrainingDraftBar: React.FC = () => {
  const { t } = useTranslation();
  const { token } = useAuth();
  const [draftId, setDraftId] = useState<string | null>(() => trainingDraftIdFromUrl());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The handoff navigates with pushState + popstate, which may land on an
  // already mounted Create.
  useEffect(() => {
    const onNav = () => { setDraftId(trainingDraftIdFromUrl()); setError(null); };
    window.addEventListener('popstate', onNav);
    return () => window.removeEventListener('popstate', onNav);
  }, []);

  if (!draftId) return null;

  const dismiss = () => { clearTrainingDraftQuery(); setDraftId(null); setError(null); };
  const apply = async () => {
    if (!token) { setError(t('createPanel.trainingDraft.signIn', 'Sign in to open this audition setup.')); return; }
    setBusy(true); setError(null);
    try {
      const data = await loadTrainingCreateDraft(draftId, token);
      applyTrainingCreateDraft(data, {
        store: useGlobalParamsStore, write: writePersistedState, storage: localStorage, scopedKey: base => scopedKey(base),
      });
      dismiss();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-4 mt-3 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-200">
      <div className="flex items-center gap-2">
        <span className="flex-1">
          {t('createPanel.trainingDraft.prompt', 'A Training Studio audition setup is ready. Applying it replaces this form\'s content and generation settings; it does not generate.')}
        </span>
        <button onClick={apply} disabled={busy}
          className="px-2.5 py-1 rounded-md font-medium text-white bg-violet-600 hover:bg-violet-500 disabled:opacity-50">
          {busy ? t('createPanel.trainingDraft.applying', 'Applying…') : t('createPanel.trainingDraft.apply', 'Apply')}
        </button>
        <button onClick={dismiss} disabled={busy}
          className="px-2.5 py-1 rounded-md text-zinc-500 hover:text-zinc-300 disabled:opacity-50">
          {t('createPanel.trainingDraft.dismiss', 'Dismiss')}
        </button>
      </div>
      {error && <div className="mt-1 text-red-500">{error}</div>}
    </div>
  );
};
