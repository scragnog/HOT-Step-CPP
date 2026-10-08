// StudioDraftPicker.tsx — the draft line under a studio's header: load one of
// the user's saved drafts (studioDraftMirror.ts) and show save/load notes.
// The list is fetched when the control is opened; choosing a draft asks before
// replacing unsaved edits and never starts a job.
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { StudioDraftBody } from '../../../../server/src/contracts/studioDrafts';
import type { StudioDraftDocument } from '../../services/studioDraftsApi';
import type { StudioDraftControl } from '../../services/studioDraftMirror';

const labelOf = (doc: StudioDraftDocument, textKey: string) => {
  const text = doc.body.fields[textKey];
  const name = typeof text === 'string' && text.trim() ? text.trim().slice(0, 40) : doc.id.slice(0, 8);
  return `${new Date(doc.updatedAt).toLocaleString()} · ${name}`;
};

export const StudioDraftPicker: React.FC<{
  control: StudioDraftControl;
  /** The field that names a draft in the list (a title, caption or source name). */
  textKey: string;
  apply: (body: StudioDraftBody) => void;
}> = ({ control, textKey, apply }) => {
  const { t } = useTranslation();
  const [drafts, setDrafts] = useState<StudioDraftDocument[] | null>(null);
  const refresh = () => { void control.list().then(setDrafts).catch(() => setDrafts([])); };
  return (
    <div className="flex items-center gap-2 px-4 py-1 text-[11px]">
      <select value="" onFocus={refresh} onMouseDown={() => { if (!drafts) refresh(); }}
        onChange={e => { if (e.target.value) void control.load(e.target.value, apply); }}
        aria-label={t('studioDrafts.load', 'Load a saved draft')}
        className="max-w-[16rem] bg-transparent text-zinc-500 dark:text-zinc-400 outline-none cursor-pointer">
        <option value="">{t('studioDrafts.load', 'Load a saved draft')}…</option>
        {(drafts ?? []).map(doc => <option key={doc.id} value={doc.id}>{labelOf(doc, textKey)}</option>)}
      </select>
      {control.error && <span className="text-amber-600 dark:text-amber-400">{control.error}</span>}
    </div>
  );
};
