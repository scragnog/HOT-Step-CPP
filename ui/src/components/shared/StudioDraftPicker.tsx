// StudioDraftPicker.tsx — the draft line under a studio's header: load one of
// the user's saved drafts (studioDraftMirror.ts) and show save/load notes.
// The list refreshes on mount and whenever the control is pressed; choosing a
// draft asks before replacing unsaved edits and never starts a job.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { StudioDraftBody } from '../../../../server/src/contracts/studioDrafts';
import type { StudioDraftDocument } from '../../services/studioDraftsApi';
import type { StudioDraftControl } from '../../services/studioDraftMirror';
import { StyledSelect } from './StyledSelect';

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
  /** A reason this studio cannot take a draft, shown instead of loading it. */
  refuse?: (body: StudioDraftBody) => string;
}> = ({ control, textKey, apply, refuse }) => {
  const { t } = useTranslation();
  const [drafts, setDrafts] = useState<StudioDraftDocument[]>([]);
  const { list } = control;
  const refresh = () => { void list().then(setDrafts).catch(() => setDrafts([])); };
  useEffect(refresh, [list]);
  return (
    <div className="flex items-center gap-2 px-4 py-1 text-[11px]">
      <div className="max-w-[18rem]" onPointerDownCapture={refresh}>
        <StyledSelect<string> value="" size="sm" options={drafts.map(doc => ({ value: doc.id, label: labelOf(doc, textKey) }))}
          onChange={id => { void control.load(id, apply, refuse); }}
          placeholder={`${t('studioDrafts.load', 'Load a saved draft')}…`}
          emptyLabel={t('studioDrafts.none', 'No saved drafts')}
          aria-label={t('studioDrafts.load', 'Load a saved draft')} />
      </div>
      {control.error && <span className="text-amber-600 dark:text-amber-400">{control.error}</span>}
    </div>
  );
};
