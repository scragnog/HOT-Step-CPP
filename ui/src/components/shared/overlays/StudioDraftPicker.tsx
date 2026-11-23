// StudioDraftPicker.tsx — the draft line under a studio's header: load one of
// the user's saved drafts (studioDraftMirror.ts) and show save/load notes.
// The list refreshes on mount and whenever the control is pressed; choosing a
// draft asks before replacing unsaved edits and never starts a job.
import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { StudioDraftBody } from '../../../../../server/src/contracts/studioDrafts';
import type { StudioDraftDocument } from '@/services/studioDraftsApi';
import { draftNameKey, type StudioDraftControl } from '@/services/studioDraftMirror';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { formatDateTime } from '@/utils/format/dateUtils';

const labelOf = (doc: StudioDraftDocument, textKey: string) => {
  const draftName = doc.body.fields[draftNameKey(doc.body.studio)];
  if (typeof draftName === 'string' && draftName.trim()) return draftName;
  const text = doc.body.fields[textKey];
  const name = typeof text === 'string' && text.trim() ? text.trim().slice(0, 40) : doc.id.slice(0, 8);
  return `${formatDateTime(doc.updatedAt)} · ${name}`;
};

export const StudioDraftPicker: React.FC<{
  control: StudioDraftControl;
  /** The field that names a draft in the list (a title, caption or source name). */
  textKey: string;
  apply: (body: StudioDraftBody) => void;
  /** A reason this studio cannot take a draft, shown instead of loading it. */
  refuse?: (body: StudioDraftBody) => string;
  namedSave?: boolean;
  /** Inside a Drafts fold at the bottom of a column: no own padding. */
  inline?: boolean;
}> = ({ control, textKey, apply, refuse, namedSave, inline }) => {
  const { t } = useTranslation();
  const [drafts, setDrafts] = useState<StudioDraftDocument[]>([]);
  const [name, setName] = useState('');
  const { list } = control;
  const refresh = () => { void list().then(setDrafts).catch(() => setDrafts([])); };
  const saveNamed = async () => {
    if (!name.trim()) return;
    if (await control.saveNamed(name)) { setName(''); refresh(); }
  };
  useEffect(refresh, [list]);
  return (
    // With a named save, the picker, name and button share the row in thirds,
    // the picker growing past its third to fit its placeholder, all three the
    // select's height and level with the centre column's header row.
    <div className={`${namedSave ? `grid grid-cols-[minmax(max-content,1fr)_1fr_1fr] items-stretch${inline ? '' : ' pt-3'}` : `flex items-center${inline ? '' : ' py-1'}`} gap-2${inline ? '' : ' px-4'} text-[11px]`}>
      <div className={namedSave ? 'min-w-0' : inline ? 'flex-1 min-w-0' : 'max-w-[18rem]'} onPointerDownCapture={refresh}>
        <StyledSelect<string> value="" size="sm" options={drafts.map(doc => ({ value: doc.id, label: labelOf(doc, textKey) }))}
          onChange={id => { void control.load(id, apply, refuse); }}
          placeholder={`${t('studioDrafts.load', 'Load saved draft')}…`}
          emptyLabel={t('studioDrafts.none', 'No saved drafts')}
          aria-label={t('studioDrafts.load', 'Load saved draft')} />
      </div>
      {namedSave && <input value={name} onChange={event => setName(event.target.value)}
        aria-label="Draft name" placeholder="Draft name"
        className="min-w-0 w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-transparent px-2 py-1.5 text-xs" />}
      {namedSave && <button type="button" onClick={() => void saveNamed()} disabled={!name.trim()}
        className="min-w-0 truncate rounded-lg border border-zinc-300 dark:border-white/10 px-2 py-1.5 text-xs text-zinc-700 dark:text-zinc-200">
        Save named draft
      </button>}
      {control.error && <span className={`${namedSave ? 'col-span-3 ' : ''}text-amber-600 dark:text-amber-400`}>{control.error}</span>}
    </div>
  );
};
