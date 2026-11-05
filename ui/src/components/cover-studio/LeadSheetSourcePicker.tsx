import React from 'react';
import { useTranslation } from 'react-i18next';
import type { CoverLeadSheetSource } from '../../../../server/src/contracts/studioWorkflows';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { LEAD_SHEET_SOURCES } from './leadSheetSource';

const LABELS = ['SheetSage2', 'SheetSage2-Refined', 'MuScriptor', 'YourMT3+ / ChordNet'];

export const LeadSheetSourcePicker: React.FC<{
  value?: CoverLeadSheetSource; disabled: boolean; onChange: (value: CoverLeadSheetSource) => void;
}> = ({ value, disabled, onChange }) => {
  const { t } = useTranslation();
  return <div className="space-y-1">
    <ParamLabel label={t('cover.leadSheetSource', 'Lead-sheet source')}
      info={t('cover.leadSheetSourceInfo', 'Choose how the recording becomes a melody and chord score. A new draft starts with the Models default. Changing a saved source clears its score approval. MuScriptor chords are estimates.')} />
    <StyledSelect accent="cyan" value={value ?? ''} disabled={disabled}
      options={[...(!value ? [{ value: '', label: t('cover.leadSheetDefault', 'Models default') }] : []),
        ...LEAD_SHEET_SOURCES.map((source, i) => ({ value: source, label: LABELS[i] }))]}
      onChange={source => onChange(source as CoverLeadSheetSource)} />
  </div>;
};
