// SkinPicker.tsx — chooses a theme skin (ui/src/themes, docs/dev/skins.md).
// Studio and the kit page only; classic never shows it.

import React from 'react';
import { Tabs } from './Tabs';
import { themesFor } from '../../../themes/registry';

const DEFAULT = 'default';

export const SkinPicker: React.FC<{ skin: string | null; onChange: (id: string | null) => void }> = ({ skin, onChange }) => (
  <Tabs
    aria-label="Skin"
    value={skin ?? DEFAULT}
    onChange={v => onChange(v === DEFAULT ? null : v)}
    items={[{ value: DEFAULT, label: 'Default' }, ...themesFor('studio').map(t => ({ value: t.id, label: t.name }))]}
  />
);
