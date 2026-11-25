// StyleGenreSelect.tsx — append a genre from the built-in dictionary to the caption.
//
// Same taxonomy Insta-Gen's genre picker uses (data/genres). Unlike the preset
// button it never replaces the box: each pick is added as ", Genre" so styles
// can be stacked. The control holds no selection, so the same genre can be
// picked again after the caption is edited.

import React from 'react';
import { useTranslation } from 'react-i18next';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { GENRE_TAXONOMY } from '../../data/genres';

const OPTIONS = GENRE_TAXONOMY.flatMap(c => c.genres.map(g => ({ value: g, label: g, hint: c.name })));

interface Props {
  caption: string;
  onCaptionChange: (value: string) => void;
}

export const StyleGenreSelect: React.FC<Props> = ({ caption, onCaptionChange }) => {
  const { t } = useTranslation();

  const add = (genre: string) => {
    const has = caption.split(',').some(s => s.trim().toLowerCase() === genre.toLowerCase());
    if (has) return;
    const base = caption.replace(/[\s,.;]+$/, '');
    onCaptionChange(base ? `${base}, ${genre}` : genre);
  };

  return (
    <StyledSelect
      accent="pink"
      size="sm"
      className="!py-0.5 !px-2 !text-[10px] !rounded-md"
      triggerLabel={t('contentSection.styleGenres', 'Genres')}
      panelWidth={260}
      searchable
      searchPlaceholder={t('contentSection.styleGenreSearch', 'Search genres…')}
      title={t('contentSection.styleGenreInfo', 'Add a genre from the built-in list to the end of the Style Description. Pick several to stack them; a genre already in the box is skipped.')}
      aria-label={t('contentSection.styleGenreAria', 'Add genre')}
      value=""
      onChange={add}
      options={OPTIONS}
    />
  );
};
