// StylePresetSelect.tsx — one-click ensemble descriptions for the caption box.
//
// "Neural Band-in-a-Box" (#211): the arrangement is the thing a lead sheet can't
// say. A preset writes one sentence of instrumentation and mood straight into the
// Style Description, so a pasted or previewed ABC score gets played by a named
// band instead of whatever the model guesses. Same shape as BUILTIN_STYLE_PRESETS
// in storm/AiContinuePresetModal: {id, label, value} fired into a text field.
//
// The pick is derived from the caption rather than stored, so typing in the box
// after choosing a preset clears the selection instead of leaving a stale label
// on the trigger.

import React from 'react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';

interface StylePreset {
  id: string;
  label: string;
  /** The exact caption text the preset writes. Doubles as the option value. */
  value: string;
}

// One sentence each: what plays it, how it plays it, what it feels like. Genre
// coverage is deliberate — jazz to metal to orchestral — and no named artists.
// Not exported: this file exports the component only, which keeps React's fast
// refresh happy.
const STYLE_PRESETS: StylePreset[] = [
  { id: 'jazz-quartet',      label: 'Smoky jazz quartet',      value: 'A smoky late-night jazz quartet of tenor sax, upright bass, brushed drums and a piano, played slow and unhurried with a warm, smoky mood.' },
  { id: 'big-band-swing',    label: 'Big-band swing',          value: 'A big-band swing arrangement with a brass section, walking double bass, swinging snare and stride piano, bright and energetic with a loose dance-floor mood.' },
  { id: 'indie-folk',        label: 'Intimate indie folk',     value: 'An indie folk arrangement of acoustic guitar, banjo, a light hand drum and close harmony vocals, intimate and gentle with a wistful, homesick mood.' },
  { id: 'worship-ballad',    label: 'Contemporary worship ballad', value: 'A contemporary worship ballad built on piano, acoustic guitar and a soft choir backing, rising gently toward a broad, reverent, uplifting close.' },
  { id: 'synthwave',         label: 'Eighties synthwave',      value: 'An eighties synthwave setup of analog synthesizers, gated drums and a plucking bass line, neon and driving with a nostalgic, cinematic mood.' },
  { id: 'heavy-metal',       label: 'Heavy metal band',        value: 'A heavy metal arrangement of distorted guitars, pounding double-bass drums and a bass guitar, dense and aggressive with a dark, defiant mood.' },
  { id: 'ambient-electronic',label: 'Ambient electronic',      value: 'An ambient electronic piece of soft synthesizer pads, slow pulses and sparse percussion, spacious and slow-moving with a calm, meditative mood.' },
  { id: 'country-band',      label: 'Country band',            value: 'A country band of electric guitar, fiddle, bass and a plain drum kit, straightforward and warm with a dusty, good-hearted mood.' },
  { id: 'orchestral',        label: 'Full orchestra',          value: 'A full orchestral arrangement of strings, brass, woodwinds and timpani, sweeping and dynamic with a solemn, cinematic mood.' },
  { id: 'reggae',            label: 'Reggae band',             value: 'A reggae band of offbeat guitar chops, deep bass, organ and light percussion, relaxed and steady with a sunny, upbeat mood.' },
  { id: 'bossa-nova',        label: 'Bossa nova group',        value: 'A bossa nova group of nylon-string guitar, light brushes, upright bass and soft flute, breezy and understated with an easygoing, sunlit mood.' },
  { id: 'funk-band',         label: 'Funk band',               value: 'A funk band with a tight horn section, wah guitar, electric bass and a driving drum kit, syncopated and lively with a gritty dance-floor mood.' },
];

interface Props {
  caption: string;
  onCaptionChange: (value: string) => void;
}

export const StylePresetSelect: React.FC<Props> = ({ caption, onCaptionChange }) => {
  const { t } = useTranslation();
  const selected = STYLE_PRESETS.find(p => p.value === caption)?.value ?? '';

  return (
    <div className="flex items-center gap-2">
      <ParamLabel
        label={t('contentSection.stylePreset', 'Style preset')}
        info={t('contentSection.stylePresetInfo', 'A ready-made band and mood written into the Style Description. Picking one replaces whatever is in the box; edit the box afterwards and the selection clears. Useful with a lead sheet, where the melody and chords are already fixed and only the arrangement is left to say.')}
        className="text-[10px] text-zinc-500"
      />
      <StyledSelect
        accent="pink"
        size="sm"
        className="flex-1"
        aria-label={t('contentSection.stylePresetAria', 'Style preset')}
        placeholder={t('contentSection.stylePresetPlaceholder', 'Pick an ensemble style')}
        value={selected}
        onChange={value => onCaptionChange(value)}
        options={STYLE_PRESETS.map(p => ({ value: p.value, label: p.label, hint: p.value }))}
      />
    </div>
  );
};
