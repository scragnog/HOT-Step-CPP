// MetadataSection.tsx — BPM, Key, Time Signature, Duration, Language, Vocal Gender
// Ported to Tailwind styling matching hot-step-9000's grid layout.
//
// MiniMax-Music3 reads none of these off the wire — bpm, key and vocal gender
// reach it only by being written into the Structured Caption (see
// services/lireek/mm3Compose.ts). Two controls have no path to MM3 at all and
// are gated accordingly:
//   Time Signature — no wire slot in engine/src/minimax/, and 26 of MiniMax's
//                    1,000 reference captions state a meter (4/4 x24, 3/4 x1,
//                    6/8 x1), never in Basic Attributes. Hidden in MM3 mode.
//   Language       — MM3 has no language input; its tokenizer is a byte-level
//                    BPE, so the language follows the characters of the lyrics.
//                    Relabelled in MM3 mode to say what it actually drives.

import React from 'react';
import { useTranslation } from 'react-i18next';
import { Slider } from '../shared/Slider';
import { VOCAL_LANGUAGES } from '../../constants/languages';
import { useCapabilities } from '../../hooks/useCapabilities';
import { useBackendStore } from '../../stores/backendStore';

/** UI-side ceiling, unchanged from before capability gating existed. Used
 *  whenever the active backend's manifest hasn't defined a duration max yet
 *  (undefined/loading) so the slider never grows or shrinks unexpectedly. */
const DEFAULT_DURATION_MAX = 240;

const KEY_SIGNATURES = [
  '', 'C major', 'C minor', 'C# major', 'C# minor',
  'D major', 'D minor', 'D# major', 'D# minor',
  'E major', 'E minor', 'F major', 'F minor',
  'F# major', 'F# minor', 'G major', 'G minor',
  'G# major', 'G# minor', 'A major', 'A minor',
  'A# major', 'A# minor', 'B major', 'B minor',
];

const TIME_SIGNATURES = ['', '4/4', '3/4', '6/8', '2/4'];

/** MiniMax's corpus states one of these in every Vocal Gender & Timbre line. */
const VOCAL_GENDERS = ['', 'female', 'male', 'duet'] as const;

interface MetadataSectionProps {
  bpm: number;
  onBpmChange: (v: number) => void;
  keyScale: string;
  onKeyScaleChange: (v: string) => void;
  timeSignature: string;
  onTimeSignatureChange: (v: string) => void;
  duration: number;
  onDurationChange: (v: number) => void;
  vocalLanguage: string;
  onVocalLanguageChange: (v: string) => void;
  vocalGender: string;
  onVocalGenderChange: (v: string) => void;
}

const selectClasses = "w-full px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";

export const MetadataSection: React.FC<MetadataSectionProps> = ({
  bpm, onBpmChange, keyScale, onKeyScaleChange,
  timeSignature, onTimeSignatureChange,
  duration, onDurationChange,
  vocalLanguage, onVocalLanguageChange,
  vocalGender, onVocalGenderChange,
}) => {
  const { t } = useTranslation();
  const { capabilities } = useCapabilities();
  const mm3Mode = useBackendStore(s => s.activeBackendId) === 'minimax-m3';
  // Clamp to the active backend's manifest when it defines one (§4.2/§4.5) —
  // never a hardcoded ACE-only ceiling. capabilities?.core.duration.max is
  // undefined while loading, so DEFAULT_DURATION_MAX (the prior hardcoded
  // value) covers that gap.
  const durationMax = capabilities?.core.duration.max ?? DEFAULT_DURATION_MAX;
  return (
    <div className="space-y-3 pt-3 border-t border-zinc-200 dark:border-white/5">
      <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">{t('metadataSection.musicParameters')}</h4>

      <div className="grid grid-cols-2 gap-3">
        {/* BPM */}
        <div>
          <Slider label={t('metadataSection.bpm')} value={bpm} onChange={onBpmChange}
            min={0} max={240} step={1} showInput suffix="" />
          {bpm === 0 && <span className="text-[10px] text-zinc-600">{t('metadataSection.auto')}</span>}
        </div>

        {/* Duration */}
        <div>
          <Slider label={t('metadataSection.duration')} value={duration} onChange={onDurationChange}
            min={-1} max={durationMax} step={1} suffix="s" showInput />
          {duration <= 0 && <span className="text-[10px] text-zinc-600">{t('metadataSection.auto')}</span>}
        </div>

        {/* Key */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('metadataSection.key')}</label>
          <select className={selectClasses} value={keyScale}
            onChange={e => onKeyScaleChange(e.target.value)}>
            {KEY_SIGNATURES.map(k => (
              <option key={k} value={k}>{k || t('metadataSection.auto')}</option>
            ))}
          </select>
        </div>

        {/* Time Signature — no path to MiniMax-Music3, so hidden there */}
        {!mm3Mode && (
          <div>
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('metadataSection.timeSig')}</label>
            <select className={selectClasses} value={timeSignature}
              onChange={e => onTimeSignatureChange(e.target.value)}>
              {TIME_SIGNATURES.map(tSig => (
                <option key={tSig} value={tSig}>{tSig || t('metadataSection.auto')}</option>
              ))}
            </select>
          </div>
        )}

        {/* Vocal Gender — written into the caption's Vocal Details in MM3 mode */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('metadataSection.vocalGender')}</label>
          <select className={selectClasses} value={vocalGender}
            onChange={e => onVocalGenderChange(e.target.value)}>
            {VOCAL_GENDERS.map(g => (
              <option key={g} value={g}>
                {g === '' ? t('metadataSection.genderAny')
                  : g === 'female' ? t('metadataSection.genderFemale')
                  : g === 'male' ? t('metadataSection.genderMale')
                  : t('metadataSection.genderDuet')}
              </option>
            ))}
          </select>
        </div>

        {/* Language */}
        <div className="col-span-2">
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">
            {mm3Mode ? t('metadataSection.lyricsLanguage') : t('metadataSection.vocalLanguage')}
          </label>
          <select className={selectClasses} value={vocalLanguage}
            onChange={e => onVocalLanguageChange(e.target.value)}>
            {VOCAL_LANGUAGES.map(l => (
              <option key={l.value} value={l.value}>{l.label}</option>
            ))}
          </select>
          {mm3Mode && (
            <p className="mt-1 text-[10px] leading-snug text-zinc-500">{t('metadataSection.lyricsLanguageHint')}</p>
          )}
        </div>
      </div>
    </div>
  );
};
