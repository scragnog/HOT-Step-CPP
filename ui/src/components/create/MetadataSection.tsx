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
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { VOCAL_LANGUAGES } from '../../constants/languages';
import { useCapabilities } from '../../hooks/useCapabilities';
import { useBackendStore } from '../../stores/backendStore';

/** UI-side ceiling, unchanged from before capability gating existed. Used
 *  whenever the active backend's manifest hasn't defined a duration max yet
 *  (undefined/loading) so the slider never grows or shrinks unexpectedly. */
const DEFAULT_DURATION_MAX = 240;

/** What the Auto chip drops to when it is switched OFF. Not the persisted
 *  value: coming back from Auto with the old number restored would make the
 *  chip look like it had done nothing. */
const DURATION_ON_LEAVING_AUTO = 120;

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

// Every cell is a 24px label row over a 34px control row (a small select's
// height), so a slider beside a select lines up.
const CONTROL_ROW = 'h-[34px] flex items-center';

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
  const tagsMode = useBackendStore(s => s.activeBackendId) === 'heartmula';
  // YuE2's request carries style, lyrics and an optional score — nothing else.
  // Time signature, vocal gender and language have no slot on the wire and no
  // place in the trained style sentence, so they are dead knobs there. BPM and
  // Key stay: they compose into that sentence's trained tail.
  const yue2Mode = useBackendStore(s => s.activeBackendId) === 'yue2';
  // Clamp to the active backend's manifest when it defines one (§4.2/§4.5) —
  // never a hardcoded ACE-only ceiling. capabilities?.core.duration.max is
  // undefined while loading, so DEFAULT_DURATION_MAX (the prior hardcoded
  // value) covers that gap.
  const durationMax = capabilities?.core.duration.max ?? DEFAULT_DURATION_MAX;
  // Auto is a real option, not a magic -1 at the bottom of the slider: the
  // backend either has a stop token of its own or it does not. MM3's planner LM
  // emits EOS and the render ends there, so a duration is only a ceiling; ACE
  // is told a length and aims for it, and has nothing to offer here.
  const durationAuto = capabilities?.core.duration.auto === true;
  const isAutoDuration = durationAuto && duration <= 0;
  const autoChip = (
    <button
      type="button"
      onClick={() => onDurationChange(isAutoDuration ? DURATION_ON_LEAVING_AUTO : -1)}
      title={t('metadataSection.durationAutoHint', { max: durationMax })}
      className={`px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider rounded-lg border transition-colors ${
        isAutoDuration
          ? 'text-pink-300 bg-pink-500/15 border-pink-500/30'
          : 'text-zinc-500 bg-zinc-100 dark:bg-zinc-800 border-zinc-300 dark:border-white/10 hover:text-zinc-700 dark:hover:text-zinc-300'
      }`}
    >
      {t('metadataSection.auto')}
    </button>
  );
  // "0 = Auto", said beside the number rather than under the slider, so the
  // cell keeps the same height as its neighbour.
  const autoHint = <span className="text-[10px] text-zinc-500">{t('metadataSection.auto')}</span>;
  return (
    <div className="space-y-3 pt-3 border-t border-zinc-200 dark:border-white/5">
      <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">{t('metadataSection.musicParameters')}</h4>

      <div className="grid grid-cols-[repeat(auto-fit,minmax(min(150px,100%),1fr))] gap-3">
        {/* BPM */}
        <div>
          <Slider label={t('metadataSection.bpm')} value={bpm} onChange={onBpmChange}
            min={0} max={240} step={1} showInput suffix="" defaultValue={0} controlClassName={CONTROL_ROW}
            headerRight={bpm === 0 ? autoHint : undefined}
            info={t('metadataSection.bpmInfo')} infoMeta={t('metadataSection.bpmMeta')} />
        </div>

        {/* Duration — hidden in MM3 mode.
            MiniMax-Music3 has no length input: the number becomes a frame cap
            and nothing else, so the only thing it can do is cut the song off
            before the planner's own ending. Every MM3 render is auto, enforced
            server-side (backends/minimax/generate.ts), and a control whose
            single setting is "Auto" is a control worth removing. */}
        {(capabilities?.core.duration.editable ?? !mm3Mode) && (
        <div>
          {durationAuto ? (
            isAutoDuration ? (
              <div className="flex items-center justify-between h-6 mb-1.5">
                <ParamLabel
                  label={t('metadataSection.duration')}
                  info={t('metadataSection.durationAutoHint', { max: durationMax })}
                  className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
                />
                {autoChip}
              </div>
            ) : (
              <Slider label={t('metadataSection.duration')} value={Math.min(duration, durationMax)} onChange={onDurationChange}
                min={1} max={durationMax} step={1} suffix="s" showInput controlClassName={CONTROL_ROW} headerRight={autoChip}
                info={t('metadataSection.durationInfo')} />
            )
          ) : (
            <>
              <Slider label={t('metadataSection.duration')} value={duration} onChange={onDurationChange}
                min={-1} max={durationMax} step={1} suffix="s" showInput controlClassName={CONTROL_ROW}
                headerRight={duration <= 0 ? autoHint : undefined}
                info={t('metadataSection.durationInfo')} infoMeta={t('metadataSection.durationMeta')} />
            </>
          )}
        </div>
        )}

        {/* Key */}
        <div>
          <ParamLabel
            label={t('metadataSection.key')}
            info={t('metadataSection.keyInfo')}
            meta={t('metadataSection.keyMeta')}
            className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
            rootClassName="flex items-center h-6 mb-1.5"
            onReset={keyScale !== '' ? () => onKeyScaleChange('') : undefined}
          />
          <StyledSelect
            accent="pink"
            size="sm"
            value={keyScale}
            onChange={onKeyScaleChange}
            className="w-full"
            options={KEY_SIGNATURES.map(k => ({ value: k, label: k || t('metadataSection.auto') }))}
          />
        </div>

        {/* Time Signature — no path to MiniMax-Music3 or YuE2, so hidden there */}
        {(capabilities?.core.timeSignature ?? (!mm3Mode && !yue2Mode && !tagsMode)) && (
          <div>
            <ParamLabel
              label={t('metadataSection.timeSig')}
              info={t('metadataSection.timeSigInfo')}
              meta={t('metadataSection.timeSigMeta')}
              className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
              rootClassName="flex items-center h-6 mb-1.5"
              onReset={timeSignature !== '' ? () => onTimeSignatureChange('') : undefined}
            />
            <StyledSelect
              accent="pink"
              size="sm"
              value={timeSignature}
              onChange={onTimeSignatureChange}
              className="w-full"
              options={TIME_SIGNATURES.map(tSig => ({ value: tSig, label: tSig || t('metadataSection.auto') }))}
            />
          </div>
        )}

        {/* Vocal Gender — written into the caption's Vocal Details in MM3 mode,
            and nowhere at all on YuE2 */}
        {!tagsMode && (capabilities?.core.languageMeans !== 'lyrics' || mm3Mode) && (
        <div>
          <ParamLabel
            label={t('metadataSection.vocalGender')}
            info={t('metadataSection.vocalGenderInfo')}
            className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
            rootClassName="flex items-center h-6 mb-1.5"
            onReset={vocalGender !== '' ? () => onVocalGenderChange('') : undefined}
          />
          <StyledSelect
            accent="pink"
            size="sm"
            value={vocalGender}
            onChange={onVocalGenderChange}
            className="w-full"
            options={VOCAL_GENDERS.map(g => ({
              value: g,
              label: g === '' ? t('metadataSection.genderAny')
                : g === 'female' ? t('metadataSection.genderFemale')
                : g === 'male' ? t('metadataSection.genderMale')
                : t('metadataSection.genderDuet'),
            }))}
          />
        </div>
        )}

      </div>

      {/* Language — YuE2 infers it from the lyrics; there is no field for it.
          Kept outside the two-column grid so it stays full width without
          forcing a second column in a narrow Create panel. */}
      {!tagsMode && (capabilities?.core.languageMeans !== 'lyrics' || mm3Mode) && (
      <div>
        <ParamLabel
          label={mm3Mode ? t('metadataSection.lyricsLanguage') : t('metadataSection.vocalLanguage')}
          info={mm3Mode ? t('metadataSection.lyricsLanguageHint') : t('metadataSection.vocalLanguageInfo')}
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider"
          rootClassName="flex items-center h-6 mb-1.5"
          onReset={vocalLanguage !== 'en' ? () => onVocalLanguageChange('en') : undefined}
        />
        <StyledSelect
          accent="pink"
          size="sm"
          value={vocalLanguage}
          onChange={onVocalLanguageChange}
          className="w-full"
          options={VOCAL_LANGUAGES.map(l => ({ value: l.value, label: l.label }))}
        />
      </div>
      )}
    </div>
  );
};
