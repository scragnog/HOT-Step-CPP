// MetadataSection.tsx — BPM, Key, Time Signature, Duration, Language
// Ported to Tailwind styling matching hot-step-9000's grid layout.

import React from 'react';
import { useLanguage } from '../../context/LanguageContext';

const KEY_SIGNATURES = [
  '', 'C major', 'C minor', 'C# major', 'C# minor',
  'D major', 'D minor', 'D# major', 'D# minor',
  'E major', 'E minor', 'F major', 'F minor',
  'F# major', 'F# minor', 'G major', 'G minor',
  'G# major', 'G# minor', 'A major', 'A minor',
  'A# major', 'A# minor', 'B major', 'B minor',
];

const TIME_SIGNATURES = ['', '4/4', '3/4', '6/8', '2/4', '5/4', '7/8'];
const LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: 'zh', label: '中文' },
  { value: 'ja', label: '日本語' },
  { value: 'ko', label: '한국어' },
  { value: 'es', label: 'Español' },
  { value: 'fr', label: 'Français' },
  { value: 'de', label: 'Deutsch' },
  { value: 'ru', label: 'Русский' },
  { value: 'it', label: 'Italiano' },
  { value: 'pt', label: 'Português' },
  { value: 'vi', label: 'Vietnamese' },
];

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
}

const selectClasses = "w-full px-3 py-2 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors cursor-pointer";
const inputClasses = "w-full px-3 py-2.5 rounded-xl bg-zinc-900 border border-white/10 text-sm text-zinc-200 placeholder:text-zinc-600 focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/20 outline-none transition-colors";

export const MetadataSection: React.FC<MetadataSectionProps> = ({
  bpm, onBpmChange, keyScale, onKeyScaleChange,
  timeSignature, onTimeSignatureChange,
  duration, onDurationChange,
  vocalLanguage, onVocalLanguageChange,
}) => {
  const { t } = useLanguage();
  return (
    <div className="space-y-3 pt-3 border-t border-white/5">
      <div className="px-3 py-2.5">
        <h4 className="text-xs font-bold text-zinc-200 uppercase tracking-wider">{t('nav_settings')}</h4>
      </div>

      <div className="px-3 pb-3 grid grid-cols-2 gap-3">
        {/* BPM */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('meta_bpm')}</label>
          <input
            type="number"
            className={inputClasses}
            placeholder={t('meta_random')}
            value={bpm === 0 ? '' : bpm}
            onChange={e => onBpmChange(e.target.value === '' ? 0 : parseInt(e.target.value, 10))}
          />
        </div>

        {/* Duration */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('meta_duration')}</label>
          <input
            type="number"
            className={inputClasses}
            placeholder={t('meta_random')}
            value={duration <= 0 ? '' : duration}
            onChange={e => onDurationChange(e.target.value === '' ? 0 : parseInt(e.target.value, 10))}
          />
        </div>

        {/* Key */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('meta_key')}</label>
          <select className={selectClasses} value={keyScale}
            onChange={e => onKeyScaleChange(e.target.value)}>
            {KEY_SIGNATURES.map(k => (
              <option key={k} value={k}>{k || t('meta_random')}</option>
            ))}
          </select>
        </div>

        {/* Time Signature */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('meta_time_sig')}</label>
          <select className={selectClasses} value={timeSignature}
            onChange={e => onTimeSignatureChange(e.target.value)}>
            {TIME_SIGNATURES.map(t_val => (
              <option key={t_val} value={t_val}>{t_val || t('meta_random')}</option>
            ))}
          </select>
        </div>

        {/* Language */}
        <div className="col-span-2">
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">{t('meta_language')}</label>
          <select className={selectClasses} value={vocalLanguage}
            onChange={e => onVocalLanguageChange(e.target.value)}>
            {LANGUAGES.map(l => (
              <option key={l.value} value={l.value}>{l.label}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
};
