// MetadataSection.tsx — BPM, Key, Time Signature, Duration, Language
// Ported to Tailwind styling matching hot-step-9000's grid layout.

import React from 'react';
import { Slider } from '../shared/Slider';

const KEY_SIGNATURES = [
  '', 'C major', 'C minor', 'C# major', 'C# minor',
  'D major', 'D minor', 'D# major', 'D# minor',
  'E major', 'E minor', 'F major', 'F minor',
  'F# major', 'F# minor', 'G major', 'G minor',
  'G# major', 'G# minor', 'A major', 'A minor',
  'A# major', 'A# minor', 'B major', 'B minor',
];

const TIME_SIGNATURES = ['', '4/4', '3/4', '6/8', '2/4'];
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

export const MetadataSection: React.FC<MetadataSectionProps> = ({
  bpm, onBpmChange, keyScale, onKeyScaleChange,
  timeSignature, onTimeSignatureChange,
  duration, onDurationChange,
  vocalLanguage, onVocalLanguageChange,
}) => {
  return (
    <div className="space-y-3 pt-3 border-t border-white/5">
      <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Music Parameters</h4>

      <div className="grid grid-cols-2 gap-3">
        {/* BPM */}
        <div>
          <Slider label="BPM" value={bpm} onChange={onBpmChange}
            min={0} max={240} step={1} showInput suffix="" />
          {bpm === 0 && <span className="text-[10px] text-zinc-600">Auto</span>}
        </div>

        {/* Duration */}
        <div>
          <Slider label="Duration" value={duration} onChange={onDurationChange}
            min={-1} max={240} step={1} suffix="s" showInput />
          {duration <= 0 && <span className="text-[10px] text-zinc-600">Auto</span>}
        </div>

        {/* Key */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Key</label>
          <select className={selectClasses} value={keyScale}
            onChange={e => onKeyScaleChange(e.target.value)}>
            {KEY_SIGNATURES.map(k => (
              <option key={k} value={k}>{k || 'Auto'}</option>
            ))}
          </select>
        </div>

        {/* Time Signature */}
        <div>
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Time Sig</label>
          <select className={selectClasses} value={timeSignature}
            onChange={e => onTimeSignatureChange(e.target.value)}>
            {TIME_SIGNATURES.map(t => (
              <option key={t} value={t}>{t || 'Auto'}</option>
            ))}
          </select>
        </div>

        {/* Language */}
        <div className="col-span-2">
          <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Vocal Language</label>
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
