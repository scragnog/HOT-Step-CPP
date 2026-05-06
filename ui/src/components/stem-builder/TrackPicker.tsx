// TrackPicker.tsx — Single-select track name grid for Stem Builder
//
// Displays 12 standard ACE-Step track names as pill buttons.
// Only one track can be selected at a time (radio mode).
// Matches the upstream acestep.cpp TRACK_NAMES list exactly.

import React from 'react';

/** Standard ACE-Step track names (matches C++ TRACK_NAMES constant) */
export const TRACK_NAMES = [
  'vocals', 'backing_vocals', 'drums', 'bass',
  'guitar', 'keyboard', 'percussion', 'strings',
  'synth', 'fx', 'brass', 'woodwinds',
] as const;

export type TrackName = typeof TRACK_NAMES[number];

/** Display labels with emoji icons for each track */
const TRACK_DISPLAY: Record<TrackName, { emoji: string; label: string }> = {
  vocals:          { emoji: '🎤', label: 'Vocals' },
  backing_vocals:  { emoji: '🎙️', label: 'Backing Vocals' },
  drums:           { emoji: '🥁', label: 'Drums' },
  bass:            { emoji: '🎵', label: 'Bass' },
  guitar:          { emoji: '🎸', label: 'Guitar' },
  keyboard:        { emoji: '🎹', label: 'Keyboard' },
  percussion:      { emoji: '🪘', label: 'Percussion' },
  strings:         { emoji: '🎻', label: 'Strings' },
  synth:           { emoji: '🎛️', label: 'Synth' },
  fx:              { emoji: '🔊', label: 'FX' },
  brass:           { emoji: '🎺', label: 'Brass' },
  woodwinds:       { emoji: '🪕', label: 'Woodwinds' },
};

interface TrackPickerProps {
  selectedTrack: TrackName | null;
  onTrackChange: (track: TrackName) => void;
  disabled?: boolean;
}

export const TrackPicker: React.FC<TrackPickerProps> = ({
  selectedTrack,
  onTrackChange,
  disabled = false,
}) => {
  return (
    <div>
      <div className="text-[11px] font-semibold text-zinc-500 uppercase tracking-wider mb-2">
        Target Track
      </div>
      <div className="grid grid-cols-3 gap-1.5">
        {TRACK_NAMES.map(name => {
          const display = TRACK_DISPLAY[name];
          const isActive = selectedTrack === name;
          return (
            <button
              key={name}
              type="button"
              disabled={disabled}
              onClick={() => onTrackChange(name)}
              className={`
                px-2 py-1.5 rounded-lg text-xs font-medium transition-all duration-150
                flex items-center gap-1.5 justify-center
                ${isActive
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 shadow-sm shadow-amber-500/10'
                  : 'bg-white/[0.03] text-zinc-400 border border-white/[0.06] hover:bg-white/[0.06] hover:text-zinc-300 hover:border-white/[0.1]'
                }
                disabled:opacity-40 disabled:cursor-not-allowed
              `}
              title={`Generate ${display.label} track`}
            >
              <span className="text-sm">{display.emoji}</span>
              <span className="truncate">{display.label}</span>
            </button>
          );
        })}
      </div>
      {!selectedTrack && (
        <div className="text-[10px] text-zinc-600 mt-1.5 text-center">
          Select which instrument track to generate
        </div>
      )}
    </div>
  );
};
