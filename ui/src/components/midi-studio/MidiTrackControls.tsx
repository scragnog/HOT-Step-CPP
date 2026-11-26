import { useTranslation } from 'react-i18next';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { Toggle } from '@/components/shared/controls/Toggle';

const gmPrograms = [
  'Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano',
  'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavinet',
  'Celesta', 'Glockenspiel', 'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer',
  'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ', 'Reed Organ', 'Accordion', 'Harmonica', 'Tango Accordion',
  'Nylon Guitar', 'Steel Guitar', 'Jazz Guitar', 'Clean Guitar', 'Muted Guitar', 'Overdriven Guitar', 'Distortion Guitar', 'Guitar Harmonics',
  'Acoustic Bass', 'Fingered Bass', 'Picked Bass', 'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2',
  'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani',
  'String Ensemble 1', 'String Ensemble 2', 'Synth Strings 1', 'Synth Strings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Voice', 'Orchestra Hit',
  'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet', 'French Horn', 'Brass Section', 'Synth Brass 1', 'Synth Brass 2',
  'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax', 'Oboe', 'English Horn', 'Bassoon', 'Clarinet',
  'Piccolo', 'Flute', 'Recorder', 'Pan Flute', 'Blown Bottle', 'Shakuhachi', 'Whistle', 'Ocarina',
  'Square Lead', 'Sawtooth Lead', 'Calliope Lead', 'Chiff Lead', 'Charang Lead', 'Voice Lead', 'Fifths Lead', 'Bass and Lead',
  'New Age Pad', 'Warm Pad', 'Polysynth Pad', 'Choir Pad', 'Bowed Pad', 'Metallic Pad', 'Halo Pad', 'Sweep Pad',
  'Rain FX', 'Soundtrack FX', 'Crystal FX', 'Atmosphere FX', 'Brightness FX', 'Goblins FX', 'Echoes FX', 'Sci-fi FX',
  'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bagpipe', 'Fiddle', 'Shanai',
  'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom', 'Synth Drum', 'Reverse Cymbal',
  'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet', 'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot',
].map((label, value) => ({ value, label: value + ' · ' + label }));

export function MidiProgramSelect({ value, onChange, disabled, defaultRemap }: {
  value: number;
  onChange: (program: number) => void;
  disabled?: boolean;
  defaultRemap?: boolean;
}) {
  const { t } = useTranslation();
  const label = t(defaultRemap ? 'midiStudio.defaultGuitarProgram' : 'midiStudio.player.program');
  const info = t(defaultRemap ? 'midiStudio.defaultGuitarProgramInfo' : 'midiStudio.player.programInfo');
  return (
    <div className="flex items-center gap-1.5">
      <ParamLabel label={label} info={info} />
      <StyledSelect size="sm" accent="purple" className="w-44" aria-label={label}
        value={value} options={gmPrograms} onChange={onChange} disabled={disabled} />
    </div>
  );
}

export function MidiFullVelocity({ checked, disabled, onChange }: {
  checked: boolean;
  disabled: boolean;
  onChange: (checked: boolean) => void;
}) {
  const { t } = useTranslation();
  return <Toggle size="sm" accent="purple" checked={checked} disabled={disabled} onChange={onChange}
    label={t('midiStudio.player.fullVelocity')} aria-label={t('midiStudio.player.fullVelocity')}
    info={t('midiStudio.player.fullVelocityInfo')} />;
}
