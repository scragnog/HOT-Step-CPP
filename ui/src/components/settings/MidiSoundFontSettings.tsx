import { Music2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { MidiSoundFontPicker } from '../midi-studio/MidiSoundFontPicker';

export function MidiSoundFontSettings() {
  const { t } = useTranslation();

  return (
    <section className="settings-section">
      <div className="settings-section-header">
        <Music2 size={16} className="settings-section-icon" />
        <span className="settings-section-title">{t('settings.midiSoundFont.title', 'MIDI Studio')}</span>
      </div>
      <div className="setting-row setting-row--wide">
        <MidiSoundFontPicker compact={false} />
      </div>
    </section>
  );
}
