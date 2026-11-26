import { useEffect, useRef, useState } from 'react';
import { Music2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { clearCustomSoundFont, readCustomSoundFont, saveCustomSoundFont } from '../midi-studio/soundFontPreference';

export function MidiSoundFontSettings() {
  const { t } = useTranslation();
  const picker = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState('');
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<'read' | 'save' | 'clear' | null>(null);

  useEffect(() => {
    let active = true;
    void readCustomSoundFont().then(
      file => { if (active) setFileName(file?.name ?? ''); },
      () => { if (active) setError('read'); },
    ).finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, []);

  const choose = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      await saveCustomSoundFont(file);
      setFileName(file.name);
    } catch {
      setError('save');
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    setBusy(true);
    setError(null);
    try {
      await clearCustomSoundFont();
      setFileName('');
    } catch {
      setError('clear');
    } finally {
      setBusy(false);
    }
  };

  const errors = {
    read: t('settings.midiSoundFont.readError', 'Could not read the saved SoundFont.'),
    save: t('settings.midiSoundFont.saveError', 'Could not save this file. Choose a valid .sf2 or .sf3 SoundFont and check available browser storage.'),
    clear: t('settings.midiSoundFont.clearError', 'Could not remove the saved SoundFont.'),
  };

  return (
    <section className="settings-section">
      <div className="settings-section-header">
        <Music2 size={16} className="settings-section-icon" />
        <span className="settings-section-title">{t('settings.midiSoundFont.title', 'MIDI Studio')}</span>
      </div>
      <div className="setting-row">
        <div className="setting-info">
          <ParamLabel
            className="setting-label"
            label={t('settings.midiSoundFont.label', 'Preview SoundFont')}
            info={t('settings.midiSoundFont.info', 'Choose a General MIDI .sf2 or .sf3 file for MIDI Studio playback. This browser stores your file. Download GeneralUser GS from Model Manager to use the default bank.')}
          />
          <p className="setting-description break-all">
            {busy ? t('settings.midiSoundFont.working', 'Checking SoundFont…') : fileName || t('settings.midiSoundFont.default', 'Default (GeneralUser GS if installed)')}
          </p>
        </div>
        <div className="settings-actions">
          <input
            ref={picker}
            type="file"
            accept=".sf2,.sf3"
            hidden
            onChange={event => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = '';
              if (file) void choose(file);
            }}
          />
          <button type="button" className="env-browse-btn" disabled={busy} onClick={() => picker.current?.click()}>
            {t('settings.midiSoundFont.choose', 'Choose SoundFont…')}
          </button>
          {fileName && (
            <button type="button" className="env-browse-btn" disabled={busy} onClick={() => void reset()}>
              {t('settings.midiSoundFont.reset', 'Use default')}
            </button>
          )}
        </div>
      </div>
      {error && <p role="alert" className="text-sm text-red-600 dark:text-red-400">{errors[error]}</p>}
    </section>
  );
}
