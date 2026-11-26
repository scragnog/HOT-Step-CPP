import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/shared/kit/Button';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { StyledSelect, type SelectOption } from '@/components/shared/controls/StyledSelect';
import { useModelRegistry } from '@/components/model-manager/useModelRegistry';
import {
  clearCustomSoundFont, CUSTOM_SOUND_FONT, DEFAULT_SOUND_FONT, DEFAULT_SOUND_FONT_ID,
  notifySoundFontAvailability, readSoundFontPreference, saveCustomSoundFont, selectSoundFont, subscribeToSoundFontChanges,
  type SoundFontPreference,
} from './soundFontPreference';

export function MidiSoundFontPicker({ compact = true }: { compact?: boolean }) {
  const { t } = useTranslation();
  const picker = useRef<HTMLInputElement>(null);
  const lastAvailability = useRef<{ selection: string; bankId?: string } | null>(null);
  const { registry } = useModelRegistry();
  const [preference, setPreference] = useState<SoundFontPreference | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<'read' | 'save' | 'clear' | 'select' | null>(null);

  useEffect(() => {
    let version = 0;
    const refresh = () => {
      const requested = ++version;
      void readSoundFontPreference().then(
        value => { if (version === requested) setPreference(value); },
        () => { if (version === requested) setError('read'); },
      );
    };
    const unsubscribe = subscribeToSoundFontChanges(refresh);
    refresh();
    return () => { ++version; unsubscribe(); };
  }, []);

  const update = async (operation: () => Promise<void>, failure: 'save' | 'clear' | 'select') => {
    setBusy(true);
    setError(null);
    try {
      await operation();
    } catch {
      setError(failure);
    } finally {
      setBusy(false);
    }
  };

  const selection = preference?.selection ?? DEFAULT_SOUND_FONT;
  const banks = registry?.files.filter(file => file.tags?.includes('soundfont')) ?? [];
  const selectedBank = banks.find(file => file.id === (selection === DEFAULT_SOUND_FONT ? DEFAULT_SOUND_FONT_ID : selection));
  const missing = Boolean(registry && (
    selection === CUSTOM_SOUND_FONT ? !preference?.custom : !selectedBank?.installed
  ));
  const bankId = selectedBank?.installed ? selectedBank.id
    : banks.find(file => file.id === DEFAULT_SOUND_FONT_ID && file.installed)?.id;
  const availabilityKnown = Boolean(preference && registry && selection !== CUSTOM_SOUND_FONT);
  useEffect(() => {
    const previous = lastAvailability.current;
    lastAvailability.current = availabilityKnown ? { selection, bankId } : null;
    if (availabilityKnown && previous?.selection === selection && previous.bankId !== bankId) {
      notifySoundFontAvailability();
    }
  }, [availabilityKnown, selection, bankId]);

  const label = t('settings.midiSoundFont.label', 'Preview SoundFont');
  const options: SelectOption<string>[] = [
    { value: DEFAULT_SOUND_FONT, label: t('settings.midiSoundFont.defaultBank', 'Default (GeneralUser GS)') },
    ...banks.filter(file => file.installed || file.id === selection).map(file => ({
      value: file.id,
      label: file.displayName,
      disabled: !file.installed,
    })),
  ];
  if (preference?.custom) options.push({ value: CUSTOM_SOUND_FONT, label: preference.custom.name });

  const errors = {
    read: t('settings.midiSoundFont.readError', 'Could not read the saved SoundFont.'),
    save: t('settings.midiSoundFont.saveError', 'Could not save this file. Choose a valid .sf2 or .sf3 SoundFont and check available browser storage.'),
    clear: t('settings.midiSoundFont.clearError', 'Could not remove the saved SoundFont.'),
    select: t('settings.midiSoundFont.selectError', 'Could not save the SoundFont selection.'),
  };

  return (
    <div className="min-w-0 w-full">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <ParamLabel
          className={compact ? 'text-xs' : 'setting-label'}
          label={label}
          info={t('settings.midiSoundFont.pickerInfo', 'Choose an installed SoundFont from Model Manager or a General MIDI .sf2 or .sf3 file. This browser stores your choice and your custom file.')}
        />
        <div className="min-w-0 w-64 max-w-full">
          <StyledSelect
            aria-label={label}
            size={compact ? 'sm' : 'md'}
            value={selection}
            options={options}
            title={options.find(option => option.value === selection)?.label}
            disabled={busy || !preference}
            onChange={value => void update(() => selectSoundFont(value), 'select')}
          />
        </div>
        <input
          ref={picker}
          type="file"
          accept=".sf2,.sf3"
          hidden
          onChange={event => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = '';
            if (file) void update(() => saveCustomSoundFont(file), 'save');
          }}
        />
        <Button size={compact ? 'sm' : 'md'} className="shrink-0 whitespace-nowrap" disabled={busy} onClick={() => picker.current?.click()}>
          {t('settings.midiSoundFont.choose', 'Choose SoundFont…')}
        </Button>
        {!compact && preference?.custom && (
          <Button size={compact ? 'sm' : 'md'} className="shrink-0 whitespace-nowrap" disabled={busy} onClick={() => void update(clearCustomSoundFont, 'clear')}>
            {t('settings.midiSoundFont.removeCustom', 'Remove custom file')}
          </Button>
        )}
      </div>
      {missing && (
        <p role="status" className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
          {t('settings.midiSoundFont.missing', 'Selected SoundFont is not installed. The preview uses the fallback sounds.')}
        </p>
      )}
      {error && <p role="alert" className="mt-1 text-xs text-red-600 dark:text-red-400">{errors[error]}</p>}
    </div>
  );
}
