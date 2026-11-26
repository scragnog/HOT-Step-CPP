import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/shared/kit/Button';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { StyledSelect, type SelectOption } from '@/components/shared/controls/StyledSelect';
import { ConfirmDialog } from '@/components/shared/overlays/ConfirmDialog';
import { useModelRegistry } from '@/components/model-manager/useModelRegistry';
import { ApiError, apiClient, HttpStatus } from '@/services/httpClient';
import {
  CUSTOM_SOUND_FONT, DEFAULT_SOUND_FONT, DEFAULT_SOUND_FONT_ID,
  libraryBankUrl, librarySelection, notifySoundFontAvailability, readSoundFontPreference,
  removeLibrarySoundFont, renameLibrarySoundFont, resetMissingSoundFont, saveCustomSoundFont, selectSoundFont,
  soundFontLibraryOptions, subscribeToSoundFontChanges, type SoundFontEntry, type SoundFontPreference,
} from './soundFontPreference';

type Operation = 'save' | 'rename' | 'remove' | 'select';

export function MidiSoundFontPicker({ compact = true }: { compact?: boolean }) {
  const { t } = useTranslation();
  const picker = useRef<HTMLInputElement>(null);
  const lastAvailability = useRef<{ selection: string; bankId?: string } | null>(null);
  const { registry } = useModelRegistry();
  const library = useQuery(soundFontLibraryOptions());
  const [preference, setPreference] = useState<SoundFontPreference | null>(null);
  const [busy, setBusy] = useState(false);
  const [readError, setReadError] = useState(false);
  const [error, setError] = useState<Operation | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const [confirm, setConfirm] = useState<SoundFontEntry | null>(null);
  const size = compact ? 'sm' : 'md';

  useEffect(() => {
    let version = 0;
    const refresh = () => {
      const requested = ++version;
      void readSoundFontPreference().then(
        value => { if (version === requested) { setPreference(value); setReadError(false); } },
        () => { if (version === requested) setReadError(true); },
      );
    };
    const unsubscribe = subscribeToSoundFontChanges(refresh);
    refresh();
    return () => { ++version; unsubscribe(); };
  }, []);

  const update = async (operation: () => Promise<void>, failure: Operation) => {
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
  const libraryUrl = libraryBankUrl(selection);
  const missingLibraryBank = Boolean(library.data && libraryUrl
    && !library.data.some(entry => librarySelection(entry.id) === selection));

  useEffect(() => {
    if (!missingLibraryBank || !libraryUrl) return;
    const controller = new AbortController();
    // A tab may have picked a newly uploaded bank before this tab's list refreshed.
    void apiClient.request('HEAD', libraryUrl, undefined, { signal: controller.signal, silent: true })
      .catch(async error => {
        if (!controller.signal.aborted && error instanceof ApiError && error.status === HttpStatus.notFound) {
          await resetMissingSoundFont(selection);
        }
      }).catch(() => { if (!controller.signal.aborted) setReadError(true); });
    return () => controller.abort();
  }, [missingLibraryBank, libraryUrl, selection]);

  const banks = registry?.files.filter(file => file.tags?.includes('soundfont')) ?? [];
  const selectedBank = banks.find(file => file.id === (selection === DEFAULT_SOUND_FONT ? DEFAULT_SOUND_FONT_ID : selection));
  const registrySelection = !libraryUrl && selection !== CUSTOM_SOUND_FONT;
  const missing = Boolean(registry && registrySelection && !selectedBank?.installed);
  const bankId = selectedBank?.installed ? selectedBank.id
    : banks.find(file => file.id === DEFAULT_SOUND_FONT_ID && file.installed)?.id;
  const availabilityKnown = Boolean(preference && registry && registrySelection);
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
    ...(library.data ?? []).map(entry => ({
      value: librarySelection(entry.id),
      label: entry.name,
      hint: entry.format.toUpperCase(),
    })),
  ];
  if (preference?.custom && selection === CUSTOM_SOUND_FONT) {
    options.push({ value: CUSTOM_SOUND_FONT, label: preference.custom.name });
  }

  const errors = {
    save: t('settings.midiSoundFont.addError', 'Could not add this SoundFont. Choose a valid .sf2 or .sf3 file up to 1 GB and check the server’s available space.'),
    rename: t('settings.midiSoundFont.renameError', 'Could not rename the SoundFont.'),
    remove: t('settings.midiSoundFont.removeError', 'Could not remove the SoundFont.'),
    select: t('settings.midiSoundFont.selectError', 'Could not save the SoundFont selection.'),
  };
  const rename = () => {
    if (!renaming?.name.trim() || busy) return;
    void update(async () => {
      await renameLibrarySoundFont(renaming.id, renaming.name);
      setRenaming(null);
    }, 'rename');
  };
  const rows = library.data?.map(entry => (
    <li key={entry.id} className="flex min-w-0 flex-wrap items-center gap-2">
      {renaming?.id === entry.id ? (
        <>
          <input
            aria-label={t('settings.midiSoundFont.name', 'SoundFont name')}
            className="min-w-0 flex-1 rounded-md border border-zinc-400/40 bg-transparent px-2 py-1 text-sm"
            autoFocus
            maxLength={255}
            value={renaming.name}
            onChange={event => setRenaming({ id: entry.id, name: event.target.value })}
            onKeyDown={event => {
              if (event.key === 'Enter' || event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
                if (event.key === 'Enter') rename();
                else if (!busy) setRenaming(null);
              }
            }}
            disabled={busy}
          />
          <Button size="sm" disabled={busy || !renaming.name.trim()} onClick={rename}>{t('common.save')}</Button>
          <Button size="sm" disabled={busy} onClick={() => setRenaming(null)}>{t('common.cancel')}</Button>
        </>
      ) : (
        <>
          <span title={entry.name} className="min-w-0 flex-1 truncate text-sm">{entry.name}</span>
          <span className="text-xs text-zinc-500 dark:text-zinc-400">{entry.format.toUpperCase()}</span>
          <Button size="sm" disabled={busy} onClick={() => { setError(null); setRenaming({ id: entry.id, name: entry.name }); }}>
            {t('settings.midiSoundFont.rename', 'Rename')}
          </Button>
          <Button size="sm" disabled={busy} onClick={() => setConfirm(entry)}>
            {t('settings.midiSoundFont.remove', 'Remove')}
          </Button>
        </>
      )}
    </li>
  ));

  return (
    <div className="min-w-0 w-full">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <ParamLabel
          className={compact ? 'text-xs' : 'setting-label'}
          label={label}
          info={t('settings.midiSoundFont.libraryInfo', 'Choose an installed bank or add a General MIDI .sf2 or .sf3 file to this server’s SoundFont library. Your selection is saved in this browser.')}
        />
        <div className="min-w-0 w-64 max-w-full">
          <StyledSelect
            aria-label={label}
            size={size}
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
        <Button size={size} className="shrink-0 whitespace-nowrap" disabled={busy || !preference} onClick={() => picker.current?.click()}>
          {t('settings.midiSoundFont.add', 'Add SoundFont…')}
        </Button>
      </div>
      {rows?.length ? (compact ? (
        <details className="mt-2">
          <summary className="w-fit cursor-pointer text-xs">{t('settings.midiSoundFont.library', 'SoundFont library')}</summary>
          <ul className="mt-2 max-h-60 max-w-xl space-y-1 overflow-y-auto">{rows}</ul>
        </details>
      ) : <ul aria-label={t('settings.midiSoundFont.library', 'SoundFont library')} className="mt-2 max-h-60 max-w-xl space-y-1 overflow-y-auto">{rows}</ul>) : null}
      {missing && (
        <p role="status" className="mt-1 text-xs text-zinc-600 dark:text-zinc-300">
          {t('settings.midiSoundFont.missing', 'Selected SoundFont is not installed. The preview uses the fallback sounds.')}
        </p>
      )}
      {(readError || library.isError || preference?.custom || error) && (
        <p role="alert" className="mt-1 text-xs text-red-600 dark:text-red-400">
          {error ? errors[error]
            : readError ? t('settings.midiSoundFont.readError', 'Could not read the saved SoundFont.')
            : library.isError ? t('settings.midiSoundFont.libraryError', 'Could not load the SoundFont library.')
            : t('settings.midiSoundFont.migrationError', 'Could not move the saved SoundFont into the library. This browser will retry when you reopen the picker.')}
        </p>
      )}
      {confirm && createPortal(
        <ConfirmDialog
          isOpen
          title={t('settings.midiSoundFont.removeTitle', 'Remove SoundFont')}
          message={t('settings.midiSoundFont.removeConfirm', { defaultValue: 'Remove “{{name}}” from the server’s SoundFont library?', name: confirm.name })}
          confirmLabel={t('settings.midiSoundFont.remove', 'Remove')}
          danger
          onCancel={() => setConfirm(null)}
          onConfirm={() => {
            setConfirm(null);
            void update(() => removeLibrarySoundFont(confirm.id), 'remove');
          }}
        />,
        document.body,
      )}
    </div>
  );
}
