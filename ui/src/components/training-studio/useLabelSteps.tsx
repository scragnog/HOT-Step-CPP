// useLabelSteps.tsx — the labelling step controls (BPM/key and their sources,
// Genius, local lyrics, captioner), shared by LabelPanel and the batch import
// wizard so a bulk run offers exactly what a single dataset does. Each step
// degrades to disabled + an explanation when its capability is missing.

import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { LabelOptions } from '../../services/trainingApi';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { useTrainingStore } from '../../stores/trainingStore';

const DEFAULTS = {
  useEssentia: true,
  // beat_this and SheetSage2 beat Essentia on tempo and key (HOTSTEP-730);
  // Essentia stays as the fallback for anything they cannot answer.
  tempoSource: 'beat_this' as 'beat_this' | 'essentia',
  keySource: 'sheetsage' as 'sheetsage' | 'essentia',
  minTempo: '',
  maxTempo: '',
  // Off by default (2026-10-24): lyrics come from the audio (local lyrics below).
  useGenius: false,
  // Music Flamingo fills the lyrics Genius could not find (section-tagged).
  localLyrics: true,
  useCaption: true,
  // '' = the local default (MOSS, else Music Flamingo), else the server's provider.
  captionProvider: '',
  // '' = the provider's default model (config-level).
  captionModel: '',
};

export type LabelStepOptions = Pick<LabelOptions,
  'useEssentia' | 'tempoSource' | 'keySource' | 'essentiaTempo' | 'useGenius' | 'localLyrics' | 'useCaption' | 'caption'>;

export function useLabelSteps() {
  const { t } = useTranslation();
  const caps = useTrainingStore(s => s.capabilities);
  const [s, setS] = useState(DEFAULTS);
  const set = <K extends keyof typeof DEFAULTS>(k: K) => (v: (typeof DEFAULTS)[K]) => setS(prev => ({ ...prev, [k]: v }));
  const reset = useCallback(() => setS(DEFAULTS), []);

  const essentiaOk = caps?.essentia.available !== false;
  const geniusOk = !!caps?.genius.configured;
  // MOSS needs no API key, so an unconfigured cloud provider must not disable
  // captioning outright — that hid the only offline option exactly when it was
  // the only one available.
  const mossOk = !!caps?.moss.available;
  const flamingoOk = !!caps?.flamingo?.available;
  const captionOk = !!caps?.llm.configured || mossOk || flamingoOk;
  // The local default: MOSS first (the established captioner), else Flamingo.
  const localDefault = mossOk ? 'moss' : flamingoOk ? 'flamingo' : '';

  const effectiveEssentia = s.useEssentia && essentiaOk;
  const effectiveGenius = s.useGenius && geniusOk;
  const effectiveCaption = s.useCaption && captionOk;
  const effectiveLocalLyrics = s.localLyrics && flamingoOk;
  const anyStep = effectiveEssentia || effectiveGenius || effectiveCaption || effectiveLocalLyrics;
  const chosenProvider = s.captionProvider || localDefault;
  // What this run would need the card for: local captioning or lyric transcription.
  const needsEngine = effectiveLocalLyrics
    || (effectiveCaption && (chosenProvider === 'moss' || chosenProvider === 'flamingo'));
  const minBpm = s.minTempo ? Number(s.minTempo) : undefined;
  const maxBpm = s.maxTempo ? Number(s.maxTempo) : undefined;
  const invalidTempoBounds = effectiveEssentia && (
    (minBpm !== undefined && (!Number.isInteger(minBpm) || minBpm < 40 || minBpm > 180)) ||
    (maxBpm !== undefined && (!Number.isInteger(maxBpm) || maxBpm < 60 || maxBpm > 250)) ||
    (minBpm !== undefined && maxBpm !== undefined && minBpm >= maxBpm)
  );

  const options: LabelStepOptions = {
    useEssentia: effectiveEssentia,
    tempoSource: s.tempoSource,
    keySource: s.keySource,
    ...(effectiveEssentia && (minBpm !== undefined || maxBpm !== undefined) ? { essentiaTempo: {
      ...(minBpm !== undefined ? { minTempo: minBpm } : {}),
      ...(maxBpm !== undefined ? { maxTempo: maxBpm } : {}),
    } } : {}),
    useGenius: effectiveGenius,
    localLyrics: effectiveLocalLyrics,
    useCaption: effectiveCaption,
    // Without this the server falls back to config.lireek.defaultProvider,
    // which is why labeling always went to Gemini even with MOSS installed.
    caption: {
      provider: chosenProvider || undefined,
      ...(s.captionModel ? { model: s.captionModel } : {}),
    },
  };

  const fields = (
    <div className="flex flex-col gap-2">
      <Toggle
        accent="amber"
        checked={effectiveEssentia}
        disabled={!essentiaOk}
        onChange={set('useEssentia')}
        label={t('trainingStudio.label.useEssentia')}
        info={t('trainingStudio.label.useEssentiaInfo')}
        defaultValue={true}
      />
      {!essentiaOk && (
        <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.caps.essentiaMissing')}</div>
      )}
      {effectiveEssentia && (
        <div className="ml-6 flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1">
            <ParamLabel label={t('trainingStudio.label.tempoSource')} info={t('trainingStudio.label.tempoSourceInfo')}
              className="text-[11px] text-zinc-500" />
            <StyledSelect accent="amber" size="sm" className="w-auto" value={s.tempoSource} onChange={set('tempoSource')}
              options={[{ value: 'beat_this', label: 'beat_this' }, { value: 'essentia', label: 'Essentia' }]} />
          </div>
          <div className="flex flex-col gap-1">
            <ParamLabel label={t('trainingStudio.label.keySource')} info={t('trainingStudio.label.keySourceInfo')}
              className="text-[11px] text-zinc-500" />
            <StyledSelect accent="amber" size="sm" className="w-auto" value={s.keySource} onChange={set('keySource')}
              options={[{ value: 'sheetsage', label: 'SheetSage2' }, { value: 'essentia', label: 'Essentia' }]} />
          </div>
        </div>
      )}
      {effectiveEssentia && (
        <div className="ml-6 flex flex-wrap items-end gap-2">
          {([['min', 'minTempo', 40, 180], ['max', 'maxTempo', 60, 250]] as const).map(([kind, key, min, max]) => (
            <label key={kind} className="flex flex-col gap-1 text-[11px] text-zinc-600 dark:text-zinc-400">
              {t(`trainingStudio.label.${kind}Tempo`, `${kind === 'min' ? 'Minimum' : 'Maximum'} tempo (optional)`)}
              <input type="number" min={min} max={max} step="1" value={s[key]}
                onChange={e => set(key)(e.target.value)}
                className="w-28 rounded border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-900 dark:border-white/10 dark:bg-suno-card dark:text-white" />
            </label>
          ))}
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.label.tempoBoundsInfo', 'Leave blank for Essentia defaults.')}</span>
          {invalidTempoBounds && <span className="text-[11px] text-red-500">{t('trainingStudio.label.tempoBoundsError', 'Use 40–180 for minimum, 60–250 for maximum, with minimum below maximum.')}</span>}
        </div>
      )}

      <Toggle
        accent="amber"
        checked={effectiveGenius}
        disabled={!geniusOk}
        onChange={set('useGenius')}
        label={t('trainingStudio.label.useGenius')}
        info={t('trainingStudio.label.useGeniusInfo')}
        defaultValue={false}
      />
      {!geniusOk && (
        <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.geniusMissing')}</div>
      )}

      <Toggle
        accent="amber"
        checked={effectiveLocalLyrics}
        disabled={!flamingoOk}
        onChange={set('localLyrics')}
        label={t('trainingStudio.label.localLyrics', 'Transcribe missing lyrics locally (Music Flamingo)')}
        info={t('trainingStudio.label.localLyricsInfo', 'Each track with no lyrics (after Genius, if that is on) is transcribed from its audio by Music Flamingo, with [Verse]/[Chorus] section tags. Tracks marked instrumental are left alone, and so are lyrics already in a sidecar unless the merge policy overwrites lyrics, which re-transcribes them (the way to move a dataset off Genius). A transcript that loops is redone once with a light repetition penalty. Runs on your GPU, about 5 seconds a track on a fast card.')}
        defaultValue={true}
      />
      {!flamingoOk && caps?.flamingo?.missing && (
        <div className="ml-6 text-[11px] text-zinc-500">
          {t('trainingStudio.label.flamingoMissing', 'Music Flamingo (local) unavailable')}: {caps.flamingo.missing}
        </div>
      )}

      <Toggle
        accent="amber"
        checked={effectiveCaption}
        disabled={!captionOk}
        onChange={set('useCaption')}
        label={t('trainingStudio.label.useCaption')}
        info={t('trainingStudio.label.useCaptionInfo')}
        defaultValue={true}
      />
      {effectiveCaption && (mossOk || flamingoOk || caps?.llm.configured) && (() => {
        const cloud = caps?.llm.providers.filter(pr => pr.available) ?? [];
        const sel = chosenProvider || caps?.llm.defaultProvider || '';
        const active = cloud.find(pr => pr.id === sel);
        return (
          <div className="ml-6 flex flex-wrap items-center gap-2">
            <ParamLabel
              label={t('trainingStudio.label.captioner')}
              info={t('trainingStudio.label.captionerInfo')}
              className="text-[11px] text-zinc-500"
            />
            <StyledSelect
              accent="amber"
              size="sm"
              value={sel}
              onChange={(v) => setS(prev => ({ ...prev, captionProvider: v, captionModel: '' }))}
              className="w-auto"
              /* Two independent facts: WHERE the model runs, which the provider
                 declares rather than the UI inferring from `id === 'gemini'`,
                 and WHETHER it receives the audio, which only Gemini and the
                 local captioners do. */
              options={[
                ...(mossOk ? [{ value: 'moss', label: 'MOSS — local, hears the audio' }] : []),
                ...(flamingoOk ? [{ value: 'flamingo', label: 'Music Flamingo — local, hears the audio' }] : []),
                ...cloud.map(pr => ({
                  value: pr.id,
                  label: `${pr.name} (${pr.local ? 'local' : 'cloud'}, ${pr.id === 'gemini' ? 'hears the audio' : 'text only'})`,
                })),
              ]}
            />
            {/* Model picker for cloud providers — Gemini's list is fetched
                live from the API, so new models appear by themselves. */}
            {active && active.models.length > 0 && (
              <StyledSelect
                accent="amber"
                size="sm"
                value={s.captionModel || active.defaultModel}
                onChange={set('captionModel')}
                className="w-auto"
                options={active.models.map(m => ({ value: m, label: m }))}
              />
            )}
          </div>
        );
      })()}
      {/* Say why MOSS is not on offer: the server works out whether the binary
          or the weights are missing, and the user's next action differs. */}
      {effectiveCaption && !mossOk && caps?.moss.missing && (
        <div className="ml-6 text-[11px] text-zinc-500">
          MOSS (local) unavailable: {caps.moss.missing}
        </div>
      )}
      {!captionOk && (
        <div className="ml-6 text-[11px] text-amber-600 dark:text-amber-400">{t('trainingStudio.enhance.captionMissing')}</div>
      )}
    </div>
  );

  return { fields, options, anyStep, needsEngine, invalidTempoBounds, reset };
}
