import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { backendModelsOptions } from '@/queries/global-models/options';
import { useBackendStore } from '@/stores/backendStore';
import { useGlobalParamsStore } from '@/stores/globalParamsStore';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { BackendExtensionControls } from './BackendExtensionControls';
import { isPresent } from '@/utils/guards';

const fileName = (value: string) => value.split(/[\\/]/).pop() || value;

export function HeartmulaAdapterDropdown() {
  const { t } = useTranslation();
  const backend = useBackendStore(s => s.activeBackendId);
  const selectModels = useBackendStore(s => s.selectModels);
  const { data, error, isPending } = useQuery(backendModelsOptions(backend));
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const selected = String(data?.defaults?.lmAdapter ?? '');
  const adapters = data?.lmAdapters ?? [];
  // A pick whose file was deleted or renamed: shown, not silently read as None (YuE2 does the same).
  const missing = !!selected && !isPending && !error && !adapters.includes(selected);
  const meta = data?.lmAdapterMeta?.[selected];
  const hasMlp = !!meta?.mlp;
  // Trained on another variant: the engine re-bases it onto the loaded one (needs the home model installed).
  const loadedVariant = String(data?.defaults?.variant || '3B');
  const home = meta?.variant && meta.variant.toLowerCase() !== loadedVariant.toLowerCase() ? meta.variant : '';
  const hidden = [...(hasMlp ? [] : ['heartmulaAdapterScaleMlp']), ...(home ? [] : ['heartmulaAdapterRebase'])];
  return (
    <div className="space-y-3">
      <ParamLabel label={t('globalBar.heartmulaAdapter', 'HeartMuLa adapter')}
        info={t('globalBar.heartmulaAdapterInfo', 'Applies a compatible LoRA to the global language model. None uses the base model. Incompatible adapters are refused when selected.')}
        rootClassName="flex mb-1.5" />
      <StyledSelect accent="emerald" className="w-full" value={selected}
        disabled={busy || isPending || !!error || !data?.selectable}
        options={[
          { value: '', label: t('globalBar.heartmulaAdapterNone', 'None, base model') },
          ...(missing ? [{ value: selected, label: `${fileName(selected)} ${t('globalBar.fileNotFound', '(file not found)')}` }] : []),
          ...adapters.map(value => {
            const m = data?.lmAdapterMeta?.[value];
            const tags = [m?.trigger ? `"${m.trigger}"` : '', m?.variant && m.variant.toLowerCase() !== loadedVariant.toLowerCase() ? m.variant : '', m?.mlp ? 'MLP' : ''].filter(isPresent);
            return { value, label: [fileName(value), ...tags].join(' · ') };
          }),
        ]}
        onChange={async value => {
          setBusy(true);
          try { setFailed(!(await selectModels({ lmAdapter: value }, backend))); }
          finally { setBusy(false); }
        }} />
      {error && <p className="text-xs text-amber-500">{error.message}</p>}
      {failed && <p className="text-xs text-amber-500">{t('globalBar.heartmulaAdapterSelectFailed', 'Could not switch adapter; the previous one is still selected.')}</p>}
      {missing && <p className="text-xs text-amber-500">{t('globalBar.heartmulaAdapterMissing', 'This adapter file is no longer installed. Pick another one, or None, before generating.')}</p>}
      {!isPending && !error && !adapters.length && (
        <p className="text-xs text-zinc-500">{t('globalBar.heartmulaNoAdapters', 'No HeartMuLa adapters installed yet. Train one on the Train page, or convert a PEFT LoRA with engine/tools/convert-heartmula-lora.py into the heartmula-lm-adapters folder.')}</p>
      )}
      {selected && !missing && <>
        {meta?.trigger && <p className="text-[11px] text-zinc-500">{t('globalBar.heartmulaAdapterTrigger', 'Trigger word "{{trigger}}" is added to the front of your tags, as in every training sample.', { trigger: meta.trigger })}</p>}
        {home && <p className="text-[11px] text-emerald-600 dark:text-emerald-400">{t('globalBar.heartmulaAdapterRebased', 'Trained on {{home}}: re-based onto {{loaded}}. Re-base strength sets how far its layers move toward {{home}}.', { home, loaded: loadedVariant })}</p>}
        <BackendExtensionControls group="adapter" accentColor="emerald" hide={hidden} />
        {!hasMlp && <p className="text-[11px] text-zinc-500">{t('globalBar.heartmulaAdapterNoMlp', 'MLP: this adapter was trained on attention only, so it has no MLP dial. Turn on "Train MLP layers" when training to get one.')}</p>}
      </>}
    </div>
  );
}

export function HeartmulaAdapterBadge() {
  const selected = useBackendStore(s => s.models[s.activeBackendId]?.defaults?.lmAdapter);
  const strength = useGlobalParamsStore(s => s.backendParams?.heartmulaAdapterStrength ?? 1);
  const { t } = useTranslation();
  return <span className="text-[10px] text-zinc-500 font-mono truncate">{selected
    ? `${fileName(String(selected))} · ${strength}`
    : t('globalBar.heartmulaAdapterNone', 'None, base model')}</span>;
}
