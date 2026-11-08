// chips.ts — each target's asset chips and readiness, free of React and the cards.
//
// Kept apart from the plug-ins so the chip matrix is unit-tested without
// loading the training cards. A missing backend block renders OFF.

import type { Sa3DatasetAssets } from '../../../../../server/src/contracts/training';
import type { DatasetAssets } from '../../../services/trainingApi';
import type { TrainingChipContext, TrainingChipSpec, TrainingReadiness } from '../trainingPluginTypes';

type Chips = (assets: DatasetAssets, ctx: TrainingChipContext) => TrainingChipSpec[];
type Ready = (assets: DatasetAssets | undefined) => TrainingReadiness;

/** The asset block arrives with the SA3 asset reader; until then every chip is OFF. */
const sa3Assets = (assets: object | undefined): Sa3DatasetAssets | undefined =>
  (assets as { sa3?: Sa3DatasetAssets } | undefined)?.sa3;

export const aceChips: Chips = (assets, { t, formatWhen }) => [
  {
    id: 'tensors',
    on: assets.tensorVariants > 0,
    label: t('trainingStudio.assets.tensors'),
    suffix: assets.tensorSamples > 0 ? String(assets.tensorSamples) : '',
    title: assets.tensorVariants > 0
      ? t('trainingStudio.assets.tensorsOn', { count: assets.tensorSamples, variant: assets.tensorVariantKey || '?' })
      : t('trainingStudio.assets.tensorsOff'),
  },
  {
    id: 'dit',
    on: !!assets.dit,
    label: t('trainingStudio.assets.dit'),
    title: assets.dit
      ? t('trainingStudio.assets.ditOn', { base: assets.dit.detail || '?', when: formatWhen(assets.dit.trainedAt) || '—' })
      : t('trainingStudio.assets.ditOff'),
  },
  {
    id: 'lm',
    on: !!assets.lm,
    label: t('trainingStudio.assets.lm'),
    suffix: assets.lm?.detail ?? '',
    title: assets.lm
      ? t('trainingStudio.assets.lmOn', { size: assets.lm.detail || '?', when: formatWhen(assets.lm.trainedAt) || '—' })
      : t('trainingStudio.assets.lmOff'),
  },
];

export const aceReadiness: Ready = assets => ({
  prepared: (assets?.tensorVariants ?? 0) > 0,
  trained: !!assets?.dit || !!assets?.lm,
});

export const mm3Chips: Chips = (assets, { t, formatWhen }) => {
  const mm3 = assets.mm3;
  return [
    {
      id: 'codes',
      on: !!mm3?.codesReady,
      label: t('trainingStudio.assets.codes'),
      suffix: mm3 && mm3.codesCount > 0 ? String(mm3.codesCount) : '',
      title: mm3?.codesReady ? t('trainingStudio.assets.codesOn') : t('trainingStudio.assets.codesOff'),
    },
    {
      id: 'lmAdapter',
      on: !!mm3?.lmAdapter,
      label: t('trainingStudio.assets.lmAdapter'),
      title: mm3?.lmAdapter
        ? t('trainingStudio.assets.lmAdapterOn', { when: formatWhen(mm3.lmAdapter.trainedAt) || '—' })
        : t('trainingStudio.assets.lmAdapterOff'),
    },
  ];
};

export const mm3Readiness: Ready = assets => ({ prepared: !!assets?.mm3?.codesReady, trained: !!assets?.mm3?.lmAdapter });

export const yue2Chips: Chips = (assets, { t, formatWhen }) => {
  const yue2 = assets.yue2;
  return [
    {
      id: 'latents',
      on: !!yue2?.latentsReady,
      label: t('trainingStudio.assets.latents'),
      suffix: yue2 && yue2.clips > 0 ? String(yue2.clips) : '',
      title: yue2?.latentsReady
        ? t('trainingStudio.assets.latentsOn', { count: yue2.clips })
        : t('trainingStudio.assets.latentsOff'),
    },
    {
      id: 'codes',
      on: !!yue2?.codesReady,
      label: t('trainingStudio.assets.codes'),
      title: yue2?.codesReady ? t('trainingStudio.assets.codesOn') : t('trainingStudio.assets.codesOff'),
    },
    {
      id: 'cursorSpans',
      on: !!yue2?.cursorReady,
      label: t('trainingStudio.assets.cursorSpans'),
      title: yue2?.cursorReady ? t('trainingStudio.assets.cursorSpansOn') : t('trainingStudio.assets.cursorSpansOff'),
    },
    {
      id: 'narAdapter',
      on: !!yue2?.narAdapter,
      label: t('trainingStudio.assets.narAdapter'),
      title: yue2?.narAdapter
        ? t('trainingStudio.assets.narAdapterOn', { when: formatWhen(yue2.narAdapter.trainedAt) || '—' })
        : t('trainingStudio.assets.narAdapterOff'),
    },
    {
      id: 'arAdapter',
      on: !!yue2?.arAdapter,
      label: t('trainingStudio.assets.arAdapter'),
      title: yue2?.arAdapter
        ? t('trainingStudio.assets.arAdapterOn', { when: formatWhen(yue2.arAdapter.trainedAt) || '—' })
        : t('trainingStudio.assets.arAdapterOff'),
    },
  ];
};

export const yue2Readiness: Ready = assets => ({
  prepared: !!assets?.yue2?.latentsReady && !!assets.yue2.codesReady,
  trained: !!assets?.yue2?.narAdapter || !!assets?.yue2?.arAdapter,
});

export const heartmulaChips: Chips = (assets, { t, formatWhen }) => {
  const hm = assets.heartmula;
  return [
    {
      id: 'codes',
      on: !!hm?.prepared,
      label: t('trainingStudio.assets.heartmulaCodes'),
      suffix: hm && hm.preparedSamples > 0 ? String(hm.preparedSamples) : '',
      title: hm?.prepared
        ? t('trainingStudio.assets.heartmulaCodesOn', { count: hm.preparedSamples })
        : t('trainingStudio.assets.heartmulaCodesOff'),
    },
    {
      id: 'adapter',
      on: !!hm?.adapter,
      label: t('trainingStudio.assets.heartmulaAdapter'),
      title: hm?.adapter
        ? t('trainingStudio.assets.heartmulaAdapterOn', { name: hm.adapter.name, when: formatWhen(hm.adapter.trainedAt) || '—' })
        : t('trainingStudio.assets.heartmulaAdapterOff'),
    },
  ];
};

export const heartmulaReadiness: Ready = assets => ({ prepared: !!assets?.heartmula?.prepared, trained: !!assets?.heartmula?.adapter });

/** "Done" for the dataset list and the Train-multiple wizard: the adapter(s) the family trains are linked. YuE2 needs the AR+NAR pair; HeartMuLa and MM3 their single adapter. A new family adds one line. */
export const TRAINED_BY_FAMILY: Record<'yue2' | 'heartmula' | 'mm3', (a: DatasetAssets | undefined) => boolean> = {
  yue2: a => !!(a?.yue2?.arAdapter && a.yue2.narAdapter),
  heartmula: a => !!a?.heartmula?.adapter,
  mm3: a => !!a?.mm3?.lmAdapter,
};

export const sa3Chips: Chips = (assets, { t, formatWhen }) => {
  const sa3 = sa3Assets(assets);
  return [
    {
      id: 'audio',
      on: !!sa3?.audioPrepared,
      label: t('trainingStudio.assets.sa3Audio'),
      suffix: sa3 && sa3.preparedClips > 0 ? String(sa3.preparedClips) : '',
      title: sa3?.audioPrepared
        ? t('trainingStudio.assets.sa3AudioOn', { count: sa3.preparedClips })
        : t('trainingStudio.assets.sa3AudioOff'),
    },
    {
      id: 'encoded',
      on: !!sa3?.encodingCacheValid,
      label: t('trainingStudio.assets.sa3Encoded'),
      suffix: sa3 && sa3.encodedClips > 0 ? String(sa3.encodedClips) : '',
      title: sa3?.encodingCacheValid
        ? t('trainingStudio.assets.sa3EncodedOn', { count: sa3.encodedClips })
        : t('trainingStudio.assets.sa3EncodedOff'),
    },
    {
      id: 'snapshots',
      on: (sa3?.snapshots ?? 0) > 0,
      label: t('trainingStudio.assets.sa3Snapshots'),
      suffix: sa3 && sa3.snapshots > 0 ? String(sa3.snapshots) : '',
      title: sa3 && sa3.snapshots > 0
        ? t('trainingStudio.assets.sa3SnapshotsOn', { count: sa3.snapshots })
        : t('trainingStudio.assets.sa3SnapshotsOff'),
    },
    {
      id: 'adapter',
      on: !!sa3?.adapter,
      label: t('trainingStudio.assets.sa3Adapter'),
      title: sa3?.adapter
        ? t('trainingStudio.assets.sa3AdapterOn', { name: sa3.adapter.name, when: formatWhen(sa3.adapter.trainedAt) || '—' })
        : t('trainingStudio.assets.sa3AdapterOff'),
    },
  ];
};

export const sa3Readiness: Ready = assets => {
  const sa3 = sa3Assets(assets);
  return { prepared: !!sa3?.audioPrepared && !!sa3.encodingCacheValid, trained: !!sa3?.adapter };
};
