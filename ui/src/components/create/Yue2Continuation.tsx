import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useUpload } from '../../queries/writes';
import { ParamLabel } from '../shared/ParamLabel';
import { Toggle } from '../shared/Toggle';
import type { Yue2Continuation as Section } from '../../../../server/src/contracts/yue2Continuation';

export const Yue2Continuation: React.FC<{ value?: Section; onChange: (value?: Section) => void }> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const upload = useUpload<{ audio_url: string }>();
  const [error, setError] = useState('');
  const [lastSource, setSource] = useState<Section>({ sourceAudioUrl: '', start: 0, end: 10 });
  const source = value ?? lastSource;
  const update = (patch: Partial<Section>) => {
    const next = { ...source, ...patch };
    setSource(next);
    onChange(next);
  };
  return <div className="space-y-2 rounded-xl border border-zinc-200 dark:border-white/10 p-3">
    <div className="flex items-center justify-between gap-2">
      <ParamLabel label={t('yue2Continuation.label')} info={t('yue2Continuation.info')} />
      <Toggle checked={!!value} disabled={upload.isPending} onChange={on => onChange(on ? source : undefined)} aria-label={t('yue2Continuation.label')} />
    </div>
    {value && <>
      <input type="file" accept="audio/*" disabled={upload.isPending} aria-label={t('yue2Continuation.upload')}
        onChange={async e => {
          const file = e.target.files?.[0];
          if (!file) return;
          setError('');
          try {
            const result = await upload.mutateAsync({ path: '/upload/audio', files: [{ field: 'audio', file }] });
            update({ sourceAudioUrl: result.audio_url });
          } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
        }} className="w-full text-xs" />
      {source.sourceAudioUrl && <audio controls src={source.sourceAudioUrl} className="w-full" />}
      <div className="grid grid-cols-2 gap-2">
        {(['start', 'end'] as const).map(key => <div key={key}>
          <ParamLabel label={t(`yue2Continuation.${key}`)} info={t('yue2Continuation.rangeInfo')} />
          <input type="number" min={0} step={0.04} value={source[key]} disabled={upload.isPending} aria-label={t(`yue2Continuation.${key}`)}
            onChange={e => update({ [key]: e.target.valueAsNumber })}
            className="w-full rounded-lg border border-zinc-200 dark:border-white/10 bg-transparent px-2 py-1 text-sm" />
        </div>)}
      </div>
      <p className="text-xs text-zinc-500">{t('yue2Continuation.hint')}</p>
      {upload.isPending && <p className="text-xs">{t('yue2Continuation.uploading')}</p>}
      {error && <p role="alert" className="text-xs text-red-500">{error}</p>}
    </>}
  </div>;
};
