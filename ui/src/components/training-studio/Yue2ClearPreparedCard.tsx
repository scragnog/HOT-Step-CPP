// Yue2ClearPreparedCard.tsx — "wipe it all", beside "Perform all stages" at the
// top of the YuE2 Train page. Moved out of the Joint Training card, which still
// owns the stored form and prepare keys this clears after a wipe.

import React, { useEffect, useState } from 'react';
import { Trash2 } from 'lucide-react';
import { clearPreparedData, getPreparedData, type PreparedCache } from '../../services/trainingApi';
import { FORM_KEY, PREP_KEY } from './Yue2AitkTrainCard';

const CARD = 'rounded-xl border border-red-500/20 bg-white dark:bg-suno-card p-4';

export const Yue2ClearPreparedCard: React.FC<{ datasetId: string; disabled: boolean }> = ({ datasetId, disabled }) => {
  const [info, setInfo] = useState<{ slug: string; caches: PreparedCache[]; busy: boolean } | null>(null);
  const [clearing, setClearing] = useState(false);
  const [note, setNote] = useState('');

  useEffect(() => {
    let cancelled = false;
    void getPreparedData(datasetId).then(value => { if (!cancelled) setInfo(value); })
      .catch(() => { if (!cancelled) setInfo(null); });
    return () => { cancelled = true; };
  }, [datasetId]);

  const clear = async () => {
    if (!info || clearing) return;
    const summary = info.caches.map(item => `${item.name}: ${item.files} files, ${(item.bytes / 1048576).toFixed(1)} MiB`).join('\n');
    if (!window.confirm(`Clear all prepared data for ${info.slug}?\n\n${summary || 'No generated caches found.'}\n\nSource tracks, sidecars, labels and adapters will remain.`)) return;
    setClearing(true); setNote('');
    try {
      await clearPreparedData(datasetId, info.slug);
      // The prepared manifest the form points at is gone with the caches.
      try {
        const form = JSON.parse(window.localStorage.getItem(`${FORM_KEY}${datasetId}`) || '{}') as Record<string, unknown>;
        window.localStorage.setItem(`${FORM_KEY}${datasetId}`, JSON.stringify({ ...form, resume: '', dataset: '' }));
      } catch { /* no stored form */ }
      window.localStorage.removeItem(`${PREP_KEY}${datasetId}:manifest`);
      window.localStorage.removeItem(`${PREP_KEY}${datasetId}:applied`);
      window.location.reload();
    } catch (err) {
      setNote(err instanceof Error ? err.message : String(err));
      void getPreparedData(datasetId).then(setInfo).catch(() => {});
    } finally { setClearing(false); }
  };

  const bytes = info ? info.caches.reduce((sum, item) => sum + item.bytes, 0) : 0;
  return (
    <div className={CARD}>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        Clears the generated latents, codes, lead sheets, alignment, stems, MM3 caches and ACE tensors for this dataset.
        Source files, labels and adapters are kept. Older runs may then be unavailable to resume.
      </p>
      <button type="button" onClick={() => void clear()}
        disabled={!info?.caches.length || info.busy || clearing || disabled}
        className="px-4 py-2 rounded-lg text-sm font-semibold border border-red-500/50 text-red-600 dark:text-red-400 hover:bg-red-500/10 disabled:opacity-40 flex items-center gap-2">
        <Trash2 size={15} />{clearing ? 'Clearing…' : 'Clear all prepared data'}
      </button>
      {info && <p className="mt-2 text-[11px] text-zinc-500">{info.caches.length} cache folders · {(bytes / 1048576).toFixed(1)} MiB</p>}
      {note && <p className="mt-1 text-xs text-red-600 dark:text-red-400">{note}</p>}
    </div>
  );
};
