// Yue2ClearPreparedCard.tsx — "clear prepared data", beside "Perform all stages"
// on the YuE2 Prepare page. The Joint Training card still owns the stored form
// and prepare keys this resets after a full wipe.
//
// By default it keeps YuE2's core prepared data (latents, codes, lead sheets,
// alignment, prepared set): ~40 MiB an album that costs minutes of GPU to
// rebuild, and rebuilt anyway when it goes stale. It deletes the vocal stems
// and the other backends' caches. A toggle deletes the core too.

import React, { useEffect, useState } from 'react';
import { Trash2 } from 'lucide-react';
import { clearPreparedData, getPreparedData, type PreparedCache } from '../../services/trainingApi';
import { Toggle } from '../shared/Toggle';
import { FORM_KEY, PREP_KEY } from './Yue2AitkTrainCard';

const CARD = 'rounded-xl border border-red-500/20 bg-white dark:bg-suno-card p-4';
const mib = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MiB`;

export const Yue2ClearPreparedCard: React.FC<{ datasetId: string; disabled: boolean }> = ({ datasetId, disabled }) => {
  const [info, setInfo] = useState<{ slug: string; caches: PreparedCache[]; busy: boolean } | null>(null);
  const [clearing, setClearing] = useState(false);
  const [includeCore, setIncludeCore] = useState(false);
  const [note, setNote] = useState('');

  const load = () => getPreparedData(datasetId).then(setInfo).catch(() => setInfo(null));
  useEffect(() => { setIncludeCore(false); void load(); }, [datasetId]);

  const targets = info ? info.caches.filter(c => includeCore || !c.core) : [];
  const kept = info ? info.caches.filter(c => c.core && !includeCore) : [];
  const clear = async () => {
    if (!info || clearing || !targets.length) return;
    const summary = targets.map(item => `${item.name}: ${item.files} files, ${mib(item.bytes)}`).join('\n');
    const keptLine = kept.length ? `\n\nKept: the YuE2 latents, codes, lead sheets and prepared set (${mib(kept.reduce((s, c) => s + c.bytes, 0))}).` : '';
    if (!window.confirm(`Clear prepared data for ${info.slug}?\n\n${summary}${keptLine}\n\nSource tracks, sidecars, labels and adapters will remain.`)) return;
    setClearing(true); setNote('');
    try {
      await clearPreparedData(datasetId, info.slug, includeCore);
      if (includeCore) {
        // The prepared manifest the form points at is gone with the core.
        try {
          const form = JSON.parse(window.localStorage.getItem(`${FORM_KEY}${datasetId}`) || '{}') as Record<string, unknown>;
          window.localStorage.setItem(`${FORM_KEY}${datasetId}`, JSON.stringify({ ...form, resume: '', dataset: '' }));
        } catch { /* no stored form */ }
        window.localStorage.removeItem(`${PREP_KEY}${datasetId}:manifest`);
        window.localStorage.removeItem(`${PREP_KEY}${datasetId}:applied`);
        window.location.reload();
        return;
      }
      setNote('Cleared.');
      void load();
    } catch (err) {
      setNote(err instanceof Error ? err.message : String(err));
      void load();
    } finally { setClearing(false); }
  };

  const bytes = targets.reduce((sum, item) => sum + item.bytes, 0);
  return (
    <div className={CARD}>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
        Clears the vocal stems and this dataset&apos;s MM3 and ACE caches. The YuE2 latents, codes, lead sheets and prepared set are
        kept: they are small, slow to rebuild, and rebuilt anyway when the audio, captions or loudness change.
        Source files, labels and adapters are always kept.
      </p>
      <div className="mb-3">
        <Toggle
          accent="amber"
          checked={includeCore}
          disabled={clearing || disabled}
          onChange={setIncludeCore}
          label="Also delete the YuE2 latents, codes, lead sheets and prepared set"
          info="For a deliberate rebuild from scratch. The next preparation runs every stage again: about 3-5 minutes of GPU per album, most of it the lead sheets. Older runs may then be unavailable to resume."
        />
      </div>
      <button type="button" onClick={() => void clear()}
        disabled={!targets.length || info?.busy || clearing || disabled}
        className="px-4 py-2 rounded-lg text-sm font-semibold border border-red-500/50 text-red-600 dark:text-red-400 hover:bg-red-500/10 disabled:opacity-40 flex items-center gap-2">
        <Trash2 size={15} />{clearing ? 'Clearing…' : includeCore ? 'Clear all prepared data' : 'Clear stems and other caches'}
      </button>
      {info && <p className="mt-2 text-[11px] text-zinc-500">{targets.length} cache folders · {mib(bytes)}{kept.length ? ` · keeps ${mib(kept.reduce((s, c) => s + c.bytes, 0))}` : ''}</p>}
      {note && <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">{note}</p>}
    </div>
  );
};
