import React from 'react';
import type { Yue2SectionMatch } from '../../services/yue2CoverApi';

type Row = { kind: 'same' | 'del' | 'add'; text: string };

/** Line diff by longest common subsequence; lyrics are a few hundred lines at most. */
export function lineDiff(before: string, after: string): Row[] {
  const a = before.replace(/\r\n?/g, '\n').split('\n'), b = after.split('\n');
  const lcs = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--)
    lcs[i][j] = a[i] === b[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
  const rows: Row[] = [];
  let i = 0, j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) { rows.push({ kind: 'same', text: a[i] }); i++; j++; }
    else if (j < b.length && (i === a.length || lcs[i][j + 1] >= lcs[i + 1][j])) rows.push({ kind: 'add', text: b[j++] });
    else rows.push({ kind: 'del', text: a[i++] });
  }
  return rows;
}

interface Props {
  before: string;
  match: Yue2SectionMatch;
  /** The lyrics were edited after matching. */
  stale: boolean;
  saving: boolean;
  onApply: () => void;
  onSave: () => void;
  onDiscard: () => void;
}

export const SectionMatchReview: React.FC<Props> = ({ before, match, stale, saving, onApply, onSave, onDiscard }) => {
  const unsure = match.blocks.filter(b => b.status === 'unsure');
  const copied = match.filled.filter(f => f.copiedFrom !== null);
  return (
    <div className="rounded-lg border border-cyan-500/30 bg-cyan-500/5 p-2 space-y-2">
      <p className="text-xs text-zinc-600 dark:text-zinc-400">
        {match.unchanged ? 'The lyric tags already match the score.' : 'Proposed tags from where the source vocal sings each block. Words are unchanged.'}
      </p>
      {unsure.length > 0 && <p role="status" className="text-xs text-amber-700 dark:text-amber-300">
        Left as they were (the aligner was unsure where they are sung): {unsure.map(b => b.tag ? `[${b.tag}]` : `block ${b.index}`).join(', ')}. Check these by hand.
      </p>}
      {copied.length > 0 && <p className="text-xs text-zinc-600 dark:text-zinc-400">
        Copied for repeats the lyrics left out: {copied.map(f => `[${f.label}]`).join(', ')}.
      </p>}
      {!match.unchanged && <pre aria-label="Section changes" className="max-h-48 overflow-y-auto rounded bg-white dark:bg-black/30 p-2 text-[11px] font-mono leading-snug whitespace-pre-wrap">
        {lineDiff(before, match.lyrics).map((row, index) => (
          <div key={index} className={row.kind === 'add' ? 'bg-emerald-500/15 text-emerald-800 dark:text-emerald-300'
            : row.kind === 'del' ? 'bg-red-500/15 text-red-700 dark:text-red-300 line-through' : 'text-zinc-600 dark:text-zinc-400'}>
            {row.kind === 'add' ? '+ ' : row.kind === 'del' ? '- ' : '  '}{row.text || ' '}
          </div>
        ))}
      </pre>}
      {stale && <p role="alert" className="text-xs text-amber-700 dark:text-amber-300">
        The lyrics changed after matching. Match again before applying.
      </p>}
      <div className="flex flex-wrap gap-2">
        {!match.unchanged && <button type="button" onClick={onApply} disabled={stale}
          className="disabled:opacity-40 rounded bg-cyan-600 px-2 py-1 text-xs font-semibold text-white">Apply</button>}
        {match.datasetSong && <button type="button" onClick={onSave} disabled={saving || stale}
          className="rounded bg-emerald-600 px-2 py-1 text-xs font-semibold text-white disabled:opacity-40">
          {saving ? 'Saving…' : match.unchanged ? 'Save to dataset' : 'Apply and save to dataset'}
        </button>}
        <button type="button" onClick={onDiscard} className="rounded px-2 py-1 text-xs text-zinc-600 dark:text-zinc-400">Discard</button>
      </div>
    </div>
  );
};
