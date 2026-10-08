// PresetSyncNotices.tsx — what a PresetCollection needs the user to decide:
// a legacy preset that collides with a saved one (keep both or replace), and
// any preset whose save or delete failed or was refused because it changed
// elsewhere (refresh, reapply or discard). The user's content stays in the
// entry until they choose.
import type { PresetCollection, PresetCollectionSnapshot } from '../../services/presetCollection';

interface Props<B extends Record<string, unknown>> {
  collection: PresetCollection<B>;
  snapshot: PresetCollectionSnapshot<B>;
  labelOf: (body: B) => string;
  compact?: boolean;
}

const btn = 'underline hover:no-underline flex-shrink-0';

export function PresetSyncNotices<B extends Record<string, unknown>>({ collection, snapshot, labelOf, compact }: Props<B>) {
  const text = compact ? 'text-[9px]' : 'text-[10px]';
  const problems = snapshot.entries.filter(e => e.error);
  if (!snapshot.loadError && snapshot.importConflicts.length === 0 && problems.length === 0) return null;
  return (
    <div className={`space-y-1 ${text}`} role="alert">
      {snapshot.loadError && (
        <div className="flex items-center gap-2 px-2 py-1 rounded-md bg-red-500/10 border border-red-500/20 text-red-500 dark:text-red-400">
          <span className="flex-1">Presets could not be loaded: {snapshot.loadError}</span>
          <button className={btn} onClick={() => void collection.load()}>Retry</button>
        </div>
      )}
      {snapshot.importConflicts.map(c => (
        <div key={c.storageKey} className="flex items-center gap-2 px-2 py-1 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400">
          <span className="flex-1">
            A preset saved in this browser, &ldquo;{c.name}&rdquo;, has the same name as a different saved preset.
            {c.error ? ` ${c.error}` : ''}
          </span>
          <button className={btn} onClick={() => void collection.resolveImport(c.storageKey, 'keep-both')}>Keep both</button>
          <button className={btn} onClick={() => void collection.resolveImport(c.storageKey, 'replace')}>Replace saved</button>
        </div>
      ))}
      {problems.map(e => (
        <div key={e.key} className="flex items-center gap-2 px-2 py-1 rounded-md bg-red-500/10 border border-red-500/20 text-red-500 dark:text-red-400">
          <span className="flex-1">&ldquo;{labelOf(e.body)}&rdquo;: {e.error}</span>
          {e.status === 'conflict' && (
            <button className={btn} onClick={() => void collection.refresh().catch(() => {})}>Refresh</button>
          )}
          {(e.status === 'conflict' || e.status === 'failed') && (
            <button className={btn} onClick={() => void collection.reapply(e.key)}>
              {e.pendingOp === 'delete' ? 'Delete again' : 'Reapply'}
            </button>
          )}
          {(e.status === 'conflict' || e.status === 'failed') && (
            <button className={btn} onClick={() => collection.discard(e.key)}>Discard</button>
          )}
        </div>
      ))}
    </div>
  );
}
