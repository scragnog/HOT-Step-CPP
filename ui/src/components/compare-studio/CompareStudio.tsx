// CompareStudio.tsx - A/B listening sessions: list, new-session picker, session view.

import React, { useDeferredValue, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Plus, Search, X, GitCompare } from 'lucide-react';
import { compareQueries } from '../../queries/compare/queries';
import { useCreateCompareSession } from '../../queries/compare/writes';
import {
  takePendingCompareSongs,
  type CompareSessionSummary, type CompareStatus,
} from '../../services/compareApi';
import { SessionView } from './SessionView';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { QueryError } from '../shared/QueryError';

const DOT: Record<CompareStatus, string> = {
  scored: 'bg-emerald-400', partial: 'bg-amber-400', unscored: 'bg-zinc-400',
};

const PAGE = 25;
const inputCls = 'w-full h-10 px-3 rounded-lg text-sm bg-white dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-zinc-900 dark:text-zinc-100';

// ── New comparison picker ────────────────────────────────────────────────────

const NewSessionPanel: React.FC<{ initialIds: string[]; onClose: () => void; onCreated: (id: string) => void }> = ({ initialIds, onClose, onCreated }) => {
  const { t } = useTranslation();
  const [name, setName] = useState('');
  const [group, setGroup] = useState('');
  const [q, setQ] = useState('');
  const [offset, setOffset] = useState(0);
  const [picked, setPicked] = useState<string[]>(initialIds);
  const [titles, setTitles] = useState<Record<string, string>>({});
  const create = useCreateCompareSession();

  // React defers the query string while typing; keepPreviousData holds the
  // current page so the list never blanks between keystrokes.
  const deferredQ = useDeferredValue(q);
  const songsQuery = useQuery(compareQueries.songSearch(deferredQ, PAGE, offset));
  const hits = songsQuery.data?.songs ?? [];
  const total = songsQuery.data?.total ?? 0;

  const toggle = (id: string) => {
    setPicked(p => (p.includes(id) ? p.filter(x => x !== id) : [...p, id]));
    const hit = hits.find(h => h.id === id);
    if (hit) setTitles(prev => (prev[id] ? prev : { ...prev, [id]: hit.title }));
  };

  const submit = () => {
    if (picked.length < 2 || create.isPending) return;
    create.mutate({
      name: name.trim() || t('compare.defaultName', 'Comparison {{date}}', { date: new Date().toLocaleString() }),
      group: group.trim(), songIds: picked,
    }, { onSuccess: s => onCreated(s.id) });
  };

  return (
    <div className="flex flex-col gap-3 p-3 sm:p-4 max-w-3xl w-full mx-auto">
      <div className="flex items-center gap-2">
        <h2 className="text-base font-bold flex-1 text-zinc-900 dark:text-zinc-100">{t('compare.newTitle', 'New comparison')}</h2>
        <button onClick={onClose} className="p-2 rounded-lg text-zinc-500 hover:text-pink-400" aria-label={t('compare.cancel', 'Cancel')}><X size={18} /></button>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <ParamLabel label={t('compare.name', 'Name')} info={t('compare.nameInfo', 'Shown in the session list. Leave blank to use the date.')} />
          <input className={inputCls} value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div>
          <ParamLabel label={t('compare.group', 'Group')} info={t('compare.groupInfo', 'Sessions with the same group are listed together, for example one group per A/B experiment.')} />
          <input className={inputCls} value={group} onChange={e => setGroup(e.target.value)} />
        </div>
      </div>

      {picked.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {picked.map((id, i) => (
            <button key={id} onClick={() => toggle(id)} className="flex items-center gap-1 px-2 h-8 rounded-full text-xs bg-pink-500/15 text-pink-400 border border-pink-500/30">
              <span className="font-bold">{i + 1}</span> <span className="max-w-[160px] truncate">{titles[id] || id}</span> <X size={12} />
            </button>
          ))}
        </div>
      )}

      <div className="relative">
        <Search size={14} className="absolute left-3 top-3 text-zinc-500" />
        <input className={`${inputCls} pl-8`} value={q} placeholder={t('compare.searchPlaceholder', 'Search the library by title or style')}
          onChange={e => { setQ(e.target.value); setOffset(0); }} />
      </div>

      {songsQuery.isError && (
        <QueryError error={songsQuery.error} onRetry={() => void songsQuery.refetch()} />
      )}

      <div className="flex flex-col gap-1">
        {hits.map(s => {
          const on = picked.includes(s.id);
          return (
            <button key={s.id} onClick={() => toggle(s.id)}
              className={`flex items-center gap-3 text-left px-3 min-h-[44px] rounded-lg border ${on ? 'border-pink-500/60 bg-pink-500/5' : 'border-zinc-200 dark:border-white/5 hover:border-zinc-400'}`}>
              <span className={`w-4 h-4 rounded border shrink-0 ${on ? 'bg-pink-500 border-pink-500' : 'border-zinc-400'}`} />
              <span className="min-w-0 flex-1">
                <span className="block text-sm truncate text-zinc-900 dark:text-zinc-100">{s.title}</span>
                <span className="block text-[11px] text-zinc-500 truncate">{s.backend} · {s.createdAt} · {s.style}</span>
              </span>
            </button>
          );
        })}
        {songsQuery.isPending && hits.length === 0 && <div className="text-sm text-zinc-500 py-2">{t('compare.loading', 'Loading...')}</div>}
        {!songsQuery.isPending && !songsQuery.isError && hits.length === 0 && (
          <div className="text-sm text-zinc-500 py-2">{t('compare.noResults', 'No songs found.')}</div>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-zinc-500">
        <button className="h-10 px-3 rounded-lg border border-zinc-300 dark:border-white/10 disabled:opacity-40" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>{t('compare.prev', 'Previous')}</button>
        <span className="flex-1 text-center">{total ? `${offset + 1}-${Math.min(offset + PAGE, total)} / ${total}` : ''}</span>
        <button className="h-10 px-3 rounded-lg border border-zinc-300 dark:border-white/10 disabled:opacity-40" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>{t('compare.next', 'Next')}</button>
      </div>

      <button onClick={submit} disabled={picked.length < 2 || create.isPending}
        className="h-12 rounded-xl bg-pink-500 hover:bg-pink-400 text-white font-semibold disabled:opacity-40">
        {picked.length < 2 ? t('compare.pickTwo', 'Pick at least two songs') : t('compare.createN', 'Compare {{count}} songs', { count: picked.length })}
      </button>
    </div>
  );
};

// ── Session list ─────────────────────────────────────────────────────────────

const SessionList: React.FC<{ onOpen: (id: string) => void; onNew: () => void }> = ({ onOpen, onNew }) => {
  const { t } = useTranslation();
  const [showArchived, setShowArchived] = useState(false);
  const sessionsQuery = useQuery(compareQueries.sessions(showArchived ? {} : { archived: false }));
  const sessions = sessionsQuery.data?.sessions ?? [];

  const groups = useMemo(() => {
    const m = new Map<string, CompareSessionSummary[]>();
    for (const s of sessions) {
      const k = s.group || '';
      if (!m.has(k)) m.set(k, []);
      m.get(k)!.push(s);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [sessions]);
  const unscored = sessions.filter(s => s.status !== 'scored');
  const statusLabel = (s: CompareSessionSummary) =>
    s.status === 'scored' ? t('compare.scored', 'scored')
      : s.status === 'partial' ? t('compare.partialN', 'partial {{n}}/{{m}}', { n: s.scored, m: s.total })
        : t('compare.unscored', 'unscored');

  return (
    <div className="flex flex-col gap-4 p-3 sm:p-4 max-w-3xl w-full mx-auto">
      <div className="flex flex-wrap items-center gap-3">
        <GitCompare size={20} className="text-pink-400" />
        <h1 className="text-lg font-bold flex-1 text-zinc-900 dark:text-zinc-100">{t('sidebar.compare', 'Compare')}</h1>
        <Toggle size="sm" accent="pink" checked={showArchived} onChange={setShowArchived}
          label={t('compare.showArchived', 'Show archived')}
          info={t('compare.showArchivedInfo', 'A session archives itself once every take has a score. Turn this on to list those too.')} />
        <button onClick={onNew} className="flex items-center gap-1.5 h-10 px-4 rounded-lg bg-pink-500 hover:bg-pink-400 text-white text-sm font-semibold">
          <Plus size={16} /> {t('compare.new', 'New comparison')}
        </button>
      </div>

      {sessionsQuery.isError && (
        <QueryError error={sessionsQuery.error} onRetry={() => void sessionsQuery.refetch()} />
      )}

      {unscored.length > 0 && (
        <div>
          <div className="text-[11px] uppercase tracking-wide text-zinc-500 mb-1.5">{t('compare.unscoredRow', 'Unscored')}</div>
          <div className="flex flex-wrap gap-1.5">
            {unscored.map(s => (
              <button key={s.id} onClick={() => onOpen(s.id)} className="flex items-center gap-1.5 h-9 px-3 rounded-full text-xs border border-zinc-300 dark:border-white/10 hover:border-pink-400 text-zinc-700 dark:text-zinc-300">
                <span className={`w-2 h-2 rounded-full ${DOT[s.status]}`} /> <span className="max-w-[180px] truncate">{s.name}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {sessionsQuery.isPending && !sessionsQuery.data && <div className="text-sm text-zinc-500">{t('compare.loading', 'Loading...')}</div>}
      {!sessionsQuery.isPending && !sessionsQuery.isError && sessions.length === 0 && <div className="text-sm text-zinc-500">{t('compare.empty', 'No comparisons yet. Start one, or select songs in the Library and press Compare.')}</div>}

      {groups.map(([g, list]) => (
        <div key={g || '_'}>
          <div className="text-xs font-semibold text-zinc-500 mb-1.5">{g || t('compare.ungrouped', 'Ungrouped')}</div>
          <div className="flex flex-col gap-1.5">
            {list.map(s => (
              <button key={s.id} onClick={() => onOpen(s.id)}
                className="flex items-center gap-3 text-left px-3 min-h-[48px] rounded-xl border border-zinc-200 dark:border-white/5 bg-white/60 dark:bg-zinc-900/40 hover:border-pink-400">
                <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${DOT[s.status]}`} />
                <span className="min-w-0 flex-1">
                  <span className="block text-sm truncate text-zinc-900 dark:text-zinc-100">{s.name}{s.archived ? ` (${t('compare.archived', 'archived')})` : ''}</span>
                  <span className="block text-[11px] text-zinc-500">{statusLabel(s)} · {t('compare.nTakes', '{{count}} takes', { count: s.total })}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

// ── Studio shell ─────────────────────────────────────────────────────────────

export const CompareStudio: React.FC = () => {
  const { t } = useTranslation();
  const [mode, setMode] = useState<'list' | 'new' | 'session'>('list');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pending] = useState<string[]>(() => takePendingCompareSongs());
  const [pendingOpened, setPendingOpened] = useState(false);
  const sessionQuery = useQuery({
    ...compareQueries.session(sessionId ?? ''),
    enabled: mode === 'session' && !!sessionId,
  });

  // Hand-off from the library opens the picker once; closing it must not
  // reopen it on the next render.
  if (pending.length && !pendingOpened) {
    setPendingOpened(true);
    setMode('new');
  }

  const open = (id: string) => { setSessionId(id); setMode('session'); };
  const created = (id: string) => { setSessionId(id); setMode('session'); };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 min-h-0 overflow-y-auto">
        {mode === 'session' ? (
          sessionQuery.isError ? (
            <div className="p-4 space-y-2">
              <QueryError error={sessionQuery.error} onRetry={() => void sessionQuery.refetch()} />
              <button onClick={() => setMode('list')} className="text-xs text-zinc-500 hover:text-pink-400">
                {t('compare.back', 'Back to sessions')}
              </button>
            </div>
          ) : sessionQuery.data ? (
            <SessionView key={sessionQuery.data.id} session={sessionQuery.data} onBack={() => setMode('list')} />
          ) : (
            <div className="p-4 text-sm text-zinc-500">{t('compare.loading', 'Loading...')}</div>
          )
        ) : mode === 'new' ? (
          <NewSessionPanel initialIds={pending} onClose={() => setMode('list')} onCreated={created} />
        ) : (
          <SessionList onOpen={open} onNew={() => setMode('new')} />
        )}
      </div>
    </div>
  );
};

export default CompareStudio;
