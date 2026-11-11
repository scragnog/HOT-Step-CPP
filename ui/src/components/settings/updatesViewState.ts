import type { UpdateApplyResult, UpdatesStatus, UpdateRestartAttempt } from '../../services/updatesApi';

const activePhases = new Set(['checking', 'downloading', 'verifying', 'extracting']);
export interface UpdateRefusal { error: string; blockers: string[]; errors: string[] }

export function updateRefusal(error: unknown): UpdateRefusal {
  const e = error as { status?: number; body?: { error?: unknown; blockers?: unknown; errors?: unknown }; message?: string };
  const strings = (v: unknown) => Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
  return {
    error: typeof e?.body?.error === 'string' ? e.body.error : e?.message ?? String(error),
    blockers: e?.status === 409 ? strings(e.body?.blockers) : [],
    errors: e?.status === 400 ? strings(e.body?.errors) : [],
  };
}

/** Offer decisions use the current snapshot, never the notices-only apply stub. */
export function updatesViewState(s: UpdatesStatus, refusal: UpdateRefusal | null = null) {
  const p = s.prepare;
  // offered is false while work is running and once ready: it means a new
  // download can start, not that the existing preparation should disappear.
  const offered = s.status === 'update-available' && s.channel === 'release'
    && s.installed.kind === 'portable' && !!s.install?.supported && !!p?.candidate
    && (!!p.offered || activePhases.has(p.phase) || p.phase === 'ready');
  const active = offered && activePhases.has(p!.phase);
  const ready = offered && p!.phase === 'ready';
  return {
    showChannel: s.installed.kind !== 'portable',
    resetToRelease: s.installed.kind === 'portable' && s.channel === 'master',
    gitInstructions: s.installed.kind === 'git' && s.status === 'update-available',
    offered, active, ready,
    action: ready ? 'install' as const : offered && !active ? 'download' as const : null,
    candidate: offered ? p!.candidate : null,
    phase: p?.phase ?? 'idle',
    progress: p && p.totalBytes > 0 ? Math.min(1, Math.max(0, p.downloadedBytes / p.totalBytes)) : null,
    prepareError: offered && p!.phase === 'failed' ? p!.error : null,
    reason: offered ? null : p?.reason || s.install?.reason || null,
    stopping: s.install?.phase === 'stopping',
    lastResult: s.install?.lastResult ?? null,
    refusal,
  };
}

/** The old process can answer health briefly; only a new matching receipt ends polling. */
export function completedUpdate(s: UpdatesStatus, attempt: UpdateRestartAttempt): UpdateApplyResult | null {
  const r = s.install?.lastResult;
  if (!r || s.install?.phase === 'stopping') return null;
  if (attempt.to && r.to !== attempt.to) return null;
  if (attempt.from && r.from !== attempt.from) return null;
  if (r.at === attempt.previousResult?.at) return null;
  return r;
}

export function updateVersion(result: UpdateApplyResult): string {
  return (result.to ?? '').replace(/-[0-9a-f]{12}-.*$/i, '');
}
