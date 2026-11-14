// CreateBuildTab.tsx — Custom-Gen's second tab: Build by section for YuE2.
//
// The same Yue2BuildPage Cover Studio uses, hosted here with no source
// recording. The tab plans the score itself from the Create form (the plan_only
// call the "Preview the score first" toggle makes), shows its sections and any
// lyric mismatch, and starts a build session from the form's params plus that
// score. A pasted lead sheet (yue2Abc) is used as is.

import React, { useCallback, useEffect, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { useBackendStore } from '../../stores/backendStore';
import { useGlobalParamsStore } from '../../stores/globalParamsStore';
import { generateApi } from '../../services/api';
import { yue2CoverApi, type Yue2SectionReview } from '../../services/yue2CoverApi';
import { Yue2BuildPage, buildCaption, useYue2Build, type BuildSettings } from '../cover-studio/Yue2Build';
import { COVER_AR_DEFAULT, clampCoverAr } from '../cover-studio/Yue2CoverPanel';
import { barRanges } from '../../utils/abcSectionMove';
import { notify } from '../shared/Toast';

const YUE2 = 'yue2';
const SESSION_KEY = 'hs-create-buildSessionId';
const PLAN_KEY = 'hs-create-buildPlan';

const read = <T,>(key: string, fallback: T): T => {
  try { const raw = localStorage.getItem(key); return raw === null ? fallback : JSON.parse(raw) as T; } catch { return fallback; }
};
const write = (key: string, value: unknown) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full or blocked: the build just is not remembered */ } };


/** Why Build by section is unavailable on this backend or setting, or ''. */
export function useBuildTabBlock(): string {
  const backend = useBackendStore(s => s.activeBackendId);
  const cot = useGlobalParamsStore((s: any) => s.backendParams?.yue2Cot ?? 'full');
  if (backend !== YUE2) return 'Build by section needs the YuE2 backend.';
  if (cot === 'off') return 'Build by section needs Chain of Thought on (melody or full): it builds on a planned score.';
  return '';
}

/** The studio's tab row. Falls back to One-shot when Build is unavailable. */
export const CreateTabs: React.FC<{ tab: 'oneshot' | 'build'; onTab: (tab: 'oneshot' | 'build') => void }> = ({ tab, onTab }) => {
  const block = useBuildTabBlock();
  useEffect(() => { if (block && tab === 'build') onTab('oneshot'); }, [block, tab, onTab]);
  const button = (id: 'oneshot' | 'build', label: string, disabled = false, title?: string) => (
    <button type="button" role="tab" aria-selected={tab === id} disabled={disabled} title={title} onClick={() => onTab(id)}
      className={`px-4 py-2 text-sm font-semibold border-b-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${tab === id
        ? 'border-pink-500 text-zinc-900 dark:text-white' : 'border-transparent text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200'}`}>{label}</button>
  );
  return (
    <div role="tablist" aria-label="Custom-Gen" className="flex flex-shrink-0 items-center gap-1 border-b border-zinc-200 dark:border-white/5 px-3">
      {button('oneshot', 'One-shot')}
      {button('build', 'Build by section', !!block, block || undefined)}
    </div>
  );
};

interface Plan { abc: string; seed: number }

interface Props {
  token: string | null;
  /** The Create form's params as a render would send them, or null while the form is not ready. */
  getParams: () => Record<string, unknown> | null;
  /** Save song: an ordinary render of the kept build. */
  onSave: (params: Record<string, unknown>, sessionId: string) => void;
  onOneShot: () => void;
  /** Build session ids that already have a saved song in the library: a build whose song is saved is finished. */
  savedSessions: ReadonlySet<string>;
}

export const CreateBuildTab: React.FC<Props> = ({ token, getParams, onSave, onOneShot, savedSessions }) => {
  const [sessionId, setSessionId] = useState(() => read<string>(SESSION_KEY, ''));
  const [plan, setPlan] = useState<Plan | null>(() => read<Plan | null>(PLAN_KEY, null));
  const [planning, setPlanning] = useState(false);
  const [planError, setPlanError] = useState('');
  const [review, setReview] = useState<Yue2SectionReview | null>(null);
  // Settings the user changed before a session exists; null leaves the form's own.
  const [edited, setEdited] = useState<{ cfg?: number; ar?: number }>({});
  useEffect(() => write(SESSION_KEY, sessionId), [sessionId]);
  useEffect(() => write(PLAN_KEY, plan), [plan]);
  useEffect(() => { if (sessionId && savedSessions.has(sessionId)) setSessionId(''); }, [sessionId, savedSessions]);

  const build = useYue2Build(token, sessionId, setSessionId, { text: null, set: () => undefined });
  const session = build.session && build.session.id === sessionId ? build.session : null;
  const arDefault = useBackendStore(s => Number((s.models[YUE2]?.defaults as Record<string, unknown> | undefined)?.lmAdapterArScale));
  const form = getParams();
  const supplied = typeof form?.yue2Abc === 'string' && form.yue2Abc.trim() ? String(form.yue2Abc) : '';
  const abc = session ? String(session.params.yue2Abc ?? '') : supplied || plan?.abc || '';
  const lyrics = session ? String(session.params.lyrics ?? '') : String(form?.lyrics ?? '');
  const instrumental = !session && !!form?.instrumental;

  // The planned (or pasted) score against the lyric tags, from the server's own section lint.
  useEffect(() => {
    if (session || !abc) { setReview(null); return; }
    let live = true;
    yue2CoverApi.reviewSections(abc, lyrics).then(r => { if (live) setReview(r); }, () => { if (live) setReview(null); });
    return () => { live = false; };
  }, [abc, lyrics, !!session]);

  const planScore = useCallback(async () => {
    const params = getParams();
    if (!params) return;
    setPlanning(true); setPlanError('');
    try {
      const data = await generateApi.yue2Plan({ ...params, randomSeed: true, seed: -1 } as never);
      setPlan({ abc: data.abc, seed: data.seed });
    } catch (err) { setPlanError(err instanceof Error ? err.message : String(err)); }
    finally { setPlanning(false); }
  }, [getParams]);

  /** The params a build starts from: the open session's own, or the form's with the planned score and any settings edited here. */
  const buildParams = (): Record<string, unknown> | null => {
    if (session) return session.params;
    const params = getParams();
    if (!params) { notify('The Create form is not ready.'); return null; }
    const pick = edited.ar === undefined ? params.yue2Pick : { ...(params.yue2Pick as Record<string, unknown> | undefined), lmAdapterArScale: edited.ar };
    return { ...params, ...(supplied ? {} : { yue2Abc: plan?.abc ?? '' }), ...(plan && !supplied ? { seed: plan.seed } : {}), randomSeed: false,
      yue2BatchSize: 1, yue2Variations: 1, batchSize: 1,
      ...(edited.cfg === undefined ? {} : { yue2CfgScale: edited.cfg }), ...(pick === undefined ? {} : { yue2Pick: pick }) };
  };

  const settings: BuildSettings = (() => {
    const p = session?.params ?? form ?? {};
    const ar = Number((p.yue2Pick as Record<string, unknown> | undefined)?.lmAdapterArScale);
    const cfg = Number(p.yue2CfgScale);
    return { voices: 'both', tempo: 'free', keyShift: 0,
      cfgScale: session ? (Number.isFinite(cfg) && cfg > 0 ? cfg : 1) : edited.cfg ?? (Number.isFinite(cfg) && cfg > 0 ? cfg : 1),
      arStrength: (session ? undefined : edited.ar) ?? (Number.isFinite(ar) ? ar : Number.isFinite(arDefault) ? arDefault : COVER_AR_DEFAULT) };
  })();
  const applySettings = async (s: BuildSettings) => {
    const next = { cfg: s.cfgScale, ar: clampCoverAr(s.arStrength) };
    if (session && !session.kept.length) { if (!await build.settings({ cfgScale: next.cfg, arStrength: next.ar })) return; }
    setEdited(next);
    if (session?.kept.length) build.startOver();
  };

  const save = () => {
    if (!session?.kept.length) return;
    onSave({ ...session.params, yue2Build: { session: session.id }, yue2BatchSize: 1, yue2Variations: 1, batchSize: 1, randomSeed: true }, session.id);
    notify('Saving the build as a song: it is in the queue');
  };

  const sections = session?.sections ?? review?.sections ?? [];
  const totalBars = barRanges(abc).reduce((n, r) => Math.max(n, r.bar), 0);
  const empty = !form || (!String(form.caption ?? '').trim() && !String(form.lyrics ?? '').trim());
  const blocked = session ? '' : empty ? 'Add a style or lyrics on the One-shot tab first.'
    : instrumental ? 'Build by section needs lyrics: it pairs each section with its words. Turn Instrumental off on the One-shot tab.'
    : !abc ? 'Plan the score first.' : review && !review.lint.ok ? `The score and lyrics do not pair: ${review.lint.message}. Re-plan, or fix the lyric tags on the One-shot tab.` : '';

  const planBar = !session && (
    <div className="flex flex-wrap items-center gap-3 border-b border-zinc-200 dark:border-white/10 px-4 py-2 text-xs text-zinc-500">
      {supplied
        ? <span className="flex-1">Using the lead sheet you pasted as the score.</span>
        : <>
          <span className="flex-1">{planning ? 'Planning the score…' : plan ? `Planned score: ${sections.map(x => x.label).join(', ') || 'no sections'} (seed ${plan.seed}).` : 'The build follows a planned score. Plan one from the Create form, then build it section by section.'}</span>
          <button type="button" onClick={() => void planScore()} disabled={planning || empty || instrumental}
            className="flex items-center gap-1 rounded-lg bg-cyan-600 px-3 py-1 font-semibold text-white disabled:opacity-40">
            {planning ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}{plan ? 'Re-plan with a new seed' : 'Plan the score'}
          </button>
        </>}
      {review && !review.lint.ok && <span className="basis-full text-amber-600 dark:text-amber-400">{review.lint.message}</span>}
      {planError && <span className="basis-full text-red-500">{planError}</span>}
    </div>
  );

  return (
    <div className="flex flex-1 min-h-0 min-w-0 flex-col">
      {planBar}
      <div className="flex flex-1 min-h-0 min-w-0 flex-col">
        {(session || abc) ? <Yue2BuildPage session={session} sections={sections} totalBars={totalBars}
          lyrics={lyrics} prepareLyrics={String(form?.lyrics ?? '')}
          onApplyLyrics={async () => { notify(session ? 'Edit the lyrics on the One-shot tab and start a fresh build.' : 'Start the build first, then edit its lyrics here; until then it uses the Create lyrics.'); return false; }}
          loading={build.loading} settings={settings} onSettings={s => void applySettings(s)} coverSettings={false} backLabel="One-shot"
          keyLabel="" sourceBpm={120}
          title={String((session?.params ?? form)?.title || 'Untitled')}
          scoreAbc={abc} recordingUrl="" blocked={blocked}
          busy={build.busy} error={build.error} caption={buildCaption((session?.params ?? form) as Record<string, unknown> | undefined)}
          onOccurrenceLyrics={edits => build.occurrenceLyrics(edits)}
          onGenerate={(to, count, caption, lyricEdits, newChords) => {
            const params = buildParams();
            return params ? build.generate(params, to, count, caption, lyricEdits, newChords).then(Boolean) : Promise.resolve(false);
          }}
          onPlan={(upcoming, finished) => { const params = buildParams(); if (params) build.plan(params, upcoming, finished); }}
          onCancel={() => build.cancel()} onKeep={(take, crop, count) => build.keep(take, crop, count)} onDelete={take => build.remove(take)}
          onUndo={section => build.undo(section)} onStartOver={build.startOver}
          onDeleteBuild={() => void build.deleteBuild()} onSave={save}
          onCrop={(take, frame, mode) => build.crop(take, frame, mode)} onTrim={(take, seconds) => build.trim(take, seconds)} onPreview={() => build.preview()}
          onBack={onOneShot} />
          : <div className="p-6 text-sm text-zinc-500">{blocked || 'Plan the score to start.'}</div>}
      </div>
    </div>
  );
};
