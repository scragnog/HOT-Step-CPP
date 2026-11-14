import { useMutation, useQuery } from '@tanstack/react-query';
import { queryClient } from '../../queries/client';
import { snapshotOptions } from '../../queries/cover-lyrics/reads';
import { withKeyId } from '../../queries/keys';
import { pollWhile } from '../../queries/contract';
import React, { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ChevronDown, ChevronLeft, ChevronRight, Loader2, Lock, MoreHorizontal, Pause, Play, RotateCcw, Scissors, Trash2, X } from 'lucide-react';
import { yue2CoverApi, type Yue2BuildSession, type Yue2BuildTake } from '../../services/yue2CoverApi';
import { lyricBlockRanges, sectionColours } from '../../utils/abcSections';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import { CoverArStrength } from './Yue2CoverPanel';
import { SectionPlayPopup, useHoverPopup, useSectionPlayer } from './SectionPlayback';
import { persist, restore, transposeKey } from './coverStudioUtils';
import { EditableSlider } from '../shared/EditableSlider';

const FPS = 25;
// A take plays from this long before its first section, so the join is heard.
const LEAD_IN = 4;
const clock = (seconds: number) => {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

const savedBuildListeners = new Set<(id: string) => void>();
/** Mounted Cover Studios subscribe here so a Save that finishes after they
 *  were rebuilt (the user left and came back) still clears the build. */
export function onBuildSaved(listener: (id: string) => void) {
  savedBuildListeners.add(listener);
  return () => { savedBuildListeners.delete(listener); };
}
/** A Save song finished: forget build `id`, unless a newer build has replaced it.
 *  Lives outside the component because the Save outlives it. */
export function finishSavedBuild(id: string): boolean {
  if (restore<string>('buildSessionId', '') !== id) return false;
  persist('buildSessionId', '');
  savedBuildListeners.forEach(l => l(id));
  return true;
}

/** The build's own caption, read from its params the way a render reads it (yue2/generate.ts). */
export const buildCaption = (params: Record<string, unknown> | undefined) =>
  String(params?.prompt || params?.songDescription || params?.caption || params?.style || '');
/** The caption a Generate sends: none while the field still shows the build's, so the takes inherit it. */
export const submittedCaption = (shown: string, original: string) => (shown.trim() === original.trim() ? '' : shown.trim());

/** The cover settings the Build page's row edits: params.yue2Cover's voices,
 *  tempo, keyShift and cfgScale, and the Cover AR strength. */
export interface BuildSettings { voices: 'vocal' | 'both'; tempo: 'free' | 'source' | number; keyShift: number; cfgScale: number; arStrength: number }

/** The first section not yet built. */
export function buildNext(session: Yue2BuildSession | null): number {
  const spine = session?.takes.find(t => t.id === session.kept.at(-1));
  return spine ? spine.to + 1 : 0;
}

/** One occurrence of the output: a score section placed in the song. Kept ones are built and locked.
 *  `lyrics` is its own sung lines, when it has been given some. An improv has section -1 and
 *  `improv` bars: rests in the score, composed without it. */
export interface PlanItem { id: string; section: number; kept: boolean; lyrics?: string; improv?: number }

/** The bar counts the Improv button offers. */
export const IMPROV_BARS = [2, 4, 8, 16];
const IMPROV_COLOUR = '#a855f7';

/** The output in order. Before a session exists it is empty: the song is built up from the source bar. */
export function planItems(session: Yue2BuildSession | null): PlanItem[] {
  const built = buildNext(session);
  if (!session) return [];
  const { sections, plan } = session.arrangement;
  return plan.map((o, i) => ({ id: o.id, section: sections.findIndex(s => s.id === o.section), kept: i < built,
    ...(o.lyrics === undefined ? {} : { lyrics: o.lyrics }), ...(o.section === 'improv' ? { improv: o.bars } : {}) }));
}

/** The ids `POST /build/:id/plan` takes for the unbuilt part of the output after `edit`. */
export function editedPlan(items: PlanItem[], next: number,
  edit: { add: number } | { improv: number } | { remove: string } | { move: string; by: -1 | 1 }): string[] {
  const upcoming = items.slice(next).map(i => i.id);
  if ('add' in edit) return [...upcoming, `s${edit.add}`];
  if ('improv' in edit) return [...upcoming, `improv:${edit.improv}`];
  if ('remove' in edit) return upcoming.filter(id => id !== edit.remove);
  const at = upcoming.indexOf(edit.move), to = at + edit.by;
  if (at < 0 || to < 0 || to >= upcoming.length) return upcoming;
  [upcoming[at], upcoming[to]] = [upcoming[to], upcoming[at]];
  return upcoming;
}

/** The lyric block an occurrence sings, as text. A session pairs sections with the sung (non-empty) blocks. */
export function occurrenceLyrics(lyrics: string, session: Yue2BuildSession | null, item: PlanItem): string {
  const ranges = lyricBlockRanges(lyrics);
  if (!session) { const r = ranges[item.section]; return r ? lyrics.slice(r.start, r.end).trim() : ''; }
  const sung = ranges.filter(r => lyrics.slice(r.start, r.end).replace(/^[^\n]*/, '').trim());
  const r = sung[session.arrangement.sections[item.section]?.lyricBlock ?? -1];
  return r ? lyrics.slice(r.start, r.end).trim() : '';
}

/** The lines an occurrence sings, without its tag: its own, else its block's. Null for an unsung section. */
export function occurrenceLines(lyrics: string, session: Yue2BuildSession | null, item: PlanItem): string | null {
  const block = occurrenceLyrics(lyrics, session, item);
  if (!block) return null;
  return item.lyrics ?? (/^\[[^\]\n]*\]/.test(block) ? block.replace(/^[^\n]*\n?/, '') : block).trim();
}

/** The lyric edits a Generate applies first: the drafts that differ from what their occurrence sings now. */
export function pendingLyrics(drafts: Record<string, string>, current: Record<string, string | null>): Record<string, string> {
  return Object.fromEntries(Object.entries(drafts).filter(([id, text]) => id in current && current[id] !== null && text.trim() !== current[id]!.trim()));
}

/** Whether a take can be kept now: it resumed the kept build, at the current revision, and composed the planned occurrences. */
export function keepable(session: Yue2BuildSession, take: Yue2BuildTake): boolean {
  const next = buildNext(session);
  const want = session.arrangement.plan.slice(next, take.to + 1).map(o => o.id);
  return take.revision === session.revision && take.parent === (session.kept.at(-1) ?? null) && take.from === next
    && want.length === take.occurrences.length && want.every((id, i) => id === take.occurrences[i]);
}

/** Codec frames a take resumed: its new part starts here. */
export const takePrefix = (take: Yue2BuildTake) => take.prefix ?? take.starts[take.from] ?? 0;

/** Where a cut at `frame` falls in `take` (as the server's cropPlan reads it): the occurrence,
 *  and whether it is inside it, so the user must say whether the next take composes it again.
 *  Past `to` it is the section the take ran on into. */
export function cropPoint(take: Yue2BuildTake, frame: number): { occurrence: number; inside: boolean } {
  let k = take.from;
  for (let i = take.from + 1; i <= Math.min(take.to + 1, take.starts.length - 1); i++) {
    const at = take.starts[i];
    if (at != null && at <= frame) k = i;
  }
  return { occurrence: k, inside: take.starts[k] !== frame };
}

/** A crop set on a take before it is kept, in seconds on the song's clock. `in` (first-section takes only)
 *  is where the saved song starts, faded in; `out` cuts the take there; null leaves that end alone.
 *  `mode` answers cropPoint's question when `out` falls inside a section. */
export interface TakeCrop { in: number | null; out: number | null; mode: 'regenerate' | 'advance' | null }

/** The seconds a take's crop handles may reach: `in` from the top to a second before the end,
 *  `out` inside the part the take composed. */
export function cropLimits(take: Yue2BuildTake): { inMax: number; outMin: number; end: number } {
  const end = take.frames / FPS;
  return { inMax: take.from === 0 ? end - 1 : 0, outMin: (takePrefix(take) + 1) / FPS, end };
}

/** `crop` with one edge moved to `at` seconds, kept in its limits and a second clear of the other edge.
 *  An edge dragged back to its end of the take is no crop (null). */
export function moveCropEdge(take: Yue2BuildTake, crop: TakeCrop, edge: 'in' | 'out', at: number): TakeCrop {
  const { inMax, outMin, end } = cropLimits(take);
  const snap = (t: number) => Math.round(t * FPS) / FPS;
  if (edge === 'in') {
    if (inMax <= 0) return crop;
    const t = snap(Math.min(inMax, (crop.out ?? end) - 1, Math.max(0, at)));
    return { ...crop, in: t <= 0 ? null : t };
  }
  const t = snap(Math.max(outMin, (crop.in ?? 0) + 1, Math.min(end, at)));
  return { ...crop, out: t >= end ? null : t, mode: null };
}

/** Whether a crop's out point falls inside a section the take composed, or inside the one it ran on into,
 *  so it needs a mode before it can be kept. */
export function cropNeedsMode(take: Yue2BuildTake, crop: TakeCrop): boolean {
  return crop.out !== null && cropPoint(take, Math.round(crop.out * FPS)).inside;
}

/** What 'advance' does for a cut inside `name`: move on to `following`, or, with nothing after it,
 *  end the song only when the plan ends there; an open plan carries on from the cut. */
export const advanceLabel = (name: string, following: string | null, finished: boolean) =>
  following ? `moves on to ${following}` : finished ? `keeps ${name} cut short and ends the song` : `keeps ${name} cut short; the next section you add starts at the cut`;

/** Where an Apply goes: the loaded session, a draft when there is no session at all, or
 *  nowhere while a saved session is still loading (or failed to), so it cannot be lost to a draft. */
export const lyricsTarget = (sessionId: string, live: unknown): 'session' | 'draft' | 'loading' =>
  live ? 'session' : sessionId ? 'loading' : 'draft';

/** A saved draft's text if it was written for `source`, else null (follow Prepare). */
export const draftText = (draft: { source: string; text: string } | null, source: string): string | null =>
  draft?.source === source ? draft.text : null;

/** The build's own lyrics before a session exists: null follows Prepare. */
export interface BuildLyricsDraft { text: string | null; set: (text: string | null) => void }

/** A Build-by-section session kept on the server, polled while takes render.
 *  Every change names the revision the page saw, so a stale page is refused. */
export function useYue2Build(token: string | null, sessionId: string, setSessionId: (id: string) => void, draft: BuildLyricsDraft) {
  const options = snapshotOptions<Yue2BuildSession>('cover', `/yue2-cover/build/${sessionId}`);
  const snapshot = useQuery({ ...options, enabled: !!token && !!sessionId, refetchInterval: pollWhile<Yue2BuildSession>(s => !s || s.pending.length > 0 || s.preview?.status === 'rendering') });
  const session = sessionId ? snapshot.data ?? null : null;
  const [error, setError] = useState('');
  const live = session && session.id === sessionId ? session : null;

  /** Every command ends with the session's new snapshot: seed it under the new session's pathname id (not the bare id),
   *  keeping this query's scope, and clear the error line. */
  const commit = (next: Yue2BuildSession) => {
    const key = withKeyId(options.queryKey, `/yue2-cover/build/${next.id}`);
    queryClient.setQueryData<Yue2BuildSession>(key, previous =>
      previous && previous.revision > next.revision ? previous : next);
    setError('');
  };
  /** The error line is the sole notifier (the commands are silent): show it, and re-read what the server still holds. */
  const fail = (err: unknown) => { setError(err instanceof Error ? err.message : String(err)); void snapshot.refetch(); };

  // One mutation per command; they share commit/fail because every one returns the session's new snapshot.
  const generateMut = useMutation({
    mutationFn: async (v: { params: Record<string, unknown>; to: number; count: number; caption: string; lyrics: Record<string, string>; newChords: boolean }) => {
      let started = live;
      if (!started) { started = await yue2CoverApi.buildStart(v.params, draft.text ?? undefined); setSessionId(started.id); }
      if (Object.keys(v.lyrics).length) started = await yue2CoverApi.buildOccurrenceLyrics(started.id, started.revision, v.lyrics);
      return yue2CoverApi.buildGenerate(started.id, started.revision, v.to, v.count, v.caption || undefined, v.newChords);
    },
    onSuccess: commit, onError: fail,
  });
  const occurrenceLyricsMut = useMutation({
    mutationFn: (v: { edits: Record<string, string | null> }) => yue2CoverApi.buildOccurrenceLyrics(live!.id, live!.revision, v.edits),
    onSuccess: commit, onError: fail,
  });
  const settingsMut = useMutation({
    mutationFn: (v: { settings: Record<string, unknown> }) => yue2CoverApi.buildSettings(live!.id, live!.revision, v.settings),
    onSuccess: commit, onError: fail,
  });
  const planMut = useMutation({
    mutationFn: async (v: { params: Record<string, unknown>; upcoming: string[]; finished: boolean }) => {
      let started = live;
      if (!started) { started = await yue2CoverApi.buildStart(v.params, draft.text ?? undefined); setSessionId(started.id); }
      return yue2CoverApi.buildPlan(started.id, started.revision, v.upcoming, v.finished);
    },
    onSuccess: commit, onError: fail,
  });
  const keepMut = useMutation({
    mutationFn: async (v: { take: string; crop?: TakeCrop; count?: number }) => {
      const kept = live!.takes.find(t => t.id === v.take);
      let next = await yue2CoverApi.buildKeep(live!.id, live!.revision, v.take);
      if (v.crop?.out != null) next = await yue2CoverApi.buildCrop(next.id, next.revision, v.take, Math.round(v.crop.out * FPS), v.crop.mode ?? undefined);
      if (v.crop?.in != null) next = await yue2CoverApi.buildTrim(next.id, next.revision, v.take, v.crop.in);
      // Not for a cut in the section the take ran on into: that section may not be planned yet.
      if (v.count && kept && v.crop?.out != null && v.crop.mode === 'regenerate' && cropPoint(kept, Math.round(v.crop.out * FPS)).occurrence <= kept.to) next = await yue2CoverApi.buildGenerate(next.id, next.revision, kept.to, v.count, kept.caption || undefined);
      return next;
    },
    onSuccess: commit, onError: fail,
  });
  const previewMut = useMutation({ mutationFn: () => yue2CoverApi.buildPreview(live!.id), onSuccess: commit, onError: fail });
  const trimMut = useMutation({
    mutationFn: (v: { take: string; seconds: number | null }) => yue2CoverApi.buildTrim(live!.id, live!.revision, v.take, v.seconds),
    onSuccess: commit, onError: fail,
  });
  const undoMut = useMutation({
    mutationFn: (v: { section?: number }) => yue2CoverApi.buildUndo(live!.id, live!.revision, v.section),
    onSuccess: commit, onError: fail,
  });
  const removeMut = useMutation({
    mutationFn: (take: string) => yue2CoverApi.buildDelete(live!.id, live!.revision, take),
    onSuccess: commit, onError: fail,
  });
  const cancelMut = useMutation({ mutationFn: () => yue2CoverApi.buildCancel(live!.id), onSuccess: commit, onError: fail });
  const cropMut = useMutation({
    mutationFn: (v: { take: string; frame: number | null; mode?: 'regenerate' | 'advance' }) => yue2CoverApi.buildCrop(live!.id, live!.revision, v.take, v.frame, v.mode),
    onSuccess: commit, onError: fail,
  });
  const applyLyricsMut = useMutation({
    mutationFn: async (text: string): Promise<{ session?: Yue2BuildSession }> => {
      if (live) return { session: await yue2CoverApi.buildLyrics(live.id, live.revision, text) };
      await yue2CoverApi.buildLyricsCheck(text); draft.set(text); return {};
    },
    onSuccess: r => { if (r.session) commit(r.session); else setError(''); },
    onError: fail,
  });

  const deleteBuildMut = useMutation({
    mutationFn: () => yue2CoverApi.buildDeleteAll(live!.id),
    onSuccess: () => { setSessionId(''); setError(''); },
    onError: fail,
  });

  const busy = [generateMut, occurrenceLyricsMut, settingsMut, planMut, keepMut, previewMut, trimMut, undoMut, removeMut, cancelMut, cropMut, applyLyricsMut, deleteBuildMut]
    .some(m => m.isPending);
  return {
    session: live, loading: lyricsTarget(sessionId, live) === 'loading', error: error || snapshot.error?.message || '', busy,
    /** Generates takes, starting the session from `params` if there is none. A non-empty
     *  `caption` is these takes' own; `lyrics` (occurrence id to lines) is applied first. */
    generate: (params: Record<string, unknown>, to: number, count: number, caption: string, lyrics: Record<string, string> = {}, newChords = false) =>
      generateMut.mutateAsync({ params, to, count, caption, lyrics, newChords }).catch(() => null),
    /** Gives unbuilt occurrences their own lines, or (null) the build's again. */
    occurrenceLyrics: (edits: Record<string, string | null>) => { void occurrenceLyricsMut.mutateAsync({ edits }).catch(() => null); },
    /** Applies new cover settings to the session (before its first keep). */
    settings: (settings: Record<string, unknown>) => settingsMut.mutateAsync({ settings }).catch(() => null),
    /** Replaces the unbuilt part of the output, starting the session from `params` if there is none. */
    plan: (params: Record<string, unknown>, upcoming: string[], finished: boolean) => { void planMut.mutateAsync({ params, upcoming, finished }).catch(() => null); },
    /** Keeps a take, then applies the crop set on it: the out point cuts the kept take, the in point starts the saved song later.
     *  With `count`, a 'regenerate' cut then queues that many takes composing from the cut to the take's end, with its caption. */
    keep: (take: string, crop?: TakeCrop, count?: number) => { void keepMut.mutateAsync({ take, ...(crop ? { crop } : {}), ...(count ? { count } : {}) }).catch(() => null); },
    /** Renders everything kept so far once, for Play build. */
    preview: () => { void previewMut.mutateAsync().catch(() => null); },
    /** Where Save song starts the song: seconds into a first-section take, or null for the top. */
    trim: (take: string, seconds: number | null) => { void trimMut.mutateAsync({ take, seconds }).catch(() => null); },
    undo: (section?: number) => { void undoMut.mutateAsync({ ...(section === undefined ? {} : { section }) }).catch(() => null); },
    remove: (take: string) => { void removeMut.mutateAsync(take).catch(() => null); },
    cancel: () => { void cancelMut.mutateAsync().catch(() => null); },
    /** Cuts the last kept take at codec frame `frame`, or restores it (null). The take rows' crop goes through keep. */
    crop: (take: string, frame: number | null, mode?: 'regenerate' | 'advance') => { void cropMut.mutateAsync({ take, frame, ...(mode ? { mode } : {}) }).catch(() => null); },
    /** Applies the build's lyrics: to the session, or to the draft the first session starts from.
     *  True once the server (or, before a session, its validation) accepted them. */
    applyLyrics: async (text: string) => {
      const target = lyricsTarget(sessionId, live);
      if (target === 'loading') { setError('This build is still loading; try again once it has.'); return false; }
      return !!(await applyLyricsMut.mutateAsync(text).catch(() => null));
    },
    startOver: () => { setSessionId(''); setError(''); },
    /** Deletes the build on the server (refused while takes render), then returns to the empty state. */
    deleteBuild: async () => { await deleteBuildMut.mutateAsync().catch(() => null); },
  };
}

/** A take's waveform: the playhead always shows, and a tap or drag seeks (and plays). Crops are set
 *  with the buttons under it. What a crop leaves out is dimmed; green and red mark its ends. */
function TakeWave({ peaks, from, to, playhead, crop, canIn, colour, onSeek }: {
  peaks: number[]; from: number; to: number; playhead: number | null; crop: TakeCrop | null; canIn: boolean; colour: string;
  onSeek: (at: number) => void;
}) {
  const scrubbing = useRef(false);
  const span = Math.max(0.1, to - from);
  const x = (t: number) => Math.min(100, Math.max(0, ((t - from) / span) * 100));
  const at = (clientX: number, box: Element) => {
    const r = box.getBoundingClientRect();
    return from + Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width))) * span;
  };
  const shown = crop ? [crop.in ?? from, crop.out ?? to] as [number, number] : null;
  const inX = shown ? x(shown[0]) : 0, outX = shown ? x(shown[1]) : 100;
  const w = peaks.length ? 100 / peaks.length : 0;
  return (
    <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="block h-14 w-full cursor-crosshair touch-none rounded bg-zinc-100/60 dark:bg-white/5" role="group"
      aria-label="Tap or drag to play from a point"
      onPointerDown={e => { e.currentTarget.setPointerCapture(e.pointerId); scrubbing.current = true; onSeek(at(e.clientX, e.currentTarget)); }}
      onPointerMove={e => { if (scrubbing.current) onSeek(at(e.clientX, e.currentTarget)); }}
      onPointerUp={() => { scrubbing.current = false; }}
      onPointerCancel={() => { scrubbing.current = false; }}>
      {peaks.map((p, k) => <rect key={k} x={k * w + w * 0.15} width={w * 0.7} y={20 - Math.max(0.5, p * 19)} height={Math.max(1, p * 38)}
        fill={k * w + w / 2 >= inX && k * w + w / 2 <= outX ? colour : '#a1a1aa'} />)}
      {shown && <>
        <rect x={0} width={inX} y={0} height={40} fill="#000" opacity={0.35} />
        <rect x={outX} width={100 - outX} y={0} height={40} fill="#000" opacity={0.35} />
        {canIn && inX > 0 && <line x1={inX} x2={inX} y1={0} y2={40} stroke="#22c55e" strokeWidth={1} />}
        {outX < 100 && <line x1={outX} x2={outX} y1={0} y2={40} stroke="#ef4444" strokeWidth={1} />}
      </>}
      {playhead !== null && <line x1={x(playhead)} x2={x(playhead)} y1={0} y2={40} stroke={colour} strokeWidth={0.8} />}
    </svg>
  );
}

/** "vocal melody only, free tempo, source key, condition strength 1, AR strength 0.6" */
export function settingsSummary(s: BuildSettings, keyLabel: string, cover = true): string {
  if (!cover) return `condition strength ${s.cfgScale}, AR strength ${s.arStrength}`;
  const tempo = typeof s.tempo === 'number' ? `${s.tempo} BPM` : s.tempo === 'free' ? 'free tempo' : 'source tempo';
  const key = s.keyShift === 0 ? 'source key' : `${s.keyShift > 0 ? '+' : ''}${s.keyShift} st${keyLabel ? ` (${transposeKey(keyLabel, s.keyShift)})` : ''}`;
  return [s.voices === 'vocal' ? 'vocal melody only' : 'vocal + instrumental line', tempo, key,
    `condition strength ${s.cfgScale}`, `AR strength ${s.arStrength}`].join(', ');
}

/** The build's cover settings in one row. Before a build exists they are the
 *  Prepare page's and change at once. A build has its own: edits wait for
 *  Apply, which rebuilds the build's request before its first keep and starts
 *  a fresh build after it. */
function SettingsRow({ value, onApply, build, kept, disabled, keyLabel, sourceBpm, cover }: {
  value: BuildSettings; onApply: (settings: BuildSettings) => void; build: boolean; kept: boolean; disabled: boolean; keyLabel: string; sourceBpm: number; cover: boolean;
}) {
  const saved = JSON.stringify(value);
  const [draft, setDraft] = useState(value);
  useEffect(() => { setDraft(JSON.parse(saved)); }, [saved]);
  const shown = build ? draft : value;
  const set = (patch: Partial<BuildSettings>) => { const next = { ...shown, ...patch }; if (build) setDraft(next); else onApply(next); };
  const changed = build && JSON.stringify(draft) !== saved;
  const tempoMode = typeof shown.tempo === 'number' ? 'set' : shown.tempo;
  // Once a section is kept the settings are the build's for good: changing them
  // starts a fresh build. The row folds to a summary and opens only on request.
  const [open, setOpen] = useState(false);
  if (kept && !open) return (
    <div className="flex flex-wrap items-center gap-2 border-b border-zinc-200 dark:border-white/10 px-4 py-2 text-xs text-zinc-500">
      <Lock className="w-3 h-3" />
      <span className="min-w-0 flex-1">Settings locked: the kept sections were built with {settingsSummary(value, keyLabel, cover)}. Changing them starts a fresh build.</span>
      <button type="button" onClick={() => setOpen(true)} disabled={disabled}
        className="flex items-center gap-1 text-cyan-700 dark:text-cyan-300 disabled:opacity-40">Change<ChevronDown className="w-3 h-3" /></button>
    </div>
  );
  return (
    <fieldset disabled={disabled} className="border-b border-zinc-200 dark:border-white/10 px-4 py-2 space-y-2">
      <div className="flex flex-wrap items-start gap-x-5 gap-y-3 [&>*]:min-w-[150px] [&>*]:flex-1" aria-label="Build settings">
        {cover && <div className="space-y-2">
          <ParamLabel label="Voices" info="Vocal melody alone gives the new style room to choose its instruments. Both keeps the transcribed instrumental line too." />
          <StyledSelect accent="pink" value={shown.voices} disabled={disabled} onChange={v => set({ voices: v as BuildSettings['voices'] })}
            options={[{ value: 'vocal', label: 'Vocal melody only' }, { value: 'both', label: 'Vocal + instrumental line' }]} />
        </div>}
        {cover && <div className="space-y-2">
          <ParamLabel label="Tempo" info="Free removes the score tempo; Source retains it; Set writes a target BPM using the score's beat unit." />
          <StyledSelect accent="pink" value={tempoMode} disabled={disabled}
            onChange={v => set({ tempo: v === 'set' ? (typeof shown.tempo === 'number' ? shown.tempo : Math.max(20, Math.min(300, sourceBpm))) : v as 'free' | 'source' })}
            options={[{ value: 'free', label: 'Free (style decides)' }, { value: 'source', label: 'Source tempo' }, { value: 'set', label: 'Set BPM' }]} />
        </div>}
        {cover && typeof shown.tempo === 'number' && <EditableSlider label="Target BPM" value={shown.tempo} min={20} max={300} step={1} disabled={disabled}
          onChange={v => set({ tempo: v })} helpText={`Source: ${sourceBpm} BPM`} />}
        {cover && <EditableSlider label="Pitch Shift" value={shown.keyShift} min={-12} max={12} step={1} defaultValue={0} disabled={disabled || !keyLabel}
          onChange={v => set({ keyShift: v })}
          formatDisplay={v => v === 0 ? 'Source key' : `${v > 0 ? '+' : ''}${v} st → ${transposeKey(keyLabel, v)}`}
          helpText={keyLabel ? `Source: ${keyLabel}.` : 'No supported score key found.'}
          tooltip="Moves the score up or down by whole semitones, within twelve." />}
        <EditableSlider label="Condition strength" value={shown.cfgScale} min={0.1} max={2} step={0.05} defaultValue={1} disabled={disabled}
          onChange={v => set({ cfgScale: v })} tooltip="Tightens style, lyrics and score together." />
        <CoverArStrength value={shown.arStrength} onChange={v => set({ arStrength: v })} disabled={disabled} />
      </div>
      {build && (changed || kept) && <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
        <span className="flex-1">{kept
          ? 'The kept sections were built with the settings above. Changing them starts a fresh build; this one is set aside.'
          : 'Applying rebuilds this build with the new settings. Takes made before it can no longer be kept.'}</span>
        {kept && !changed && <button type="button" onClick={() => setOpen(false)} className="rounded-lg px-3 py-1 text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200">Close</button>}
        {changed && <>
          <button type="button" onClick={() => { setDraft(JSON.parse(saved)); if (kept) setOpen(false); }} className="rounded-lg px-3 py-1 text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200">Discard</button>
          <button type="button" onClick={() => onApply(draft)}
            className="rounded-lg bg-cyan-600 px-3 py-1 font-semibold text-white disabled:opacity-40">{kept ? 'Start a fresh build with these' : 'Apply'}</button>
        </>}
      </div>}
    </fieldset>
  );
}

interface PageProps {
  session: Yue2BuildSession | null;
  /** The score's sections before a session exists; the session's after. */
  sections: Array<{ label: string; startBar: number }>;
  totalBars: number;
  /** The lyrics the build follows (the session's once it exists). */
  lyrics: string;
  /** The Prepare page's lyrics, which the build starts from until it is given its own. */
  prepareLyrics: string;
  /** Applies new lyrics to this build only; true when accepted. */
  onApplyLyrics: (text: string) => Promise<boolean>;
  /** A saved build is still loading, so its lyrics cannot be edited yet. */
  loading: boolean;
  /** The settings the build uses: the session's once it exists, else the Prepare page's. */
  settings: BuildSettings;
  /** Sets them: on the Prepare page before a build exists, on the build before its
   *  first keep, and through a fresh build after it. */
  onSettings: (settings: BuildSettings) => void;
  /** The score's key ('' when it has none) and tempo, for the pitch and BPM readouts. */
  keyLabel: string;
  sourceBpm: number;
  title: string;
  /** False for a build that is not a cover (Custom-Gen's): the voices, tempo and key shift controls do not show. Default true. */
  coverSettings?: boolean;
  /** The back button's label. Default "Prepare". */
  backLabel?: string;
  /** The score in source order, and the source recording ('' when there is none or a cut means the
   *  score no longer follows it), for the source bar's playback popup. */
  scoreAbc: string;
  recordingUrl: string;
  /** Why a build cannot start yet, or ''. */
  blocked: string;
  busy: boolean;
  error: string;
  /** The build's caption, which the caption field starts from for each section. */
  caption: string;
  /** `caption`, when non-empty, is these takes' own; `lyrics` (occurrence id to lines) is applied first. True once queued. */
  onGenerate: (to: number, count: number, caption: string, lyrics: Record<string, string>, newChords: boolean) => Promise<boolean>;
  /** Gives unbuilt occurrences their own lines, or (null) the build's again. */
  onOccurrenceLyrics: (edits: Record<string, string | null>) => void;
  /** Sets what the unbuilt part of the output is: section ids (s0...) or upcoming occurrence ids (o3...). */
  onPlan: (upcoming: string[], finished: boolean) => void;
  onCancel: () => void;
  /** Keeps a take, with the crop set on it if any; `count` regenerates from a 'regenerate' cut at once. */
  onKeep: (take: string, crop?: TakeCrop, count?: number) => void;
  onDelete: (take: string) => void;
  onUndo: (section?: number) => void;
  /** Restores the last kept take's uncropped original (frame null). */
  onCrop?: (take: string, frame: number | null, mode?: 'regenerate' | 'advance') => void;
  /** Renders everything kept so far once, so Play build can play it. */
  onPreview: () => void;
  /** Where Save song starts the song: seconds into a first-section take, or null for the top. */
  onTrim?: (take: string, seconds: number | null) => void;
  onStartOver: () => void;
  /** Deletes the whole build on the server, then returns to the empty state. */
  onDeleteBuild: () => void;
  onSave: () => void;
  onBack: () => void;
}

/** Build by section: a song timeline, the next sections' lyrics, and their
 *  takes to compare, keep or delete. */
export const Yue2BuildPage: React.FC<PageProps> = p => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState<{ key: string; from: number; to: number; offset: number } | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [paused, setPaused] = useState(true);
  const generation = useRef(0);
  const [target, setTarget] = useState<string | null>(null);
  const [count, setCount] = useState(4);
  // New chords per take is the default for a build that is not a cover; a cover follows its score.
  const [newChords, setNewChords] = useState(p.coverSettings === false);
  const [menu, setMenu] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  // Crops set on takes not yet kept (applied by Keep), and the take whose crop is being edited.
  const [crops, setCrops] = useState<Record<string, TakeCrop>>({});
  // The previous section's crop while its panel is open (null: closed).
  const [keptCrop, setKeptCrop] = useState<TakeCrop | null>(null);
  const [caption, setCaption] = useState(p.caption);
  const dragging = useRef(false);
  const s = p.session;
  const next = buildNext(s);
  const spine = s?.takes.find(t => t.id === s.kept.at(-1)) ?? null;
  const colours = sectionColours(p.sections.map(x => x.label));
  const items = planItems(s);
  const finished = s?.arrangement.finished ?? false;
  const targetAt = items.findIndex(i => i.id === target);
  const range = targetAt >= next ? targetAt : null;
  const candidates = s?.takes.filter(t => t.from === next && t.parent === (spine?.id ?? null)) ?? [];
  const waiting = s?.pending.reduce((n, x) => n + x.count, 0) ?? 0;
  const chunk = s?.pending[0];
  const chunkStage = ({ starting: 'Starting', semantic: 'Composing', plan: 'Planning', nar: 'Rendering', vae: 'Decoding audio', aligning: 'Aligning lyrics' } as Record<string, string>)[chunk?.phase ?? ''] ?? 'Rendering';
  const done = !!items.length && next >= items.length;
  const editable = !p.busy && waiting === 0 && !p.loading;
  const nameOf = (i: PlanItem) => (i.improv ? 'Improv' : p.sections[i.section]?.label ?? '?');
  // The source section after the song's last source section (an improv adds none): where the source would go next, if the user wants it.
  const nextSource = (items.filter(i => !i.improv).at(-1)?.section ?? -1) + 1;
  const colourOf = (i: PlanItem) => (i.improv ? IMPROV_COLOUR : colours[i.section]);
  const [improvBars, setImprovBars] = useState(8);
  const sourcePopup = useHoverPopup<number>();
  const [sourceLoop, setSourceLoop] = useState(false);
  const section = useSectionPlayer(p.scoreAbc, p.recordingUrl);
  /** Bars of source section `i`, first and last (1-based). */
  const sourceBars = (i: number): [number, number] => [p.sections[i].startBar, (p.sections[i + 1]?.startBar ?? p.totalBars + 1) - 1];

  // Each section starts from the build's caption.
  useEffect(() => { setTarget(null); setCaption(p.caption); }, [next, s?.id, p.caption]);
  // Sections just added from the source bar arrive selected, so Generate is ready.
  const selectAdded = useRef(false);
  const lastItem = items.at(-1)?.id;
  useEffect(() => {
    if (!selectAdded.current || !lastItem) return;
    selectAdded.current = false;
    setTarget(lastItem);
  }, [items.length, lastItem]);
  // Crops belong to the takes of the section being built.
  useEffect(() => { setCrops({}); setKeptCrop(null); }, [spine?.id, s?.id]);
  // Lyric drafts belong to the build they were typed in.
  useEffect(() => { setDrafts({}); }, [s?.id]);
  useEffect(() => {
    const up = () => { dragging.current = false; };
    window.addEventListener('pointerup', up);
    return () => window.removeEventListener('pointerup', up);
  }, []);

  /** Stops playback and releases the audio element. */
  const release = (audio: HTMLAudioElement | null) => {
    generation.current++;
    if (audio) { audio.pause(); audio.removeAttribute('src'); audio.load(); }
  };
  const stop = () => { release(audioRef.current); setPlaying(null); setProgress(null); };
  /** Plays `url` from `at` seconds on the song's clock; the playhead covers [from, to].
   *  The file starts `offset` seconds into the song (a take holds only its own part). */
  const start = (key: string, url: string, from: number, to: number, at = from, offset = 0) => {
    const audio = audioRef.current;
    if (!audio) return;
    section.stop();
    const mine = ++generation.current;
    audio.src = url;
    audio.currentTime = Math.max(0, at - offset);
    setPlaying({ key, from, to, offset });
    setProgress(Math.min(1, Math.max(0, (at - from) / Math.max(0.1, to - from))));
    audio.play().catch(() => { if (generation.current === mine) stop(); });
  };
  /** Play, pause and resume for `key`; a different `key` starts it from the top. */
  const toggle = (key: string, url: string, from: number, to: number, offset = 0) => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing?.key !== key) return start(key, url, from, to, from, offset);
    if (!audio.paused) { generation.current++; audio.pause(); return; }
    const mine = ++generation.current;
    audio.play().catch(() => { if (generation.current === mine) stop(); });
  };
  /** Seeks to `at` seconds on the song's clock within [from, to] of a take, keeping it paused if it is; starts it if idle. */
  const seek = (key: string, url: string, from: number, to: number, at: number, offset = 0) => {
    const audio = audioRef.current;
    if (!audio || playing?.key !== key || playing.from !== from || playing.to !== to) return start(key, url, from, to, at, offset);
    audio.currentTime = at - offset;
    setProgress(Math.min(1, Math.max(0, (at - from) / Math.max(0.1, to - from))));
  };
  // Playback ends with the build it belongs to: a session change or leaving the page.
  // React clears the ref before this cleanup runs on unmount, so hold the element.
  useEffect(() => {
    const audio = audioRef.current;
    return () => { release(audio); setPlaying(null); setProgress(null); };
  }, [s?.id]);
  const keep = (take: string) => { stop(); p.onKeep(take, crops[take]); };
  const undo = (section?: number) => { stop(); p.onUndo(section); };
  const remove = (take: string) => { if (playing?.key === take) stop(); p.onDelete(take); };
  /** What a take plays: from a lead-in before its new part (or its in point) to its end (or its out point). */
  const takeWindow = (take: Yue2BuildTake, crop?: TakeCrop | null) => {
    const start = takePrefix(take) / FPS;
    const from = crop?.in ?? Math.max(0, start - (start ? LEAD_IN : 0)), to = crop?.out ?? take.frames / FPS;
    return { from, to, newPart: to - Math.max(start, crop?.in ?? 0) };
  };

  /** Where the playhead is on the song's clock while `key` plays, else null. */
  const playheadOf = (key: string) => playing?.key === key && progress !== null ? playing.from + progress * (playing.to - playing.from) : null;
  /** A take's crop controls under its waveform: what it keeps, its ends set at the playhead, and,
   *  for a cut inside a section, what the next take does. `onMode`, when given, acts on that choice at once. */
  const cropTools = (take: Yue2BuildTake, crop: TakeCrop | null, head: number | null, set: (crop: TakeCrop | null) => void,
    onMode?: (crop: TakeCrop) => void) => {
    const limits = cropLimits(take);
    const c: TakeCrop = crop ?? { in: null, out: null, mode: null };
    const cropped = c.in !== null || c.out !== null;
    const point = c.out !== null ? cropPoint(take, Math.round(c.out * FPS)) : null;
    const pointItem = point ? items[point.occurrence] : undefined;
    const following = point ? items[point.occurrence + 1] : undefined;
    // A cut past `to` falls in the section the take ran on into, which may not be planned yet.
    const runOn = !!point && point.occurrence > take.to;
    const runOnName = take.after === 'improv' ? 'Improv' : s?.arrangement.sections.find(x => x.id === take.after)?.label
      ?? (pointItem ? nameOf(pointItem) : 'the next section');
    const pointName = runOn ? runOnName : pointItem ? nameOf(pointItem) : null;
    const button = 'rounded border border-zinc-300 dark:border-white/10 px-2 py-0.5 hover:border-cyan-500 disabled:opacity-40';
    return <div className="space-y-1 text-xs text-zinc-500">
      <div className="flex flex-wrap items-center gap-2">
        <span className="basis-full"><Scissors className="mr-1 inline w-3 h-3" />{cropped
          ? `Keeps ${limits.inMax > 0 ? clock(c.in ?? 0) : 'it from the start'} to ${clock(c.out ?? limits.end)}${point && pointName ? `, ending ${point.inside ? 'inside' : 'at the start of'} ${pointName}` : ''}${c.in !== null ? '; the saved song fades in there' : ''}.`
          : 'Tap or drag the waveform to play from a point; End at playhead crops the take there.'}</span>
        {limits.inMax > 0 && <button type="button" className={button} disabled={head === null} onClick={() => set(moveCropEdge(take, c, 'in', head!))}>Start at playhead</button>}
        <button type="button" className={button} disabled={head === null} onClick={() => set(moveCropEdge(take, c, 'out', head!))}>End at playhead</button>
        {cropped && <button type="button" className={button} onClick={() => set(null)}>No crop</button>}
      </div>
      {cropNeedsMode(take, c) && pointName && <div className="flex flex-wrap items-center gap-2" role="radiogroup" aria-label="What the next take does">
        <span>The next take:</span>
        {(runOn
          ? [['regenerate', `keeps this start of ${pointName}; the next take carries on from the cut`],
            ['advance', `drops it; the next take starts ${pointName} afresh`]] as const
          : [['regenerate', `composes ${pointName} again from the cut`],
            ['advance', advanceLabel(pointName, following ? nameOf(following) : null, finished)]] as const)
          .map(([mode, text]) => <button type="button" key={mode} role="radio" aria-checked={c.mode === mode} onClick={() => (onMode ?? set)({ ...c, mode })}
            className={`rounded-lg border px-2 py-1 ${c.mode === mode ? 'border-cyan-500 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300' : 'border-zinc-300 dark:border-white/10'}`}>{text}</button>)}
      </div>}
    </div>;
  };

  // Lyrics of the chosen sections (block i pairs with section i).
  const shown = done ? [] : items.slice(next, (range ?? next) + 1).map(i => ({ item: i, lines: occurrenceLines(p.lyrics, s, i) }));
  const current = Object.fromEntries(shown.map(x => [x.item.id, x.lines]));
  const generate = async () => {
    if (range === null) return;
    if (await p.onGenerate(range, count, submittedCaption(caption, p.caption), pendingLyrics(drafts, current), newChords)) setDrafts({});
  };
  const bars = (i: number) => s?.arrangement.sections[i]?.bars ?? (p.sections[i + 1]?.startBar ?? p.totalBars + 1) - p.sections[i].startBar;
  const itemBars = (i: PlanItem) => i.improv ?? bars(i.section);
  const rangeLabel = range === null ? '' : `${nameOf(items[next])}${range > next ? ` → ${nameOf(items[range])}` : ''}`;
  const edit = (change: Parameters<typeof editedPlan>[2], ends = finished) => {
    // The only section, nothing kept: the server keeps at least one, so go back to no build at all.
    if ('remove' in change && !s?.kept.length && items.length === 1) { p.onStartOver(); return; }
    p.onPlan(editedPlan(items, next, change), ends);
  };
  /** Adds source section `i` to the end of the song, selected. Adding the last one ticks Ends the song. */
  const add = (i: number) => {
    selectAdded.current = true;
    edit({ add: i }, finished || i === p.sections.length - 1);
  };
  const crop = (frame: number | null) => { stop(); if (spine) p.onCrop?.(spine.id, frame); };
  // The saved song's start, when the first kept take is the trimmed one.
  const first = s?.takes.find(t => t.id === s.kept[0]);
  const trimAt = first && s?.trim && (first.cropOf ?? first.id) === s.trim.take ? s.trim.seconds : null;
  // What Play build plays: the build's own render while it is current, or a
  // spine from an older build whose audio still holds the whole song.
  const preview = s?.preview && spine && s.preview.spine === spine.id ? s.preview : null;
  const buildAudio = preview?.status === 'done' ? preview.audioUrl ?? null : spine && !spine.audioStart ? spine.audioUrl : null;

  return (
    <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
      <audio ref={audioRef} className="hidden" onEnded={() => { setPlaying(null); setProgress(null); }}
        onPlay={() => setPaused(false)} onPause={() => setPaused(true)}
        onTimeUpdate={e => {
          if (!playing) return;
          const t = e.currentTarget.currentTime + playing.offset;
          if (t >= playing.to) { stop(); return; }
          setProgress(Math.min(1, Math.max(0, (t - playing.from) / Math.max(0.1, playing.to - playing.from))));
        }} />
      {/* Top bar: what is being built and how. */}
      <div className="flex flex-wrap items-center gap-3 border-b border-zinc-200 dark:border-white/10 px-4 py-2">
        <button type="button" onClick={p.onBack} className="flex items-center gap-1 text-xs text-zinc-600 dark:text-zinc-300 hover:text-cyan-600">
          <ArrowLeft className="w-3.5 h-3.5" />{p.backLabel ?? 'Prepare'}</button>
        <p className="min-w-0 flex-1 truncate text-sm font-semibold text-zinc-900 dark:text-white">{p.title}</p>
        <div className="relative">
          <button type="button" onClick={() => setMenu(m => !m)} disabled={!s} aria-label="Build actions"
            className="rounded p-1.5 text-zinc-500 hover:bg-zinc-200 dark:hover:bg-white/10 disabled:opacity-40"><MoreHorizontal className="w-4 h-4" /></button>
          {menu && s && <div className="absolute right-0 top-8 z-20 w-44 rounded-lg border border-zinc-200 dark:border-white/10 bg-white dark:bg-zinc-900 py-1 text-xs shadow-xl"
            onMouseLeave={() => setMenu(false)}>
            <button type="button" disabled={!s.kept.length} onClick={() => { setMenu(false); undo(); }}
              className="block w-full px-3 py-1.5 text-left hover:bg-zinc-100 dark:hover:bg-white/5 disabled:opacity-40">Undo last keep</button>
            <button type="button" disabled={!s.kept.length} onClick={() => { setMenu(false); p.onSave(); }}
              title="Render everything kept so far as a library song; this ends the build"
              className="block w-full px-3 py-1.5 text-left hover:bg-zinc-100 dark:hover:bg-white/5 disabled:opacity-40">Save the song so far</button>
            <button type="button" onClick={() => { setMenu(false); p.onStartOver(); }}
              className="block w-full px-3 py-1.5 text-left text-red-500 hover:bg-zinc-100 dark:hover:bg-white/5">Start over</button>
            <button type="button" onClick={() => { setMenu(false); if (window.confirm('Delete this build and all its takes? This cannot be undone.')) p.onDeleteBuild(); }}
              className="block w-full px-3 py-1.5 text-left text-red-500 hover:bg-zinc-100 dark:hover:bg-white/5">Delete build</button>
          </div>}
        </div>
      </div>

      <SettingsRow value={p.settings} onApply={p.onSettings} build={!!s} kept={!!s?.kept.length} disabled={!editable}
        keyLabel={p.keyLabel} sourceBpm={p.sourceBpm} cover={p.coverSettings !== false} />

      {/* Source bar: the score as it is. Song bar: what the output will be, kept occurrences first. */}
      <div className="border-b border-zinc-200 dark:border-white/10 px-4 py-3 space-y-2">
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span className="font-semibold uppercase tracking-wide">Source</span>
          <span className="ml-auto">Click a section to add it to the end of the song; repeat it as often as you like.</span>
          <span className="flex items-center rounded-md border-t-4" style={{ borderColor: IMPROV_COLOUR, backgroundColor: `${IMPROV_COLOUR}22` }}>
            <button type="button" disabled={!editable} onClick={() => { selectAdded.current = true; edit({ improv: improvBars }); }}
              title="Adds bars the model makes up itself to the end of the song: its takes compose them without the score or the lyrics, then the next section follows the score again."
              className="px-2 py-1 text-[11px] font-semibold text-zinc-800 dark:text-zinc-100 disabled:opacity-40">+ Improv</button>
            <div className="w-24">
              <StyledSelect accent="purple" size="sm" aria-label="Improv length in bars" value={improvBars} disabled={!editable}
                onChange={v => setImprovBars(Number(v))} options={IMPROV_BARS.map(n => ({ value: n, label: `${n} bars` }))} />
            </div>
          </span>
        </div>
        <div className="flex gap-1 select-none" aria-label="Source sections">
          {p.sections.map((src, i) => {
            const uses = items.filter(x => x.section === i).length;
            const upNext = i === nextSource;
            return <div key={i} className="flex min-w-0" style={{ flexGrow: Math.max(1, bars(i)), flexBasis: 0 }}
              onMouseEnter={e => sourcePopup.show(i, e.currentTarget)} onMouseLeave={sourcePopup.close}>
              <button type="button" disabled={!editable} onClick={() => add(i)} aria-label={`Add ${src.label} to the end of the song`}
                style={{ borderColor: colours[i], backgroundColor: `${colours[i]}22` }}
                className={`w-full min-w-0 rounded-md border-t-4 px-1.5 py-1 text-left disabled:opacity-40 ${upNext ? 'outline-dashed outline-1 outline-offset-1 outline-pink-500' : ''}`}>
                <p className="truncate text-[11px] font-semibold text-zinc-800 dark:text-zinc-100">{src.label}</p>
                <p className="truncate text-[10px] text-zinc-500">{upNext ? <span className="text-pink-600 dark:text-pink-400">next in source · </span> : ''}{bars(i)} bars{uses > 1 ? ` · used ${uses}×` : ''}</p>
              </button>
              {sourcePopup.open?.key === i && <SectionPlayPopup at={sourcePopup.open} name={src.label}
                onMouseEnter={sourcePopup.keep} onMouseLeave={sourcePopup.close}
                onHearScore={() => { stop(); void section.hearScore(...sourceBars(i), sourceLoop); }}
                onHearRecording={() => { stop(); section.hearRecording(...sourceBars(i), sourceLoop); }}
                recordingDisabled={!section.canHearRecording}
                recordingTitle={section.canHearRecording ? 'Plays the source recording from one bar before the section to its end, by the score\'s own tempo, so it can drift.'
                  : 'Not available: there is no recording, or a cut means the score no longer follows it.'}
                playing={section.playing !== null} onStop={section.stop} loop={sourceLoop} onLoop={setSourceLoop} />}
            </div>;
          })}
        </div>
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span className="font-semibold uppercase tracking-wide">Song</span>
          {buildAudio || !spine
            ? <button type="button" disabled={!spine} onClick={() => spine && buildAudio && toggle('build', buildAudio, trimAt ?? 0, spine.frames / FPS)}
              className="flex items-center gap-1 text-cyan-700 dark:text-cyan-300 disabled:opacity-40">
              {playing?.key === 'build' && !paused ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}Play build</button>
            : preview?.status === 'rendering'
              ? <span role="status" className="flex items-center gap-1 text-amber-600"><Loader2 className="w-3 h-3 animate-spin" />Rendering the build: {preview.progress ?? 0}%</span>
              : <button type="button" onClick={p.onPreview} disabled={p.busy || waiting > 0}
                title="Takes hold only their own part, so the whole build is rendered once to play it. It is quicker than a take: nothing is composed."
                className="flex items-center gap-1 text-cyan-700 dark:text-cyan-300 disabled:opacity-40"><Play className="w-3 h-3" />Render the build so far</button>}
          {preview?.status === 'failed' && !buildAudio && <span role="alert" className="text-red-500" title={preview.error}>Build render failed</span>}
          <Toggle size="sm" accent="cyan" checked={finished} disabled={!editable} label="Ends the song"
            info="Tick it before building the last section: that take composes an ending. Until then each take runs a few seconds into the section that follows it in the source, so the next one has a seam to start from."
            onChange={ends => p.onPlan(items.slice(next).map(i => i.id), ends)} />
          <span className="ml-auto">{!items.length ? 'Add sections from the source bar above.' : done ? 'Every section is built.' : `Click a section to build up to it, starting from ${nameOf(items[next])}.`}</span>
        </div>
        <div className="flex gap-1 select-none" aria-label="Song sections">
          {items.map((item, i) => {
            const picked = range !== null && i >= next && i <= range;
            const start = spine?.starts[i];
            return <div key={item.id} style={{ flexGrow: Math.max(1, itemBars(item)), flexBasis: 0, borderColor: colourOf(item),
              backgroundColor: item.kept ? `${colourOf(item)}55` : picked ? `${colourOf(item)}33` : `${colourOf(item)}12` }}
              onClick={() => { if (item.kept) { if (spine && buildAudio && start != null) toggle(`s${item.id}`, buildAudio, start / FPS, spine.frames / FPS); } else setTarget(t => t === item.id ? null : item.id); }}
              title={item.kept ? (buildAudio ? `Play the build from ${nameOf(item)}` : 'Render the build so far to play it') : item.id === target ? 'Click again to clear the selection' : `Build up to ${nameOf(item)}`}
              className={`group relative min-w-0 cursor-pointer rounded-md border-t-4 px-1.5 py-1 ${picked ? 'ring-2 ring-pink-500' : ''} ${!item.kept && i > next && !picked ? 'opacity-60' : ''}`}>
              <p className="truncate text-[11px] font-semibold text-zinc-800 dark:text-zinc-100">
                {item.kept ? '✓ ' : i === next ? '→ ' : ''}{nameOf(item)}</p>
              <p className="truncate text-[10px] text-zinc-500">{item.kept ? (start != null ? clock(start / FPS) : 'built') : i === next ? 'next' : `${itemBars(item)} bars`}</p>
              {item.kept && <button type="button" onClick={e => { e.stopPropagation(); undo(i); }}
                title={`Undo back to ${nameOf(item)}, to build it again`}
                className="absolute right-1 top-1 hidden rounded p-0.5 text-zinc-500 hover:text-pink-500 group-hover:block"><RotateCcw className="w-3 h-3" /></button>}
              {!item.kept && <div className="absolute right-0.5 top-0.5 hidden items-center group-hover:flex">
                <button type="button" disabled={!editable || i === next} onClick={e => { e.stopPropagation(); edit({ move: item.id, by: -1 }); }}
                  aria-label={`Move ${nameOf(item)} earlier`} className="rounded p-0.5 text-zinc-500 hover:text-pink-500 disabled:opacity-30"><ChevronLeft className="w-3 h-3" /></button>
                <button type="button" disabled={!editable || i === items.length - 1} onClick={e => { e.stopPropagation(); edit({ move: item.id, by: 1 }); }}
                  aria-label={`Move ${nameOf(item)} later`} className="rounded p-0.5 text-zinc-500 hover:text-pink-500 disabled:opacity-30"><ChevronRight className="w-3 h-3" /></button>
                <button type="button" disabled={!editable} onClick={e => { e.stopPropagation(); edit({ remove: item.id }); }}
                  aria-label={`Remove ${nameOf(item)} from the song`} className="rounded p-0.5 text-zinc-500 hover:text-red-500 disabled:opacity-30"><X className="w-3 h-3" /></button>
              </div>}
            </div>;
          })}
        </div>
        {spine && (spine.cropOf || trimAt !== null) && <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500">
          <Scissors className="w-3 h-3" />
          {trimAt !== null && <>
            <span>The saved song starts at {clock(trimAt)}, faded in.</span>
            {p.onTrim && <button type="button" onClick={() => first && p.onTrim?.(first.id, null)} disabled={!editable}
              className="text-cyan-700 dark:text-cyan-300 disabled:opacity-40">Start from the top</button>}
          </>}
          {spine.cropOf && p.onCrop && <>
            <span>The last kept take is cropped at {clock(spine.frames / FPS)}.</span>
            <button type="button" onClick={() => crop(null)} disabled={!editable}
              className="text-cyan-700 dark:text-cyan-300 disabled:opacity-40">Undo crop</button>
          </>}
        </div>}
      </div>

      {/* The next sections: their lyrics, and their takes. */}
      <div className="flex-1 min-h-0 grid grid-cols-[minmax(240px,1fr)_minmax(320px,1.4fr)] overflow-hidden">
        <div className="min-h-0 flex flex-col border-r border-zinc-200 dark:border-white/10">
        <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3">
          <div className="flex items-center gap-2">
            <p className="flex-1 text-xs font-semibold uppercase tracking-wide text-zinc-500">
              {done ? 'Done' : items.length ? `Building: ${rangeLabel || nameOf(items[next])}` : 'Nothing to build yet'}</p>
            {p.lyrics !== p.prepareLyrics && <button type="button" onClick={() => void p.onApplyLyrics(p.prepareLyrics)} disabled={!editable}
              title="Replace this build's lyrics with the Prepare page's, for every section not yet built"
              className="text-xs text-zinc-500 hover:text-cyan-600 disabled:opacity-40">Use Prepare lyrics</button>}
          </div>
          {shown.map(({ item, lines }) => <div key={item.id} className="space-y-1">
            <div className="flex items-center gap-2">
              <p className="flex-1 text-xs font-semibold text-zinc-700 dark:text-zinc-200">{nameOf(item)}</p>
              {lines !== null && (item.lyrics !== undefined || item.id in drafts) && <button type="button" disabled={!editable}
                onClick={() => { setDrafts(d => Object.fromEntries(Object.entries(d).filter(([id]) => id !== item.id))); if (item.lyrics !== undefined) p.onOccurrenceLyrics({ [item.id]: null }); }}
                title="Sing this build's lyrics for this section again"
                className="flex items-center gap-1 text-xs text-zinc-500 hover:text-cyan-600 disabled:opacity-40"><RotateCcw className="w-3 h-3" />Reset</button>}
            </div>
            {lines === null ? <p className="text-xs text-zinc-500">Instrumental: no lyrics for this section.</p>
              : <textarea value={drafts[item.id] ?? lines} disabled={!editable} maxLength={20000} spellCheck={false}
                rows={Math.min(14, Math.max(3, (drafts[item.id] ?? lines).split('\n').length + 1))}
                aria-label={`Lyrics for ${nameOf(item)}`}
                onChange={e => { const text = e.target.value; setDrafts(d => ({ ...d, [item.id]: text })); }}
                className="w-full rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/30 p-2 font-mono text-sm leading-relaxed text-zinc-800 dark:text-zinc-200" />}
          </div>)}
          {shown.length > 0 && <p className="text-xs text-zinc-500">
            Edits apply to this section only, when you generate. The Prepare page, the dataset and other copies of the section keep theirs.
            {candidates.length && Object.keys(pendingLyrics(drafts, current)).length ? ' Takes made with the old lyrics can no longer be kept.' : ''}
          </p>}
        </div>
        {/* This section's caption, under its lyrics at the foot of the column. */}
        {!done && <div className="space-y-1 border-t border-zinc-200 dark:border-white/10 p-4">
          <div className="flex items-center gap-2">
            <ParamLabel label="Caption" className="flex-1 text-xs text-zinc-500"
              info="The caption for this section's takes, starting from the build's. Edit it to steer this section only; the next section starts from the build's again. It changes what is composed (the AR), and the adapter trigger is still added. The sound of the whole song is rendered (the NAR) with one caption: a take's preview uses its own for everything it plays, and Render Final Song uses the build's for the whole song. Hover a take's name to see the prompt it was sent; ✎ marks a take with its own caption." />
            {caption !== p.caption && <button type="button" onClick={() => setCaption(p.caption)} title="Back to the build's caption"
              aria-label="Back to the build's caption" className="rounded p-1 text-zinc-500 hover:text-cyan-600"><RotateCcw className="w-3 h-3" /></button>}
          </div>
          <textarea value={caption} maxLength={4000} rows={4} placeholder="The build's caption" aria-label="Caption for this section's takes"
            onChange={e => setCaption(e.target.value)}
            className="block w-full min-h-[5.5rem] resize-y rounded-lg border border-zinc-300 dark:border-white/10 bg-white dark:bg-black/30 p-2 text-xs text-zinc-800 dark:text-zinc-200" />
        </div>}
        </div>
        <div className="min-h-0 flex flex-col p-4 gap-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Takes</p>
          <div className="flex-1 min-h-0 overflow-y-auto space-y-2">
            {/* The last kept take, behind a button so it is not mistaken for a new one: crop where it ends
                (and, for the first section, where the song starts). Always cut from its uncropped original. */}
            {spine && s && p.onCrop && (() => {
              const base = s.takes.find(t => t.id === spine.cropOf) ?? spine;
              const current: TakeCrop = { in: base.from === 0 ? trimAt : null, out: spine.cropOf ? spine.frames / FPS : null, mode: null };
              if (!keptCrop) return <button type="button" onClick={() => setKeptCrop(current)} disabled={!editable}
                title="Crop what is already kept: where the last kept take ends (and, for the first section, where the song starts). Takes made from it stop being keepable."
                className="flex w-full items-center justify-center gap-1 rounded-lg border border-dashed border-zinc-300 dark:border-white/10 px-3 py-1.5 text-xs text-zinc-600 dark:text-zinc-300 hover:border-cyan-500 disabled:opacity-40">
                <Scissors className="w-3 h-3" />Crop previous section{spine.cropOf || trimAt !== null ? ' (cropped)' : ''}</button>;
              const full = takeWindow(base);
              const offset = (base.audioStart ?? 0) / FPS;
              const colour = (items[base.to] ? colourOf(items[base.to]) : undefined) ?? '#06b6d4';
              const changed = keptCrop.in !== current.in || keptCrop.out !== current.out;
              const needsMode = keptCrop.out !== current.out && cropNeedsMode(base, keptCrop);
              const apply = () => {
                stop();
                setKeptCrop(null);
                if (keptCrop.out !== current.out) p.onCrop?.(spine.id, keptCrop.out === null ? null : Math.round(keptCrop.out * FPS), keptCrop.mode ?? undefined);
                if (keptCrop.in !== current.in) p.onTrim?.(base.id, keptCrop.in);
              };
              return <div className="space-y-2 rounded-lg border border-cyan-500/40 bg-cyan-500/5 px-3 py-2">
                <div className="flex items-center gap-3">
                  <span className="flex-1 text-xs font-semibold text-zinc-700 dark:text-zinc-200" title="The last take you kept, uncropped">Previous section</span>
                  <button type="button" onClick={() => toggle('kept', base.audioUrl, full.from, full.to, offset)}
                    aria-label={playing?.key === 'kept' && !paused ? 'Pause the previous section' : 'Play the previous section'}
                    className="rounded-full bg-zinc-100 dark:bg-white/10 p-2 text-zinc-700 dark:text-zinc-200 hover:text-pink-500">
                    {playing?.key === 'kept' && !paused ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}</button>
                  <button type="button" onClick={apply} disabled={!editable || !changed || (needsMode && !keptCrop.mode)}
                    title={needsMode && !keptCrop.mode ? 'Choose what the next take does first' : 'Apply the crop to the kept build'}
                    className="rounded-lg bg-cyan-600 px-3 py-1 text-xs font-semibold text-white disabled:opacity-40">Apply</button>
                  <button type="button" onClick={() => { if (playing?.key === 'kept') stop(); setKeptCrop(null); }} aria-label="Close without cropping"
                    className="rounded p-1 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"><X className="w-4 h-4" /></button>
                </div>
                <TakeWave peaks={base.peaks ?? []} from={full.from} to={full.to} playhead={playheadOf('kept')} crop={keptCrop}
                  canIn={cropLimits(base).inMax > 0} colour={colour}
                  onSeek={at => seek('kept', base.audioUrl, full.from, full.to, at, offset)} />
                {cropTools(base, keptCrop, playheadOf('kept'), c => setKeptCrop(c ?? { in: null, out: null, mode: null }))}
              </div>;
            })()}
            {candidates.map((take, i) => {
              const set = crops[take.id] ?? null;
              const full = takeWindow(take);
              const win = takeWindow(take, set);
              const offset = (take.audioStart ?? 0) / FPS;
              const colour = (items[take.from] ? colourOf(items[take.from]) : undefined) ?? '#ec4899';
              const needsMode = !!set && cropNeedsMode(take, set);
              const setCrop = (c: TakeCrop | null) => setCrops(all => {
                const rest = { ...all };
                delete rest[take.id];
                return c && (c.in !== null || c.out !== null) ? { ...rest, [take.id]: c } : rest;
              });
              return <div key={take.id} className="space-y-2 rounded-lg border border-zinc-200 dark:border-white/10 bg-white dark:bg-black/20 px-3 py-2">
                <div className="flex items-center gap-3">
                  <span className="w-12 text-xs font-semibold text-zinc-700 dark:text-zinc-200" title={take.style ? `Sent: ${take.style}` : take.cropOf ? 'A cropped copy of a take' : undefined}>
                    Take {i + 1}{take.cropOf || set ? ' ✂' : ''}{take.caption ? ' ✎' : ''}</span>
                  <button type="button" onClick={() => toggle(take.id, take.audioUrl, full.from, full.to, offset)} aria-label={playing?.key === take.id && !paused ? `Pause take ${i + 1}` : `Play take ${i + 1}`}
                    className="rounded-full bg-zinc-100 dark:bg-white/10 p-2 text-zinc-700 dark:text-zinc-200 hover:text-pink-500">
                    {playing?.key === take.id && !paused ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}</button>
                  <span className="ml-auto w-10 text-right font-mono text-xs text-zinc-500" title="Length of the new part">{clock(win.newPart)}</span>
                  <button type="button" onClick={() => keep(take.id)} disabled={p.busy || !s || !keepable(s, take) || (needsMode && !set!.mode)}
                    title={s && !keepable(s, take) ? 'Made before the lyrics or the arrangement changed; generate it again' : needsMode && !set!.mode ? 'Choose what the next take does first' : set ? 'Keep it, cropped' : undefined}
                    className="rounded-lg bg-cyan-600 px-3 py-1 text-xs font-semibold text-white disabled:opacity-40">Keep</button>
                  <button type="button" onClick={() => remove(take.id)} disabled={p.busy} aria-label={`Delete take ${i + 1}`}
                    className="rounded p-1 text-zinc-400 hover:text-red-500 disabled:opacity-40"><Trash2 className="w-4 h-4" /></button>
                </div>
                {take.planNote && <p className="text-xs text-amber-600">{take.planNote}</p>}
                <TakeWave peaks={take.peaks ?? []} from={full.from} to={full.to} playhead={playheadOf(take.id)} crop={set}
                  canIn={cropLimits(take).inMax > 0} colour={colour}
                  onSeek={at => seek(take.id, take.audioUrl, full.from, full.to, at, offset)} />
                {(set || playing?.key === take.id) && cropTools(take, set, playheadOf(take.id), setCrop,
                  // Choosing what the next take does keeps this one; 'regenerate' then composes again from the cut.
                  editable && s && keepable(s, take) ? c => { stop(); p.onKeep(take.id, c, c.mode === 'regenerate' ? count : undefined); } : undefined)}
              </div>;
            })}
            {waiting > 0 && <div className="flex items-center gap-2 rounded-lg border border-dashed border-amber-500/50 px-3 py-3 text-xs text-amber-600">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span role="status" className="flex-1">
                {chunk?.activeCount ? `${chunkStage} ${chunk.activeCount} take${chunk.activeCount > 1 ? 's' : ''}: ${chunk.progress ?? 0}%` : 'Queued'}
                {` · ${waiting} take${waiting > 1 ? 's' : ''} remaining`}
              </span>
              <button type="button" onClick={p.onCancel} disabled={p.busy}
                className="rounded px-2 py-1 font-semibold hover:bg-amber-500/10 disabled:opacity-40">Cancel</button>
            </div>}
            {!candidates.length && !waiting && !done && <p className="text-xs text-zinc-500">{items.length ? 'No takes yet. Choose sections on the song bar, then generate.' : 'No takes yet. Add the sections you want from the source bar, then generate.'}</p>}
            {s?.error && <p role="alert" className="text-xs text-red-500">Last take failed: {s.error}</p>}
          </div>
          {!done && <div className="space-y-1">
            <div role="radiogroup" aria-label="Chords for these takes" className="flex gap-2 text-xs"
              title="Takes of a section you have not built yet can each write their own chords, planned after what you kept, or follow the score as it stands. A repeat of a section you kept follows the kept one either way.">
              {([[true, 'New chords per take'], [false, 'Follow the score']] as const).map(([on, text]) =>
                <button type="button" key={text} role="radio" aria-checked={newChords === on} onClick={() => setNewChords(on)}
                  className={`rounded-lg border px-2 py-1 ${newChords === on ? 'border-cyan-500 bg-cyan-500/10 text-cyan-700 dark:text-cyan-300' : 'border-zinc-300 dark:border-white/10'}`}>{text}</button>)}
            </div>
            <div className="flex items-center gap-2">
              <div className="w-20 flex-shrink-0" title="Takes per run">
                <StyledSelect accent="pink" value={count} onChange={v => setCount(Number(v))}
                  options={Array.from({ length: 10 }, (_, k) => ({ value: k + 1, label: String(k + 1) }))} />
              </div>
              <button type="button" onClick={() => void generate()} disabled={range === null || p.busy || waiting > 0 || !!p.blocked}
                className="flex-1 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 py-2.5 text-sm font-bold text-white disabled:opacity-40">
                {range === null ? 'Choose sections on the song bar' : `Generate ${count} take${count > 1 ? 's' : ''} of ${rangeLabel}`}</button>
            </div>
            {p.blocked && <p className="text-xs text-amber-600">{p.blocked}</p>}
            {p.error && <p role="alert" className="text-xs text-red-500">{p.error}</p>}
          </div>}
          {done && <button type="button" onClick={p.onSave} disabled={p.busy}
            title="Render every kept section as one library song, with the usual post-processing"
            className="w-full rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 py-4 text-base font-bold text-white disabled:opacity-40">Render Final Song</button>}
        </div>
      </div>
    </div>
  );
};
