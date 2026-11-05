import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { backendModelsOptions } from '../../queries/global-models/options';
import { QueryError } from '../shared/QueryError';
import { queryManager } from '../../queries/invalidation';
import { apiClient } from '../../services/httpClient';
// Yue2LmAdapterDropdown.tsx — Adapters cluster for a backend whose LM adapters
// are ENGINE STATE rather than per-request parameters.
//
// The sibling of Mm3LmAdapterDropdown, and the difference between them is the
// whole reason there are two: MM3 applies its LoRA per generation, so its
// picker writes to the request bag and nothing happens until Generate. YuE2
// MERGES the delta into the resident LM at load (engine/src/yue2/yue2-adapter.h)
// — there is no way to change it in place, and merging again on top of an
// already-merged model would double it — so committing a pick is a POST that
// tears the model down and the next generation pays a reload.
//
// TWO SLOTS, NOT ONE. YuE2's LM is a Mixture of Transformers: the AR half
// composes (it writes the plan and the semantic tokens) and the NAR half
// renders, and they share no weights at all. An adapter trained on one is
// REFUSED at load on the other, gated on the file's own __metadata__.format.
// So they are two independent picks with their own strengths, stacked into one
// merge — not one list with both halves jumbled into it, which is a choice
// nobody can get right by reading it.
//
// THE DIALS ARE NOT COMMITTED PER TICK. Each slot keeps a local draft and
// commits it when a drag ends, or when a typed value is confirmed (blur or
// Enter), never on every drag frame.
//
// The trigger word is not decoration. A YuE2 adapter is addressed by the word
// it was trained under, and a style prompt that does not lead with it gets a
// model that has been nudged and not much else — so the trigger is shown
// large, copyable, and called out when an adapter has none.
//
// LICENCE: YuE2 weights are CC BY-NC 4.0 — with the authors' carve-out for
// individual creators — and a trained adapter carries the same terms. The
// text is YUE2_LICENSE_NOTICE in the server's backends/yue2/index.ts and
// arrives here as `capabilities.license` — rendered verbatim, never retyped.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, ChevronDown, ChevronRight, FileInput, FolderOpen, Loader2, RotateCcw, Search } from 'lucide-react';
import { useBackendStore } from '../../stores/backendStore';
import { useCapabilities } from '../../hooks/useCapabilities';
import { ParamLabel } from '../shared/ParamLabel';
import { Slider } from '../shared/Slider';
import { StyledSelect } from '../shared/StyledSelect';
import { FileBrowserModal } from '../shared/FileBrowserModal';

const inputClasses =
  'w-full px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 ' +
  'text-sm text-zinc-800 dark:text-zinc-200 outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 ' +
  'disabled:opacity-50';

/** The two halves, in the order the server stacks them. */
type SlotKind = 'ar' | 'nar';
const SLOTS: SlotKind[] = ['ar', 'nar'];

/** Mirrors the engine's Yue2LmAdapterScales: a master, two module-kind dials,
 *  and three depth-band dials, every one defaulting to 1.0 and multiplying
 *  independently. */
interface Dials {
  scale: number;
  attn: number;
  mlp: number;
  early: number;
  mid: number;
  late: number;
  /** Planner strength (AR only): the master while the score is written.
   *  null = follow `scale`. */
  plan: number | null;
  /** Basin re-base beta, 0 to 1 (engine default 0.75): used only when the
   *  adapter was trained on another YuE2 variant than the loaded LM. */
  rebase: number;
}

const UNITY: Dials = { scale: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1, plan: null, rebase: 0.75 };

/** Selection keys the server reads, per slot. Kept in one place because the
 *  spelling has to match backends/yue2/index.ts exactly — a typo here is a
 *  dial that silently does nothing. */
function keysFor(kind: SlotKind) {
  const Slot = kind === 'ar' ? 'Ar' : 'Nar';
  return {
    path: `lmAdapter${Slot}`,
    scale: `lmAdapter${Slot}Scale`,
    attn: `lmAdapter${Slot}ScaleAttn`,
    mlp: `lmAdapter${Slot}ScaleMlp`,
    early: `lmAdapter${Slot}ScaleEarly`,
    mid: `lmAdapter${Slot}ScaleMid`,
    late: `lmAdapter${Slot}ScaleLate`,
    plan: `lmAdapter${Slot}ScalePlan`,
    rebase: `lmAdapter${Slot}Rebase`,
    inForce: `lmAdapter${Slot}InForce`,
  };
}

function formatSize(bytes?: number): string {
  if (!bytes || !Number.isFinite(bytes) || bytes <= 0) return '';
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

/** Last two path segments, extension dropped — enough to tell two runs apart
 *  without spilling an absolute path across the badge. */
function shortName(p: string): string {
  return p.split(/[\\/]/).slice(-2).join('/').replace(/\.safetensors$/i, '');
}

function sameDials(a: Dials, b: Dials): boolean {
  return a.scale === b.scale && a.attn === b.attn && a.mlp === b.mlp
    && a.early === b.early && a.mid === b.mid && a.late === b.late && a.plan === b.plan && a.rebase === b.rebase;
}

/** The catalogue rows the picker renders, plus what the engine says is
 *  actually merged right now. Both come from the one models catalogue, which
 *  the store refetches after every successful selection. */
function useYue2Adapters() {
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const { data: catalogue, isError, error, refetch } = useQuery(backendModelsOptions(activeBackendId));


  const meta = catalogue?.lmAdapterMeta ?? {};
  const paths = useMemo(() => catalogue?.lmAdapters ?? [], [catalogue]);
  const defaults = (catalogue?.defaults ?? {}) as Record<string, unknown>;

  /** An entry whose `kind` the server did not record belongs to neither slot
   *  cleanly. Show it in BOTH rather than hiding it: the engine's format gate
   *  is the real authority, and a picker that silently drops a file someone
   *  trained is worse than one that lets them try it. */
  const pathsFor = (kind: SlotKind) =>
    paths.filter(p => {
      const k = (meta[p] as { kind?: string } | undefined)?.kind;
      return k === undefined || k === kind;
    });

  const dialsFor = (kind: SlotKind): Dials => {
    const k = keysFor(kind);
    const num = (key: string) => {
      const v = Number(defaults[key]);
      return Number.isFinite(v) && v >= 0 ? v : 1;
    };
    return {
      scale: num(k.scale), attn: num(k.attn), mlp: num(k.mlp),
      early: num(k.early), mid: num(k.mid), late: num(k.late),
      plan: defaults[k.plan] === undefined ? null : num(k.plan),
      rebase: Number.isFinite(Number(defaults[k.rebase])) && defaults[k.rebase] !== undefined
        ? Math.min(1, Math.max(0, Number(defaults[k.rebase]))) : UNITY.rebase,
    };
  };

  return {
    activeBackendId,
    catalogue,
    isError,
    error,
    refetch,
    meta,
    pathsFor,
    dialsFor,
    selectedFor: (kind: SlotKind) => String(defaults[keysFor(kind).path] ?? ''),
    /** false until the next warm/synth after a pick — the merge happens at
     *  load, so "chosen" and "in force" genuinely differ for that window. */
    inForceFor: (kind: SlotKind) => defaults[keysFor(kind).inForce] === true,
    /** The picked LM's variant ('' = base); an adapter whose `home` differs is re-based. */
    loadedVariant: String(defaults.lmVariant ?? ''),
  };
}

interface SlotPanelProps {
  kind: SlotKind;
  title: string;
  blurb: string;
  paths: string[];
  meta: Record<string, any>;
  selected: string;
  dials: Dials;
  inForce: boolean;
  loadedVariant: string;
  disabled: boolean;
  onCommit: (kind: SlotKind, path: string | null, dials: Dials) => Promise<void>;
  /** Save a typed trigger word for an adapter file that carries none. */
  onTrigger: (path: string, trigger: string) => Promise<void>;
}

const SlotPanel: React.FC<SlotPanelProps> = ({
  kind, title, blurb, paths, meta, selected, dials, inForce, loadedVariant, disabled, onCommit, onTrigger,
}) => {
  const { t } = useTranslation();
  const [draft, setDraftState] = useState<Dials>(dials);
  const [showDepth, setShowDepth] = useState(false);
  // The latest draft, readable in the same event that changed it: a reset
  // sets the value and commits in one handler, before React re-renders.
  const draftRef = useRef(draft);
  const setDraft = (next: Dials) => { draftRef.current = next; setDraftState(next); };
  const edit = (patch: Partial<Dials>) => setDraft({ ...draftRef.current, ...patch });
  // Called when a drag ends, a typed value is confirmed, or on reset.
  const requestCommit = () => {
    const d = draftRef.current;
    if (!disabled && !sameDials(d, dials)) void onCommit(kind, selected, d);
  };

  // The server is the source of truth for what is committed. Whenever it says
  // something different — another tab, a reconcile after an engine restart,
  // this panel's own successful commit — the draft is replaced rather than
  // merged: a half-dragged slider is not a change anyone asked to keep.
  useEffect(() => { setDraft(dials); }, [dials.scale, dials.attn, dials.mlp, dials.early, dials.mid, dials.late, dials.plan, dials.rebase]);

  const entry = selected ? meta[selected] : undefined;
  // Trained on another variant: the engine re-bases it onto the loaded LM.
  const variantName = (v: string) => v || t('globalBar.yue2BaseVariant', 'YuE2 base');
  const home = typeof entry?.home === 'string' && entry.home !== loadedVariant ? variantName(entry.home) : '';
  // A trigger the file itself (or its run/dataset) carries is shown as-is; one
  // the user typed, or none at all, is an editable LoRA trigger word field.
  const triggerEmbedded = !!entry?.trigger && !entry.triggerManual;
  const [triggerDraft, setTriggerDraft] = useState(entry?.triggerManual ? entry.trigger ?? '' : '');
  useEffect(() => { setTriggerDraft(entry?.triggerManual ? entry.trigger ?? '' : ''); }, [selected, entry?.trigger, entry?.triggerManual]);
  const saveTrigger = () => {
    const typed = triggerDraft.trim();
    const saved = entry?.triggerManual ? entry.trigger ?? '' : '';
    if (selected && typed !== saved) void onTrigger(selected, typed);
  };

  return (
    <div className="space-y-3">
      <div>
        <ParamLabel
          label={title}
          info={blurb}
          rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <StyledSelect
          accent="emerald"
          className="w-full"
          value={selected}
          disabled={disabled}
          onChange={v => void onCommit(kind, v || null, draft)}
          placeholder={t('globalBar.yue2AdapterBase', 'None — base model') as string}
          options={[
            { value: '', label: t('globalBar.yue2AdapterBase', 'None — base model') },
            ...paths.map(p => {
              const label = meta[p]?.label || shortName(p);
              const h = meta[p]?.home;
              return { value: p, label: typeof h === 'string' && h !== loadedVariant ? `${label} · ${variantName(h)}` : label };
            }),
            // A pick whose file has since moved or been deleted still has to be
            // visible: silently showing blank for a value that is not among
            // the options reads as "nothing is selected" when something very
            // much is.
            ...(selected && !paths.includes(selected)
              ? [{ value: selected, label: `${shortName(selected)} ${t('globalBar.yue2AdapterMissing', '(file not found)')}` }]
              : []),
          ]}
        />
      </div>

      {selected && (
        <>
          {/* ── What this adapter is ── */}
          <div className="px-3 py-2 rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 space-y-1">
            {triggerEmbedded ? (
              <div className="flex items-baseline gap-2">
                <span className="text-[10px] uppercase tracking-wider text-zinc-500">
                  {t('globalBar.yue2AdapterTrigger', 'Trigger')}
                </span>
                <span className="text-xs font-mono text-emerald-500 select-all break-all">{entry!.trigger}</span>
              </div>
            ) : (
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <ParamLabel
                    label={t('globalBar.yue2AdapterManualTrigger', 'LoRA trigger word')}
                    info={t('globalBar.yue2AdapterManualTriggerInfo',
                      'This adapter carries no trigger word. If you know the word it was trained with, type it here: it is put at the start of the style prompt for every render with this adapter. Saved for this adapter file when you leave the box or press Enter; clear it to stop.')}
                    className="text-[10px] uppercase tracking-wider text-zinc-500 shrink-0"
                  />
                  <input
                    type="text"
                    value={triggerDraft}
                    onChange={e => setTriggerDraft(e.target.value)}
                    onBlur={saveTrigger}
                    onKeyDown={e => { if (e.key === 'Enter') saveTrigger(); }}
                    disabled={disabled}
                    placeholder={t('globalBar.yue2AdapterManualTriggerPlaceholder', 'none')}
                    className="flex-1 min-w-0 px-2 py-1 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 text-xs font-mono text-zinc-700 dark:text-zinc-300 placeholder:text-zinc-400 dark:placeholder:text-zinc-600 outline-none focus:border-emerald-500/40 disabled:opacity-50"
                  />
                </div>
                {!entry?.triggerManual && (
                  <div className="flex gap-2 items-start">
                    <AlertTriangle size={12} className="text-amber-500 flex-shrink-0 mt-0.5" />
                    <p className="text-[10px] text-amber-500/90 leading-relaxed">
                      {t('globalBar.yue2AdapterNoTrigger',
                        'This adapter records no trigger word, so there is no phrase that reliably calls its style up. It was trained without one.')}
                    </p>
                  </div>
                )}
              </div>
            )}
            {entry?.triggerInferred && (
              <p className="text-[10px] text-amber-500/90 leading-relaxed">
                {t('globalBar.yue2AdapterInferredTrigger',
                  'Dataset tag inferred for this joint checkpoint. The adapter was trained without the trigger phrase; adding it to the prompt is experimental.')}
              </p>
            )}
            {triggerEmbedded && !entry!.triggerInferred && (
              <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
                {t('globalBar.yue2AdapterTriggerUse',
                  'Start the style prompt with this word — it is how the adapter is addressed.')}
              </p>
            )}
            <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
              {[
                entry?.dataset,
                entry?.rank ? `rank ${entry.rank}` : '',
                entry?.steps ? `${entry.steps} steps` : '',
                entry?.final ? 'final export' : '',
                formatSize(entry?.bytes),
              ].filter(Boolean).join(' · ')}
            </p>
            <p className="text-[10px] text-zinc-500 font-mono break-all">{selected}</p>
            <p className="text-[11px] text-zinc-700 dark:text-zinc-300 tabular-nums">
              {typeof entry?.loss === 'number' && Number.isFinite(entry.loss)
                ? t('globalBar.yue2AdapterLoss', 'Mean of last 20 logged losses: {{loss}}', { loss: entry.loss.toFixed(4) })
                : t('globalBar.yue2AdapterLossUnknown', 'Training loss: not recorded')}
              {typeof entry?.valLoss === 'number' && Number.isFinite(entry.valLoss)
                && ` · ${t('globalBar.yue2AdapterVal', 'Held-out val: {{val}}', { val: entry.valLoss.toFixed(4) })}`}
            </p>
            {/* Chosen vs merged: the delta is baked in at load, so a fresh pick
                is not in force until the next generation warms the model.
                Saying "merged" while nothing is merged would be a lie for that
                window, and with two slots it is a per-slot answer. */}
            <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
              {inForce
                ? t('globalBar.yue2AdapterInForce', 'Merged into the resident model now.')
                : t('globalBar.yue2AdapterPending', 'Merges when the model next loads.')}
            </p>
          </div>

          {home && (
            <>
              <p className="text-[11px] text-emerald-600 dark:text-emerald-400 leading-relaxed">
                {t('globalBar.yue2AdapterRebased',
                  'Trained on {{home}}: re-based onto {{loaded}}. The layers it targets move toward {{home}} before it merges, which needs a {{home}} LM installed.',
                  { home, loaded: variantName(loadedVariant) })}
              </p>
              <Slider
                label={t('globalBar.yue2AdapterRebase', 'Re-base strength (β)')}
                info={t('globalBar.yue2AdapterRebaseHint',
                  'How far the layers this adapter targets move toward the model it was trained on before it merges. 1 runs them as that model, 0 applies the adapter to the loaded model unchanged.')}
                value={draft.rebase}
                onChange={v => edit({ rebase: v })}
                min={0} max={1} step={0.05} showInput
                defaultValue={UNITY.rebase}
                onCommit={requestCommit}
                disabled={disabled}
              />
            </>
          )}

          {/* ── Strength dials ── */}
          {kind === 'ar' && (
            <Slider
              label={t('globalBar.yue2AdapterPlannerStrength', 'Planner Strength')}
              info={t('globalBar.yue2AdapterPlannerStrengthHint',
                'Strength while the AR writes the score: the lead sheet of melody, chords and sections it plans before any audio. Lower it to keep the base model\'s songwriting while the adapter still shapes the sound. It follows Composer Strength until you move it; Reset makes it follow again. Only used with Chain of Thought on. When it differs from Composer Strength, planning and rendering each load the model at their own strength, so every generation reloads the model twice.')}
              value={draft.plan ?? draft.scale}
              onChange={v => edit({ plan: v })}
              min={0} max={2} step={0.05} showInput
              defaultValue={UNITY.scale}
              onCommit={requestCommit}
              disabled={disabled}
            />
          )}
          <Slider
            label={kind === 'ar'
              ? t('globalBar.yue2AdapterComposerStrength', 'Composer Strength')
              : t('globalBar.yue2AdapterStrength', 'Strength')}
            info={kind === 'ar'
              ? t('globalBar.yue2AdapterComposerStrengthHint',
                'Strength while the AR writes the audio tokens: it turns the score into the actual performance, arrangement and sound. Every other dial below multiplies on top of both strengths, so 0 here turns the adapter off for the render without clearing the pick.')
              : t('globalBar.yue2AdapterStrengthHint',
                'Master strength for this half. Every other dial multiplies on top of it, so 0 here turns the adapter off without clearing the pick.')}
            value={draft.scale}
            onChange={v => edit({ scale: v })}
            min={0} max={2} step={0.05} showInput
            defaultValue={UNITY.scale}
            onCommit={requestCommit}
            disabled={disabled}
          />
          <Slider
            label={t('globalBar.yue2AdapterAttn', 'Attention')}
            info={t('globalBar.yue2AdapterGroupHint',
              'Hold the attention and MLP sites at different strengths. Timbre tends to sit in attention and phrasing in the MLPs, so pulling one back is how you keep half of what an adapter learned.')}
            value={draft.attn}
            onChange={v => edit({ attn: v })}
            min={0} max={2} step={0.05} showInput
            defaultValue={UNITY.attn}
            onCommit={requestCommit}
            disabled={disabled}
          />
          <Slider
            label={t('globalBar.yue2AdapterMlp', 'MLP')}
            info={t('globalBar.yue2AdapterGroupHint',
              'Hold the attention and MLP sites at different strengths. Timbre tends to sit in attention and phrasing in the MLPs, so pulling one back is how you keep half of what an adapter learned.')}
            value={draft.mlp}
            onChange={v => edit({ mlp: v })}
            min={0} max={2} step={0.05} showInput
            defaultValue={UNITY.mlp}
            onCommit={requestCommit}
            disabled={disabled}
          />

          <div className="flex items-center justify-between gap-2 pt-1">
            <button
              type="button"
              onClick={() => setShowDepth(v => !v)}
              className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              {showDepth ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {t('globalBar.yue2AdapterDepth', 'Depth thirds (advanced)')}
            </button>
            <button
              type="button"
              disabled={disabled}
              onClick={() => { setDraft(UNITY); requestCommit(); }}
              title={t('globalBar.yue2AdapterResetHint', 'Back to 1.0 on every dial') as string}
              className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors disabled:opacity-40"
            >
              <RotateCcw size={11} />
              {t('globalBar.yue2AdapterReset', 'Reset')}
            </button>
          </div>

          {showDepth && (
            <div className="space-y-3 pl-3 border-l-2 border-amber-500/30">
              <div className="flex gap-2 items-start">
                <AlertTriangle size={12} className="text-amber-500 flex-shrink-0 mt-0.5" />
                <ParamLabel
                  label={t('globalBar.yue2AdapterDepthWarningShort', 'Leave these at 1.0 for renders you want to keep.')}
                  info={t('globalBar.yue2AdapterDepthWarning',
                    'Each dial covers a third of the planner\'s transformer blocks by depth (the first, middle and last third of the stack), not the start, middle and end of the song: every block processes the whole song. Each multiplies the adapter\'s strength in those blocks on top of the master and attention/MLP dials, so 1.0 changes nothing. On the MM3 planner the late third turned out to carry sequence termination, and halving it produced songs that faded out or never ended; the early third drove tempo. Nothing says YuE2 divides the work the same way, which is exactly why these are a probe rather than a setting.')}
                  className="text-[10px] text-amber-500/90 leading-relaxed" />
              </div>
              <Slider
                label={t('globalBar.yue2AdapterEarly', 'Early third')}
                info={t('globalBar.yue2AdapterEarlyInfo',
                  'Multiplies this adapter\'s strength in the first third of the planner\'s blocks (depth, not song position), on top of the master and attention/MLP dials. On the MM3 planner these blocks drove tempo and basic feel; nothing yet confirms YuE2 keeps the same things there. Lower it if the adapter is bending tempo or groove in ways you do not want, raise it if the render ignores the artist\'s feel, and leave it at 1.0 for renders you want to keep.')}
                value={draft.early}
                onChange={v => edit({ early: v })}
                min={0} max={2} step={0.05} showInput
                defaultValue={UNITY.early}
                onCommit={requestCommit}
                disabled={disabled}
              />
              <Slider
                label={t('globalBar.yue2AdapterMid', 'Middle third')}
                info={t('globalBar.yue2AdapterMidInfo',
                  'Multiplies this adapter\'s strength in the middle third of the planner\'s blocks (depth, not song position), on top of the master and attention/MLP dials. Nothing yet says what YuE2 keeps in its middle blocks, so this is a probe: lower it and listen for what changes, and leave it at 1.0 for renders you want to keep.')}
                value={draft.mid}
                onChange={v => edit({ mid: v })}
                min={0} max={2} step={0.05} showInput
                defaultValue={UNITY.mid}
                onCommit={requestCommit}
                disabled={disabled}
              />
              <Slider
                label={t('globalBar.yue2AdapterLate', 'Late third')}
                info={t('globalBar.yue2AdapterLateInfo',
                  'Multiplies this adapter\'s strength in the last third of the planner\'s blocks (depth, not song position), on top of the master and attention/MLP dials. On the MM3 planner these blocks carried sequence termination: halving this dial produced songs that faded out or never ended, so if a render loses its ending after you lower it, that is the same effect. Nothing confirms YuE2 divides its blocks the same way; treat it as a probe and leave it at 1.0 for renders you want to keep.')}
                value={draft.late}
                onChange={v => edit({ late: v })}
                min={0} max={2} step={0.05} showInput
                defaultValue={UNITY.late}
                onCommit={requestCommit}
                disabled={disabled}
              />
            </div>
          )}

        </>
      )}
    </div>
  );
};

export const Yue2LmAdapterDropdown: React.FC = () => {
  const { t } = useTranslation();
  const { capabilities } = useCapabilities();
  const selectModels = useBackendStore(s => s.selectModels);
  const {
    activeBackendId, catalogue, isError, error, refetch, meta, pathsFor, dialsFor, selectedFor, inForceFor, loadedVariant,
  } = useYue2Adapters();
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const savedFolder = String((catalogue?.defaults as Record<string, unknown> | undefined)?.lmAdapterFolder ?? '');
  const [folder, setFolder] = useState(savedFolder);
  const [browsing, setBrowsing] = useState(false);
  const [importing, setImporting] = useState(false);
  useEffect(() => { setFolder(savedFolder); }, [savedFolder]);
  const queryClient = useQueryClient();

  // Silent: a failure gets the localized note below, not the global toast.
  const importMut = useMutation({
    mutationFn: (file: string) => apiClient.post<{ arPath: string; narPath: string; name?: string }>(
      '/yue2/import-adapter', { path: file }, { silent: true }),
    // The imported adapter must appear in the lists below even when the
    // selectModels leg fails — the "pick it by hand" note promises it is there,
    // and only a successful selection refetches the catalogue.
    onSuccess: () => { void queryManager.invalidate(queryClient, backendModelsOptions(activeBackendId).queryKey); },
  });

  /** A ComfyUI / ai-toolkit YuE2 adapter is one file holding both halves in
   *  the upstream naming. The server splits it into our AR + NAR pair; both
   *  slots are then pointed at the result, since they are one training run. */
  const importComfy = async (file: string) => {
    setBusy(true);
    setNote(null);
    try {
      const body = await importMut.mutateAsync(file);
      const ok = await selectModels({ lmAdapterAr: body.arPath, lmAdapterNar: body.narPath }, activeBackendId);
      setNote(ok
        ? t('globalBar.yue2AdapterImported',
            'Imported "{{name}}" and loaded it into both halves. The next generation reloads the model and merges it.',
            { name: body.name })
        : t('globalBar.yue2AdapterImportedNotSet',
            'Imported "{{name}}" — it is in both lists below, but selecting it failed; pick it by hand.',
            { name: body.name }));
    } catch (err) {
      setNote(t('globalBar.yue2AdapterImportFailed', 'Import failed: {{error}}',
        { error: err instanceof Error ? err.message : String(err) }));
    } finally {
      setBusy(false);
    }
  };

  /** Adapters copied from another machine are not in this one's training
   *  index; the server scans this folder for them as well. */
  const scanFolder = async (dir: string) => {
    setFolder(dir);
    setBusy(true);
    setNote(null);
    const ok = await selectModels({ lmAdapterFolder: dir.trim() }, activeBackendId);
    setBusy(false);
    if (!ok) {
      setNote(t('globalBar.yue2AdapterFolderFailed',
        'That folder could not be used. It needs to be a full path to a folder that exists.'));
    }
  };

  const license = typeof capabilities?.license === 'string' ? capabilities.license : undefined;

  const onTrigger = async (adapterPath: string, trigger: string) => {
    setBusy(true);
    const ok = await selectModels({ lmAdapterTriggerPath: adapterPath, lmAdapterTrigger: trigger }, activeBackendId);
    setBusy(false);
    setNote(ok
      ? (trigger
        ? t('globalBar.yue2AdapterTriggerSaved', 'Trigger word saved for this adapter.')
        : t('globalBar.yue2AdapterTriggerCleared', 'Trigger word cleared for this adapter.'))
      : t('globalBar.yue2AdapterTriggerFailed', 'Could not save that trigger word.'));
  };

  const onCommit = async (kind: SlotKind, pick: string | null, dials: Dials) => {
    setBusy(true);
    setNote(null);
    const k = keysFor(kind);
    // Only this slot's keys travel. An omitted key means "leave it alone"
    // server-side, so committing the AR half cannot disturb the NAR one.
    const ok = await selectModels({
      [k.path]: pick ?? '',
      [k.scale]: String(dials.scale),
      [k.attn]: String(dials.attn),
      [k.mlp]: String(dials.mlp),
      [k.early]: String(dials.early),
      [k.mid]: String(dials.mid),
      [k.late]: String(dials.late),
      // '' = follow the master strength.
      [k.plan]: dials.plan === null ? '' : String(dials.plan),
      [k.rebase]: String(dials.rebase),
    }, activeBackendId);
    setBusy(false);
    const half = kind.toUpperCase();
    setNote(ok
      ? (pick
        ? t('globalBar.yue2AdapterSwitched',
            '{{half}} set — the next generation reloads the model and merges it (adds a warm-up).', { half })
        : t('globalBar.yue2AdapterCleared',
            '{{half}} cleared — the next generation reloads without it.', { half }))
      : t('globalBar.yue2AdapterFailed',
          'Could not apply that {{half}} pick; the previous one is still in force.', { half }));
  };

  if (isError && !catalogue) {
    return <QueryError error={error} onRetry={() => void refetch()} />;
  }

  if (!catalogue) {
    return <p className="text-[11px] text-zinc-500">{t('globalBar.yue2AdapterLoading', 'Loading adapters…')}</p>;
  }

  const nothingTrained = SLOTS.every(k => pathsFor(k).length === 0 && !selectedFor(k));
  const locked = busy || catalogue.selectable === false;

  return (
    <div className="space-y-4">
      {isError && <QueryError error={error} onRetry={() => void refetch()} compact />}
      {nothingTrained && (
        <p className="text-[11px] text-zinc-500 leading-relaxed">
          {t('globalBar.yue2AdapterNone',
            'No YuE2 adapters trained yet. Train one in the Training Studio (NAR or AR LoRA) and it appears here — no restart needed.')}
        </p>
      )}

      <div>
        <ParamLabel
          label={t('globalBar.yue2AdapterFolder', 'Adapter folder')}
          info={t('globalBar.yue2AdapterFolderHint',
            'Adapters trained here appear on their own. For ones trained on another machine, point this at the folder you copied them into. Every AR and NAR adapter file inside it, including in subfolders, is added to the lists below.')}
          rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <div className="flex gap-1.5">
          <input
            className={inputClasses}
            value={folder}
            disabled={locked}
            onChange={e => setFolder(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') void scanFolder(folder); }}
            placeholder={t('globalBar.yue2AdapterFolderPlaceholder', 'Optional: folder with copied adapters') as string}
          />
          <button type="button" disabled={locked} onClick={() => void scanFolder(folder)}
            title={t('adapter.scanFolder') as string}
            className="px-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 disabled:opacity-40">
            <Search size={14} />
          </button>
          <button type="button" disabled={locked} onClick={() => setBrowsing(true)}
            title={t('adapter.browseFolder') as string}
            className="px-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 disabled:opacity-40">
            <FolderOpen size={14} />
          </button>
        </div>
      </div>
      <FileBrowserModal
        open={browsing}
        onClose={() => setBrowsing(false)}
        onSelect={p => { setBrowsing(false); void scanFolder(p); }}
        mode="folder"
        startPath={folder || undefined}
        title={t('adapter.selectAdapterFolder') as string}
      />

      <div>
        <ParamLabel
          label={t('globalBar.yue2AdapterImport', 'Import ComfyUI adapter')}
          info={t('globalBar.yue2AdapterImportHint',
            'For YuE2 LoRA or LoKr adapters made in ComfyUI or ai-toolkit. Those come as a single .safetensors file that holds both halves (NAR and AR) at once, which this app cannot load directly. Importing splits it into an AR and a NAR adapter, lists them under "Imported" below and selects both. The original file is left untouched. Adapters trained in this app never need importing.')}
          rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <button type="button" disabled={locked} onClick={() => setImporting(true)}
          title={t('globalBar.yue2AdapterImportButtonHint', 'Pick a ComfyUI / ai-toolkit YuE2 adapter (.safetensors) to convert') as string}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-white/10 text-sm text-zinc-600 dark:text-zinc-300 hover:text-zinc-800 dark:hover:text-white disabled:opacity-40">
          <FileInput size={14} />
          {t('globalBar.yue2AdapterImportButton', 'Import adapter…')}
        </button>
      </div>
      <FileBrowserModal
        open={importing}
        onClose={() => setImporting(false)}
        onSelect={p => { setImporting(false); void importComfy(p); }}
        mode="file"
        filter="adapters"
        title={t('globalBar.yue2AdapterImportPick', 'Select a ComfyUI YuE2 adapter') as string}
      />

      <SlotPanel
        kind="nar"
        title={t('globalBar.yue2AdapterNar', 'NAR adapter (renderer)')}
        blurb={t('globalBar.yue2AdapterNarHint',
          'A LoRA on the NAR half, which renders the audio — this is where timbre and production live. Merged into the model weights when it loads, so switching evicts the resident model and the next generation pays a reload.')}
        paths={pathsFor('nar')}
        meta={meta}
        selected={selectedFor('nar')}
        dials={dialsFor('nar')}
        inForce={inForceFor('nar')}
        loadedVariant={loadedVariant}
        disabled={locked}
        onCommit={onCommit}
        onTrigger={onTrigger}
      />

      <div className="border-t border-zinc-200 dark:border-white/5" />

      <SlotPanel
        kind="ar"
        title={t('globalBar.yue2AdapterAr', 'AR adapter (composer)')}
        blurb={t('globalBar.yue2AdapterArHint',
          'A LoRA on the AR half, which writes the plan and the semantic tokens — this is where structure and phrasing live. AR and NAR share no weights, so both can be loaded at once and each is merged separately.')}
        paths={pathsFor('ar')}
        meta={meta}
        selected={selectedFor('ar')}
        dials={dialsFor('ar')}
        inForce={inForceFor('ar')}
        loadedVariant={loadedVariant}
        disabled={locked}
        onCommit={onCommit}
        onTrigger={onTrigger}
      />

      {selectedFor('ar') && selectedFor('nar') && (
        <p className="text-[10px] text-zinc-600 dark:text-zinc-500 leading-relaxed">
          {t('globalBar.yue2AdapterBothNote',
            'Both halves are loaded. The style prompt can only carry one trigger word, and the NAR pick gets it — put the AR trigger in the caption yourself if the two were trained apart.')}
        </p>
      )}

      {busy && (
        <p className="text-[10px] text-zinc-500 flex items-center gap-1.5">
          <Loader2 size={11} className="animate-spin" />
          {t('globalBar.yue2AdapterBusy', 'Switching…')}
        </p>
      )}
      {note && <p className="text-[10px] text-zinc-500 leading-relaxed">{note}</p>}

      {license && (
        <p className="text-[10px] text-amber-600 dark:text-amber-500/80 leading-relaxed border-t border-zinc-200 dark:border-white/5 pt-2">
          {license}{' '}
          {t('globalBar.yue2AdapterLicenseDerivative',
            'An adapter trained on these weights is a derivative and carries the same terms.')}
        </p>
      )}
    </div>
  );
};

/** Summary badge — what is picked in each half, or the base model. */
export const Yue2LmAdapterBadge: React.FC = () => {
  const catalogue = useBackendStore(s => s.models[s.activeBackendId] ?? null);
  const defaults = (catalogue?.defaults ?? {}) as Record<string, unknown>;

  const picked = SLOTS
    .map(kind => {
      const p = String(defaults[keysFor(kind).path] ?? '');
      if (!p) return null;
      const m = catalogue?.lmAdapterMeta?.[p];
      return { kind, label: m?.trigger || m?.runName || shortName(p), path: p };
    })
    .filter(Boolean) as Array<{ kind: SlotKind; label: string; path: string }>;

  if (picked.length === 0) return <span className="text-[10px] text-zinc-600">None</span>;

  return (
    <div className="flex items-center gap-1.5 min-w-0">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
      <span className="text-[10px] text-emerald-400 font-mono truncate max-w-[150px]"
            title={picked.map(p => `${p.kind.toUpperCase()}: ${p.path}`).join('\n')}>
        {picked.map(p => `${p.kind.toUpperCase()} ${p.label}`).join(' + ')}
      </span>
    </div>
  );
};
