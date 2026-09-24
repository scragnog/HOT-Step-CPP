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
// THE DIALS ARE NOT COMMITTED PER TICK. Each slot keeps a local draft and an
// Apply button. A slider that posted on every drag frame would tear the model
// down and re-merge it dozens of times per drag, and on a K-quant base each
// re-merge is a requant of the patched tensors. The draft/commit split is the
// one real departure from MM3's panel, where the same dials are free because
// they ride on the request.
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

import React, { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, ChevronDown, ChevronRight, FolderOpen, Loader2, RotateCcw, Search } from 'lucide-react';
import { useBackendStore } from '../../stores/backendStore';
import { useCapabilities } from '../../hooks/useCapabilities';
import { ParamLabel } from '../shared/ParamLabel';
import { Slider } from '../shared/Slider';
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
}

const UNITY: Dials = { scale: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 };

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
    && a.early === b.early && a.mid === b.mid && a.late === b.late;
}

/** The catalogue rows the picker renders, plus what the engine says is
 *  actually merged right now. Both come from the one models catalogue, which
 *  the store refetches after every successful selection. */
function useYue2Adapters() {
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const catalogue = useBackendStore(s => s.models[s.activeBackendId] ?? null);
  const fetchModels = useBackendStore(s => s.fetchModels);

  useEffect(() => {
    if (!activeBackendId) return;
    void fetchModels(activeBackendId);
  }, [activeBackendId, fetchModels]);

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
    };
  };

  return {
    activeBackendId,
    catalogue,
    meta,
    pathsFor,
    dialsFor,
    selectedFor: (kind: SlotKind) => String(defaults[keysFor(kind).path] ?? ''),
    /** false until the next warm/synth after a pick — the merge happens at
     *  load, so "chosen" and "in force" genuinely differ for that window. */
    inForceFor: (kind: SlotKind) => defaults[keysFor(kind).inForce] === true,
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
  disabled: boolean;
  onCommit: (kind: SlotKind, path: string | null, dials: Dials) => Promise<void>;
}

const SlotPanel: React.FC<SlotPanelProps> = ({
  kind, title, blurb, paths, meta, selected, dials, inForce, disabled, onCommit,
}) => {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<Dials>(dials);
  const [showDepth, setShowDepth] = useState(false);

  // The server is the source of truth for what is committed. Whenever it says
  // something different — another tab, a reconcile after an engine restart,
  // this panel's own successful commit — the draft is replaced rather than
  // merged: a half-dragged slider is not a change anyone asked to keep.
  useEffect(() => { setDraft(dials); }, [dials.scale, dials.attn, dials.mlp, dials.early, dials.mid, dials.late]);

  const entry = selected ? meta[selected] : undefined;
  const dirty = !sameDials(draft, dials);

  return (
    <div className="space-y-3">
      <div>
        <ParamLabel
          label={title}
          info={blurb}
          rootClassName="flex mb-1.5"
          className="text-xs font-medium text-zinc-500 uppercase tracking-wider" />
        <select
          className={inputClasses}
          value={selected}
          disabled={disabled}
          onChange={e => void onCommit(kind, e.target.value || null, draft)}
        >
          <option value="">{t('globalBar.yue2AdapterBase', 'None — base model')}</option>
          {paths.map(p => (
            <option key={p} value={p}>{meta[p]?.label || shortName(p)}</option>
          ))}
          {/* A pick whose file has since moved or been deleted still has to be
              visible: a native select silently shows blank for a value that is
              not among its options, which reads as "nothing is selected" when
              something very much is. */}
          {selected && !paths.includes(selected) && (
            <option value={selected}>
              {shortName(selected)} {t('globalBar.yue2AdapterMissing', '(file not found)')}
            </option>
          )}
        </select>
      </div>

      {selected && (
        <>
          {/* ── What this adapter is ── */}
          <div className="px-3 py-2 rounded-xl bg-zinc-100/50 dark:bg-zinc-800/50 border border-zinc-200 dark:border-white/5 space-y-1">
            {entry?.trigger ? (
              <div className="flex items-baseline gap-2">
                <span className="text-[10px] uppercase tracking-wider text-zinc-500">
                  {t('globalBar.yue2AdapterTrigger', 'Trigger')}
                </span>
                <span className="text-xs font-mono text-emerald-500 select-all break-all">{entry.trigger}</span>
              </div>
            ) : (
              <div className="flex gap-2 items-start">
                <AlertTriangle size={12} className="text-amber-500 flex-shrink-0 mt-0.5" />
                <p className="text-[10px] text-amber-500/90 leading-relaxed">
                  {t('globalBar.yue2AdapterNoTrigger',
                    'This adapter records no trigger word, so there is no phrase that reliably calls its style up. It was trained without one.')}
                </p>
              </div>
            )}
            {entry?.triggerInferred && (
              <p className="text-[10px] text-amber-500/90 leading-relaxed">
                {t('globalBar.yue2AdapterInferredTrigger',
                  'Dataset tag inferred for this joint checkpoint. The adapter was trained without the trigger phrase; adding it to the prompt is experimental.')}
              </p>
            )}
            {entry?.trigger && !entry.triggerInferred && (
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
                ? t('globalBar.yue2AdapterLoss', 'Training loss: {{loss}}', { loss: entry.loss.toFixed(4) })
                : t('globalBar.yue2AdapterLossUnknown', 'Training loss: not recorded')}
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

          {/* ── Strength dials ── */}
          <Slider
            label={t('globalBar.yue2AdapterStrength', 'Strength')}
            info={t('globalBar.yue2AdapterStrengthHint',
              'Master strength for this half. Every other dial multiplies on top of it, so 0 here turns the adapter off without clearing the pick.')}
            value={draft.scale}
            onChange={v => setDraft(d => ({ ...d, scale: v }))}
            min={0} max={2} step={0.05} showInput
          />
          <Slider
            label={t('globalBar.yue2AdapterAttn', 'Attention')}
            info={t('globalBar.yue2AdapterGroupHint',
              'Hold the attention and MLP sites at different strengths. Timbre tends to sit in attention and phrasing in the MLPs, so pulling one back is how you keep half of what an adapter learned.')}
            value={draft.attn}
            onChange={v => setDraft(d => ({ ...d, attn: v }))}
            min={0} max={2} step={0.05} showInput
          />
          <Slider
            label={t('globalBar.yue2AdapterMlp', 'MLP')}
            info={t('globalBar.yue2AdapterGroupHint',
              'Hold the attention and MLP sites at different strengths. Timbre tends to sit in attention and phrasing in the MLPs, so pulling one back is how you keep half of what an adapter learned.')}
            value={draft.mlp}
            onChange={v => setDraft(d => ({ ...d, mlp: v }))}
            min={0} max={2} step={0.05} showInput
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
              onClick={() => setDraft(UNITY)}
              title={t('globalBar.yue2AdapterResetHint', 'Back to 1.0 on every dial') as string}
              className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
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
                    'Each dial covers a third of the block stack. On the MM3 planner the late third turned out to carry sequence termination, and halving it produced songs that faded out or never ended; the early third drove tempo. Nothing says YuE2 divides the work the same way, which is exactly why these are a probe rather than a setting.')}
                  className="text-[10px] text-amber-500/90 leading-relaxed" />
              </div>
              <Slider
                label={t('globalBar.yue2AdapterEarly', 'Early third')}
                value={draft.early}
                onChange={v => setDraft(d => ({ ...d, early: v }))}
                min={0} max={2} step={0.05} showInput
              />
              <Slider
                label={t('globalBar.yue2AdapterMid', 'Middle third')}
                value={draft.mid}
                onChange={v => setDraft(d => ({ ...d, mid: v }))}
                min={0} max={2} step={0.05} showInput
              />
              <Slider
                label={t('globalBar.yue2AdapterLate', 'Late third')}
                value={draft.late}
                onChange={v => setDraft(d => ({ ...d, late: v }))}
                min={0} max={2} step={0.05} showInput
              />
            </div>
          )}

          {/* The commit. Deliberately a button and not a live write: see the
              file header — each commit evicts the resident model. */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={disabled || !dirty}
              onClick={() => void onCommit(kind, selected, draft)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500/15 text-emerald-600 dark:text-emerald-400
                         hover:bg-emerald-500/25 disabled:opacity-40 disabled:hover:bg-emerald-500/15 transition-colors"
            >
              {t('globalBar.yue2AdapterApply', 'Apply strengths')}
            </button>
            {dirty && (
              <span className="text-[10px] text-zinc-500 leading-snug">
                {t('globalBar.yue2AdapterDirty', 'Not applied yet — applying reloads the model.')}
              </span>
            )}
          </div>
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
    activeBackendId, catalogue, meta, pathsFor, dialsFor, selectedFor, inForceFor,
  } = useYue2Adapters();
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const savedFolder = String((catalogue?.defaults as Record<string, unknown> | undefined)?.lmAdapterFolder ?? '');
  const [folder, setFolder] = useState(savedFolder);
  const [browsing, setBrowsing] = useState(false);
  useEffect(() => { setFolder(savedFolder); }, [savedFolder]);

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

  if (!catalogue) {
    return <p className="text-[11px] text-zinc-500">{t('globalBar.yue2AdapterLoading', 'Loading adapters…')}</p>;
  }

  const nothingTrained = SLOTS.every(k => pathsFor(k).length === 0 && !selectedFor(k));
  const locked = busy || catalogue.selectable === false;

  return (
    <div className="space-y-4">
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
        disabled={locked}
        onCommit={onCommit}
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
        disabled={locked}
        onCommit={onCommit}
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
