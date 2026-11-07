// TrainingPreviewPanel.tsx — the checkpoint-preview presentation Training Studio
// shares between backends. Extracted from the YuE2 train card's "Checkpoint
// previews" settings and the ladder's take list; YuE2 and SA3 both draw it.
//
// Presentation only: it holds no server state and calls no API. A caller maps
// its own records to PreviewTake rows and keeps its own actions, scoring and
// labels in the slots, so YuE2's lead-sheet and ladder features stay in YuE2.
import React, { type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '../shared/ParamLabel';
import { Toggle } from '../shared/Toggle';
import { PreviewPlayer } from './PreviewPlayer';
import { StyledSelect } from '../shared/StyledSelect';

export const PREVIEW_INPUT = 'px-2 py-1.5 rounded-lg text-xs bg-white/70 dark:bg-black/20 border border-zinc-300/70 dark:border-white/10 text-zinc-800 dark:text-zinc-100';

export function TrainingPresetChoice({ selected, disabled, onClick, children }: {
  selected: boolean; disabled?: boolean; onClick: () => void; children: ReactNode;
}) {
  return <button type="button" disabled={disabled} onClick={onClick}
    className={`flex flex-col items-center gap-1 px-4 py-3 rounded-xl border-2 transition-colors disabled:opacity-40 ${selected
      ? 'border-blue-500 bg-blue-500/15 text-blue-700 dark:text-blue-300'
      : 'border-zinc-300 dark:border-white/15 text-zinc-800 dark:text-zinc-100 hover:border-blue-500/50 hover:bg-blue-500/5'}`}>
    {children}
  </button>;
}

/** The run picker and ladder frame used by the YuE2 training page. */
export function TrainingRunPicker({ label, info, value, onChange, options, disabled }: {
  label: string; info: string; value: string; onChange: (value: string) => void;
  options: Array<{ value: string; label: string; disabled?: boolean }>; disabled?: boolean;
}) {
  return <>
    <ParamLabel label={label} info={info} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" />
    <StyledSelect accent="amber" className="w-full" value={value} onChange={onChange} options={options} disabled={disabled} />
  </>;
}

export function TrainingCheckpointLadder({ children }: { children: ReactNode }) {
  return <div className="mt-3 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">{children}</div>;
}

export function TrainingPreviewLadderRows({ children }: { children: ReactNode }) {
  return <div className="mt-3 flex flex-col gap-3">{children}</div>;
}

/** The collapsible settings block: an optional lead slot, the enable toggle,
 *  an optional slot under it, and the fields grid while previews are on. */
export const TrainingPreviewSettings: React.FC<{
  title: string;
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  disabled?: boolean;
  enableLabel: string;
  enableInfo: string;
  enableDefault?: boolean;
  /** Above the enable toggle (YuE2: stop the engine). */
  lead?: ReactNode;
  /** Between the toggle and the grid (YuE2: parallel / engine-stopped note). */
  afterToggle?: ReactNode;
  /** The fields grid; drawn only while enabled. */
  children?: ReactNode;
}> = ({ title, enabled, onEnabledChange, disabled, enableLabel, enableInfo, enableDefault, lead, afterToggle, children }) => (
  <details className="mt-3 rounded-lg border border-zinc-300/70 dark:border-white/10 bg-white/30 dark:bg-black/10 p-3">
    <summary className="cursor-pointer text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">{title}</summary>
    {lead}
    <Toggle accent="amber" className="mt-2" checked={enabled} disabled={disabled} onChange={onEnabledChange}
      label={enableLabel} info={enableInfo} defaultValue={enableDefault} />
    {afterToggle}
    {enabled && <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">{children}</div>}
  </details>
);

/** One labelled field of that grid. `wide` spans both columns. */
export const PreviewField: React.FC<{
  label: string; info?: string; meta?: string; onReset?: () => void; wide?: boolean; children: ReactNode;
}> = ({ label, info, meta, onReset, wide, children }) => (
  <label className={`flex flex-col gap-1${wide ? ' md:col-span-2' : ''}`}>
    <ParamLabel label={label} info={info} meta={meta} className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider" onReset={onReset} />
    {children}
  </label>
);

/** One take of a checkpoint. Playable rows draw a PreviewPlayer; the rest draw `text`. */
export interface PreviewTake {
  id: string;
  /** Stream source. Absent or empty = not playable. */
  src?: string | null;
  /** The original file, when `src` is a lighter stream of it. */
  downloadSrc?: string;
  downloadName?: string;
  label: string;
  sublabel?: string;
  /** Shown when the take cannot play (pending, failed, pruned). */
  text?: string;
}

export const TrainingPreviewTakes: React.FC<{ takes: PreviewTake[] }> = ({ takes }) => takes.length === 0 ? null : (
  <div className="mt-2 flex flex-col gap-2">
    {takes.map(take => take.src
      ? <PreviewPlayer key={take.id} src={take.src} downloadSrc={take.downloadSrc} downloadName={take.downloadName} label={take.label} sublabel={take.sublabel} />
      : <div key={take.id} className="text-[11px] text-zinc-500">{take.text ?? take.label}</div>)}
  </div>
);

/** A rung that is the adapter in use. */
export const LADDER_IN_USE = 'border-emerald-500/60 bg-emerald-500/5';

/** A checkpoint ladder rung's actions, the same for every engine: "Render"
 *  (more takes of this step) and "Use this …", which makes it the adapter. */
export const TrainingLadderActions: React.FC<{
  hasTakes: boolean;
  rendering: boolean;
  renderDisabled: boolean;
  onRender: () => void;
  inUse: boolean;
  useDisabled: boolean;
  onUse?: () => void;
  useLabel: string;
}> = ({ hasTakes, rendering, renderDisabled, onRender, inUse, useDisabled, onUse, useLabel }) => {
  const { t } = useTranslation();
  return <>
    <span className="flex-1" />
    <button type="button" onClick={onRender} disabled={renderDisabled}
      className="px-2 py-1 rounded-lg text-[11px] border border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10 disabled:opacity-40">
      {rendering ? t('trainingStudio.refine.rendering', 'Rendering…') : hasTakes ? t('trainingStudio.refine.render', 'Render more') : t('trainingStudio.refine.renderFirst', 'Render')}
    </button>
    {onUse && <button type="button" onClick={onUse} disabled={useDisabled}
      className={`px-2 py-1 rounded-lg text-[11px] font-semibold border disabled:opacity-40 ${inUse ? 'border-emerald-500 text-emerald-700 dark:text-emerald-300' : 'border-zinc-300/70 dark:border-white/10 hover:bg-zinc-500/10'}`}>
      {inUse ? t('trainingStudio.refine.picked', 'In use') : useLabel}
    </button>}
  </>;
};

/** A checkpoint's card: title and badges on the left, actions on the right,
 *  the caller's own rows (scores, notes) as children, then the takes. */
export const TrainingPreviewRung: React.FC<{
  id?: string;
  title: ReactNode;
  badges?: ReactNode;
  actions?: ReactNode;
  /** Border and tint, e.g. a picked or best rung. */
  className?: string;
  takes: PreviewTake[];
  children?: ReactNode;
}> = ({ id, title, badges, actions, className = 'border-zinc-300/70 dark:border-white/10', takes, children }) => (
  <div id={id} className={`rounded-lg border p-3 ${className}`}>
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
      <span className="font-semibold text-zinc-800 dark:text-zinc-100">{title}</span>
      {badges}
      <span className="flex-1" />
      {actions}
    </div>
    {children}
    <TrainingPreviewTakes takes={takes} />
  </div>
);
