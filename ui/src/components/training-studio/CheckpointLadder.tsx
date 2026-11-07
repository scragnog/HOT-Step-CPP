// CheckpointLadder.tsx — the checkpoint ladder every engine draws: one card
// per saved checkpoint with its takes, "Render" / "Render more", and "Use this
// …", which makes that step the adapter. The plug-ins whose previewSource is a
// ladder ('yue2-ladder', 'mm3-ladder') map their records to rungs and draw it.
//
// Engine features ride in slots: `header` above the rungs (YuE2's blind toggle
// and album score), and per rung `badges` and `children` (YuE2's KL, replans,
// score buttons and notes). An engine without scores leaves them empty.
import React, { type ReactNode } from 'react';
import { LADDER_IN_USE, TrainingLadderActions, TrainingPreviewLadderRows, TrainingPreviewRung, type PreviewTake } from './TrainingPreviewPanel';

export interface CheckpointLadderRung {
  key: string | number;
  step: number;
  /** Element id, for scroll targets. */
  id?: string;
  title: ReactNode;
  takes: PreviewTake[];
  /** Border and tint when the rung is not in use (best, KL rung, base). */
  className?: string;
  badges?: ReactNode;
  /** Engine rows under the title, e.g. scores and notes. */
  children?: ReactNode;
  inUse: boolean;
  /** False when the step has no weights left to render from. */
  canRender: boolean;
  /** Absent: no Use button (the base model, a deleted checkpoint). */
  onUse?: () => void;
}

export const CheckpointLadder: React.FC<{
  rungs: CheckpointLadderRung[];
  header?: ReactNode;
  /** The step being rendered, or null. */
  rendering: number | null;
  onRender: (step: number) => void;
  /** A pick is in flight. */
  picking: boolean;
  /** Render and Use are off on every rung (MM3: the run is still training). */
  locked?: boolean;
  useLabel: string;
}> = ({ rungs, header, rendering, onRender, picking, locked = false, useLabel }) => rungs.length === 0 ? null : (
  <TrainingPreviewLadderRows>
    {header}
    {rungs.map(r => <TrainingPreviewRung key={r.key} id={r.id} title={r.title} takes={r.takes}
      className={r.inUse ? LADDER_IN_USE : r.className} badges={r.badges}
      actions={<TrainingLadderActions hasTakes={r.takes.length > 0} rendering={rendering === r.step}
        renderDisabled={locked || rendering !== null || !r.canRender} onRender={() => onRender(r.step)}
        inUse={r.inUse} useDisabled={locked || picking} onUse={r.onUse} useLabel={useLabel} />}>
      {r.children}
    </TrainingPreviewRung>)}
  </TrainingPreviewLadderRows>
);

export default CheckpointLadder;
