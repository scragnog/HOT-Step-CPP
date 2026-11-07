// Mm3LadderRows.tsx — the MM3 checkpoint ladder's rungs, with no server state
// (Mm3CheckpointLadder.tsx owns the queries). Drawn with YuE2's rung and
// actions: TrainingPreviewRung and TrainingLadderActions.
import React from 'react';
import { useTranslation } from 'react-i18next';
import type { Mm3Ladder, Mm3LadderPreview } from '../../services/trainingApi';
import { LADDER_IN_USE, TrainingLadderActions, TrainingPreviewLadderRows, TrainingPreviewRung, type PreviewTake } from './TrainingPreviewPanel';

const sameRef = (a: string | null | undefined, b: string) => !!a && a.replace(/\\/g, '/').toLowerCase() === b.replace(/\\/g, '/').toLowerCase();

interface Mm3Rung { step: number; base: boolean; ref?: string; loss?: number; takes: Mm3LadderPreview[] }

/** Base first (when the run rendered one), then every saved checkpoint, plus
 *  any step that has previews but whose checkpoint is gone (listen only). */
function mm3Rungs(ladder: Pick<Mm3Ladder, 'checkpoints' | 'previews'>): Mm3Rung[] {
  const rungs = new Map<string, Mm3Rung>();
  const base = ladder.previews.filter(p => p.base);
  if (base.length) rungs.set('base', { step: 0, base: true, takes: base });
  for (const c of ladder.checkpoints) rungs.set(String(c.step), { step: c.step, base: false, ref: c.ref, loss: c.loss, takes: [] });
  for (const p of ladder.previews.filter(v => !v.base)) {
    const r = rungs.get(String(p.step)) ?? { step: p.step, base: false, loss: p.loss, takes: [] };
    r.takes.push(p);
    rungs.set(String(p.step), r);
  }
  return [...rungs.values()].sort((a, b) => Number(b.base) - Number(a.base) || a.step - b.step);
}

/** The rungs, with no server state: the live component and its test draw it. */
export const Mm3LadderRows: React.FC<{
  ladder: Mm3Ladder;
  /** Step being rendered, or null. */
  rendering: number | null;
  picking: boolean;
  onRender: (step: number) => void;
  onUse: (step: number) => void;
}> = ({ ladder, rendering, picking, onRender, onUse }) => {
  const { t } = useTranslation();
  const rungs = mm3Rungs(ladder);
  if (!rungs.length) return null;
  const takeRows = (r: Mm3Rung): PreviewTake[] => r.takes.map(p => ({
    id: p.id, src: p.audioUrl,
    label: `${p.kind === 'control' ? t('trainingStudio.mm3.previewControlCaption', 'Control caption') : t('trainingStudio.mm3.previewArtistCaption', 'Artist caption')} · ${t('trainingStudio.mm3.previewTake', '{{seconds}} s · seed {{seed}} · rendered in {{renderSeconds}} s', {
      seconds: p.seconds, seed: p.seed, renderSeconds: (p.ms / 1000).toFixed(0) })}`,
    sublabel: p.caption || undefined,
  }));
  return <TrainingPreviewLadderRows>
    {rungs.map(r => {
      const inUse = !!r.ref && sameRef(ladder.inUse, r.ref);
      return <TrainingPreviewRung key={r.base ? 'base' : r.step}
        title={r.base ? t('trainingStudio.mm3.previewBase', 'Base') : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step: r.step })}
        takes={takeRows(r)}
        className={inUse ? LADDER_IN_USE : r.base ? 'border-sky-500/30 bg-sky-500/5' : undefined}
        badges={r.loss !== undefined && <span className="text-[11px] text-zinc-500 tabular-nums">
          {t('trainingStudio.mm3.previewLoss', 'Loss {{loss}}', { loss: r.loss.toFixed(3) })}</span>}
        actions={<TrainingLadderActions hasTakes={r.takes.length > 0} rendering={rendering === r.step}
          renderDisabled={ladder.running || rendering !== null || (!r.base && !r.ref)} onRender={() => onRender(r.step)}
          inUse={inUse} useDisabled={ladder.running || picking} onUse={r.ref ? () => onUse(r.step) : undefined}
          useLabel={t('trainingStudio.mm3.useCheckpoint', 'Use this checkpoint')} />} />;
    })}
  </TrainingPreviewLadderRows>;
};
