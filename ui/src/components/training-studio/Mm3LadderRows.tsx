// Mm3LadderRows.tsx — an MM3 run's records as CheckpointLadder rungs with the
// shared review parts (LadderReview.tsx): blind letters, likeness/corruption
// and notes, the overall score and best rung. No server state:
// Mm3CheckpointLadder.tsx owns the queries and writes.
import React, { type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import type { Mm3Ladder, Mm3LadderPreview, Mm3RungScore } from '../../services/trainingApi';
import { CheckpointLadder } from './CheckpointLadder';
import { plainOverall, RungScoreRow, type RungPatch } from './LadderReview';
import type { PreviewTake } from './TrainingPreviewPanel';

const sameRef = (a: string | null | undefined, b: string) => !!a && a.replace(/\\/g, '/').toLowerCase() === b.replace(/\\/g, '/').toLowerCase();
/** A "Render more" take carries its random seed in the name (`…-s123.wav`);
 *  the take without one is the run's fixed seed, the one rungs compare on. */
const extraTake = (p: Mm3LadderPreview) => /-s\d+\.wav$/i.test(p.file);

export interface Mm3Rung { step: number; base: boolean; ref?: string; loss?: number; takes: Mm3LadderPreview[] }

/** Base first (when the run rendered one), then every saved checkpoint, plus
 *  any step that has previews but whose checkpoint is gone (listen only). */
export function mm3Rungs(ladder: Pick<Mm3Ladder, 'checkpoints' | 'previews'>): Mm3Rung[] {
  const rungs = new Map<string, Mm3Rung>();
  const base = ladder.previews.filter(p => p.base);
  if (base.length) rungs.set('base', { step: 0, base: true, takes: base });
  for (const c of ladder.checkpoints) rungs.set(String(c.step), { step: c.step, base: false, ref: c.ref, loss: c.loss, takes: [] });
  for (const p of ladder.previews.filter(v => !v.base)) {
    const r = rungs.get(String(p.step)) ?? { step: p.step, base: false, loss: p.loss, takes: [] };
    r.takes.push(p);
    rungs.set(String(p.step), r);
  }
  for (const r of rungs.values()) r.takes.sort((a, b) => Number(extraTake(a)) - Number(extraTake(b)) || a.ts - b.ts);
  return [...rungs.values()].sort((a, b) => Number(b.base) - Number(a.base) || a.step - b.step);
}

/** Overall per scored checkpoint, and the best: highest, ties to the earlier step. */
export function mm3Overall(ladder: Pick<Mm3Ladder, 'checkpoints' | 'scores'>) {
  const byStep = new Map(ladder.scores.map(s => [s.step, s]));
  const overall = new Map(ladder.checkpoints.map(c => [c.step, plainOverall(byStep.get(c.step)?.likeness, byStep.get(c.step)?.corruption)]));
  let best: number | undefined;
  for (const [step, o] of [...overall].sort((a, b) => a[0] - b[0])) if (o !== null && (best === undefined || o > overall.get(best)!)) best = step;
  return { overall, best };
}

export interface Mm3ReviewState {
  blind: boolean;
  scores: Record<number, Partial<Mm3RungScore> | undefined>;
  notes: Record<number, string | undefined>;
  onNote: (step: number, note: string) => void;
  onScore: (step: number, patch: RungPatch) => void;
  idPrefix: string;
}

/** The rungs, with no server state: the live component and its test draw it. */
export const Mm3LadderRows: React.FC<{
  ladder: Mm3Ladder;
  review: Mm3ReviewState;
  header?: ReactNode;
  /** Step being rendered, or null. */
  rendering: number | null;
  /** The last Render that failed, shown on its rung as YuE2 shows a failed take. */
  renderError?: { step: number; message: string } | null;
  picking: boolean;
  pickError?: { step: number; message: string } | null;
  onRender: (step: number) => void;
  onUse: (step: number) => void;
}> = ({ ladder, review, header, rendering, renderError, picking, pickError, onRender, onUse }) => {
  const { t } = useTranslation();
  const { blind } = review;
  const label = (step: number) => ladder.blindLabels[step] ?? '';
  const { overall, best } = mm3Overall(ladder);
  const takeRows = (r: Mm3Rung): PreviewTake[] => [...r.takes.map((p, i): PreviewTake => {
    const caption = p.kind === 'control' ? t('trainingStudio.mm3.previewControlCaption', 'Control caption') : t('trainingStudio.mm3.previewArtistCaption', 'Artist caption');
    if (p.pruned || !p.audioUrl) return { id: p.id, label: caption,
      text: `${t('trainingStudio.refine.take', 'Take {{n}}', { n: i + 1 })}: ${t('trainingStudio.refine.audioPruned', 'audio removed by cleanup')}` };
    return {
      id: p.id, src: p.audioUrl,
      downloadName: `${ladder.datasetName || 'preview'}_${r.base ? 'base' : blind ? `rung${label(r.step)}` : `step${r.step}`}_take${i + 1}_seed${p.seed}.wav`,
      label: `${caption} · ${t('trainingStudio.mm3.previewTake', '{{seconds}} s · seed {{seed}} · rendered in {{renderSeconds}} s', {
        seconds: p.seconds, seed: p.seed, renderSeconds: (p.ms / 1000).toFixed(0) })}${r.base ? '' : extraTake(p)
        ? ` · ${t('trainingStudio.refine.seedRandom', 'random seed: a new song, not comparable with other rungs')}`
        : ` · ${t('trainingStudio.refine.seedFixed', 'same seed on every rung: compare rungs on this one')}`}`,
      sublabel: p.caption || undefined,
    };
  }),
  // The render in flight, or the one that failed, as a take row of its rung.
  ...(rendering === r.step ? [{ id: `pending-${r.step}`, label: t('trainingStudio.refine.take', 'Take {{n}}', { n: r.takes.length + 1 }),
    text: `${t('trainingStudio.refine.take', 'Take {{n}}', { n: r.takes.length + 1 })}: ${t('trainingStudio.mm3.takePending', 'pending')}` }] : []),
  ...(rendering !== r.step && renderError?.step === r.step ? [{ id: `failed-${r.step}`, label: t('trainingStudio.refine.take', 'Take {{n}}', { n: r.takes.length + 1 }),
    text: `${t('trainingStudio.refine.take', 'Take {{n}}', { n: r.takes.length + 1 })}: ${t('trainingStudio.mm3.takeFailed', 'failed')} — ${renderError.message}` }] : [])];
  const rungs = mm3Rungs(ladder);
  // Blind: the checkpoints in letter order, so step order cannot show
  // through; listen-only steps (checkpoint deleted) after them.
  if (blind) rungs.sort((a, b) => Number(b.base) - Number(a.base) || Number(!a.ref) - Number(!b.ref)
    || (a.ref && b.ref ? label(a.step).localeCompare(label(b.step)) : a.step - b.step));
  return <CheckpointLadder rendering={rendering} onRender={onRender} picking={picking} locked={ladder.running}
    useLabel={t('trainingStudio.mm3.useCheckpoint', 'Use this checkpoint')} header={header}
    rungs={rungs.map(r => {
      const o = r.ref ? overall.get(r.step) ?? null : null;
      return {
        key: r.base ? 'base' : r.step, step: r.step, id: r.base ? undefined : `${review.idPrefix}-${r.step}`,
        title: r.base ? t('trainingStudio.mm3.previewBase', 'Base')
          : blind && label(r.step) ? t('trainingStudio.refine.blindRung', 'Rung {{label}}', { label: label(r.step) })
          : t('trainingStudio.refine.rungStep', 'Step {{step}}', { step: r.step }),
        takes: takeRows(r),
        className: r.base ? 'border-sky-500/30 bg-sky-500/5' : r.ref && r.step === best ? 'border-sky-500/70' : undefined,
        badges: <>
          {!blind && r.loss !== undefined && <span className="text-[11px] text-zinc-500 tabular-nums">
            {t('trainingStudio.mm3.previewLoss', 'Loss {{loss}}', { loss: r.loss.toFixed(3) })}</span>}
          {r.ref && r.step === best && <span className="px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-700 dark:text-sky-300 text-[10px] font-semibold">{t('trainingStudio.refine.bestBadge', 'best by score')}</span>}
          {o !== null && <span className="font-mono text-sky-700 dark:text-sky-300" title={t('trainingStudio.mm3.overallInfo', 'overall = (likeness + (6 − corruption)) / 2')}>{t('trainingStudio.refine.overall', 'overall {{n}}', { n: o.toFixed(2) })}</span>}
        </>,
        children: <>
          {r.ref && <RungScoreRow score={review.scores[r.step]} note={review.notes[r.step]}
            onNote={v => review.onNote(r.step, v)} onScore={patch => review.onScore(r.step, patch)} />}
          {pickError?.step === r.step && <div role="alert" className="mt-2 text-xs text-rose-600 dark:text-rose-400">{pickError.message}</div>}
        </>,
        inUse: !!r.ref && sameRef(ladder.inUse, r.ref),
        canRender: r.base || !!r.ref,
        onUse: r.ref ? () => onUse(r.step) : undefined,
      };
    })} />;
};
