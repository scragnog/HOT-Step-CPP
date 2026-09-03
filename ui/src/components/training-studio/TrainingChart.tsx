// TrainingChart.tsx — the live training curve
//
// Replaces LossSparkline everywhere it was used. Dependency-free inline SVG for
// the same reason the sparkline was: the studio ships no charting library and
// one is not worth 40 kB.
//
// Two coordinate systems, deliberately:
//  • The lines live in a 100×100 viewBox stretched to the container
//    (preserveAspectRatio="none"), so strokes carry vector-effect
//    ="non-scaling-stroke" to stay hairlines regardless of the stretch.
//  • Every label and dot is an absolutely-positioned HTML element on top. Text
//    inside that viewBox would be squashed horizontally and circles would come
//    out as ellipses; 0–100 in viewBox units is exactly 0–100 % of the box, so
//    the two systems line up for free.
//
// Colours come from Tailwind text utilities via `currentColor` — never a
// hard-coded hex, so every layer reads correctly in both themes.
//
// NO LEARNING-RATE CURVE (the optional layer): five overlapping series in
// 180 px is unreadable, and a normalized lr line shares no y-meaning with loss,
// so it would need a second axis this card has no room to label honestly. Both
// the LM and DiT cards already show the live lr as a metric tile directly
// underneath.
//
// HOVER (2026-09-03): moving the mouse over the plot snaps a vertical cursor to
// the nearest point (a step when the step layer exists, else an epoch) and
// shows a tooltip with the step, epoch, loss, the epoch's MA5, lr / grad norm /
// step time when the series carries them, and the time from the start of the
// run to that point. Elapsed comes from the server clock on live runs
// (TrainStepPoint.elapsedMs) and from the summed per-epoch `ms` on finished
// runs, whose logs keep only the epoch series.

import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { TrainStepPoint } from '../../stores/trainingStore';

/** Structural shape of TrainLmEpoch / TrainDitEpoch — the two are identical.
 *  `ms` is the epoch's wall time when the caller has it (both live and persisted
 *  epoch records carry it); it feeds the tooltip's elapsed readout. */
export interface ChartEpochPoint {
  epoch: number;
  loss: number;
  ms?: number;
  lr?: number;
  gradNorm?: number;
}

export interface ChartMilestone {
  epoch: number;
  loss: number;
  path: string;
}

const VB = 100;
/** Vertical breathing room so the target line and the extremes clear the edges. */
const PAD_T = 8;
const PAD_B = 10;
/** Above this many epochs the per-epoch dots turn into a solid smear. */
const MAX_DOTS = 60;
/** Window of the moving average the auto-stop watches. */
const MA_WINDOW = 5;

/** Trailing moving average, partial window at the head (so it starts at
 *  losses[0] rather than after four blank epochs). Deliberately NOT exported —
 *  a non-component export here breaks Vite's fast refresh for the whole file. */
function movingAverage(values: number[], window = MA_WINDOW): number[] {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(sum / Math.min(i + 1, window));
  }
  return out;
}

function fmtElapsed(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    : `${m}:${String(sec).padStart(2, '0')}`;
}

function fmtSci(v: number): string {
  return v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4) ? v.toExponential(2) : v.toFixed(4);
}

/** Index of the element of `xs` (ascending) closest to `x`. */
function nearestIndex(xs: number[], x: number): number {
  if (!xs.length) return -1;
  let lo = 0, hi = xs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(xs[lo - 1] - x) <= Math.abs(xs[lo] - x)) return lo - 1;
  return lo;
}

interface Props {
  epochs: ChartEpochPoint[];
  /** Per-step loss — the faint noise layer. Omitted on the done-state cards,
   *  which only have the epoch series persisted to disk. */
  steps?: TrainStepPoint[];
  milestones?: ChartMilestone[];
  /** Target loss; <= 0 hides the target line. */
  target: number;
  /** Held-out loss, in the SAME fractional-epoch x domain as the other series.
   *  Sparse by nature (one point per eval), and the only line here that can
   *  distinguish learning from memorising — a training loss on a random crop
   *  cannot. Drawn in emerald with dots, because six points are a series the
   *  eye has to be able to follow. */
  evals?: Array<{ ep: number; loss: number }>;
  /** Epoch cap, for the x-axis caption. 0 = unknown. */
  maxEpochs?: number;
}

export const TrainingChart: React.FC<Props> = ({
  epochs, steps = [], milestones = [], target, maxEpochs = 0, evals = [],
}) => {
  const { t } = useTranslation();
  /** Cursor position as a fraction of the plot width, null when not hovering. */
  const [hoverFrac, setHoverFrac] = useState<number | null>(null);

  const epochPts = epochs.filter(e => Number.isFinite(e.loss));
  const stepPts = steps.filter(s => Number.isFinite(s.loss));
  if (epochPts.length < 2 && stepPts.length < 2) return null;

  const epochLosses = epochPts.map(e => e.loss);
  const ma = movingAverage(epochLosses);
  const hasTarget = target > 0;

  // The y-range is taken from the EPOCH series (plus the target), not from the
  // steps: a single per-step spike — routine on a DiT run — would otherwise
  // squash the epoch curve into a flat line at the bottom. The step layer is
  // clamped into the plot band instead, which is honest for a texture layer.
  // Before two epochs exist there is nothing else to scale by, so the steps
  // take over.
  const evalPts = evals.filter(e => Number.isFinite(e.loss));
  const rangeSrc = (epochLosses.length >= 2 ? epochLosses : stepPts.map(s => s.loss))
    // Include the held-out points: they routinely sit ABOVE the training band,
    // and clipping them to the top edge would draw a flat line that looks like
    // "no signal" when it is the most important signal here.
    .concat(evalPts.map(e => e.loss));
  let lo = Math.min(...rangeSrc, hasTarget ? target : Infinity);
  let hi = Math.max(...rangeSrc, hasTarget ? target : -Infinity);
  const span = hi - lo;
  if (span > 0) { lo -= span * 0.08; hi += span * 0.08; } else { lo -= 0.5; hi += 0.5; }

  // x is the true epoch position, shared by both layers: an epoch point sits at
  // its own epoch number, a step point at step/stepsPerEpoch. The run starts at
  // 0, so the gap before the first epoch average is real, not a layout bug.
  const lastEpoch = epochPts.length ? epochPts[epochPts.length - 1].epoch : 0;
  const lastStepEp = stepPts.length ? stepPts[stepPts.length - 1].ep : 0;
  const xMax = Math.max(lastEpoch, lastStepEp, 1);

  const xFor = (ep: number) => (ep / xMax) * VB;
  const yFor = (v: number) => {
    const f = (v - lo) / (hi - lo);
    return VB - PAD_B - Math.min(1, Math.max(0, f)) * (VB - PAD_T - PAD_B);
  };

  const line = (pts: Array<{ x: number; y: number }>) =>
    pts.map(p => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ');

  const stepLine = line(stepPts.map(s => ({ x: xFor(s.ep), y: yFor(s.loss) })));
  const epochLine = line(epochPts.map(e => ({ x: xFor(e.epoch), y: yFor(e.loss) })));
  const maLine = line(epochPts.map((e, i) => ({ x: xFor(e.epoch), y: yFor(ma[i]) })));

  const evalLine = line(evalPts.map(e => ({ x: xFor(e.ep), y: yFor(e.loss) })));

  const targetY = hasTarget ? yFor(target) : 0;
  const currentLoss = epochPts.length
    ? epochPts[epochPts.length - 1].loss
    : stepPts[stepPts.length - 1].loss;
  const showDots = epochPts.length > 0 && epochPts.length <= MAX_DOTS;
  // Only ticks for milestones that actually landed inside the drawn range.
  const ticks = milestones.filter(m => m.epoch > 0 && m.epoch <= xMax);

  // ── hover: snap to the nearest point and describe it ───────────────────
  // Cumulative epoch wall time, for the elapsed readout when no step carries a
  // server-clock elapsed (finished runs, whose logs keep only the epoch series).
  const epochCumMs: number[] = [];
  {
    let acc = 0;
    let known = true;
    for (const e of epochPts) {
      if (typeof e.ms === 'number' && Number.isFinite(e.ms)) acc += e.ms; else known = false;
      epochCumMs.push(known ? acc : NaN);
    }
  }
  type Hover = {
    x: number; y: number;
    lines: string[];
  } | null;
  let hover: Hover = null;
  if (hoverFrac !== null) {
    const epPos = hoverFrac * xMax;
    const epIdx = nearestIndex(epochPts.map(e => e.epoch), epPos);
    const stIdx = stepPts.length >= 2 ? nearestIndex(stepPts.map(s => s.ep), epPos) : -1;
    const lines: string[] = [];
    let x = 0, y = 0;
    if (stIdx >= 0) {
      const s = stepPts[stIdx];
      x = xFor(s.ep); y = yFor(s.loss);
      lines.push(t('trainingStudio.chart.hoverStep', { step: s.step, epoch: s.ep.toFixed(2), defaultValue: 'step {{step}} · epoch {{epoch}}' }));
      lines.push(t('trainingStudio.chart.hoverLoss', { loss: s.loss.toFixed(4), defaultValue: 'loss {{loss}}' }));
      if (typeof s.lr === 'number') lines.push(`lr ${fmtSci(s.lr)}`);
      if (typeof s.gradNorm === 'number') lines.push(`grad norm ${s.gradNorm.toFixed(3)}`);
      if (typeof s.stepMs === 'number') lines.push(`${(s.stepMs / 1000).toFixed(2)} s / step`);
      // The nearest completed epoch on or before this step, with its MA5.
      let prevEp = -1;
      for (let i = 0; i < epochPts.length; i++) if (epochPts[i].epoch <= s.ep + 1e-9) prevEp = i;
      if (prevEp >= 0) {
        lines.push(t('trainingStudio.chart.hoverEpochLine', {
          epoch: epochPts[prevEp].epoch, loss: epochPts[prevEp].loss.toFixed(4), ma5: ma[prevEp].toFixed(4),
          defaultValue: 'epoch #{{epoch}} · {{loss}} · ma5 {{ma5}}',
        }));
      }
      const elapsed = typeof s.elapsedMs === 'number' ? s.elapsedMs
        : (prevEp >= 0 && Number.isFinite(epochCumMs[prevEp]) ? epochCumMs[prevEp] : NaN);
      if (Number.isFinite(elapsed)) lines.push(t('trainingStudio.chart.hoverElapsed', { time: fmtElapsed(elapsed), defaultValue: '{{time}} from start' }));
    } else if (epIdx >= 0) {
      const e = epochPts[epIdx];
      x = xFor(e.epoch); y = yFor(e.loss);
      lines.push(t('trainingStudio.chart.hoverEpoch', { epoch: e.epoch, defaultValue: 'epoch #{{epoch}}' }));
      lines.push(t('trainingStudio.chart.hoverLoss', { loss: e.loss.toFixed(4), defaultValue: 'loss {{loss}}' }));
      lines.push(`ma5 ${ma[epIdx].toFixed(4)}`);
      if (typeof e.lr === 'number') lines.push(`lr ${fmtSci(e.lr)}`);
      if (typeof e.gradNorm === 'number') lines.push(`grad norm ${e.gradNorm.toFixed(3)}`);
      if (typeof e.ms === 'number') lines.push(`${fmtElapsed(e.ms)} this epoch`);
      if (Number.isFinite(epochCumMs[epIdx])) lines.push(t('trainingStudio.chart.hoverElapsed', { time: fmtElapsed(epochCumMs[epIdx]), defaultValue: '{{time}} from start' }));
    }
    if (lines.length) hover = { x, y, lines };
  }

  const LABEL = 'absolute text-[10px] tabular-nums pointer-events-none';

  return (
    <div className="flex flex-col gap-1.5">
      <div
        className="relative w-full h-[180px] rounded-lg bg-zinc-100 dark:bg-black/30 overflow-hidden cursor-crosshair"
        onMouseMove={(ev) => {
          const r = ev.currentTarget.getBoundingClientRect();
          if (r.width > 0) setHoverFrac(Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)));
        }}
        onMouseLeave={() => setHoverFrac(null)}
      >
        <svg
          viewBox={`0 0 ${VB} ${VB}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={t('trainingStudio.chart.aria')}
          className="absolute inset-0 w-full h-full"
        >
          {/* (a) per-step loss — noisy, thin, low opacity. yFor() clamps to the
              plot band, so an off-scale spike rides the top edge rather than
              escaping the card or dragging the y-range. */}
          {stepPts.length >= 2 && (
            <polyline
              points={stepLine}
              fill="none"
              className="text-zinc-400 dark:text-zinc-600"
              stroke="currentColor"
              strokeWidth={0.75}
              strokeOpacity={0.55}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          )}

          {/* (d) target loss */}
          {hasTarget && (
            <line
              x1={0} y1={targetY} x2={VB} y2={targetY}
              className="text-emerald-500"
              stroke="currentColor"
              strokeWidth={1}
              strokeDasharray="4 3"
              vectorEffect="non-scaling-stroke"
            />
          )}

          {/* (e) milestone ticks, along the bottom edge */}
          {ticks.map(m => (
            <line
              key={m.path}
              x1={xFor(m.epoch)} y1={VB} x2={xFor(m.epoch)} y2={VB - 9}
              className="text-sky-500"
              stroke="currentColor"
              strokeWidth={1.5}
              vectorEffect="non-scaling-stroke"
            >
              <title>{`${m.loss.toFixed(2)} · #${m.epoch}`}</title>
            </line>
          ))}

          {/* (b0) held-out loss — drawn UNDER the epoch line so the training
              curve stays readable where they cross */}
          {evalPts.length >= 2 && (
            <polyline points={evalLine} fill="none" stroke="rgb(16 185 129)" strokeWidth="1.6"
              strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
          )}
          {/* (b) epoch loss */}
          {epochPts.length >= 2 && (
            <polyline
              points={epochLine}
              fill="none"
              className="text-amber-500"
              stroke="currentColor"
              strokeWidth={1.25}
              strokeOpacity={0.7}
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          )}

          {/* (c) MA5 — the quantity the auto-stop actually compares to target */}
          {epochPts.length >= 2 && (
            <polyline
              points={maLine}
              fill="none"
              className="text-violet-500"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          )}

          {/* (h) hover cursor — a vertical hairline snapped to the nearest point */}
          {hover && (
            <line
              x1={hover.x} y1={0} x2={hover.x} y2={VB}
              className="text-zinc-500 dark:text-zinc-400"
              stroke="currentColor"
              strokeWidth={1}
              strokeOpacity={0.8}
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>

        {/* (b) epoch dots — HTML so they stay round under the stretched viewBox */}
        {showDots && epochPts.map(e => (
          <span
            key={e.epoch}
            className="absolute w-[3px] h-[3px] -ml-[1.5px] -mt-[1.5px] rounded-full bg-amber-500 pointer-events-none"
            style={{ left: `${xFor(e.epoch)}%`, top: `${yFor(e.loss)}%` }}
          />
        ))}

        {/* (h) hover marker + tooltip. The tooltip flips to the left of the
            cursor past the 60 % mark so it never runs off the right edge. */}
        {hover && (
          <>
            <span
              className="absolute w-[7px] h-[7px] -ml-[3.5px] -mt-[3.5px] rounded-full border-2 border-violet-500 bg-white dark:bg-zinc-900 pointer-events-none"
              style={{ left: `${hover.x}%`, top: `${hover.y}%` }}
            />
            <div
              className="absolute top-2 z-10 rounded-md border border-zinc-300 dark:border-white/10 bg-white/95 dark:bg-zinc-900/95 px-2 py-1.5 text-[10px] leading-4 tabular-nums text-zinc-700 dark:text-zinc-200 shadow-sm pointer-events-none whitespace-nowrap"
              style={hover.x > 60
                ? { right: `${100 - hover.x + 1}%` }
                : { left: `${hover.x + 1}%` }}
            >
              {hover.lines.map((l, i) => (
                <div key={i} className={i === 0 ? 'font-semibold text-zinc-800 dark:text-zinc-100' : ''}>{l}</div>
              ))}
            </div>
          </>
        )}

        {/* (f) axis labels */}
        <span className={`${LABEL} left-1.5 top-1 text-zinc-500`}>{hi.toFixed(3)}</span>
        <span className={`${LABEL} left-1.5 bottom-1 text-zinc-500`}>{lo.toFixed(3)}</span>
        <span className={`${LABEL} right-1.5 top-1 font-semibold text-zinc-600 dark:text-zinc-300`}>
          {t('trainingStudio.chart.current', { loss: currentLoss.toFixed(4) })}
        </span>
        <span className={`${LABEL} left-1/2 -translate-x-1/2 bottom-1 text-zinc-500`}>
          {maxEpochs > 0
            ? t('trainingStudio.chart.epochAxis', { epoch: lastEpoch, total: maxEpochs })
            : t('trainingStudio.chart.epochAxisOpen', { epoch: lastEpoch })}
        </span>
        {hasTarget && (
          <span
            className={`${LABEL} right-1.5 -translate-y-1/2 font-semibold text-emerald-500`}
            style={{ top: `${Math.min(92, Math.max(8, targetY))}%` }}
          >
            {target.toFixed(2)}
          </span>
        )}
      </div>

      {/* (g) legend */}
      <div className="flex items-center gap-3 flex-wrap text-[10px] text-zinc-500">
        {stepPts.length >= 2 && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-px bg-zinc-400 dark:bg-zinc-600" />
            {t('trainingStudio.chart.legendStep')}
          </span>
        )}
        {/* Only advertise a series that is actually drawn. These two were
            unconditional, which on a step-only run promised an epoch line and a
            5-epoch average that can never appear. */}
        {epochPts.length >= 2 && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-px bg-amber-500" />
            {t('trainingStudio.chart.legendEpoch')}
          </span>
        )}
        {epochPts.length >= 2 && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-violet-500" />
            {t('trainingStudio.chart.legendMa5')}
          </span>
        )}
        {evalPts.length >= 2 && (
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-emerald-500" />
            {t('trainingStudio.chart.legendEval', 'held-out')}
          </span>
        )}
        {hasTarget && (
          <span className="flex items-center gap-1">
            <span className="w-3 border-t border-dashed border-emerald-500" />
            {t('trainingStudio.chart.legendTarget')}
          </span>
        )}
        {ticks.length > 0 && (
          <span className="flex items-center gap-1">
            <span className="w-px h-2 bg-sky-500" />
            {t('trainingStudio.chart.legendMilestone')}
          </span>
        )}
      </div>
    </div>
  );
};

export default TrainingChart;
