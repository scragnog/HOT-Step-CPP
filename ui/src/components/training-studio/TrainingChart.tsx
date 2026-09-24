// TrainingChart.tsx — the live training curve
//
// Dependency-free inline SVG: the studio ships no charting library and one is
// not worth 40 kB.
//
// ONE AXIS PER METRIC (2026-09-21). Everything used to share a single y-range
// taken from the loss, which is why the learning rate could not be drawn at all
// — 1.5e-4 and 5.0 on one scale is a flat line against the floor — and why a
// second loss component squashed the first. Each metric now carries its own
// scale and its own colour-matched tick column in the gutter, the way a run
// viewer does it: the shape of every curve is readable at once, and only values
// on the SAME axis are comparable by height. The legend is the control: click a
// metric to drop it and its axis out of the layout.
//
// Every metric draws twice — the raw samples faint, and a trailing mean over
// them solid. On a rectified-flow run the per-step loss is dominated by the
// random timestep draw (measured corr(t, loss) = -0.52 on a YuE2 NAR run), so
// the raw layer is texture and the trend is the line to read.
//
// Two coordinate systems, deliberately:
//  • The lines live in a 100×100 viewBox stretched to the plot box
//    (preserveAspectRatio="none"), so strokes carry vector-effect
//    ="non-scaling-stroke" to stay hairlines regardless of the stretch.
//  • Every label and dot is an absolutely-positioned HTML element. Text inside
//    that viewBox would be squashed horizontally and circles would come out as
//    ellipses; 0–100 in viewBox units is exactly 0–100 % of the box, so the two
//    systems line up for free.
//
// Colours are explicit hex rather than Tailwind utilities: a series, its trend,
// its axis ticks and its legend swatch must be the same colour, and that is one
// value in one place. They are picked to read on both themes.
//
// HOVER: moving the mouse over the plot snaps a vertical cursor to the nearest
// sample and reads EVERY visible metric there, plus the step/epoch and the time
// from the start of the run. Elapsed comes from the server clock on live runs
// (TrainStepPoint.elapsedMs) and from the summed per-epoch `ms` on finished
// runs, whose logs keep only the epoch series.

import React, { useMemo, useState } from 'react';
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

/** What a step point may carry beyond the store's own shape: the 20-step stop
 *  mean (computed by the AITK card) and the AR KL term of a joint run. */
export type ChartStepPoint = TrainStepPoint & { ma20?: number; arKl?: number; klStop?: number; narMse?: number; narRecon?: number };

const VB = 100;
/** Vertical breathing room so the target line and the extremes clear the edges. */
const PAD_T = 8;
const PAD_B = 8;
/** Above this many epochs the per-epoch dots turn into a solid smear. */
const MAX_DOTS = 60;
/** Window of the moving average the auto-stop watches. */
const MA_WINDOW = 5;
/** Horizontal room one axis tick column needs, in px. Five visible metrics is
 *  three columns one side and two the other, so this is what the plot pays for
 *  readable axes on a card ~600 px wide. */
const AXIS_W = 44;
/** Tick levels, as a fraction of the plot band from the bottom up. */
const TICKS = [0.06, 0.37, 0.68, 0.96];

/** Trailing moving average, partial window at the head (so it starts at
 *  values[0] rather than after four blank samples). Deliberately NOT exported —
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

/** Axis ticks and tooltips share this: a learning rate needs exponent notation,
 *  a loss needs decimals, and a step time in ms needs neither. */
function fmtVal(v: number): string {
  const a = Math.abs(v);
  if (v === 0) return '0';
  if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(4);
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

/** One plotted quantity: its own scale, its own axis column, its own legend
 *  entry. `trend` is drawn solid over the faint `raw`. */
interface Metric {
  key: string;
  label: string;
  /** Raw samples, faint. Empty when the metric only has a trend (epoch MA5). */
  raw: Array<{ x: number; v: number }>;
  /** The line to read. Supplied by the trainer where it has one (ma5), else a
   *  trailing mean over `raw`. */
  trend: Array<{ x: number; v: number }>;
  colour: string;
  trendColour: string;
  /** Held-out evals are six points, not a band — they need their samples marked. */
  dots?: boolean;
  dashed?: boolean;
  /** Hidden until the user asks for it in the legend. */
  offByDefault?: boolean;
  /** Draw on another metric's scale (and skip an axis of its own). */
  axis?: string;
}

interface Props {
  epochs: ChartEpochPoint[];
  /** Per-step metrics — the noise layer and everything derived from it. Omitted
   *  on the done-state cards, which only have the epoch series persisted. */
  steps?: ChartStepPoint[];
  milestones?: ChartMilestone[];
  /** Target loss; <= 0 hides the target line. Drawn against the loss axis. */
  target: number;
  /** Held-out loss, in the SAME fractional-epoch x domain as the other series.
   *  Sparse by nature (one point per eval), and the only line here that can
   *  distinguish learning from memorising — a training loss on a random crop
   *  cannot. */
  evals?: Array<{ ep: number; loss: number }>;
  /** Epoch cap, for the x-axis caption. 0 = unknown. */
  maxEpochs?: number;
  /** AR KL stop target; > 0 draws it on the ar_kl axis, beside the stop
   *  reading (`klStop` on the steps) that is compared against it. */
  klTarget?: number;
  klStopLabel?: string;
}

export const TrainingChart: React.FC<Props> = ({
  epochs, steps = [], milestones = [], target, maxEpochs = 0, evals = [], klTarget = 0, klStopLabel,
}) => {
  const { t } = useTranslation();
  /** Cursor position as a fraction of the plot width, null when not hovering. */
  const [hoverFrac, setHoverFrac] = useState<number | null>(null);
  /** Metrics the user has toggled away from their default visibility. */
  const [flipped, setFlipped] = useState<Record<string, boolean>>({});

  const epochPts = epochs.filter(e => Number.isFinite(e.loss));
  // A step without a loss still carries a gradient norm and a step time (the
  // YuE2 decoder-only phase after the planner freezes): keep it, each series
  // filters its own values.
  const stepPts = steps.filter(s => Number.isFinite(s.loss) || Number.isFinite(s.narMse) || Number.isFinite(s.narRecon) || Number.isFinite(s.gradNorm) || Number.isFinite(s.stepMs));
  const evalPts = evals.filter(e => Number.isFinite(e.loss));

  // ── the metric table ──────────────────────────────────────────────────
  // Built from what the run actually carries: a trainer that reports no grad
  // norm gets no grad-norm axis rather than an empty one.
  const metrics = useMemo<Metric[]>(() => {
    const out: Metric[] = [];
    const epochLosses = epochPts.map(e => e.loss);

    /** A per-step numeric field, when enough steps carry it to be a line. */
    const fromSteps = (
      key: string, label: string, pick: (s: ChartStepPoint) => number | undefined,
      colour: string, trendColour: string, offByDefault?: boolean,
      trendOverride?: (s: ChartStepPoint) => number | undefined,
    ): void => {
      const raw: Array<{ x: number; v: number }> = [];
      for (const s of stepPts) {
        const v = pick(s);
        if (typeof v === 'number' && Number.isFinite(v)) raw.push({ x: s.ep, v });
      }
      if (raw.length < 2) return;
      // The trainer's own running mean where it publishes one — it is computed
      // over every step, not over the thinned series the chart holds.
      let trend: Array<{ x: number; v: number }> = [];
      if (trendOverride) {
        for (const s of stepPts) {
          const v = trendOverride(s);
          if (typeof v === 'number' && Number.isFinite(v)) trend.push({ x: s.ep, v });
        }
      }
      if (trend.length < 2) {
        const w = Math.min(51, Math.max(MA_WINDOW, Math.round(raw.length / 25)));
        const ma = movingAverage(raw.map(p => p.v), w);
        trend = raw.map((p, i) => ({ x: p.x, v: ma[i] }));
      }
      out.push({ key, label, raw, trend, colour, trendColour, offByDefault });
    };

    fromSteps('loss', t('trainingStudio.chart.mLoss', 'loss'),
      s => s.loss, '#5eead4', '#14b8a6', false, s => s.ma5);
    fromSteps('arKl', t('trainingStudio.chart.mArKl', 'ar_kl'),
      s => s.arKl, '#fcd34d', '#d97706');
    // The decoder's own loss: the one series that continues after the YuE2
    // planner freezes, when the composite loss above stops.
    fromSteps('narMse', t('trainingStudio.chart.mNarMse', 'decoder loss'),
      s => s.narMse, '#c4b5fd', '#7c3aed', true);
    // One reading per checkpoint: how well the decoder reproduces the album's
    // own latents. Falls and flattens; the decoder stop reads its knee.
    fromSteps('narRecon', t('trainingStudio.chart.mNarRecon', 'decoder recon'),
      s => s.narRecon, '#6ee7b7', '#059669');
    fromSteps('lr', t('trainingStudio.chart.mLr', 'learning rate'),
      s => s.lr, '#86efac', '#16a34a');
    fromSteps('grad', t('trainingStudio.chart.mGrad', 'grad norm'),
      s => s.gradNorm, '#f9a8d4', '#db2777', true);
    fromSteps('stepMs', t('trainingStudio.chart.mStepMs', 's / step'),
      s => (typeof s.stepMs === 'number' ? s.stepMs / 1000 : undefined), '#a5b4fc', '#6366f1', true);
    // The 20-step mean the auto-stop compares to target — a trend with no raw
    // layer of its own, since it summarises the loss already drawn.
    {
      const stop = stepPts
        .filter(s => typeof s.ma20 === 'number' && Number.isFinite(s.ma20))
        .map(s => ({ x: s.ep, v: s.ma20 as number }));
      if (stop.length >= 2) {
        out.push({
          key: 'stopMean', label: t('trainingStudio.chart.mStopMean', '20-step stop mean'),
          raw: [], trend: stop, colour: '#fb7185', trendColour: '#fb7185', dashed: true,
        });
      }
    }

    // The KL stop reading, on the ar_kl axis so it and the target line are
    // comparable by height with the raw KL.
    {
      const stop = stepPts
        .filter(s => typeof s.klStop === 'number' && Number.isFinite(s.klStop))
        .map(s => ({ x: s.ep, v: s.klStop as number }));
      if (stop.length >= 2 && out.some(m => m.key === 'arKl')) {
        out.push({
          key: 'klStop', label: klStopLabel ?? t('trainingStudio.chart.mKlStop', 'KL stop reading'),
          raw: [], trend: stop, colour: '#ea580c', trendColour: '#ea580c', dashed: true, axis: 'arKl',
        });
      }
    }

    if (epochPts.length >= 2) {
      const raw = epochPts.map(e => ({ x: e.epoch, v: e.loss }));
      const ma = movingAverage(epochLosses);
      out.push({
        key: 'epochLoss', label: t('trainingStudio.chart.mEpochLoss', 'epoch loss'),
        raw, trend: epochPts.map((e, i) => ({ x: e.epoch, v: ma[i] })),
        colour: '#fcd34d', trendColour: '#a78bfa', dots: epochPts.length <= MAX_DOTS,
      });
    }

    if (evalPts.length >= 1) {
      const raw = evalPts.map(e => ({ x: e.ep, v: e.loss }));
      out.push({
        key: 'eval', label: t('trainingStudio.chart.mEval', 'held-out'),
        raw, trend: raw, colour: '#34d399', trendColour: '#059669', dots: true,
      });
    }
    return out;
  }, [epochPts, stepPts, evalPts, t, klStopLabel]);

  /** On unless the user said otherwise — `flipped` holds the exceptions, so a
   *  metric that appears mid-run (the first eval, say) arrives at its default. */
  const isOn = (m: Metric): boolean => (flipped[m.key] ? !!m.offByDefault : !m.offByDefault);
  const visible = metrics.filter(isOn);
  const hasTarget = target > 0;

  if (metrics.length === 0) return null;

  // ── x domain, shared by everything ────────────────────────────────────
  // An epoch point sits at its own epoch number, a step point at
  // step/stepsPerEpoch. The run starts at 0, so the gap before the first epoch
  // average is real, not a layout bug.
  const lastEpoch = epochPts.length ? epochPts[epochPts.length - 1].epoch : 0;
  const lastStepEp = stepPts.length ? stepPts[stepPts.length - 1].ep : 0;
  const xMax = Math.max(lastEpoch, lastStepEp, 1);
  /** No epoch series at all: the step layer is carrying its own step numbers as
   *  x (see the store's epPos fallback), so the axis is steps, not epochs. */
  const stepAxis = epochPts.length === 0 && stepPts.length > 0;
  const xFor = (x: number) => (x / xMax) * VB;

  // ── per-metric scales ─────────────────────────────────────────────────
  // Each metric is normalised by its OWN extremes, which is the whole point:
  // a learning rate of 1.5e-4 and a loss of 5.0 both fill the band.
  const scales = new Map<string, { lo: number; hi: number }>();
  for (const m of metrics) {
    if (m.axis) continue;
    const vs = m.raw.concat(m.trend).map(p => p.v);
    let lo = Math.min(...vs);
    let hi = Math.max(...vs);
    // The target belongs to the loss axis, or it is a line at an arbitrary
    // height pretending to mean something.
    if (hasTarget && (m.key === 'loss' || m.key === 'epochLoss')) {
      lo = Math.min(lo, target); hi = Math.max(hi, target);
    }
    if (klTarget > 0 && m.key === 'arKl') { lo = Math.min(lo, klTarget); hi = Math.max(hi, klTarget); }
    const span = hi - lo;
    if (span > 0) { lo -= span * 0.08; hi += span * 0.08; } else { lo -= 0.5; hi += 0.5; }
    scales.set(m.key, { lo, hi });
  }
  const axisOf = new Map(metrics.filter(m => m.axis).map(m => [m.key, m.axis!]));
  const yIn = (key: string, v: number): number => {
    const s = scales.get(axisOf.get(key) ?? key)!;
    const f = (v - s.lo) / (s.hi - s.lo);
    return VB - PAD_B - Math.min(1, Math.max(0, f)) * (VB - PAD_T - PAD_B);
  };
  /** The value a tick level shows on a given axis. */
  const valueAt = (key: string, frac: number): number => {
    const s = scales.get(key)!;
    return s.lo + frac * (s.hi - s.lo);
  };
  const yAtFrac = (frac: number) => VB - PAD_B - frac * (VB - PAD_T - PAD_B);

  const line = (key: string, pts: Array<{ x: number; v: number }>) =>
    pts.map(p => `${xFor(p.x).toFixed(2)},${yIn(key, p.v).toFixed(2)}`).join(' ');

  // ── axis gutters ──────────────────────────────────────────────────────
  // Alternating so a four-metric run reads two a side, like the run viewers.
  const axisMetrics = visible.filter(m => m.key !== 'stopMean' && !m.axis);
  const leftAxes = axisMetrics.filter((_, i) => i % 2 === 0);
  const rightAxes = axisMetrics.filter((_, i) => i % 2 === 1);
  const padL = Math.max(AXIS_W, leftAxes.length * AXIS_W);
  const padR = Math.max(12, rightAxes.length * AXIS_W);

  const ticks = milestones.filter(m => m.epoch > 0 && m.epoch <= xMax);
  const targetVisible = hasTarget
    && visible.some(m => m.key === 'loss' || m.key === 'epochLoss');
  const targetKey = visible.some(m => m.key === 'epochLoss') ? 'epochLoss' : 'loss';
  const klTargetVisible = klTarget > 0 && visible.some(m => m.key === 'arKl' || m.key === 'klStop');

  // ── hover: snap to the nearest sample and read every visible metric ────
  // Cumulative epoch wall time, for the elapsed readout when no step carries a
  // server-clock elapsed (finished runs keep only the epoch series).
  const epochCumMs: number[] = [];
  {
    let acc = 0;
    let known = true;
    for (const e of epochPts) {
      if (typeof e.ms === 'number' && Number.isFinite(e.ms)) acc += e.ms; else known = false;
      epochCumMs.push(known ? acc : NaN);
    }
  }
  type Hover = { x: number; head: string; rows: Array<{ label: string; value: string; colour: string }>; foot: string[] } | null;
  let hover: Hover = null;
  if (hoverFrac !== null && visible.length) {
    const xPos = hoverFrac * xMax;
    const rows: Array<{ label: string; value: string; colour: string }> = [];
    let cursorX = xFor(xPos);
    const stIdx = stepPts.length >= 2 ? nearestIndex(stepPts.map(s => s.ep), xPos) : -1;
    const epIdx = epochPts.length >= 2 ? nearestIndex(epochPts.map(e => e.epoch), xPos) : -1;
    let head = '';
    if (stIdx >= 0) {
      const s = stepPts[stIdx];
      cursorX = xFor(s.ep);
      head = stepAxis
        ? t('trainingStudio.chart.hoverStepOnly', { step: s.step, defaultValue: 'step {{step}}' })
        : t('trainingStudio.chart.hoverStep', { step: s.step, epoch: s.ep.toFixed(2), defaultValue: 'step {{step}} · epoch {{epoch}}' });
    } else if (epIdx >= 0) {
      cursorX = xFor(epochPts[epIdx].epoch);
      head = t('trainingStudio.chart.hoverEpoch', { epoch: epochPts[epIdx].epoch, defaultValue: 'epoch #{{epoch}}' });
    }
    // Every visible metric is read at the cursor, from its own nearest sample —
    // the eval series has six points and the step series thousands, so one
    // shared index would be wrong for all but one of them.
    for (const m of visible) {
      const src = m.raw.length ? m.raw : m.trend;
      const i = nearestIndex(src.map(p => p.x), xPos);
      if (i < 0) continue;
      const tr = m.trend.length ? m.trend[Math.min(i, m.trend.length - 1)] : null;
      rows.push({
        label: m.label,
        value: m.raw.length && tr && m.raw !== m.trend
          ? `${fmtVal(src[i].v)}  (${fmtVal(tr.v)})`
          : fmtVal(src[i].v),
        colour: m.trendColour,
      });
    }
    const foot: string[] = [];
    const elapsed = stIdx >= 0 && typeof stepPts[stIdx].elapsedMs === 'number'
      ? stepPts[stIdx].elapsedMs as number
      : (epIdx >= 0 && Number.isFinite(epochCumMs[epIdx]) ? epochCumMs[epIdx] : NaN);
    if (Number.isFinite(elapsed)) {
      foot.push(t('trainingStudio.chart.hoverElapsed', { time: fmtElapsed(elapsed), defaultValue: '{{time}} from start' }));
    }
    if (head || rows.length) hover = { x: cursorX, head, rows, foot };
  }

  const LABEL = 'absolute text-[10px] tabular-nums pointer-events-none';
  /** x-axis ticks: five evenly spaced positions in the shared domain. */
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map(f => ({ f, v: f * xMax }));

  return (
    <div className="flex flex-col gap-1.5">
      <div className="relative w-full h-[200px] rounded-lg bg-zinc-100 dark:bg-black/30">
        {/* Axis gutters are padding on the frame; the plot is the box inside it,
            so every series and every tick share one coordinate origin. */}
        <div
          className="absolute inset-y-2 cursor-crosshair"
          style={{ left: padL, right: padR }}
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
            className="absolute inset-0 w-full h-full overflow-visible"
          >
            {/* gridlines at the tick levels — one set, shared by every axis */}
            {TICKS.map(f => (
              <line
                key={f} x1={0} y1={yAtFrac(f)} x2={VB} y2={yAtFrac(f)}
                className="text-zinc-400 dark:text-zinc-600" stroke="currentColor"
                strokeWidth={0.5} strokeOpacity={0.25} vectorEffect="non-scaling-stroke"
              />
            ))}

            {/* target loss, on the loss axis */}
            {targetVisible && (
              <line
                x1={0} y1={yIn(targetKey, target)} x2={VB} y2={yIn(targetKey, target)}
                stroke="#10b981" strokeWidth={1} strokeDasharray="4 3"
                vectorEffect="non-scaling-stroke"
              />
            )}

            {/* AR KL stop target, on the ar_kl axis */}
            {klTargetVisible && (
              <line
                x1={0} y1={yIn('arKl', klTarget)} x2={VB} y2={yIn('arKl', klTarget)}
                stroke="#ea580c" strokeWidth={1} strokeDasharray="2 2"
                vectorEffect="non-scaling-stroke"
              />
            )}

            {/* milestone ticks, along the bottom edge */}
            {ticks.map(m => (
              <line
                key={m.path}
                x1={xFor(m.epoch)} y1={VB} x2={xFor(m.epoch)} y2={VB - 7}
                stroke="#0ea5e9" strokeWidth={1.5} vectorEffect="non-scaling-stroke"
              >
                <title>{`${m.loss.toFixed(2)} · #${m.epoch}`}</title>
              </line>
            ))}

            {visible.map(m => (
              <g key={m.key}>
                {m.raw.length >= 2 && m.raw !== m.trend && (
                  <polyline
                    points={line(m.key, m.raw)} fill="none" stroke={m.colour}
                    strokeWidth={0.9} strokeOpacity={0.5} strokeLinejoin="round"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {m.trend.length >= 2 && (
                  <polyline
                    points={line(m.key, m.trend)} fill="none" stroke={m.trendColour}
                    strokeWidth={1.8} strokeLinejoin="round" strokeLinecap="round"
                    {...(m.dashed ? { strokeDasharray: '5 3' } : {})}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </g>
            ))}

            {/* hover cursor — a vertical hairline snapped to the nearest sample */}
            {hover && (
              <line
                x1={hover.x} y1={0} x2={hover.x} y2={VB}
                className="text-zinc-500 dark:text-zinc-400" stroke="currentColor"
                strokeWidth={1} strokeOpacity={0.8} vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>

          {/* sample dots — HTML so they stay round under the stretched viewBox */}
          {visible.filter(m => m.dots).flatMap(m => m.raw.map((p, i) => (
            <span
              key={`${m.key}-${i}`}
              className="absolute w-[3px] h-[3px] -ml-[1.5px] -mt-[1.5px] rounded-full pointer-events-none"
              style={{ left: `${xFor(p.x)}%`, top: `${yIn(m.key, p.v)}%`, background: m.trendColour }}
            />
          )))}

          {/* hover tooltip. Flips to the left of the cursor past the 55 % mark
              so it never runs off the right edge. */}
          {hover && (
            <div
              className="absolute top-1 z-10 rounded-md border border-zinc-300 dark:border-white/10 bg-white/95 dark:bg-zinc-900/95 px-2 py-1.5 text-[10px] leading-4 tabular-nums text-zinc-700 dark:text-zinc-200 shadow-sm pointer-events-none whitespace-nowrap"
              style={hover.x > 55 ? { right: `${100 - hover.x + 1}%` } : { left: `${hover.x + 1}%` }}
            >
              {hover.head && <div className="font-semibold text-zinc-800 dark:text-zinc-100">{hover.head}</div>}
              {hover.rows.map(r => (
                <div key={r.label} className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-[2px] flex-shrink-0" style={{ background: r.colour }} />
                  <span className="flex-1">{r.label}</span>
                  <span className="font-semibold">{r.value}</span>
                </div>
              ))}
              {hover.foot.map(f => <div key={f} className="text-zinc-500">{f}</div>)}
            </div>
          )}
        </div>

        {/* Axis tick columns, colour-matched to their series. They live in a box
            with the plot's own vertical inset, so a tick's `top: y%` is the same
            y the line is drawn at — mixing frame % with plot % puts every label
            a few px off its own gridline. */}
        <div className="absolute inset-y-2 left-0 right-0 pointer-events-none">
          {leftAxes.map((m, col) => TICKS.map(f => (
            <span
              key={`${m.key}-${f}`}
              className={`${LABEL} -translate-y-1/2 text-right`}
              style={{
                left: col * AXIS_W, width: AXIS_W - 6,
                top: `${yAtFrac(f)}%`, color: m.trendColour,
              }}
            >
              {fmtVal(valueAt(m.key, f))}
            </span>
          )))}
          {rightAxes.map((m, col) => TICKS.map(f => (
            <span
              key={`${m.key}-${f}`}
              className={`${LABEL} -translate-y-1/2`}
              style={{
                right: col * AXIS_W, width: AXIS_W - 6,
                top: `${yAtFrac(f)}%`, color: m.trendColour,
              }}
            >
              {fmtVal(valueAt(m.key, f))}
            </span>
          )))}
        </div>

        {/* target, on the same edge as the loss axis it belongs to */}
        {targetVisible && (
          <div className="absolute inset-y-2 left-0 right-0 pointer-events-none">
            <span
              className={`${LABEL} right-1 -translate-y-1/2 font-semibold`}
              style={{ top: `${Math.min(96, Math.max(4, yIn(targetKey, target)))}%`, color: '#10b981' }}
            >
              {target.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* x-axis ticks — the run's own units, under the plot */}
      <div className="relative h-3" style={{ marginLeft: padL, marginRight: padR }}>
        {xTicks.map(({ f, v }) => (
          <span
            key={f}
            className={`${LABEL} -translate-x-1/2 text-zinc-500`}
            style={{ left: `${f * 100}%` }}
          >
            {stepAxis || !maxEpochs ? Math.round(v) : v.toFixed(v < 10 ? 1 : 0)}
          </span>
        ))}
      </div>

      {/* legend — and the visibility control. A metric the run reports but the
          user does not want is one click away, which is the only way a card
          this size can offer six series honestly. */}
      <div className="flex items-center gap-x-3 gap-y-1 flex-wrap text-[10px] text-zinc-500">
        {metrics.map(m => {
          const on = isOn(m);
          return (
            <button
              key={m.key}
              type="button"
              onClick={() => setFlipped(f => ({ ...f, [m.key]: !f[m.key] }))}
              className={`flex items-center gap-1 rounded px-1 -mx-1 hover:bg-black/5 dark:hover:bg-white/10 transition-colors ${on ? '' : 'opacity-40'}`}
              title={t('trainingStudio.chart.toggle', 'Show or hide this series')}
            >
              <span
                className="w-2.5 h-2.5 rounded-[2px] border"
                style={{ borderColor: m.trendColour, background: on ? m.trendColour : 'transparent' }}
              />
              {m.label}
            </button>
          );
        })}
        {targetVisible && (
          <span className="flex items-center gap-1">
            <span className="w-3 border-t border-dashed" style={{ borderColor: '#10b981' }} />
            {t('trainingStudio.chart.legendTarget')}
          </span>
        )}
        {klTargetVisible && (
          <span className="flex items-center gap-1">
            <span className="w-3 border-t border-dotted" style={{ borderColor: '#ea580c' }} />
            {t('trainingStudio.chart.legendKlTarget', { value: klTarget, defaultValue: 'KL target {{value}}' })}
          </span>
        )}
        {ticks.length > 0 && (
          <span className="flex items-center gap-1">
            <span className="w-px h-2" style={{ background: '#0ea5e9' }} />
            {t('trainingStudio.chart.legendMilestone')}
          </span>
        )}
        <span className="ml-auto tabular-nums">
          {stepAxis
            ? t('trainingStudio.chart.stepAxis', { step: lastStepEp, defaultValue: 'step {{step}}' })
            : maxEpochs > 0
              ? t('trainingStudio.chart.epochAxis', { epoch: lastEpoch, total: maxEpochs })
              : t('trainingStudio.chart.epochAxisOpen', { epoch: lastEpoch })}
        </span>
      </div>
    </div>
  );
};

export default TrainingChart;
