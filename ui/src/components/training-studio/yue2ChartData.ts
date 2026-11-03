import type { TrainingMetricEvent } from '../../services/trainingApi';

export interface Yue2Point {
  step: number; ep: number; loss?: number; arCe?: number; cursorCe?: number;
  arKl?: number; narMse?: number; narRecon?: number; narDrift?: number;
  frozen?: boolean; gradNorm?: number; lr?: number; stepMs?: number;
  elapsedMs?: number; ma5?: number; ma20?: number;
}

/** Merge sparse checkpoint meters and replayed steps without inventing history. */
export function reduceYue2Metric(previous: Yue2Point[], item: TrainingMetricEvent, cap = 2000): Yue2Point[] {
  if (item.metric !== 'step' || !Number.isFinite(item.step)) return previous;
  const step = item.step!;
  const append = !previous.length || step > previous[previous.length - 1].step;
  const point: Yue2Point = { ...(!append ? previous.find(p => p.step === step) : undefined), step, ep: step };
  for (const key of ['loss', 'arCe', 'cursorCe', 'arKl', 'narMse', 'narRecon', 'narDrift', 'gradNorm', 'lr', 'stepMs'] as const) {
    const v = item[key];
    if (typeof v === 'number' && Number.isFinite(v)) point[key] = v;
  }
  if (Number.isFinite(item.trainMs)) point.elapsedMs = item.trainMs;
  if (item.plannerFrozen || point.frozen) {
    point.frozen = true;
    delete point.loss; delete point.arCe; delete point.cursorCe; delete point.arKl;
  }
  const merged = append ? [...previous, point]
    : [...previous.filter(p => p.step !== step), point].sort((a, b) => a.step - b.step);
  const next = merged.slice(-cap);
  const changed = append ? next.length - 1 : next.findIndex(p => p.step === step);
  const updateMean = (i: number) => {
    const p = next[i];
    const q = { ...p };
    delete q.ma5; delete q.ma20;
    if (p.loss !== undefined) {
      const values = next.slice(Math.max(0, i - 19), i + 1).map(p => p.loss).filter((v): v is number => v !== undefined);
      q.ma5 = values.slice(-5).reduce((a, b) => a + b, 0) / Math.min(5, values.length);
      if (values.length === 20) q.ma20 = values.reduce((a, b) => a + b, 0) / 20;
    }
    next[i] = q;
  };
  // A sparse update only affects its own average and the next 19. Evicting
  // the oldest point also changes the partial windows at the start.
  const prefix = merged.length > cap ? Math.min(19, next.length) : 0;
  for (let i = 0; i < prefix; i++) updateMean(i);
  if (changed >= 0) {
    for (let i = Math.max(prefix, changed); i < Math.min(next.length, changed + 20); i++) updateMean(i);
  }
  return next;
}

export function klReading(values: number[], mode: 'mean' | 'trend', window = 20, partial = false): number | undefined {
  const n = mode === 'trend' ? 30 : Math.max(1, Math.floor(window));
  if (values.length < n && !(partial && values.length >= 5)) return undefined;
  const v = values.slice(-n), mean = v.reduce((a, b) => a + b, 0) / v.length;
  // The engine uses the partial mean until the full trend window is available.
  if (mode === 'mean' || v.length < n) return mean;
  const xm = (n - 1) / 2;
  let xy = 0, xx = 0;
  v.forEach((y, i) => { xy += (i - xm) * (y - mean); xx += (i - xm) ** 2; });
  return mean + xy / xx * xm;
}
