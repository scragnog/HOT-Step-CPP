export type Yue2Optimizer = 'adamw' | 'prodigy' | 'muon';

export const YUE2_OPTIM_DEFAULTS = {
  optimizer: 'prodigy' as Yue2Optimizer,
  prodigyD0: 1e-6,
  muonLrScale: 1,
  muonNsSteps: 5,
};

export type Yue2OptimOptions = typeof YUE2_OPTIM_DEFAULTS;

/** Old queued jobs and saved runs predate optimizer selection and used AdamW. */
export function yue2StoredOptimizer(o: Partial<Yue2OptimOptions>): Yue2OptimOptions {
  return { ...YUE2_OPTIM_DEFAULTS, ...o, optimizer: o.optimizer ?? 'adamw' };
}

export function yue2OptimArgs(o: Partial<Yue2OptimOptions>): string[] {
  const c = yue2StoredOptimizer(o);
  const args = ['--optimizer', c.optimizer];
  if (c.optimizer === 'prodigy') args.push('--prodigy-d0', String(c.prodigyD0));
  if (c.optimizer === 'muon') args.push('--muon-lr-scale', String(c.muonLrScale), '--muon-ns-steps', String(c.muonNsSteps));
  return args;
}

export function yue2OptimRequest(b: Record<string, unknown>): Yue2OptimOptions {
  const c = { ...YUE2_OPTIM_DEFAULTS };
  if (b.optimizer !== undefined) {
    if (!['adamw', 'prodigy', 'muon'].includes(b.optimizer as string)) throw new Error('Unknown YuE2 optimizer');
    c.optimizer = b.optimizer as Yue2Optimizer;
  }
  for (const key of ['prodigyD0', 'muonLrScale', 'muonNsSteps'] as const) {
    if (b[key] === undefined) continue;
    const v = Number(b[key]);
    if (!Number.isFinite(v) || v <= 0 || (key === 'muonNsSteps' && (!Number.isInteger(v) || v > 20))) {
      throw new Error(`Invalid ${key}`);
    }
    c[key] = v;
  }
  return c;
}
