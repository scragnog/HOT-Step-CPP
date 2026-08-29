// generation/planGuard.ts — degenerate-plan detection for adapter-led LM runs
//
// Deep AS1.5 LM adapters are loop-fragile at SAMPLING time: on some seeds the
// code stream falls into a short-period attractor mid-generation (measured
// 2026-08-29 — the same chain-trained adapter emitted a clean 900-code plan on
// two seeds and a 46-code stuck loop on a third; see the lm-adapter RCA
// memory). The failure is fully visible in the raw 5 Hz code statistics, so it
// can be caught BEFORE the DiT renders anything — a retry costs one LM pass,
// not a generation.
//
// Thresholds come from the measured probe corpus, which separated cleanly:
//   healthy plans (base + coherent adapters): loop fraction 0.4–12%, max
//   period-1..25 run ≤ 13, full length (duration*5 − 4 codes);
//   degenerate plans: loop fraction 53–99%, runs 40–721, or plans that stop
//   at a fraction of the requested duration (gaslight: 131/900, smoke3: 46/900).
// Musical repetition (choruses, riffs) lives WELL below these bars — the
// widest healthy tail-loop measured was 32% on a strongly styled adapter, so
// the loop bar sits at 45% with the run bar as an independent catch.

/** Loop statistics over a raw comma-separated 5 Hz code plan. */
export interface PlanLoopStats {
  count: number;      // codes in the plan
  loopFrac: number;   // best fraction of positions repeating at period 1..25
  loopPeriod: number; // the period that fraction was measured at
  maxRun: number;     // longest contiguous repeat run at that period
}

export function planLoopStats(codesCsv: string): PlanLoopStats {
  const codes = codesCsv.split(',').map(s => parseInt(s, 10)).filter(Number.isFinite);
  const n = codes.length;
  let loopFrac = 0, loopPeriod = 0;
  for (let p = 1; p <= 25 && p < n; p++) {
    let hits = 0;
    for (let i = p; i < n; i++) if (codes[i] === codes[i - p]) hits++;
    const frac = hits / (n - p);
    if (frac > loopFrac) { loopFrac = frac; loopPeriod = p; }
  }
  let maxRun = 0, run = 0;
  const p = Math.max(1, loopPeriod);
  for (let i = p; i < n; i++) {
    run = codes[i] === codes[i - p] ? run + 1 : 0;
    if (run > maxRun) maxRun = run;
  }
  return { count: n, loopFrac, loopPeriod, maxRun };
}

/**
 * Reason string when the plan is degenerate, null when it is healthy.
 * `durationSec` is the REQUESTED duration — a healthy planner counts to
 * duration*5 − 4 codes with metronomic reliability, so a big shortfall means
 * the adapter broke the stop behaviour (early im_end), not artistic intent.
 */
export function degeneratePlanReason(codesCsv: string, durationSec: number): string | null {
  const s = planLoopStats(codesCsv);
  if (durationSec > 0 && s.count < 0.8 * (durationSec * 5)) {
    return `early EOS — ${s.count} codes for a ${durationSec}s request (expected ~${durationSec * 5})`;
  }
  if (s.loopFrac > 0.45) {
    return `stuck loop — ${(100 * s.loopFrac).toFixed(0)}% of codes repeat at period ${s.loopPeriod}`;
  }
  if (s.maxRun > 50) {
    return `stuck run — ${s.maxRun} consecutive period-${s.loopPeriod} repeats (~${(s.maxRun / 5).toFixed(0)}s)`;
  }
  return null;
}
