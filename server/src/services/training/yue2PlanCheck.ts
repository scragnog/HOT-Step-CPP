// yue2PlanCheck.ts — the planner's own songs as its stop signal.
//
// KL to base is not a portable limit (2026-09-23: the same KL was clean for
// one band and wrecked for another). What breaks is the planner's plan:
// songs that loop, lose their vocals, or run to the cap instead of ending.
// So at each planner checkpoint, have it write N plans (lead sheet only,
// seconds each, no audio) and count genuine failures. The base planner's own
// rate on the same prompt is the reference: even the base writes an
// over-long plan now and then.
import fs from 'fs';
import path from 'path';
import { aceClient } from '../aceClient.js';
import { yue2PersistedSelection } from '../backends/yue2/index.js';
import { yue2SelectModel, yue2Synth, yue2FinalDetail, yue2Unload, type Yue2Selection } from '../backends/yue2/client.js';
import { classifyYue2Score } from '../backends/yue2/scoreHealth.js';

export interface Yue2PlanCheckOptions {
  /** Plans per checkpoint. 8 separates the ear-scored checkpoints. */
  plans: number;
  seed: number;
  /** Failure rate (fraction of plans) above the base planner's that stops
   *  the planner. The 2026-09-24 sweep: overcooked checkpoints 0.38-0.67
   *  above base, clean ones ~0. */
  margin: number;
  caption: string;
  lyrics: string;
}

export interface Yue2PlanResult { seed: number; verdict: string; endReason: string; bars: number; vocalShare: number; ms: number; abc?: string }
export interface Yue2PlanCheck { step: number; plans: Yue2PlanResult[]; failRate: number; failures: number }

/** A genuine failure: the plan loops or loses its vocals (runaway), has no
 *  vocal bars at all (unknown), or the plan stage ran to its cap. "long" is
 *  a length warning against the render cap, which depends on the lyric, not
 *  the planner, so it does not count. */
export function isPlanFailure(p: { verdict: string; endReason: string }): boolean {
  return p.verdict === 'runaway' || p.verdict === 'unknown' || p.endReason === 'limit_hit';
}

async function planOnce(style: string, lyrics: string, seed: number, signal?: AbortSignal): Promise<Yue2PlanResult> {
  const t0 = Date.now();
  const sub = await yue2Synth({ style, lyrics, cot: 'full', seed, plan_only: true });
  for (;;) {
    if (signal?.aborted) { await aceClient.cancelJob(sub.job_id).catch(() => {}); throw new Error('plan check cancelled'); }
    const s = await aceClient.pollJob(sub.job_id);
    if (s.status === 'done') break;
    if (s.status === 'failed' || s.status === 'cancelled') throw new Error(`plan ${s.status}`);
    if (Date.now() - t0 > 5 * 60_000) { await aceClient.cancelJob(sub.job_id).catch(() => {}); throw new Error('plan stage timed out'); }
    await new Promise(r => setTimeout(r, 500));
  }
  const d = await yue2FinalDetail(sub.job_id);
  const abc = (d.abc ?? '').trim();
  const endReason = d.end_reason ?? 'completed';
  const h = classifyYue2Score(abc, endReason);
  return { seed, verdict: h.verdict, endReason, bars: h.bars, vocalShare: h.vocalShare, ms: Date.now() - t0, abc };
}

/** Write `plans` plans with the given planner adapter ('' = base) and score
 *  them. Selects the adapter on the engine and restores the user's picks
 *  afterwards; the caller owns the engine being up. */
export async function runYue2PlanCheck(arAdapter: string, step: number, o: Yue2PlanCheckOptions, output: string, signal?: AbortSignal): Promise<Yue2PlanCheck> {
  const base = yue2PersistedSelection();
  const sel: Yue2Selection = { lm: base.lm, vae_variant: base.vae_variant === 'legacy' ? 'legacy' : 'standard',
    lm_adapter: arAdapter ? [{ path: arAdapter, scales: { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 } }] : [] };
  await yue2SelectModel(sel);
  const plans: Yue2PlanResult[] = [];
  try {
    for (let i = 0; i < o.plans; i++) plans.push(await planOnce(o.caption, o.lyrics, o.seed + i, signal));
  } finally {
    await yue2Unload().catch(() => {});
    await yue2SelectModel({ lm: base.lm, vae_variant: sel.vae_variant, lm_adapter: [
      ...(base.adapters.ar.path ? [{ path: base.adapters.ar.path, scales: base.adapters.ar.scales }] : []),
      ...(base.adapters.nar.path ? [{ path: base.adapters.nar.path, scales: base.adapters.nar.scales }] : []),
    ] }).catch(() => {});
  }
  const failures = plans.filter(isPlanFailure).length;
  const check = { step, plans, failRate: failures / Math.max(1, plans.length), failures };
  try {
    fs.mkdirSync(path.join(output, 'plan-checks'), { recursive: true });
    fs.writeFileSync(path.join(output, 'plan-checks', `step-${step}.json`), JSON.stringify(check, null, 1));
  } catch { /* the decision is already made; the record is for reading later */ }
  return check;
}
