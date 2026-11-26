// ACE LM coalescing. The job holding the GPU lane plans compatible ACE jobs
// queued behind it in the same LM pass (one /lm call with an array), then hands
// each its plans. Followers never leave the queue: each still renders on its own
// lane turn and only skips its LM. A pass that fails or is cancelled hands them
// null, and they plan alone as if nothing had happened.
import type { AceRequest } from '../../aceClient.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

/** LM settings jobs must share to plan in one pass: the engine's
 *  ace_lm_batch_mismatch list. Prompt, lyrics, metadata and seeds may differ. */
const LM_SHARED = [
  'lm_mode', 'lm_model', 'lm_adapter', 'lm_adapter_scale', 'lm_temperature', 'lm_cfg_scale',
  'lm_cfg_cutoff_ratio', 'lm_top_p', 'lm_top_k', 'lm_min_p', 'lm_cfg_min_p', 'lm_cfg_logprob',
  'lm_cfg_rescale', 'lm_rep_penalty', 'lm_rep_window', 'lm_rep_mode', 'lm_dry_base', 'lm_dry_min_len',
  'lm_negative_prompt',
] as const;

const SOURCE_TASKS = ['cover', 'cover-nofsq', 'repaint', 'lego', 'extract', 'complete'];

export function isAceSourceTask(task: AceRequest['task_type']): boolean {
  return SOURCE_TASKS.includes(task ?? '');
}

/** Same key = may share a pass. Co-resident is part of it because it picks
 *  the engine's eviction policy for the whole call. */
export function aceLmKey(req: AceRequest, coResident: boolean): string {
  const r = req as unknown as Record<string, unknown>;
  return JSON.stringify([coResident, ...LM_SHARED.map(k => r[k] ?? null)]);
}

/** The LM runs when the job needs a plan and synthesizes without a source. */
export function needsAceLm(params: any, req: AceRequest): boolean {
  return params?.skipLm !== true && !req.audio_codes && !isAceSourceTask(req.task_type);
}

export function songsOf(job: GenerationJob): number {
  return Math.max(1, Math.floor(Number(job.params?.batchSize)) || 1);
}

export interface LmFollower {
  job: GenerationJob;
  req: AceRequest;
  songs: number;
  settle: (plans: AceRequest[] | null) => void;
}

export interface ClaimOptions {
  /** ACE jobs still waiting, oldest first. */
  pending: GenerationJob[];
  /** Songs one pass may hold (ace-server --max-batch). */
  cap: number;
  /** The request a job would send, without side effects (for the key check). */
  translate: (job: GenerationJob) => AceRequest;
  /** The request a claimed job will use, resolved for good (seeds written back). */
  prepare: (job: GenerationJob) => AceRequest;
  /** True when this request's plans are in the LM cache already. */
  cached: (job: GenerationJob, req: AceRequest) => boolean;
}

/** Claim waiting jobs that can share the lead's LM pass, up to `cap` songs. */
export function claimLmFollowers(lead: GenerationJob, leadReq: AceRequest, o: ClaimOptions): LmFollower[] {
  const key = aceLmKey(leadReq, lead.params?.coResident === true);
  let songs = songsOf(lead);
  const out: LmFollower[] = [];
  for (const cand of o.pending) {
    if (songs >= o.cap) break;
    if (cand.id === lead.id || cand.status !== 'pending' || cand.aceLmShare || cand.coalescedInto) continue;
    const n = songsOf(cand);
    if (songs + n > o.cap) continue;
    let probe: AceRequest;
    try { probe = o.translate(cand); } catch { continue; }
    if (!needsAceLm(cand.params, probe) || aceLmKey(probe, cand.params?.coResident === true) !== key) continue;
    const req = o.prepare(cand);
    if (o.cached(cand, req)) continue;
    let settle!: (plans: AceRequest[] | null) => void;
    const plans = new Promise<AceRequest[] | null>(resolve => { settle = resolve; });
    cand.aceLmShare = { lead: lead.id, req, plans };
    out.push({ job: cand, req, songs: n, settle });
    songs += n;
  }
  if (out.length) {
    const members = [lead.id, ...out.map(f => f.job.id)];
    for (const j of [lead, ...out.map(f => f.job)]) j.aceLmBatch = members;
  }
  return out;
}

/** Run one pass for the lead and its followers and hand each its plans.
 *  Returns the lead's. Any failure settles every follower with null first. */
export async function runSharedLmPass(
  leadReq: AceRequest,
  leadSongs: number,
  followers: LmFollower[],
  run: (body: AceRequest[]) => Promise<AceRequest[]>,
): Promise<AceRequest[]> {
  try {
    const body = [
      { ...leadReq, lm_batch_size: leadSongs },
      ...followers.map(f => ({ ...f.req, lm_batch_size: f.songs })),
    ];
    const out = await run(body);
    const want = leadSongs + followers.reduce((n, f) => n + f.songs, 0);
    if (!Array.isArray(out) || out.length !== want) {
      throw new Error(`Shared LM pass returned ${Array.isArray(out) ? out.length : 'no'} plans for ${want} songs`);
    }
    let at = leadSongs;
    for (const f of followers) {
      f.settle(out.slice(at, at + f.songs));
      at += f.songs;
    }
    return out.slice(0, leadSongs);
  } finally {
    // A no-op for followers settled above; the rest plan alone.
    for (const f of followers) f.settle(null);
  }
}
