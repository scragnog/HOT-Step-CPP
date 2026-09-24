// YuE2 checkpoint preview catalogue and request validation (plan 27).
// Rendering is intentionally injected by the joint runner: this module only
// owns bounded, durable metadata and safe file resolution.
import fs from 'fs';
import path from 'path';
import { randomInt, randomUUID } from 'crypto';

import type { Yue2JointPreviewOptions } from './types.js';
import { aceClient } from '../aceClient.js';
import { yue2PersistedSelection, type Yue2PersistedSelection } from '../backends/yue2/index.js';
import { yue2SelectModel, yue2Synth, yue2Warm, yue2Unload, yue2FinalDetail, type Yue2Selection } from '../backends/yue2/client.js';
import { splitMultipartMixed } from '../backends/yue2/generate.js';
import { classifyYue2Score, yue2PlanUsable } from '../backends/yue2/scoreHealth.js';

export const YUE2_JOINT_PREVIEW_DEFAULTS: Yue2JointPreviewOptions = {
  enabled: false, everySteps: 0, seconds: 90, seed: 424242,
  previewMaxFrames: 2250, baseline: false, control: false,
};

export interface Yue2JointPreviewRecord {
  id: string;
  step: number;
  kind: 'artist' | 'baseline' | 'control';
  status: 'rendering' | 'done' | 'failed';
  file?: string;
  error?: string;
  seconds: number;
  seed: number;
  previewMaxFrames: number;
  caption?: string;
  lyrics?: string;
  endReason?: string;
  stageEndReasons?: Record<string, string>;
  /** Render-free planner health: the preview's own lead sheet, classified.
   *  This is where AR over-training shows first (looping sections, no vocal),
   *  so a run can be judged checkpoint by checkpoint without listening. */
  score?: { verdict: string; reason: string; bars: number; vocalShare: number; sections: string[] };
  /** Artist takes go through the app's auto re-plan: the plan that was
   *  rendered (its seed) and every attempt before it. */
  plan?: { seed: number; accepted: boolean; attempts: Array<{ seed: number; verdict: string; reason: string }> };
  createdAt: number;
  updatedAt: number;
}
// Plan attempts per artist take: Rob's app setting (yue2ReplanAttempts 10).
const PREVIEW_REPLAN_ATTEMPTS = 10;

export class Yue2PreviewCleanupError extends Error { readonly code = 'PREVIEW_CLEANUP'; }

const MAX_RECORDS = 256;
const MAX_JSON_BYTES = 4 * 1024 * 1024;

export function parseYue2JointPreviewOptions(raw: unknown, everySteps: number): Yue2JointPreviewOptions {
  const b = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {};
  const integer = (key: string, fallback: number, lo: number, hi: number): number => {
    const n = Number(b[key]);
    return Number.isInteger(n) && n >= lo && n <= hi ? n : fallback;
  };
  const bool = (key: string, fallback: boolean): boolean =>
    b[key] === undefined || b[key] === null ? fallback : b[key] === true;
  const enabled = bool('enabled', false);
  const text = (key: string): string | undefined => {
    if (typeof b[key] !== 'string') return undefined;
    const value = (b[key] as string).trim();
    return value ? value.slice(0, 65536) : undefined;
  };
  return {
    enabled,
    everySteps: integer('everySteps', everySteps, 0, 100000),
    seconds: integer('seconds', 90, 8, 360),
    ...(integer('takes', 1, 1, 4) > 1 ? { takes: integer('takes', 1, 1, 4) } : {}),
    ...(integer('odeSteps', 0, 0, 64) > 0 ? { odeSteps: integer('odeSteps', 0, 0, 64) } : {}),
    ...(typeof b.narCacheRatio === 'number' && b.narCacheRatio >= 0 && b.narCacheRatio <= 0.9 ? { narCacheRatio: b.narCacheRatio } : {}),
    seed: integer('seed', 424242, 0, 0xffffffff),
    previewMaxFrames: integer('previewMaxFrames', 2250, 0, 9000),
    baseline: bool('baseline', false), control: bool('control', false),
    ...(bool('parallel', false) ? { parallel: true } : {}),
    ...(text('caption') ? { caption: text('caption') } : {}),
    ...(text('lyrics') ? { lyrics: text('lyrics') } : {}),
    ...(text('previewSongId') ? { previewSongId: text('previewSongId') } : {}),
  };
}

function cataloguePath(output: string): string { return path.join(output, 'previews', 'index.json'); }
function read(output: string): Yue2JointPreviewRecord[] {
  try {
    const file = cataloguePath(output);
    if (fs.statSync(file).size > MAX_JSON_BYTES) return [];
    const value = JSON.parse(fs.readFileSync(file, 'utf8')) as unknown;
    if (!Array.isArray(value)) return [];
    return value.filter((x): x is Yue2JointPreviewRecord => {
      const r = x as Yue2JointPreviewRecord;
      return !!r && typeof r.id === 'string' && typeof r.step === 'number'
        && (r.kind === 'artist' || r.kind === 'baseline' || r.kind === 'control')
        && (r.status === 'rendering' || r.status === 'done' || r.status === 'failed')
        && typeof r.seconds === 'number' && typeof r.seed === 'number';
    }).slice(-MAX_RECORDS);
  } catch { return []; }
}
function write(output: string, records: Yue2JointPreviewRecord[]): void {
  const dir = path.dirname(cataloguePath(output));
  fs.mkdirSync(dir, { recursive: true });
  const tmp = `${cataloguePath(output)}.${process.pid}.${randomUUID()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(records.slice(-MAX_RECORDS), null, 2), 'utf8');
  fs.renameSync(tmp, cataloguePath(output));
}

/** Upsert one immutable checkpoint preview record. Audio is written by the
 * renderer first; metadata is committed last so incomplete files stay hidden. */
export function recordYue2JointPreview(output: string, record: Yue2JointPreviewRecord): void {
  const records = read(output).filter(r => r.id !== record.id);
  write(output, [...records, record]);
}
/** Drop every preview record and file of a run except those at `keepStep`. */
export function pruneYue2JointPreviews(output: string, keepStep: number): number {
  const all = read(output);
  const gone = all.filter(r => r.step !== keepStep);
  for (const r of gone) {
    if (!r.file) continue;
    fs.rmSync(path.join(output, 'previews', r.file), { force: true });
    fs.rmSync(path.join(output, 'previews', r.file.replace(/\.wav$/i, '.score.abc')), { force: true });
  }
  write(output, all.filter(r => r.step === keepStep));
  return gone.length;
}
export function listYue2JointPreviews(output: string): Yue2JointPreviewRecord[] {
  return read(output).sort((a, b) => b.step - a.step || b.updatedAt - a.updatedAt);
}
export function resolveYue2JointPreview(output: string, file: string): string | null {
  if (!/^[A-Za-z0-9._-]+\.wav$/i.test(file)) return null;
  const candidate = path.resolve(output, 'previews', file);
  const root = path.resolve(output, 'previews');
  return candidate.startsWith(root + path.sep) && fs.existsSync(candidate) && fs.statSync(candidate).isFile()
    ? candidate : null;
}

export async function renderYue2JointPreview(input: {
  output: string; step: number; options: Yue2JointPreviewOptions;
  arAdapter: string; narAdapter: string; dataset?: string; signal?: AbortSignal;
  deps?: { select: typeof yue2SelectModel; warm: typeof yue2Warm; synth: typeof yue2Synth;
    poll: typeof aceClient.pollJob; result: typeof aceClient.getJobResult; cancel: typeof aceClient.cancelJob; unload: typeof yue2Unload;
    detail?: typeof yue2FinalDetail; persisted?: () => Yue2PersistedSelection };
}): Promise<Yue2JointPreviewRecord> {
  const started = Date.now();
  const base = input.deps?.persisted ? input.deps.persisted() : yue2PersistedSelection();
  const kinds: Array<'artist' | 'baseline' | 'control'> = ['artist'];
  if (input.options.baseline) kinds.push('baseline');
  if (input.options.control) kinds.push('control');
  let last: Yue2JointPreviewRecord | undefined;
  const api = input.deps ?? { select: yue2SelectModel, warm: yue2Warm, synth: yue2Synth,
    poll: aceClient.pollJob.bind(aceClient), result: aceClient.getJobResult.bind(aceClient), cancel: aceClient.cancelJob.bind(aceClient), unload: yue2Unload };
  let activeJob: string | undefined;
  let activeTerminal = true;
  let caption = input.options.caption || '';
  let lyrics = input.options.lyrics || '';
  if ((!caption || !lyrics) && input.dataset) {
    try {
      const parsed = JSON.parse(fs.readFileSync(input.dataset, 'utf8')) as { items?: unknown[] };
      const item = Array.isArray(parsed.items) ? (parsed.items.find(x => x && typeof x === 'object'
        && (!input.options.previewSongId || (x as Record<string, unknown>).id === input.options.previewSongId)) as Record<string, unknown> | undefined) : undefined;
      if (!caption) caption = typeof item?.style === 'string' ? item.style : typeof item?.prompt_style === 'string' ? item.prompt_style : '';
      if (!lyrics) lyrics = typeof item?.lyrics === 'string' ? item.lyrics : typeof item?.prompt_lyrics === 'string' ? item.prompt_lyrics : '';
    } catch { /* explicit empty prompt remains a visible render failure */ }
  }
  try {
    // Artist takes go out as ONE batched request (lm_batch_size = takes,
    // seeds seed, seed+1, ...): the planner stage is shared across the batch
    // and the engine returns one WAV part per song, song-major. Baseline and
    // control are single songs.
    const detail = input.deps?.detail ?? yue2FinalDetail;
    type Group = { kind: 'artist' | 'baseline' | 'control'; seeds: number[] };
    const wanted = Math.max(1, input.options.takes ?? 1);
    const groups: Group[] = kinds.flatMap((kind): Group[] => {
      if (kind !== 'artist') return [{ kind, seeds: [input.options.seed] }];
      const seeds = Array.from({ length: wanted }, (_, i) => input.options.seed + i);
      const out: Group[] = [];
      // One take per request: each renders its own re-planned lead sheet.
      for (const seed of seeds) out.push({ kind, seeds: [seed] });
      return out;
    });
    for (const { kind, seeds } of groups) {
      if (input.signal?.aborted) throw new Error('preview cancelled');
      const unity = { global: 1, attn: 1, mlp: 1, early: 1, mid: 1, late: 1 };
      const selected: Yue2Selection = { lm: base.lm, lm_adapter: kind === 'baseline' ? [] : [
        { path: input.arAdapter, scales: unity }, { path: input.narAdapter, scales: unity },
      ], vae_variant: base.vae_variant === 'legacy' ? 'legacy' : 'standard' };
      await api.select(selected);
      await api.warm({ vae_variant: selected.vae_variant });
      const records: Yue2JointPreviewRecord[] = seeds.map(seed => ({ id: randomUUID(), step: input.step, kind, status: 'rendering',
        seconds: input.options.seconds, seed, previewMaxFrames: input.options.previewMaxFrames,
        caption, lyrics, createdAt: started, updatedAt: Date.now() }));
      for (const r of records) recordYue2JointPreview(input.output, r);
      last = records[records.length - 1];
      // Artist takes: the app's auto re-plan (generate.ts). Plan with the
      // take's seed, redraw a plan the app would reject, render the kept one.
      let supplied: { abc: string; seed: number } | undefined;
      if (kind === 'artist') {
        const attempts: Array<{ seed: number; verdict: string; reason: string }> = [];
        for (let a = 1; a <= PREVIEW_REPLAN_ATTEMPTS; a++) {
          if (input.signal?.aborted) throw new Error('preview cancelled');
          const planSeed = a === 1 ? seeds[0] : randomInt(1, 2 ** 31 - 1);
          const planSub = await api.synth({ style: caption, lyrics, cot: 'full', seed: planSeed, plan_only: true });
          activeJob = planSub.job_id; activeTerminal = false;
          for (;;) {
            const st = await api.poll(planSub.job_id);
            if (st.status === 'done') break;
            if (st.status === 'failed' || st.status === 'cancelled') { activeTerminal = true; throw new Error(`preview plan ${st.status}`); }
            await new Promise(resolve => setTimeout(resolve, 500));
          }
          activeTerminal = true;
          const pd = await detail(planSub.job_id);
          const abc = (pd.abc ?? '').trim();
          const h = classifyYue2Score(abc, pd.end_reason);
          attempts.push({ seed: planSeed, verdict: h.verdict, reason: h.reason });
          supplied = { abc, seed: planSeed };
          if (abc && yue2PlanUsable(h.verdict, false)) break;
        }
        const accepted = !!supplied && yue2PlanUsable(attempts[attempts.length - 1].verdict, false);
        records[0].plan = { seed: supplied!.seed, accepted, attempts };
        recordYue2JointPreview(input.output, records[0]);
      }
      const sub = await api.synth({ style: kind === 'control' ? 'downtempo electronic, calm and spacious' : caption, lyrics: kind === 'control' ? '' : lyrics, cot: 'full', seed: supplied ? supplied.seed : seeds[0],
        ...(supplied ? { abc: supplied.abc } : {}),
        preview_max_frames: input.options.previewMaxFrames,
        ...(seeds.length > 1 ? { lm_batch_size: seeds.length } : {}),
        ...(input.options.odeSteps ? { ode_steps: input.options.odeSteps } : {}),
        ...(input.options.narCacheRatio !== undefined ? { nar_cache_ratio: input.options.narCacheRatio } : {}) });
      activeJob = sub.job_id;
      activeTerminal = false;
      const renderDeadline = Date.now() + 30 * 60_000;
      const markFailed = (message: string) => { for (const r of records) { if (r.status === 'rendering') { r.status = 'failed'; r.error = message; r.updatedAt = Date.now(); recordYue2JointPreview(input.output, r); } } };
      try {
        for (;;) {
          if (Date.now() >= renderDeadline) throw new Error('preview exceeded 30-minute time limit');
          if (input.signal?.aborted) {
            await api.cancel(sub.job_id).catch(() => {});
            const deadline = Date.now() + 30_000;
            while (Date.now() < deadline) {
              const stopped = await api.poll(sub.job_id).catch(() => null);
              if (stopped && (stopped.status === 'cancelled' || stopped.status === 'failed' || stopped.status === 'done')) { activeTerminal = true; break; }
              await new Promise(resolve => setTimeout(resolve, 250));
            }
            throw new Error('preview cancelled');
          }
          let status;
          try { status = await api.poll(sub.job_id); }
          catch (err) { await api.cancel(sub.job_id).catch(() => {}); throw err; }
          if (status.status === 'done') { activeTerminal = true; break; }
          if (status.status === 'failed' || status.status === 'cancelled') { activeTerminal = true; throw new Error(`preview engine job ${status.status}`); }
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        // Per-track end reasons and scores; a single song reports at the top level.
        const d = await detail(sub.job_id).catch(() => ({} as Awaited<ReturnType<typeof yue2FinalDetail>>));
        const perTrack = (i: number) => (d.tracks && d.tracks.length > i ? d.tracks[i] : i === 0 ? d : undefined);
        records.forEach((record, i) => {
          const t = perTrack(i);
          if (!t) return;
          if (typeof t.end_reason === 'string') record.endReason = t.end_reason;
          if (t.stage_end_reasons && typeof t.stage_end_reasons === 'object') record.stageEndReasons = t.stage_end_reasons as Record<string, string>;
          const abc = (t as { abc?: unknown }).abc;
          if (typeof abc === 'string' && abc.trim()) {
            const h = classifyYue2Score(abc, t.end_reason);
            record.score = { verdict: h.verdict, reason: h.reason, bars: h.bars, vocalShare: h.vocalShare, sections: h.sections };
            try { fs.mkdirSync(path.join(input.output, 'previews'), { recursive: true }); fs.writeFileSync(path.join(input.output, 'previews', `step-${input.step}-${kind}-s${record.seed}${record.plan ? `-p${record.plan.seed}` : ''}.score.abc`), abc); } catch { /* the verdict is already on the record */ }
          }
        });
        const response = await api.result(sub.job_id);
        const body = Buffer.from(await response.arrayBuffer());
        const contentType = response.headers.get('content-type') ?? '';
        const parts = seeds.length > 1 && /multipart\/mixed/i.test(contentType) ? splitMultipartMixed(body, contentType) : [body];
        if (parts.length < records.length) throw new Error(`preview batch returned ${parts.length} of ${records.length} tracks`);
        const dir = path.join(input.output, 'previews'); fs.mkdirSync(dir, { recursive: true });
        records.forEach((record, i) => {
          // Re-planned takes carry their plan seed, so a re-render never
          // collides with (or replaces) an earlier take of the same seed.
          const filename = `step-${input.step}-${kind}-s${record.seed}${record.plan ? `-p${record.plan.seed}` : ''}.wav`;
          fs.writeFileSync(path.join(dir, filename), parts[i], { flag: 'wx' });
          record.status = 'done'; record.file = filename; record.updatedAt = Date.now();
          recordYue2JointPreview(input.output, record);
        });
      } catch (err: any) {
        markFailed(err?.message || String(err));
        throw err;
      }
    }
    return last!;
  } catch (err: any) {
    if (last && last.status === 'rendering') { last.status = 'failed'; last.error = err?.message || String(err); last.updatedAt = Date.now(); recordYue2JointPreview(input.output, last); }
    throw err;
  } finally {
    const restore: Yue2Selection = { lm: base.lm, vae_variant: base.vae_variant === 'legacy' ? 'legacy' : 'standard', lm_adapter: [] };
    let cleanupError: unknown;
    if (activeJob && !activeTerminal) {
      try { await api.cancel(activeJob); } catch { /* bounded polling below */ }
      const deadline = Date.now() + 30_000;
      while (Date.now() < deadline && !activeTerminal) {
        try {
          const status = await api.poll(activeJob);
          if (status.status === 'done' || status.status === 'failed' || status.status === 'cancelled') activeTerminal = true;
        } catch { /* transient engine failure */ }
        if (!activeTerminal) await new Promise(resolve => setTimeout(resolve, 250));
      }
      if (!activeTerminal) cleanupError = new Error('preview job did not reach a terminal state');
    }
    if (activeTerminal) {
      try {
        const unloaded = await api.unload();
        if (!unloaded || unloaded.unloaded !== true || unloaded.loaded === true) cleanupError = new Error('YuE2 unload did not complete');
      } catch (err) { cleanupError = err; }
      if (!cleanupError) {
        try { await api.select({ ...restore, lm_adapter: [
          ...(base.adapters.ar.path ? [{ path: base.adapters.ar.path, scales: base.adapters.ar.scales }] : []),
          ...(base.adapters.nar.path ? [{ path: base.adapters.nar.path, scales: base.adapters.nar.scales }] : []),
        ] }); } catch (err) { cleanupError = err; }
      }
    }
    if (cleanupError) throw new Yue2PreviewCleanupError(`preview cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`);
  }
}

