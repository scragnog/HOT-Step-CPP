#!/usr/bin/env npx tsx
/**
 * dit-adapter-ab.ts — render ONE plan through several DiT adapters.
 *
 * Purpose: A/B a dataset's older adapter run against a newer one (or the bare
 * base) with everything else held fixed. Two ways to fix the plan:
 *
 *   text2music (default): the LM runs once per dataset (same seed, caption,
 *     lyrics, LM adapter); every arm renders those exact codes.
 *   cover / cover-nofsq (--task): no LM at all — a real track is the source
 *     (plan.datasets[].source, a WAV). cover-nofsq with --noise-strength > 0
 *     starts the DiT from the source latents, so the arms differ ONLY by the
 *     adapter weights and the source is a fixed reference for the ear.
 *
 * Two sampler configs per arm: P50 = the production settings Rob listens with
 * (50 steps, guidance 20, md_hamiltonian_v2 / linear_quadratic / dynamic_cfg)
 * and T8 = the plain turbo recipe (8 steps, guidance 1, euler). --configs picks.
 *
 *   npx tsx scripts/dit-adapter-ab.ts --plan <plan.json> --out <dir> [--duration 60]
 *        [--only <slug>] [--task text2music|cover|cover-nofsq] [--cover-strength 1.0]
 *        [--noise-strength 0.5] [--configs P50,T8]
 *
 * plan.json: { datasets: [{ slug, caption, lyrics, bpm, keyscale, timesignature,
 *              seed, lmAdapter?, source?, arms: [{ tag, adapter?, scale? }] }] }
 *
 * Needs ace-server on config.aceServer.port (the Node server is NOT needed).
 * Idempotent: an existing wav is skipped. Writes <out>/<slug>/<config>_<task>_<arm>.wav,
 * <out>/<slug>/lm.json (text2music only) and <out>/manifest.json.
 */
import fs from 'fs';
import path from 'path';
import { aceClient, type AceRequest } from '../src/services/aceClient.js';
import { config } from '../src/config.js';

const JOB_DEADLINE_MS = 15 * 60_000;

interface Arm { tag: string; adapter?: string; scale?: number }
interface PlanDataset {
  slug: string; caption: string; lyrics: string; bpm: number; keyscale: string; timesignature?: string;
  seed: number; lmAdapter?: string; trigger?: string; source?: string; arms: Arm[];
}
interface Plan { datasets: PlanDataset[] }

const CONFIGS: Record<string, Partial<AceRequest>> = {
  // What Rob's generations use (songs table, 2026-09-03 Demo Dirt rows).
  P50: { inference_steps: 50, guidance_scale: 20, shift: -1, infer_method: 'md_hamiltonian_v2', scheduler: 'linear_quadratic', guidance_mode: 'dynamic_cfg' },
  // The mirrored-audition turbo recipe.
  T8: { inference_steps: 8, guidance_scale: 1.0, shift: 0, infer_method: 'euler' },
};

const SYNTH_MODEL = 'acestep-v15-merge-base-sft-turbo-xl-thirds-BF16.gguf';
const LM_MODEL = 'acestep-5Hz-lm-4B-BF16.gguf';
const VAE_MODEL = 'scragvae-BF16.gguf';
const EMB_MODEL = 'Qwen3-Embedding-0.6B-BF16.gguf';
const GROUP_SCALES = { self_attn: 1, cross_attn: 1, mlp: 1, cond_embed: 1, time_embed: 0, proj_in: 0 };
const COVER_TASKS = new Set(['cover', 'cover-nofsq']);

function die(msg: string): never { console.error(`\nERROR: ${msg}`); process.exit(1); }

function parseArgs(argv: string[]): Map<string, string> {
  const m = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) { m.set(a.slice(2), next); i++; } else m.set(a.slice(2), '1');
    }
  }
  return m;
}

async function awaitJob(jobId: string, what: string): Promise<void> {
  const deadline = Date.now() + JOB_DEADLINE_MS;
  for (;;) {
    if (Date.now() > deadline) { await aceClient.cancelJob(jobId).catch(() => {}); throw new Error(`${what} timed out`); }
    const st = await aceClient.pollJob(jobId);
    if (st.status === 'done') return;
    if (st.status === 'failed') throw new Error(`${what} failed — see the engine log`);
    if (st.status === 'cancelled') throw new Error(`${what} cancelled`);
    await new Promise(r => setTimeout(r, 500));
  }
}

/** Same trigger placement translateParams/generate.ts use: prepend, skip if present. */
function withTrigger(caption: string, trigger: string): string {
  if (!trigger) return caption;
  if (caption.includes(trigger)) return caption;
  return `${trigger}, ${caption}`;
}

function baseRequest(d: PlanDataset, duration: number): AceRequest {
  const trigger = d.trigger ?? d.slug;
  const req: AceRequest = {
    caption: withTrigger(d.caption, trigger),
    lyrics: d.lyrics,
    bpm: d.bpm,
    duration,
    keyscale: d.keyscale,
    vocal_language: 'en',
    seed: d.seed,
    lm_batch_size: 1,
    lm_temperature: 0.85,
    lm_cfg_scale: 2,
    lm_top_p: 0.9,
    lm_top_k: 0,
    lm_negative_prompt: 'NO USER INPUT',
    use_cot_caption: true,
    task_type: 'text2music',
    audio_cover_strength: 0.31,
    synth_model: SYNTH_MODEL,
    lm_model: LM_MODEL,
    vae_model: VAE_MODEL,
    emb_model: EMB_MODEL,
    ...CONFIGS.P50,
  };
  if (d.timesignature) req.timesignature = d.timesignature;
  if (d.lmAdapter) {
    req.lm_adapter = d.lmAdapter;
    req.lm_adapter_scale = 1;
    req.lm_rep_penalty = 1.05; // translateParams default when an LM adapter rides the request
  }
  return req;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const planPath = args.get('plan') || die('--plan <plan.json> required');
  const out = args.get('out') || die('--out <dir> required');
  const duration = parseInt(args.get('duration') || '60', 10);
  const only = args.get('only');
  const task = args.get('task') || 'text2music';
  const coverStrength = parseFloat(args.get('cover-strength') || '1.0');
  const noiseStrength = parseFloat(args.get('noise-strength') || '0');
  const cfgTags = (args.get('configs') || 'P50,T8').split(',').map(s => s.trim()).filter(Boolean);
  for (const t of cfgTags) if (!CONFIGS[t]) die(`unknown config ${t}`);
  if (task !== 'text2music' && !COVER_TASKS.has(task)) die(`unknown task ${task}`);
  const isCover = COVER_TASKS.has(task);
  const plan = JSON.parse(fs.readFileSync(planPath, 'utf8')) as Plan;
  fs.mkdirSync(out, { recursive: true });

  const health = await fetch(`http://127.0.0.1:${config.aceServer.port}/health`).then(r => r.ok).catch(() => false);
  if (!health) die('ace-server is not reachable');

  const manifestPath = path.join(out, 'manifest.json');
  const manifest: any = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, 'utf8')) : { duration, renders: [] };

  for (const d of plan.datasets) {
    if (only && d.slug !== only) continue;
    const dir = path.join(out, d.slug);
    fs.mkdirSync(dir, { recursive: true });
    const aceReq = baseRequest(d, duration);
    const trigger = d.trigger ?? d.slug;

    let synthBase: AceRequest;
    let srcBuf: Buffer | undefined;
    let tag = 'text2music';

    if (isCover) {
      if (!d.source || !fs.existsSync(d.source)) { console.error(`[${d.slug}] no source wav (${d.source}) — skipping`); continue; }
      srcBuf = fs.readFileSync(d.source);
      synthBase = { ...aceReq, task_type: task, audio_cover_strength: coverStrength, cover_noise_strength: noiseStrength };
      tag = `${task}-c${coverStrength}-n${noiseStrength}`;
      console.log(`[${d.slug}] ${task}: source ${path.basename(d.source)} (${(srcBuf.length / 1048576).toFixed(1)} MB), cover ${coverStrength}, noise ${noiseStrength}`);
    } else {
      // ---- LM once per dataset --------------------------------------------------
      const lmPath = path.join(dir, 'lm.json');
      let lmOut: AceRequest;
      if (fs.existsSync(lmPath)) {
        lmOut = JSON.parse(fs.readFileSync(lmPath, 'utf8'));
        console.log(`[${d.slug}] LM: reusing ${lmPath}`);
      } else {
        const t0 = Date.now();
        console.log(`[${d.slug}] LM: seed ${d.seed}, adapter ${d.lmAdapter ?? '(none)'} ...`);
        const lmJob = await aceClient.submitLm(aceReq);
        await awaitJob(lmJob, `LM (${d.slug})`);
        const res = await aceClient.getJobResult(lmJob);
        const arr = await res.json() as AceRequest[];
        if (!Array.isArray(arr) || !arr.length || !arr[0].audio_codes) die(`LM returned no codes for ${d.slug}`);
        lmOut = arr[0];
        fs.writeFileSync(lmPath, JSON.stringify(lmOut, null, 2));
        console.log(`[${d.slug}] LM done in ${((Date.now() - t0) / 1000).toFixed(1)}s: bpm=${lmOut.bpm} dur=${lmOut.duration} codes=${(lmOut.audio_codes || '').split(',').length}`);
      }
      // generate.ts: rebuild from the CURRENT request + LM fields, then re-inject the trigger.
      synthBase = {
        ...aceReq,
        audio_codes: lmOut.audio_codes,
        caption: withTrigger(lmOut.caption || aceReq.caption, trigger),
        lyrics: lmOut.lyrics,
        bpm: lmOut.bpm,
        duration: lmOut.duration,
        keyscale: lmOut.keyscale,
        timesignature: lmOut.timesignature,
        lm_seed: lmOut.lm_seed,
      };
    }

    for (const arm of d.arms) {
      for (const cfgTag of cfgTags) {
        const cfg = CONFIGS[cfgTag];
        const wavName = isCover ? `${cfgTag}_${tag}_${arm.tag}.wav` : `${cfgTag}_${arm.tag}.wav`;
        const wavPath = path.join(dir, wavName);
        if (fs.existsSync(wavPath)) { console.log(`[${d.slug}] skip ${wavName} (exists)`); continue; }
        const req: AceRequest = { ...synthBase, ...cfg };
        if (arm.adapter) {
          const scale = arm.scale ?? 1;
          req.adapter = arm.adapter;
          req.adapter_scale = scale;
          req.adapters = [{ name: arm.adapter, scale }];
          req.adapter_group_scales = GROUP_SCALES;
          req.adapter_mode = 'merge';
        }
        const t0 = Date.now();
        console.log(`[${d.slug}] synth ${cfgTag} ${tag} ${arm.tag} ...`);
        try {
          const jobId = srcBuf
            ? await aceClient.submitSynthMultipart(req, srcBuf, undefined, undefined, undefined, 'wav16')
            : await aceClient.submitSynth(req, 'wav16');
          await awaitJob(jobId, `synth ${d.slug} ${cfgTag} ${tag} ${arm.tag}`);
          const res = await aceClient.getJobResult(jobId);
          if (!res.ok) throw new Error(`result fetch failed (${res.status})`);
          fs.writeFileSync(wavPath, Buffer.from(await res.arrayBuffer()));
          const ms = Date.now() - t0;
          console.log(`[${d.slug}]   -> ${wavName} (${(ms / 1000).toFixed(1)}s)`);
          manifest.renders.push({ slug: d.slug, config: cfgTag, task: tag, arm: arm.tag, adapter: arm.adapter ?? null, source: d.source ?? null, wav: wavPath, ms, request: req });
        } catch (e) {
          console.error(`[${d.slug}]   FAILED ${cfgTag} ${tag} ${arm.tag}: ${(e as Error).message}`);
          manifest.renders.push({ slug: d.slug, config: cfgTag, task: tag, arm: arm.tag, adapter: arm.adapter ?? null, source: d.source ?? null, wav: null, error: (e as Error).message, request: req });
        }
        fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
      }
    }
  }
  console.log('done');
}

main().catch(e => die(e.stack || String(e)));
