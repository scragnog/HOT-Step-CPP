/**
 * param-ab.ts — parameterization A/B batch, driven THROUGH the running app.
 *
 * Why through the app and not ace-train directly: the app's training routes
 * are the form's defaults (an empty option bag is the batch-pipeline recipe),
 * its job queue serialises the GPU, it stops and respawns its engine around
 * every run, and the Monitor tab shows progress. What this script adds is the
 * matrix, the renders and the listening-hub staging.
 *
 *   npx tsx server/scripts/param-ab.ts --phase lm      # form-default LM run + eval
 *   npx tsx server/scripts/param-ab.ts --phase dit     # 3 datasets x 8 arms, target 0.5
 *   npx tsx server/scripts/param-ab.ts --phase all
 *     [--datasets mj_dangerous,dio_holydiver] [--target 0.5] [--epochs 200] [--rank 128]
 *
 * Needs the Node server on :3001 and its engine on config.aceServer.port.
 * Re-runnable: arms whose wav already exists are skipped.
 */
import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import { aceClient, type AceRequest } from '../src/services/aceClient.js';
import { config } from '../src/config.js';

const API   = 'http://127.0.0.1:3001/api/training';
const ROOT  = path.resolve(process.cwd(), '..');  // run from server/
const STAMP = '2026-09-04';
const LST   = path.join(ROOT, '_experiments', '_LISTENING', `${STAMP}-dit-param-ab`);
const SYNTH_MODEL = 'acestep-v15-merge-base-sft-turbo-xl-thirds-BF16.gguf';
const LM_MODEL    = 'acestep-5Hz-lm-4B-BF16.gguf';
const VAE_MODEL   = 'scragvae-BF16.gguf';
const EMB_MODEL   = 'Qwen3-Embedding-0.6B-BF16.gguf';
const P50 = {
  inference_steps: 50, guidance_scale: 20, shift: -1,
  infer_method: 'md_hamiltonian_v2', scheduler: 'linear_quadratic', guidance_mode: 'dynamic_cfg',
} as Partial<AceRequest>;

type Ds = { id: string; caption: string; lyrics: string };
const DATASETS: Record<string, Ds> = {
  mj_dangerous: {
    id: '732bfdcd-fcb9-4d6a-b5bd-c6c21dac3286',
    caption: 'a slick early-90s pop record, tight funk drums, slap bass, bright synth stabs, male lead vocal with layered harmonies',
    lyrics: '[Verse]\nHold the line and let the rhythm take it\n\n[Chorus]\nWe can turn it up tonight\n',
  },
  dio_holydiver: {
    id: '8203ce8f-92c3-480f-a073-0bfe463c0a58',
    caption: 'classic early-80s heavy metal, galloping bass, twin harmony guitars, powerful male vocal, big arena drums',
    lyrics: '[Verse]\nRide the night on wheels of thunder\n\n[Chorus]\nHold on, the sky is falling down\n',
  },
  carpenterbrut_trilogy: {
    id: '3da921d9-45e0-4a8b-8dd5-c4ffe262bf33',
    caption: 'dark synthwave, pounding analog synth bass, gated retro drum machine, distorted lead synth, instrumental, cinematic',
    lyrics: '[Instrumental]\n',
  },
};

// LoRA-shaped arms at the shipped rank; rsLoRA at matched strength
// (alpha/sqrt(r) == 256/128 -> alpha 2*sqrt(128) ~ 23); HRA at r=8 because its
// step cost scales with the reflection count (10x a LoRA already at 8).
type Arm = { key: string; body: Record<string, unknown>; note: string };
function arms(rank: number): Arm[] {
  const alpha = rank * 2;
  const base = { adapterType: 'lora', rank, alpha };
  return [
    { key: 'lora',     body: { ...base },                                          note: 'control' },
    { key: 'dora',     body: { ...base, dora: true },                              note: 'learned per-row magnitude' },
    { key: 'pissa',    body: { ...base, pissa: true },                             note: 'principal-direction init, exported as rank-2r' },
    { key: 'loraplus', body: { ...base, loraPlusRatio: 16 },                       note: 'B at 16x A LR' },
    { key: 'rslora',   body: { ...base, alpha: Math.round(2 * Math.sqrt(rank)), rslora: true }, note: 'alpha/sqrt(r), strength matched to control' },
    { key: 'hira',     body: { ...base, hira: true },                              note: 'W (.) BA, merge-only' },
    { key: 'loha',     body: { ...base, loha: true },                              note: 'Hadamard of two pairs, merge-only' },
    { key: 'hra',      body: { adapterType: 'lora', rank: 8, alpha: 8, hra: true }, note: '8 Householder reflections, orthogonal' },
    // LyCORIS LoKr at the form's own defaults (dim 512, alpha 512, factor 6,
    // decompose both) — the other shipped parameterization, for coverage.
    { key: 'lokr',     body: { adapterType: 'lokr' },                             note: 'LoKr dim 512 / factor 6 (form defaults)' },
  ];
}

function log(m: string) { console.log(`[${new Date().toISOString().slice(11, 19)}] ${m}`); }
function args() {
  const a = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) a.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1] ?? '');
  return a;
}

async function post(p: string, body: unknown): Promise<string> {
  const r = await fetch(`${API}${p}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json() as { jobId?: string; error?: string };
  if (!r.ok || !j.jobId) throw new Error(`${p}: HTTP ${r.status} ${j.error ?? ''}`);
  return j.jobId;
}
async function waitJob(jobId: string, label: string): Promise<{ status: string; error: string | null; secs: number }> {
  const t0 = Date.now();
  let lastPhase = '';
  for (;;) {
    const j = await (await fetch(`${API}/jobs/${jobId}`)).json() as { status: string; phase: string; error: string | null };
    if (j.phase && j.phase !== lastPhase) { lastPhase = j.phase; log(`${label}: ${j.phase}`); }
    if (j.status === 'done' || j.status === 'failed' || j.status === 'cancelled') {
      return { status: j.status, error: j.error, secs: (Date.now() - t0) / 1000 };
    }
    await new Promise(r => setTimeout(r, 5000));
  }
}
async function waitEngine(): Promise<void> {
  for (let i = 0; i < 120; i++) {
    const ok = await fetch(`http://127.0.0.1:${config.aceServer.port}/health`).then(r => r.ok).catch(() => false);
    if (ok) return;
    await new Promise(r => setTimeout(r, 3000));
  }
  throw new Error('engine did not come back within 6 min');
}
function newestRunDir(prefixGlob: string, name: string): string | null {
  const root = config.aceServer.adapters;
  let best: { dir: string; mtime: number } | null = null;
  for (const top of fs.readdirSync(root)) {
    if (!top.startsWith(prefixGlob)) continue;
    const nameDir = path.join(root, top, name);
    if (!fs.existsSync(nameDir)) continue;
    for (const run of fs.readdirSync(nameDir)) {
      const cfg = path.join(nameDir, run, 'adapter_config.json');
      const alt = path.join(nameDir, run, 'lokr_weights.safetensors');
      if (!fs.existsSync(cfg) && !fs.existsSync(alt)) continue;
      const mtime = fs.statSync(path.join(nameDir, run)).mtimeMs;
      if (!best || mtime > best.mtime) best = { dir: path.join(nameDir, run), mtime };
    }
  }
  return best?.dir ?? null;
}

// Completion state of a DiT run from its log: complete when the epoch cap was
// reached or the target-loss stop fired; anything else is a partial run.
function runState(dir: string): { complete: boolean; epochsRun: number; reason: string } {
  try {
    const jl = JSON.parse(fs.readFileSync(path.join(dir, 'dit_train_log.json'), 'utf8')) as {
      epochs_run?: number; saved_reason?: string; epochs?: unknown[]; config?: { epochs?: number };
    };
    const run = jl.epochs_run ?? (jl.epochs?.length ?? 0);
    const cap = jl.config?.epochs ?? 0;
    const target = jl.saved_reason === 'target';
    const complete = target || (cap > 0 && run >= cap);
    return { complete, epochsRun: run, reason: target ? 'target reached' : complete ? 'epoch cap' : 'partial' };
  } catch {
    return { complete: false, epochsRun: 0, reason: 'no log' };
  }
}

// ── renders (same recipe as artist-token-ab.ts; the LM pass is shared per dataset) ──
const codesCache = new Map<string, AceRequest>();
async function awaitAce(jobId: string, what: string): Promise<void> {
  const deadline = Date.now() + 20 * 60 * 1000;
  for (;;) {
    if (Date.now() > deadline) { await aceClient.cancelJob(jobId).catch(() => {}); throw new Error(`${what} timed out`); }
    const st = await aceClient.pollJob(jobId);
    if (st.status === 'done') return;
    if (st.status === 'failed') throw new Error(`${what} failed — see the engine log`);
    if (st.status === 'cancelled') throw new Error(`${what} cancelled`);
    await new Promise(r => setTimeout(r, 500));
  }
}
type RenderOverride = { caption?: string; lyrics?: string; duration?: number; seed?: number; key?: string };
async function render(dsKey: string, adapterDir: string | null, out: string, ov: RenderOverride = {}): Promise<void> {
  const ds = DATASETS[dsKey];
  const cacheKey = ov.key ?? dsKey;
  const aceReq: AceRequest = {
    caption: ov.caption ?? ds.caption, lyrics: ov.lyrics ?? ds.lyrics, duration: ov.duration ?? 60, seed: ov.seed ?? 20260904,
    vocal_language: 'en',
    lm_batch_size: 1, lm_temperature: 0.85, lm_cfg_scale: 2, lm_top_p: 0.9, lm_top_k: 0,
    lm_negative_prompt: 'NO USER INPUT',
    use_cot_caption: true, task_type: 'text2music',
    synth_model: SYNTH_MODEL, lm_model: LM_MODEL, vae_model: VAE_MODEL, emb_model: EMB_MODEL,
    ...P50,
  };
  let lmOut = codesCache.get(cacheKey);
  if (!lmOut) {
    const lmJob = await aceClient.submitLm(aceReq);
    await awaitAce(lmJob, `LM (${cacheKey})`);
    const arr = await (await aceClient.getJobResult(lmJob)).json() as AceRequest[];
    if (!Array.isArray(arr) || !arr.length || !arr[0].audio_codes) throw new Error(`LM returned no codes for ${cacheKey}`);
    lmOut = arr[0];
    codesCache.set(cacheKey, lmOut);
    log(`LM plan (${cacheKey}): bpm=${lmOut.bpm} dur=${lmOut.duration} codes=${(lmOut.audio_codes || '').split(',').length}`);
  }
  const synthReq: AceRequest = {
    ...aceReq,
    audio_codes: lmOut.audio_codes, caption: lmOut.caption || aceReq.caption, lyrics: lmOut.lyrics,
    bpm: lmOut.bpm, duration: lmOut.duration, keyscale: lmOut.keyscale, timesignature: lmOut.timesignature,
    lm_seed: lmOut.lm_seed,
  };
  if (adapterDir) {
    synthReq.adapter = adapterDir;
    synthReq.adapter_scale = 1;
    synthReq.adapters = [{ name: adapterDir, scale: 1 }];
  }
  const sJob = await aceClient.submitSynth(synthReq, 'wav16');
  await awaitAce(sJob, `synth (${path.basename(out)})`);
  const buf = Buffer.from(await (await aceClient.getJobResult(sJob)).arrayBuffer());
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, buf);
}

// ── phase LM: the form default with the token on, evaluated ──
async function phaseLm(): Promise<void> {
  const name = 'mjd-formdefault';
  const exp = path.join(ROOT, '_experiments', 'artist-token', 'formdefault_4B');
  fs.mkdirSync(exp, { recursive: true });
  log(`LM: posting train-lm on mj_dangerous with an EMPTY option bag (adapterName ${name})`);
  const jobId = await post(`/datasets/${DATASETS.mj_dangerous.id}/train-lm`, { adapterName: name });
  const r = await waitJob(jobId, 'LM');
  log(`LM: ${r.status} in ${Math.round(r.secs)} s ${r.error ?? ''}`);
  if (r.status !== 'done') return;
  const dir = newestRunDir('lm-', name);
  if (!dir) { log('LM: no run dir found'); return; }
  fs.writeFileSync(path.join(exp, 'RUN_DIR.txt'), dir);
  await waitEngine();
  log(`LM: eval of ${dir}`);
  const gen = spawnSync('npx', ['tsx', 'scripts/lm-adapter-eval.ts', 'generate', '--dataset', 'mj_dangerous', '--adapter', dir,
    '--lm-model', LM_MODEL, '--seeds', '2', '--max-duration', '60', '--out', path.join(exp, 'lm-eval')],
    { cwd: path.join(ROOT, 'server'), shell: true, encoding: 'utf8' });
  fs.writeFileSync(path.join(exp, 'eval_generate.log'), (gen.stdout || '') + (gen.stderr || ''));
  const rep = spawnSync('npx', ['tsx', 'scripts/lm-adapter-eval.ts', 'report', '--run', path.join(exp, 'lm-eval')],
    { cwd: path.join(ROOT, 'server'), shell: true, encoding: 'utf8' });
  fs.writeFileSync(path.join(exp, 'eval_report.txt'), (rep.stdout || '') + (rep.stderr || ''));
  const verdict = (rep.stdout || '').split('\n').filter(l => /VERDICT|Unigram JS|Marginals|Transitions|OOV/.test(l)).join('\n');
  log(`LM eval:\n${verdict}`);
}

// ── phase DiT: the matrix ──
async function phaseDit(dsKeys: string[], target: number, epochs: number, rank: number): Promise<void> {
  fs.mkdirSync(LST, { recursive: true });
  const readme = path.join(LST, 'README.md');
  if (!fs.existsSync(readme)) {
    fs.writeFileSync(readme,
      `# DiT parameterization A/B — ${STAMP}\n\nAll arms: same dataset tensors, target loss ${target} (ma5), epoch cap ${epochs}, ` +
      `rank ${rank}/alpha ${rank * 2} unless noted, same seed/caption/lyrics, rendered through the app's engine at 60 s. ` +
      `00 = base model, no adapter. Trained through the app's train-dit route (form defaults for everything not listed).\n\n` +
      `| dataset | # | arm | epochs (cap ${epochs}) | final ma5 | train s | note |\n|---|---|---|---|---|---|---|\n`);
  }
  for (const dsKey of dsKeys) {
    const ds = DATASETS[dsKey];
    const outDir = path.join(LST, dsKey);
    fs.mkdirSync(outDir, { recursive: true });
    const baseWav = path.join(outDir, '00_base.wav');
    if (!fs.existsSync(baseWav)) {
      await waitEngine();
      log(`${dsKey}: base render`);
      await render(dsKey, null, baseWav);
    }
    let n = 0;
    for (const arm of arms(rank)) {
      n++;
      const wav = path.join(outDir, `${String(n).padStart(2, '0')}_${arm.key}.wav`);
      if (fs.existsSync(wav)) { log(`${dsKey}/${arm.key}: exists, skipping`); continue; }
      const name = `ab-${dsKey}-${arm.key}`;
      const body = { adapterName: name, targetLoss: target, epochs, ...arm.body };
      let row: string;
      try {
        // An adapter that already exists is either COMPLETE (a previous runner
        // instance trained it but never rendered: reuse, training is the
        // expensive half) or PARTIAL (a paused/cancelled run). The train-dit
        // route resumes from the latest same-named adapter when initAdapter is
        // omitted, so a partial arm is re-posted for its remaining epochs.
        let dir = newestRunDir('dit-', name);
        let secs = 0;
        let priorEpochs = 0;
        if (dir) {
          const st = runState(dir);
          if (st.complete) {
            log(`${dsKey}/${arm.key}: adapter complete at ${dir} (${st.epochsRun} epochs, ${st.reason}), rendering only`);
          } else {
            priorEpochs = st.epochsRun;
            log(`${dsKey}/${arm.key}: partial adapter at ${dir} (${st.epochsRun}/${epochs} epochs), resuming for ${Math.max(1, epochs - st.epochsRun)} more`);
            dir = null;
          }
        }
        if (!dir) {
          const remaining = Math.max(1, epochs - priorEpochs);
          const postBody = { ...body, epochs: remaining };
          log(`${dsKey}/${arm.key}: posting train-dit ${JSON.stringify(arm.body)} epochs=${remaining}`);
          const jobId = await post(`/datasets/${ds.id}/train-dit`, postBody);
          const r = await waitJob(jobId, `${dsKey}/${arm.key}`);
          secs = r.secs;
          if (r.status !== 'done') {
            row = `| ${dsKey} | ${n} | ${arm.key} | — | — | ${Math.round(r.secs)} | FAILED: ${r.error ?? r.status} |\n`;
            fs.appendFileSync(readme, row);
            continue;
          }
          dir = newestRunDir('dit-', name);
          if (!dir) throw new Error('no run dir');
        }
        let ep = '?', ma5 = '?';
        try {
          const jl = JSON.parse(fs.readFileSync(path.join(dir, 'dit_train_log.json'), 'utf8')) as { epochs?: { loss: number; ma5?: number }[] };
          const eps = jl.epochs ?? [];
          ep = priorEpochs > 0 ? `${priorEpochs}+${eps.length}` : String(eps.length);
          const last = eps[eps.length - 1];
          ma5 = last ? (last.ma5 ?? last.loss).toFixed(4) : '?';
        } catch { /* row keeps ? */ }
        await waitEngine();
        log(`${dsKey}/${arm.key}: render from ${dir}`);
        await render(dsKey, dir, wav);
        row = `| ${dsKey} | ${n} | ${arm.key} | ${ep} | ${ma5} | ${Math.round(secs)} | ${arm.note} |\n`;
      } catch (e: any) {
        row = `| ${dsKey} | ${n} | ${arm.key} | — | — | — | ERROR: ${String(e?.message ?? e).slice(0, 120)} |\n`;
      }
      fs.appendFileSync(readme, row);
      log(row.trim());
    }
  }
}

// ── phase render: full-length renders of every existing arm with one plan ──
// Same caption/lyrics/seed for all, only the adapter changes. Partial adapters
// (a stopped arm) render as they are; the README says which.
async function phaseRender(dsKey: string, sub: string, ov: RenderOverride, rank: number): Promise<void> {
  const ds = DATASETS[dsKey];
  if (!ds) throw new Error(`unknown dataset ${dsKey}`);
  const outDir = path.join(LST, dsKey, sub);
  fs.mkdirSync(outDir, { recursive: true });
  const readme = path.join(outDir, 'README.md');
  if (!fs.existsSync(readme)) {
    fs.writeFileSync(readme,
      `# ${dsKey} — full-length renders (${sub})\n\nOne plan for all arms (same caption, lyrics, seed ${ov.seed ?? 20260904}, ` +
      `duration ${ov.duration ?? '?'} s), thirds DiT, 50 steps, guidance 20, md_hamiltonian_v2 / linear_quadratic / dynamic_cfg, ` +
      `no post-processing. Only the adapter changes. 00 = base.\n\n| # | arm | adapter | state |\n|---|---|---|---|\n`);
  }
  await waitEngine();
  const baseWav = path.join(outDir, '00_base.wav');
  if (!fs.existsSync(baseWav)) {
    log(`${dsKey}/${sub}: base render`);
    await render(dsKey, null, baseWav, ov);
    fs.appendFileSync(readme, `| 0 | base | — | no adapter |\n`);
  }
  let n = 0;
  for (const arm of arms(rank)) {
    n++;
    const wav = path.join(outDir, `${String(n).padStart(2, '0')}_${arm.key}.wav`);
    if (fs.existsSync(wav)) { log(`${dsKey}/${sub}/${arm.key}: exists, skipping`); continue; }
    const dir = newestRunDir('dit-', `ab-${dsKey}-${arm.key}`);
    if (!dir) { log(`${dsKey}/${sub}/${arm.key}: no adapter, skipping`); fs.appendFileSync(readme, `| ${n} | ${arm.key} | — | not trained |\n`); continue; }
    const st = runState(dir);
    try {
      await waitEngine();
      log(`${dsKey}/${sub}/${arm.key}: render from ${dir} (${st.epochsRun} epochs, ${st.reason})`);
      await render(dsKey, dir, wav, ov);
      fs.appendFileSync(readme, `| ${n} | ${arm.key} | ${path.basename(path.dirname(dir))}/${path.basename(dir)} | ${st.epochsRun} epochs, ${st.reason} |\n`);
    } catch (e: any) {
      fs.appendFileSync(readme, `| ${n} | ${arm.key} | ${dir} | ERROR: ${String(e?.message ?? e).slice(0, 120)} |\n`);
    }
  }
  log(`${dsKey}/${sub}: renders done`);
}

// ── phase blind: N songs x (base + chosen arms), letters shuffled per song ──
// Files are song<k>_<letter>.wav; the letter->arm mapping is written ONLY to
// _key/KEY.json (do not open before rating). RATE.md is the scoring sheet.
function shuffled<T>(xs: T[], seed: number): T[] {
  const a = xs.slice();
  let x = seed >>> 0;
  for (let i = a.length - 1; i > 0; i--) {
    x = (x * 1664525 + 1013904223) >>> 0;
    const j = x % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
async function phaseBlind(dsKey: string, sub: string, armKeys: string[], songs: number, seed: number, rank: number): Promise<void> {
  const outDir = path.join(LST, dsKey, sub);
  const keyDir = path.join(outDir, '_key');
  fs.mkdirSync(keyDir, { recursive: true });
  const keyPath = path.join(keyDir, 'KEY.json');
  const key: Record<string, Record<string, string>> = fs.existsSync(keyPath) ? JSON.parse(fs.readFileSync(keyPath, 'utf8')) : {};
  const items = ['base', ...armKeys];
  const letters = 'ABCDEFGHIJ'.slice(0, items.length).split('');
  const rate = path.join(outDir, 'RATE.md');
  if (!fs.existsSync(rate)) {
    fs.writeFileSync(rate, `# Blind rating — ${dsKey}\n\n${items.length} items per song (one is the base model, the rest are adapters), letters shuffled per song. ` +
      `Same plan per song (seed ${seed}), thirds DiT, 50 steps, no post-processing. Score each 1-10 on quality / expressiveness / likeness, or just rank them. ` +
      `Open _key/KEY.json only when done.\n\n`);
  }
  for (let k = 1; k <= songs; k++) {
    const slug = `song${k}`;
    const lyricsFile = path.join(outDir, `${slug}.lyrics.txt`);
    const captionFile = path.join(outDir, `${slug}.caption.txt`);
    if (!fs.existsSync(lyricsFile) || !fs.existsSync(captionFile)) throw new Error(`${slug}: lyrics/caption files missing in ${outDir}`);
    const src = fs.existsSync(path.join(outDir, `${slug}.SOURCE.txt`)) ? fs.readFileSync(path.join(outDir, `${slug}.SOURCE.txt`), 'utf8').trim() : slug;
    const durMatch = src.match(/(\d+) s\b/);
    const ov: RenderOverride = {
      lyrics: fs.readFileSync(lyricsFile, 'utf8'), caption: fs.readFileSync(captionFile, 'utf8').trim(),
      duration: durMatch ? Number(durMatch[1]) : 240, seed, key: `${dsKey}:${sub}:${slug}`,
    };
    if (!key[slug]) {
      const order = shuffled(items, seed * 7919 + k * 104729);
      key[slug] = {};
      letters.forEach((L, i) => { key[slug][L] = order[i]; });
      fs.writeFileSync(keyPath, JSON.stringify(key, null, 2));
    }
    if (!fs.readFileSync(rate, 'utf8').includes(`## ${slug}`)) {
      fs.appendFileSync(rate, `## ${slug} — ${src}\n\n| file | quality | expressiveness | likeness | notes |\n|---|---|---|---|---|\n` +
        letters.map(L => `| ${slug}_${L}.wav |  |  |  |  |`).join('\n') + '\n\n');
    }
    for (const L of letters) {
      const item = key[slug][L];
      const wav = path.join(outDir, `${slug}_${L}.wav`);
      if (fs.existsSync(wav)) { log(`${slug}_${L}: exists, skipping`); continue; }
      let dir: string | null = null;
      if (item !== 'base') {
        dir = newestRunDir('dit-', `ab-${dsKey}-${item}`);
        if (!dir) throw new Error(`${item}: no adapter`);
      }
      await waitEngine();
      log(`${slug}_${L}: render`);
      await render(dsKey, dir, wav, ov);
    }
    log(`${slug}: done`);
  }
  log(`${dsKey}/${sub}: blind set complete — key sealed in _key/KEY.json`);
}

async function main() {
  const a = args();
  const phase = a.get('phase') || 'all';  // lm | dit | all | render | blind
  const dsKeys = (a.get('datasets') || 'mj_dangerous,dio_holydiver,carpenterbrut_trilogy').split(',').map(s => s.trim()).filter(Boolean);
  const target = Number(a.get('target') || 0.5);
  const epochs = Number(a.get('epochs') || 200);
  const rank = Number(a.get('rank') || 128);
  for (const k of dsKeys) if (!DATASETS[k]) throw new Error(`unknown dataset ${k}`);
  const up = await fetch(`${API}/jobs`).then(r => r.ok).catch(() => false);
  if (!up) throw new Error('Node server not reachable on :3001 — start dev.bat');
  if (phase === 'lm' || phase === 'all') await phaseLm();
  if (phase === 'dit' || phase === 'all') await phaseDit(dsKeys, target, epochs, rank);
  if (phase === 'blind') {
    // --arms rslora,lora,hira,pissa,dora --songs 3 [--sub blind] [--seed n]
    const armKeys = (a.get('arms') || 'rslora,lora,hira,pissa,dora').split(',').map(x => x.trim()).filter(Boolean);
    await phaseBlind(dsKeys[0], a.get('sub') || 'blind', armKeys, Number(a.get('songs') || 3), Number(a.get('seed') || 20260904), rank);
  }
  if (phase === 'render') {
    // --lyrics-file <path> --caption-file <path> [--duration s] [--seed n] [--sub full]
    const lyricsFile = a.get('lyrics-file'); const captionFile = a.get('caption-file');
    if (!lyricsFile || !captionFile) throw new Error('--phase render needs --lyrics-file and --caption-file');
    const ov: RenderOverride = {
      lyrics: fs.readFileSync(lyricsFile, 'utf8'), caption: fs.readFileSync(captionFile, 'utf8').trim(),
      duration: Number(a.get('duration') || 240), seed: Number(a.get('seed') || 20260904), key: `${dsKeys[0]}:${a.get('sub') || 'full'}`,
    };
    await phaseRender(dsKeys[0], a.get('sub') || 'full', ov, rank);
  }
  log('done');
}

main().catch(e => { console.error(String(e?.stack || e)); process.exit(1); });
