/**
 * artist-token-ab.ts — render one prompt through the LM+DiT and write a wav.
 *
 * Deliberately renders ONE arm per invocation. The artist token is loaded by
 * ace-server from HOTSTEP_ARTIST_TOKEN at model load, so the two arms are two
 * ace-server processes, not two requests — which also means this cannot
 * accidentally compare a token against itself.
 *
 * It also re-runs the LM per arm, unlike dit-adapter-ab.ts which caches lm.json
 * across arms. That caching is right for a DiT adapter and WRONG here: an
 * as15_lm token changes the planner's output, so reusing one LM pass would
 * compare two identical codes streams and show nothing.
 *
 *   npx tsx server/scripts/artist-token-ab.ts --out <file.wav> --tag <label> \
 *     [--caption "..."] [--lyrics-file <path>] [--seed 12345] [--duration 60]
 *
 * Needs ace-server on config.aceServer.port. The Node server is NOT needed.
 */
import fs from 'fs';
import path from 'path';
import { aceClient, type AceRequest } from '../src/services/aceClient.js';
import { config } from '../src/config.js';

const SYNTH_MODEL = 'acestep-v15-merge-base-sft-turbo-xl-thirds-BF16.gguf';
const LM_MODEL    = 'acestep-5Hz-lm-4B-BF16.gguf';
const VAE_MODEL   = 'scragvae-BF16.gguf';
const EMB_MODEL   = 'Qwen3-Embedding-0.6B-BF16.gguf';
const P50 = {
  inference_steps: 50, guidance_scale: 20, shift: -1,
  infer_method: 'md_hamiltonian_v2', scheduler: 'linear_quadratic', guidance_mode: 'dynamic_cfg',
} as Partial<AceRequest>;

const JOB_DEADLINE_MS = 20 * 60 * 1000;

function die(m: string): never { console.error(m); process.exit(1); }

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

async function main() {
  const args = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);

  const out = args.get('out') || die('--out <file.wav> required');
  const tag = args.get('tag') || 'arm';
  const seed = Number(args.get('seed') || 20260903);
  const duration = Number(args.get('duration') || 60);
  const caption = args.get('caption')
    || 'a slick early-90s pop record, tight funk drums, slap bass, bright synth stabs, male lead vocal with layered harmonies';
  const lyrics = args.get('lyrics-file') && fs.existsSync(args.get('lyrics-file')!)
    ? fs.readFileSync(args.get('lyrics-file')!, 'utf8')
    : '[Verse]\nHold the line and let the rhythm take it\n\n[Chorus]\nWe can turn it up tonight\n';

  const health = await fetch(`http://127.0.0.1:${config.aceServer.port}/health`).then(r => r.ok).catch(() => false);
  if (!health) die('ace-server is not reachable');

  // --lm-adapter <dir>: a unified adapter (LoRA + token in one file) goes through
  // the normal request field, and the engine installs its token at load. This is
  // the production path; HOTSTEP_ARTIST_TOKEN is only the standalone override.
  const lmAdapter = args.get('lm-adapter');
  const lmScale = Number(args.get('lm-adapter-scale') || 1);
  const lmModel = args.get('lm-model') || LM_MODEL;  // a 0.6B-trained adapter needs the 0.6B

  const aceReq: AceRequest = {
    caption, lyrics, duration, seed,
    vocal_language: 'en',
    lm_batch_size: 1, lm_temperature: 0.85, lm_cfg_scale: 2, lm_top_p: 0.9, lm_top_k: 0,
    lm_negative_prompt: 'NO USER INPUT',
    use_cot_caption: true,
    task_type: 'text2music',
    synth_model: SYNTH_MODEL, lm_model: lmModel, vae_model: VAE_MODEL, emb_model: EMB_MODEL,
    ...P50,
    ...(lmAdapter ? { lm_adapter: lmAdapter, lm_adapter_scale: lmScale, lm_rep_penalty: 1.05 } : {}),
  };

  const t0 = Date.now();
  console.log(`[${tag}] LM ...`);
  const lmJob = await aceClient.submitLm(aceReq);
  await awaitJob(lmJob, `LM (${tag})`);
  const arr = await (await aceClient.getJobResult(lmJob)).json() as AceRequest[];
  if (!Array.isArray(arr) || !arr.length || !arr[0].audio_codes) die(`LM returned no codes for ${tag}`);
  const lmOut = arr[0];
  const nCodes = (lmOut.audio_codes || '').split(',').length;
  console.log(`[${tag}] LM done ${((Date.now() - t0) / 1000).toFixed(1)}s: bpm=${lmOut.bpm} dur=${lmOut.duration} codes=${nCodes}`);

  // generate.ts' rule: rebuild from the CURRENT request plus the LM's own
  // fields. Never whitelist — see the LM echo sideband gotcha.
  const synthReq: AceRequest = {
    ...aceReq,
    audio_codes: lmOut.audio_codes,
    caption: lmOut.caption || aceReq.caption,
    lyrics: lmOut.lyrics,
    bpm: lmOut.bpm,
    duration: lmOut.duration,
    keyscale: lmOut.keyscale,
    timesignature: lmOut.timesignature,
    lm_seed: lmOut.lm_seed,
  };

  console.log(`[${tag}] synth ...`);
  const sJob = await aceClient.submitSynth(synthReq, 'wav16');
  await awaitJob(sJob, `synth (${tag})`);
  const buf = Buffer.from(await (await aceClient.getJobResult(sJob)).arrayBuffer());
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, buf);
  console.log(`[${tag}] wrote ${out} (${(buf.length / 1048576).toFixed(1)} MB) in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  // The codes are the evidence that an as15_lm token did anything at all: if
  // two arms produce identical codes, the token never reached the planner and
  // any audible difference would be imagination.
  fs.writeFileSync(out.replace(/\.wav$/, '.codes.txt'), lmOut.audio_codes || '');
}

main().catch(e => die(String(e?.stack || e)));
