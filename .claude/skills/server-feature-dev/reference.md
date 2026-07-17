# server-feature-dev — reference

Deep detail supporting [SKILL.md](SKILL.md). All paths repo-relative; all line numbers verified against the code as of 2026-07.

## 1. aceClient method catalog (`server/src/services/aceClient.ts`)

`const BASE = config.aceServer.url` (aceClient.ts:12) — `http://127.0.0.1:8085` by default.

| Method | Engine endpoint | Timeout tier | Returns |
|---|---|---|---|
| `health()` | `GET /health` | QUICK | `{ status }` |
| `props()` | `GET /props` | QUICK | models/adapters/defaults (`AceProps`, :22-35) |
| `plugins()` | `GET /plugins` | QUICK | Lua plugin registry (solvers/schedulers/guidance/postprocess) |
| `warm(req, keepLoaded=true)` | `POST /warm[?keep_loaded=1]` | QUICK | job id — pre-loads DiT+VAE+adapter; no-op under strict eviction |
| `listJobs()` | `GET /jobs` | QUICK | every job in the engine's in-memory table (reconcile after disconnect) |
| `submitLm(req, mode?, keepLoaded?)` | `POST /lm[?keep_loaded=1]` | QUICK | job id; `mode: 'inspire' \| 'format'` sets `lm_mode` |
| `submitSynth(req\|req[], format='wav16', keepLoaded?)` | `POST /synth[?format=...]` | QUICK | job id; JSON body may be an array (batch) |
| `submitSynthMultipart(req, srcAudio?, refAudio?, srcLatents?, refLatents?, format?, keepLoaded?, seedLatents?)` | `POST /synth` multipart | RESULT | job id; **single JSON object only** in the `request` part (:374-378) |
| `submitUnderstand(audioBuffer)` | `POST /understand` multipart | RESULT | job id |
| `pollJob(id)` | `GET /job?id=N` | POLL | `AceJobStatus` (:154-161) |
| `getJobResult(id)` | `GET /job?id=N&result=1` | RESULT | raw `Response` (audio body) |
| `getJobLatent(id)` | `GET /job?id=N&latent=1` | RESULT | `Buffer \| null` (soft-fail) |
| `cancelJob(id)` | `POST /job?id=N&cancel=1` | POLL | void |
| `isReachable()` | via `health()` | QUICK | boolean (soft-fail) |
| `submitSpectralLifter(wav, params)` | `POST /spectral-lifter?...` | RESULT | processed WAV `Buffer` (synchronous endpoint) |
| `submitPpVaeReencode(wav, blend, useOnnx?)` | `POST /pp-vae-reencode?...` | RESULT | processed WAV `Buffer` (synchronous endpoint) |

Timeout tiers (aceClient.ts:17-19): `TIMEOUT_QUICK = 15_000`, `TIMEOUT_POLL = 30_000`, `TIMEOUT_RESULT = 300_000` ms, all via `AbortSignal.timeout`.

Engine job phases (`AceJobPhase`, aceClient.ts:165-180, mirrors `job_phase_str()` in the C++ `hot-step-server.cpp` — engine side not re-verified here): `queued, loading_text_enc, encoding_text, loading_cond_enc, encoding_cond, loading_dit, loading_adapter, adapter_precompute, dit_inference, loading_vae, vae_decode, encoding_output, done, failed, cancelled`. `adapter_precompute` is the "~17 s cold-start LoKr precompute" phase — a job sitting there is working, not stuck.

Multipart part names accepted by the engine: `request` (JSON), `audio` (source WAV), `ref_audio` (timbre reference WAV), `src_latents`, `ref_latents`, `seed_latents` (raw float32) — aceClient.ts:380-403. Boundary is hand-rolled: `'----HotStepBoundary' + Date.now()` (:362).

### AceRequest field groups (aceClient.ts:38-151)

- Core: `caption, lyrics, bpm, duration, keyscale, timesignature, vocal_language, seed`
- LM: `lm_batch_size, lm_temperature, lm_cfg_scale, lm_cfg_cutoff_ratio, lm_top_p, lm_top_k, lm_negative_prompt, use_cot_caption, audio_codes`
- Synth/sampler: `inference_steps, guidance_scale, shift, infer_method, scheduler, guidance_mode, custom_timesteps, cfg_cutoff_ratio, cache_ratio`
- Cover/repaint: `task_type, audio_cover_strength, cover_noise_strength, cover_noise_method, repainting_start, repainting_end, seed_strength`
- Model routing: `synth_model, lm_model, vae_model, emb_model`
- Adapters: `adapter, adapter_scale`, multi-adapter `adapters: {name, scale}[]` (supersedes single), per-section `adapter_sections`, `adapter_section_align_at`, `adapter_section_isolation` (engine feature reverted — inert), `adapter_group_scales`, `adapter_mode` ("merge"|"runtime"), `adapter_runtime_quant`, basin re-base `rebase_source`/`rebase_beta` (**stack-level fields, applied once per adapter stack — never per adapter; do not move them into the `adapters[]` entries**, issue #72)
- Plugins: `plugin_params: Record<string, string|number|boolean>`, `postprocess_plugin`
- Post: `latent_shift, latent_rescale, denoise_*`, `pp_vae_reencode`, `get_lrc`, `use_ort_vae`, streaming `stream_mode/stream_depth/stream_chunk_dir`

Adding a field: extend the interface here, then map it in `services/generation/translateParams.ts`. The engine ignores unknown fields, so a typo fails silently — verify end-to-end in `gen_*.log` (params are dumped by `logGenerationParams`).

## 2. Engine child-process lifecycle (`server/src/index.ts:146-316`)

- Spawn args built at index.ts:166-213: `--models`, `--host`, `--port` always; `--adapters` if the dir exists; `--keep-loaded` if `config.aceServer.keepLoaded`; `--noise-profile`, `--draft-lm`, `--vae-chunk`, `--vae-overlap`, `--onnx-dir` conditionally.
- Custom env only when TensorRT libs or `CUDA_VISIBLE_DEVICES` are configured (index.ts:228-258). The Windows `Path`-key gotcha and its fix live at :221-244.
- stdout/stderr piped to console (with GGML noise filtered: `CUDA graph warmup`, `CUDA Graph id`, `ggml_backend_cuda_graph_compute` — since 2026-07-17 these are also dropped at the engine source by `acestep_ggml_log` in engine/src/backend.h, which discards GGML DEBUG-level lines unless `HOTSTEP_GGML_DEBUG=1`; the Node filter is belt-and-braces), plus `logEngine()` → `ace_engine.log` and `pushLog(line, 'engine')` → UI SSE stream (index.ts:262-282).
- Crash respawn: non-clean exit → restart after 3 s, capped at 3 crashes per 30 s window; on cap, `setEngineReady(false, '...')` and give up (index.ts:284-309). Note `setEngineReady(ready, status)` takes **two** args (engineState.ts:13-16).
- Shutdown (index.ts:538-572): Windows uses `taskkill /PID <pid> /T /F` for a proper tree kill (:549); then `server.close()`, `closeDb()`, `closeLogger()`, forced `process.exit(0)` after 1 s. This clean shutdown path is why C++ rebuilds must go through `dev-rebuild.bat`.

## 3. Settings hot-reload internals

- `GET /api/settings/env` returns only whitelisted keys plus `restartKeys` (settings.ts:93-96).
- `POST /api/settings/env` (settings.ts:109-174; header comment says PUT — code wins): filters body to `EXPOSED_ENV_KEYS`, updates matching `.env` lines in place, appends new keys, preserves original EOL style (`\r\n` vs `\n`, :156), writes the file, calls `reloadEnvConfig()`, responds `{ updated, restartRequired }` where `restartRequired = changed.some(k => RESTART_REQUIRED_KEYS.has(k))`.
- `reloadEnvConfig()` (config.ts:300-395) re-parses `.env` with `dotenvParse` and hot-patches the live `config` object via per-key `apply(envKey, setter, getter)` calls, returning the changed keys and logging `[Config] Hot-reloaded N setting(s): ...`.
- Restart-required keys (config.ts:289-294): all `ACESTEPCPP_*` engine-spawn keys, `CUDA_VISIBLE_DEVICES`, `SERVER_PORT`, `DATA_DIR`. Their values still hot-patch the config object, but the running engine child was spawned with the old args.

## 4. DB idioms (copy-paste ready)

Checked column-add (the preferred idiom — database.ts:213-259):

```ts
const songsMigrations: Array<{ check: string; alter: string }> = [
  {
    check: `SELECT COUNT(*) as c FROM pragma_table_info('songs') WHERE name='my_col'`,
    alter: `ALTER TABLE songs ADD COLUMN my_col TEXT DEFAULT ''`,
  },
];
for (const m of songsMigrations) {
  const row = db.prepare(m.check).get() as any;
  if (row.c === 0) {
    db.exec(m.alter);
    console.log(`[DB] Migration: ${m.alter}`);
  }
}
```

Auth-scoped list with dynamic filter (songs.ts:18-42):

```ts
router.get('/', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
  const source = req.query.source as string | undefined;
  let query = 'SELECT * FROM songs WHERE user_id = ?';
  const params: any[] = [userId];
  if (source) {
    query += ` AND json_extract(generation_params, '$.source') = ?`;
    params.push(source);
  }
  query += ' ORDER BY created_at DESC';
  const songs = getDb().prepare(query).all(...params);
  const parsed = songs.map((s: any) => ({
    ...s,
    tags: JSON.parse(s.tags || '[]'),
    is_public: !!s.is_public,
  }));
  res.json({ songs: parsed });
});
```

Insert with UUID PK and JSON-in-TEXT (generate.ts:1150-1161): `uuidv4()` id, `JSON.stringify(job.params)` into `generation_params`.

Bulk cross-DB migration reference (`migrateLireekData`, database.ts:293-386): `ATTACH DATABASE ... AS lireek_old`, copy common columns via `INSERT OR IGNORE ... SELECT` (columns intersected via `PRAGMA table_info`), `DETACH`, rename the old file to `.migrated`; `foreign_keys = OFF` during, restored in `finally`.

## 5. Logger and SSE log stream

`services/logger.ts`:
- `initLogger()` (called at index.ts:49 **before any console output**) creates `logs/<YYYY-MM-DD_HH-MM-SS>/` with `node_console.log` + `ace_engine.log` streams and a `generations/` dir, then monkey-patches `process.stdout.write`/`process.stderr.write` to mirror everything into `node_console.log` (logger.ts:60-72).
- `logEngine(line)` appends to `ace_engine.log` (logger.ts:84-89).
- Per-generation: `startGenerationLog(jobId, taskType)` seeds an in-memory buffer; `logGeneration(jobId, level, msg)` appends (levels `INFO|DEBUG|WARNING|ERROR`) and also pushes `[Gen:<8-char-id>] msg` to the SSE stream; `logGenerationParams(jobId, obj)` pretty-prints JSON line-by-line; `finishGenerationLog(jobId, taskType)` / `failGenerationLog(jobId, error, taskType)` write `gen_<jobId>_<taskType>.log` and drop the buffer (logger.ts:95-178). Buffer is lost if the process dies before finish/fail.

`routes/logs.ts`:
- Ring buffer of `MAX_LINES = 2000` `LogLine` objects `{ id, ts, text, source: 'engine'|'server' }` (:14-24).
- `pushLog(text, source)` — suppresses engine noise patterns (:27-31), appends, notifies subscribers (:34-46).
- `subscribeLines(cb)` returns an unsubscribe function (:49-52). **Global pub/sub, no job tagging** — the reason the generation queue is serialized. generate.ts subscribes during a run to parse engine progress lines (including stream-preview markers) into `job.stage`/`job.progress`/`job.streamPreviews`.
- `GET /api/logs` — SSE: replays backlog (optionally `?after=<id>`), streams `data: {json}\n\n` events, `: keepalive` ping every 15 s, cleans up on close (:56-96).
- `GET /api/logs/vram` — proxies the engine's `GET /vram` with a 3 s timeout, soft-failing to zeros (:100-110).

## 6. generate.ts orchestration details beyond the SKILL summary

- LM-cache hit path (generate.ts:233-252): reconstructs each synth request as `{ ...aceReq, audio_codes, caption, lyrics, bpm, duration, keyscale, timesignature }` from the cached LM outputs — the concrete embodiment of the "LM echo sideband" rule. Cache entries are LRU-refreshed on hit.
- Skip-LM defaults (generate.ts:218-226): when LM is skipped for a non-cover task, missing metadata is defaulted (bpm 120, duration 120 s, `C major`, timesig `4`) because `/lm` cannot be called for metadata only (it always generates audio codes).
- Retry policy (generate.ts:1294-1339): `MAX_RETRIES = 1`; on retryable failure the job is reset to `pending`, the seed randomized (`Math.floor(Math.random() * 2_147_483_647)`) because a bad same-seed LM output may have caused the stall, 2 s pause, retry. Terminal failure calls `failGenerationLog`.
- Duration backfill (generate.ts:1139-1148): cover/repaint tasks skip the LM so `duration` is 0; it's measured from the output WAV via `wavDurationSec` before insert, since Song Builder clip points depend on it.
- Multipart-vs-JSON decision (generate.ts:814-827): multipart iff any of source audio, timbre-ref audio, source latents, ref latents, or seed latents buffers are present.
- Status values and their UI meaning: `pending` (queued), `lm_running`, `synth_running`, `saving`, then terminal `succeeded|failed|cancelled`. Progress is 0–100, set by log-subscription callbacks and phase milestones.

## 7. Server directory census (verified 2026-07)

- `server/src/routes/`: adapters, analyze, assistant, auth, coverArt, download, generate, health, inspire, lireek.ts (+ `lireek/` subfolder), logs, mastering, modelManager, models, plugins, seeds, settings, shutdown, songBuilder, songs, stemStudio, supersep, upload, vst.
- `server/src/services/`: aceClient, audioConvert, audioCrop, audioMetadata, autoTrim, disco-analyzer, latentFormat, logger, lyricsReconcile, modelDownloadService, pathMapper, spectralLifter, whisperTranscribe, plus subfolders `coverArt/`, `generation/` (adapterSections, audioQualityEvaluator, lmCache, lufsNormalize, postProcessing, sourceAudio, sourceLatentCache, translateParams, vocalNaturalizer), `lireek/`.
- Mount-path ↔ filename mismatch: only `/api/builder` → `songBuilder.ts`.

## 8. Known non-obvious states

- **Draft-LM speculative decoding: DISABLED** (config.ts:117-122). GGML per-call overhead (~10 ms) makes sequential 0.6B forwards nearly as costly as the 4B target. Re-enable via `ACESTEPCPP_DRAFT_LM`; the auto-detect block is commented out at config.ts:160-173. The spawn still passes `--draft-lm` if the config value is set and the file exists (index.ts:193-196).
- **`lireekDb.ts` shims**: `initLireekDb`/`closeLireekDb`/`getLireekDb` are deprecated no-ops/aliases (lireekDb.ts:12-23) kept for compile compat. Do not create new separate-DB patterns.
- **`adapter_section_isolation`** (aceClient.ts:95-97): field exists; the engine-side feature was reverted (commit ee041e1, "broke musical continuity"). Treat as inert unless the engine re-lands it.
- **Warm-on-startup** is gated on `keepLoaded && warmDit` — with keep-loaded off (the default) it never fires (config.ts:133-141).
- **`uncaughtException`/`unhandledRejection` log-and-continue** (index.ts:576-581); no global Express error middleware exists, so per-handler try/catch is mandatory for clean JSON errors.

## 9. Unverified notes (from the original research dossier)

- The C++ side of the engine HTTP API (`hot-step-server.cpp`) was not read while writing this skill; engine endpoint semantics come from aceClient.ts's own comments (which state they mirror `job_phase_str()` / `request_parse_json`). Verify against engine source before changing wire behavior.
- `docs/plans/` (gitignored, local-only) may contain design docs that supersede details here; it was not inspected.
- generate.ts lines ~260–1130 (LM submit details, post-processing chain, LUFS normalize, quality evaluator internals) were sampled, not exhaustively read; the orchestration skeleton and all cited line numbers were verified directly.
