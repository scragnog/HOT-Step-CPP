# Project Map — Deep Reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Companion to [SKILL.md](SKILL.md). All paths repo-relative to the HOT-Step CPP root. Line numbers verified against `master` at `dda5f6b6` on 2026-10-23; they drift over time, so treat them as anchors and re-grep if a ref misses.

## 1. Engine source map (`engine/src/`, headers unless noted)

| Subsystem | Files |
|---|---|
| HTTP server + job queue | `engine/tools/hot-step-server.cpp` (the real ace-server binary); embedded webui built from `engine/tools/webui` + `engine/tools/public/index.html.gz` via `xxd.cmake` |
| LM phase (Qwen3 → audio codes) | pipeline-lm.cpp/.h, qwen3-lm.h, qwen3-enc.h, bpe.h, prompt.h, metadata-fsm.h, sampling.h, lm-cfg-extras.h (opt-in CFG refinements on LM logits) |
| LM runtime adapters | lm-adapter.h (planner-LM runtime LoRA), qwen3-lora.h, lm-prefix-runtime.h (trained prefix), artist-token-runtime.h (textual-inversion token), pissa-residual.h |
| DiT synth phase | pipeline-synth.cpp/.h, **pipeline-synth-ops.cpp** (fork hook at line 9: `#include "hot-step-sampler.h"`), pipeline-synth-ops.h, pipeline-synth-impl.h, dit.h (fork hooks at lines 11/14: adapter-merge.h + adapter-runtime.h), dit-graph.h, dit-sampler.h (upstream, **bypassed**), denoiser.h |
| HOT-Step sampler layer | **hot-step-sampler.h** (routes ALL solver/scheduler/guidance selection), hot-step-params.h (server→sampler sideband param channel `g_hotstep_params`, hooked into model-store.h:52), sampler-schedule.h, sampler-repaint.h, sampler-dcw.h, dcw.h, dwt-haar.h |
| Native solver/scheduler/guidance registries (LEGACY) | `engine/src/solvers/` (solver-interface.h, solver-registry.h and one header per solver), `engine/src/schedulers/`, `engine/src/guidance/`; new work goes in Lua plugins instead |
| Lua plugin system | lua-plugin.h, lua-plugin-registry.h (`init()` at :39 scans `<engine_dir>/plugins/{solvers,schedulers,guidance,postprocess}`, then `<project_dir>/plugins/...`, then the `--user-plugins` folder. A duplicate name from the project overlay is skipped with a warning, first one wins; only a user-folder plugin overrides a shipped one. `scan_dir` (:172) skips companion modules: stems containing `_constants`, `_math` or `_data`, and `*_core` files that have a matching plugin) |
| Adapters (LoRA/LoKr) | adapter-merge.h, adapter-runtime.h, adapter-cancel.h, adapter-trt.h, lokr-apply.h, lokr-common.h, dora-common.h, convrot.h, weight-source.h, weight-ctx.h |
| Concept steering | concept-steer.h (CAA activation steering), concept-extract.h |
| VAE | vae.h, vae-enc.h (GGML only; the ONNX Runtime VAE/encoder headers were deleted in `a44753f1`) |
| Model loading/eviction | model-store.cpp/.h (EVICT_STRICT default vs EVICT_NEVER under `--keep-loaded`), model-registry.h, gguf-weights.h, safetensors.h, hot-step-families.h (registry of the extra families ace-server hosts) |
| Conditioning / tokenizers | cond-enc.h, fsq-tok.h, fsq-detok.h, fsq-quant.h |
| Stem separation (GGML) | supersep.cpp/.h, supersep-stft.h, bs-roformer-ggml.h, mdx23c-ggml.h |
| Understand / analysis | pipeline-understand.cpp/.h |
| Post-DSP | mastering.h, spectral-lifter.h, silence-latent.h |
| Lyrics timing (LRC) | lrc-alignment.h, dit-alignment-graph.h, alignment-config.h |
| Request parsing | request.cpp/.h, config-json.h, task-types.h |
| TensorRT paths | dit-trt.h, lm-trt.h, lm-trtllm.h, hot-step-sampler-trt.h, adapter-trt.h, trt-runtime-probe.h |
| Stable Audio 3 | sa3-dit-ggml.h, sa3-same-ggml.h (autoencoder), sa3-t5gemma-enc.h (text conditioner), sa3-refine.h (refine and noise-only synth), sa3-families.h, sa3-preview-resolver.h |
| MiniMax-Music3 | `engine/src/minimax/` (mm3-pipeline.h, mm3-server.h for `/mm3/*`, mm3-job.h, LM/DiT/vocoder graphs, adapters) |
| YuE2 | `engine/src/yue2/` (yue2-pipeline.h, yue2-server.h for `/yue2/*`, LM/NAR/VAE graphs, adapters, MMS_FA aligner `yue2-mmsfa.h`, SheetSage lead-sheet model `sheetsage-*.h`) |
| MOSS audio encoder | `engine/src/moss/` (used by `ace-caption`) |
| Training | `engine/src/train/` (ACE-Step DiT and LM, MM3, SA3, YuE2 trainers), driven by `engine/tools/ace-train.cpp` |
| Misc infra | backend.h, audio-io.h, audio-resample.h, wav.h, philox.h, timer.h, debug.h, gpu.h, graph-arena.h, static-graph.h, hot-step-fsutf8.h, hot-step-build-flags.h |
| Upstream-sync backup | `engine/src/_backup_pre_sync/`: snapshot copies, not compiled |

**Other binaries** (`engine/tools/`, see `engine/CMakeLists.txt`): ace-lm.cpp, ace-synth.cpp, ace-understand.cpp, ace-train.cpp, ace-caption.cpp, ace-midi.cpp, neural-codec.cpp, mp3-codec.cpp, quantize.cpp, vst-host.cpp, mastering.cpp, plus synth-batch-runner.h and many `*-test.cpp` probes. `ace-server.cpp` = uncompiled upstream reference (`engine/CMakeLists.txt:482-484`).

## 2. Engine HTTP API (hot-step-server.cpp, registration lines)

| Endpoint | Line | Purpose |
|---|---|---|
| `POST /lm` | 3438 | LM phase; returns job id; result is an array of enriched AceRequests with `audio_codes` |
| `POST /synth` | 3439 | DiT + VAE synth (multipart variant carries source/ref/seed latents & audio) |
| `POST /understand` | 3440 | Audio analysis (pipeline-understand) |
| `POST /vae` | 3441 | Raw VAE encode (multipart audio in, f32 latent out) |
| `POST /codes-decode` | 3442 | 5 Hz LM codes straight to audio (FSQ detokenizer + VAE, no DiT) |
| `POST /warm` | 3443 | Preload DiT+VAE+adapter (used by warm-on-startup, `index.ts:424-466`) |
| `GET /health` | 3444 | Liveness (unanswerable during compute) |
| `GET /worker-drain` | 3451 | Answers once the worker has finished what was queued before it (used after a dropped `/sa3-*` call) |
| `GET /props` | 3454 | Model/adapter lists + CLI defaults (feeds `/api/models`) |
| `GET /logs` | 3455 | Engine log tail |
| `GET /jobs` | 3456 | Job list |
| `GET /plugins` | 3458 | Lua plugin registry dump (feeds `/api/plugins`) |
| `GET /vram` | 3481 | Device memory info (feeds `/api/logs/vram`) |
| `GET /models/loaded` | 3503 | Resident model list |
| `POST /models/unload` | 3521 | Evict one model label |
| `POST /models/restore-policy` | 3550 | Evict everything and return to EVICT_STRICT after a `?keep_loaded=1` latch |
| `GET /job?id=` | 3573 | Poll job status (`status`, `phase` e.g. `adapter_precompute`, `phase_step/total`); `&result=1` returns the result (300 s timeout in aceClient), `&latent=1` the post-DiT latent |
| `POST /job?id=` | 3659 | `&cancel=1` cancels; `&drop_songs=MASK` drops songs from a running YuE2 batch |
| `POST /pp-vae-reencode` | 3695 | Post-processing VAE re-encode (synchronous) |
| `POST /sa3-refine` | 3925 | Stable Audio 3 SDEdit refine (synchronous, worker thread) |
| `GET /sa3/props` | 4359 | SA3 readiness probe (file checks only) |
| `POST /sa3-synth` | 4413 | Stable Audio 3 text-to-audio from noise (synchronous, worker thread) |
| `POST /supersep/separate` | 4755 | Stem separation job |
| `GET /supersep/progress` | 4892 | Separation progress |
| `GET /supersep/result` | 4934 | Separation result |
| `GET /supersep/serve` | 4978 | Serve stem file |
| `POST /supersep/release` | 5021 | Release a separation job |
| `POST /supersep/recombine` | 5038 | Recombine stems |
| `POST /spectral-lifter` | 5135 | Spectral lifter DSP |
| `GET /` | 5176 | Embedded webui (gzip HTML) |

Family routes, registered from the same file (`hot-step-server.cpp:3466-3478`): `mm3_register_routes` (`engine/src/minimax/mm3-server.h`) adds `GET /mm3/props` and `POST /mm3/{warm,unload,select-model,voc-decode,dit-forward,flow-sample,depth-frame,cond-encode,lm-plan,imatrix,tokenize-check,synth-e2e}`; `mm3_register_job_routes` (`mm3-job.h`) adds `POST /mm3/synth` and `GET /mm3/{take,job,stream}`; `yue2_register_routes` (`engine/src/yue2/yue2-server.h`) adds `GET /yue2/props` and `POST /yue2/{warm,unload,select-model,tokenize-check,synth,imatrix,align}`.

## 3. AceRequest field catalogue (aceClient.ts:84-233)

Grouped; see the interface for exact optionality and comments.

- **Core**: `caption, lyrics, bpm, duration, keyscale, timesignature, vocal_language, seed`; `lm_seed` (independent LM seed); seed breeding `seed_b, seed_blend, seed_blend_end`
- **LM sampling**: `lm_batch_size, lm_temperature, lm_cfg_scale, lm_cfg_cutoff_ratio, lm_top_p, lm_top_k, lm_min_p, lm_negative_prompt, use_cot_caption, audio_codes` (LM output); repetition control `lm_rep_penalty, lm_rep_window, lm_rep_mode, lm_dry_base, lm_dry_min_len`; CFG refinements `lm_cfg_min_p, lm_cfg_logprob, lm_cfg_rescale`
- **DiT**: `inference_steps, guidance_scale, shift, infer_method` (solver), `scheduler, guidance_mode, batch_cfg, synth_batch_size, peak_clip, negative_prompt, dit_sliding_window`
- **Task routing**: `task_type, track, repainting_start/end, audio_cover_strength, cover_noise_strength/method, seed_strength, evict_lm`
- **Model routing**: `synth_model, lm_model, vae_model, emb_model`; planner-LM runtime LoRA `lm_adapter, lm_adapter_scale`
- **Adapters**: `adapter, adapter_scale` (single); `adapters: [{name, scale, gain_curve?, gain_domain?}]` (multi-stack, supersedes single); `adapter_sections: [{weights, size}]` + `adapter_section_align_at` + `adapter_section_isolation` (per-section masking; isolation is parsed by the engine but nothing reads it); `adapter_group_scales` (self_attn/cross_attn/mlp/cond_embed/time_embed/proj_in); `adapter_mode: "merge"|"runtime"|"runtime_lowrank"`, `adapter_runtime_quant: "bf16"|"q8_0"|"q4_k"`, `adapter_runtime_dora`, `adapter_merge_lowvram`; `rebase_source, rebase_beta` (basin re-base, UNVALIDATED)
- **Solver/guidance sub-params**: `stork_substeps, beat_stability, frequency_damping, temporal_smoothing, apg_momentum, apg_norm_threshold`
- **DCW** (Differential Correction in Wavelet domain): `dcw_enabled, dcw_mode, dcw_scaler, dcw_high_scaler`
- **Latent post**: `latent_shift, latent_rescale, custom_timesteps, cfg_cutoff_ratio, cache_ratio`; Latent Spectral Suppressor `lss_strength, lss_var_thresh, lss_dc_remove`
- **Post-VAE denoise**: `denoise_strength, denoise_smoothing, denoise_mix`; `pp_vae_reencode`
- **Misc**: `get_lrc` (synced lyrics), `vae_chunk`
- **Plugins**: `plugin_params: Record<string, string|number|boolean>`, `postprocess_plugin`

The streaming-preview fields (`stream_mode`, `stream_depth`, `stream_chunk_dir`) and `use_ort_vae` are gone: the engine removed ONNX Runtime and the streaming pipeline in `a44753f1`, and `/synth` answers `stream_mode: true` with 400.

Fields the C++ struct does not know about (the **sideband**) are dropped by the `/lm` echo; see SKILL.md and `services/backends/ace/generate.ts:192-213`.

## 4. Server internals worth knowing

- **Route mounts**: `index.ts:98-151`. The training-worker token gate and update gate run on every `/api` request, and `/api/workers` mounts before the body parsers so proxied bodies stream through (`index.ts:98-102`). Static: `/audio` (154), `/listening` (166), `/references` (171), `ui/dist` + SPA fallback (182-206).
- **Engine process** (`services/aceEngineProcess.ts`): `startAceServer()` :149; args :161-202; env rebuilt every spawn by `buildGpuEnv()` (GPU selection, including unsetting an empty `CUDA_VISIBLE_DEVICES`), TensorRT/TRT-LLM lib dirs prepended to PATH with case-insensitive Windows PATH-key handling :209-242; stdout/stderr piped to console + `ace_engine.log` + `pushLog` with GGML-noise filtering :247-274; crash limiter :295-339 (3 crashes / 30 s window); `stopAceServer()` :366 (tree kill on Windows) and `restartAceServer()` :448 for training jobs that borrow the GPU.
- **Portable CUDA bootstrap**: `index.ts:218-384`. `.cuda-version` marker next to the exe selects CUDA 12 vs 13 DLL set; a `.variant` marker skips the download for Vulkan/CPU builds; downloads via `modelDownloadService`. A failed download → CPU-only start.
- **Warm-on-startup**: gated on `warmOnStartup && keepLoaded && warmDit` (`index.ts:429-433`, settings in `config.ts:247-263`); polls `/health` then POSTs `/warm`.
- **generate.ts anatomy** (~1060 lines): `jobs` Map :60 (TTL 1 h sweep :85-98, ledger in `services/generation/jobLedger.ts`); `runGeneration` :156; `enqueueGeneration` (GPU lane + retry from the envelope's policy) :191; `POST /yue2/plan` :281; `submitGeneration` :344 (shared with the audio queue); `POST /` :476; `GET /status/:id` :483; `GET /mm3/stream/:id` :541; cancel :612-631; `/cancel-all` :633; `/queue` :651; `/reset-queue` :675; STORM `/storm/stream` :785, `/storm/stop` :986, `/storm/control` :994/1048.
- **ACE backend** (`services/backends/ace/generate.ts`): `translateParams` call :51; seed write-back :58-65; skip-LM decision :70-89; LM cache hit :96-118; LM submit :170; LM-echo rebuild :192-213; source audio/latent/seed-latent load :398-428; deferred parallel tasks (whisper, cover art) :543+; multipart submit :815-827; auto-trim :1114; post-processing call :1175; duration from the output WAV :1214-1225; `INSERT INTO songs` :1247-1259.
- **Other backends**: `services/backends/{minimax,yue2,sa3}/generate.ts`, each with its own song insert. The registry and `active_backend_id` setting live in `services/backends/registry.ts`.
- **aceClient.ts**: `BASE = config.aceServer.url` :13; timeouts :18-21 (15 s / 30 s / 300 s / 30 min SA3); `aceFetch` retries refused connections :342; `AceProps` :66; `AceRequest` :84; `AceJobStatus` :236; `AceJobPhase` :258; `aceClient` object :424.
- **Post-processing** (`services/generation/postProcessing.ts`): `runPostProcessingChain` :359; stage order QE(unmastered) :413 → shared vocal separation :441 → StableStep SA3 refine :528 → PP-VAE :776 → Spectral Lifter :792 → Vocal Naturalizer :813 → pre-VST gain offset :840 → VST chain :905 → Mastering :921 → final LUFS normalizer :939 → final peak guard :955 → QE(mastered) :982.
- **logs.ts**: `MAX_LINES` 2000 :22, `pushLog` :35, `subscribeLines` :50 (in-memory ring feeding the terminal panel), `GET /` :57, `GET /vram` :105, `GET /models-loaded` :125, `POST /models-unload` :138.
- **stemStudio.ts**: two modes documented in the header :1-8, `extract` (generative, sequential DiT `/synth` per track) and `supersep` (the header still says ONNX; the engine side is GGML now). Both persist to `<data dir>/stems/<jobId>/`.

## 5. Config / env (`server/src/config.ts`)

- Runtime layout comes from `server/src/dataroot/bootstrap.ts` (`RUNTIME`), imported first by config.ts. Release mode (`HOT_STEP_ROOT` set by the launcher) resolves app files from the distribution root and user data from the chosen data folder; a dev checkout uses the checkout, plus a data folder when Settings gave it one (`config.ts:12-23`).
- `.env` path: `RUNTIME.envPath` (`<data root>/settings/.env`) or the repo root (`config.ts:40`); bootstrapped from `.env.example` on first launch (`config.ts:95-109`).
- ace-server exe auto-located: `engine/ace-server` (portable) → `engine/build/Release/` (VS) → `engine/build/` (Ninja) → `engine/build/Debug/` (`config.ts:133-139`). Override: `ACESTEPCPP_EXE`.
- Defaults: models and adapters from `RUNTIME` (data folder) or `<checkout>/models` and `<checkout>/adapters` (`config.ts:140-141`); port 8085, host 127.0.0.1, vaeChunk 1024, vaeOverlap 64 (`config.ts:217-220`). Node server: port 3001, host 0.0.0.0 (`config.ts:312-315`).
- ffmpeg: portable `server/ffmpeg(.exe)` else the `ffmpeg-static` npm package (`config.ts:157-192`). Essentia: `Essentia/essentia_streaming_extractor_music` or `ESSENTIA_BIN` (`config.ts:294-296`). Whisper: `tools/whisper/whisper-cli` (or the data folder's whisper dir) + `<models>/whisper/` (`config.ts:418-421`). vst-host lives next to ace-server (`config.ts:379-386`).
- Lyric Studio LLM providers (`config.ts:331-376`): gemini (default), openai, anthropic, ollama, lmstudio, unsloth, llamacpp, openai-compat; impls in `server/src/services/lireek/llm/`.
- Settings UI whitelist `EXPOSED_ENV_KEYS` :442-470; `configReloadListeners` :480; restart-required set :483-488; hot-reload `reloadEnvConfig()` :494-636.

## 6. UI structure

- Component folders (`ui/src/components/`): assistant, compare-studio, cover-studio, create, details, global-bar, insta-gen, library, lyric-studio, midi-studio, model-manager, player, playlist, repaint-studio, settings, shared, sidebar, song-builder, stem-builder, stem-studio, storm, terminal, training-studio.
- HTTP layer (`ui/src/services/`): `httpClient.ts` (shared transport over the axios instance in `axiosClient.ts`, which injects the bearer and normalizes failures to `ApiError`), `api.ts` (domain clients such as `songApi`, `generateApi`), plus per-area clients (assistantApi, audioQueueApi, compareApi, coverWorkflowApi, dataFolderApi, exportImportApi, inspireApi, lireekApi, lyricWorkflowApi, midiStudioApi, preferencesApi, repaintLayerWorkflowApi, sa3PreviewApi, stemStudioApi, streamSessionsApi, studioDraftsApi, supersepApi, trainingApi and its siblings, updatesApi, workflowApi, yue2CoverApi).
- Server state: TanStack Query in `ui/src/queries/` (`client.ts`, `keys.ts`, `contract.ts`, one folder per area). See `docs/dev/ui-server-state.md`.
- Client stores (`ui/src/stores/`): Zustand `globalParamsStore.ts` (all generation params, the big one), `backendStore.ts`, `trainingStore.ts`, `vstChainStore.ts`, `coverStemsStore.ts`; hand-rolled `playbackStore.ts` (+playbackConverters.ts), `audioGenQueueStore.ts`, `streamingStore.ts`, `mm3StreamStore.ts`, `abCompareStore.ts`, `compareStore.ts`, `discoStore.ts`, `postProcessStore.ts`.
- View routing: `ui/src/App.tsx:103-150` maps URL paths ↔ view state; `ui/src/components/shared/overlays/CommandPalette.tsx` (`PALETTE_VIEWS`) and `ui/src/components/sidebar/Sidebar.tsx` list the studios for navigation.
- Dev proxy: `ui/vite.config.ts`, port 3000, host 0.0.0.0, proxies `/api`, `/audio`, `/references` → `http://127.0.0.1:3001`.

## 7. Status ledger (as of 2026-10-23)

- **DONE**: everything in the feature table; four backends (ACE-Step, MM3, YuE2, SA3) behind one registry; multi-adapter stacking (`adapters[]`, fixes in commit `168dcb5`); per-section adapter masking P1 (proportional map) + P2 (alignment-anchored token map); Lua plugin system (engine/plugins holds 21 solvers, 9 schedulers and 7 guidance modes, plus the `stork4_constants` and `beta_math` companion modules; the root `plugins/` overlay adds more and holds postprocess); safetensors + GGUF dual-format loading; GGML-only VAE and SuperSep.
- **REMOVED**: ONNX Runtime and the streaming preview pipeline (commit `a44753f1`).
- **DISABLED (deliberate)**: draft-LM speculative decoding (`config.ts:232-237`).
- **REVERTED**: regional self-attn isolation for per-section masking (commit `ee041e1`; broke musical continuity).
- **UNVALIDATED**: LoKr cross-base basin nudge β·(S−T) (`rebase_source`/`rebase_beta`, `aceClient.ts:196-197`).
- **Resolved on 2026-10-23**: root `plugins/` ships in release archives (`.github/workflows/release.yml` copies both `engine/plugins` and `plugins`).
- **Unverified** (check before relying): contents of `docs/plans/upstream-sync-workflow.md` (gitignored, may be absent).
