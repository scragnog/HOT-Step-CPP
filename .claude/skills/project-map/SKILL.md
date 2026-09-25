---
name: project-map
description: Maps every HOT-Step CPP feature to its route file, service, UI folder, and engine subsystem, including port topology and the browser-to-engine request path. Use when you need to find where a feature or bug lives, which tier owns a symptom, or how a request flows through the system.
---

# Project Map & Navigation — HOT-Step CPP

HOT-Step CPP is a desktop app for local AI music generation (caption + lyrics in, stereo 48 kHz audio out). It is a heavily-extended fork of acestep.cpp, a C++/GGML port of ACE-Step 1.5. Three tiers:

1. **Engine** — C++17/CUDA/GGML in `engine/`. Pipeline: **LM** (Qwen3 language model that emits `audio_codes`) → **DiT** (Diffusion Transformer, the flow-matching synth stage) → **VAE** (decodes latents to audio).
2. **Server** — Node/TypeScript/Express/better-sqlite3 in `server/src/`. Orchestrates the engine, owns SQLite and files, serves the UI.
3. **UI** — React 19/Vite/Zustand/Tailwind in `ui/src/`. One component folder per "studio" (feature area).

## When to use this skill

- "Where is the code for X?" / "Which file handles Y?"
- Triaging a bug report: deciding whether the fault is UI, Node server, or C++ engine.
- Tracing a request end-to-end (browser → Express → ace-server → audio file).
- Onboarding to any unfamiliar feature before editing it.

## Golden rules (hard constraints — each prevents expensive damage)

1. **C++ changes: rebuild with `.\dev-rebuild.bat` at repo root, NEVER `engine\build.cmd` directly, under any circumstances.** WHY: you cannot reliably tell whether the app is running; the Node server auto-respawns `ace-server.exe` on crash (`server/src/index.ts:284-308`); killing the engine mid-run causes an infinite respawn + file-lock loop that blocks the build. `dev-rebuild.bat` POSTs `/api/shutdown`, waits for the process (force-kills at 10 s), then builds. It does NOT relaunch — restart with `.\dev.bat` or `.\LAUNCH.bat` yourself. Recompile immediately after editing any `engine/src/` or `engine/tools/` file.
2. **NEVER `cmake --build . --clean-first`** unless the GGML/CUDA layer itself changed. WHY: CUDA kernel recompilation takes 20+ minutes. For stale `.obj` issues delete only `engine\build\acestep-core.dir\` and `engine\build\Release\acestep-core.lib`.
3. **Engine HTTP/API changes go in `engine/tools/hot-step-server.cpp`, NOT `engine/tools/ace-server.cpp`.** WHY: upstream `ace-server.cpp` is kept as reference and is NOT compiled — the `ace-server` binary is built from `hot-step-server.cpp` (`engine/CMakeLists.txt:416-419`). Edits to `ace-server.cpp` silently do nothing.
4. **Don't `npm run build` during dev.** Type-check with `npx tsc --noEmit` (run in `server\` and `ui\`). Only build before user testing.
5. **Git: all work on `master`, no branches. Never `git add -A` or `git add -f`** (re-adds gitignored `.agents/`, `checkpoints/`, `node_modules/`). Stage explicit paths. **Push requires explicit user approval.** Commit locally often. **Any pushed `v*` tag triggers a full multi-platform CI release build** — use a `-CI-Test` suffix for throwaway tags.
6. **Three fork-hook files break on upstream sync** (see Failure signatures). After any sync run `powershell -File engine\verify-hooks.ps1`. The `pipeline-synth-ops.cpp` hook loss is SILENT — it compiles but all solvers/schedulers/guidance go dead.
7. **Don't visually verify UI with a browser agent — ask the human user.** Browser agent is fine for API-only checks.
8. **Node 18–22 LTS only.** Node 24+ breaks dependencies (`engines` field enforces `<24`).
9. **PowerShell is the shell: use `;` not `&&`** (unless pwsh 7 chaining). Windows 11 environment.

## Process / port topology

```
LAUNCH.bat (prod) or dev.bat (dev)
  └─ Node server  server/src/index.ts → Express :3001 (host 0.0.0.0, config.ts:186-187)
       ├─ serves prebuilt ui/dist/ with SPA fallback (index.ts:123-140)
       ├─ /api/* routes → SQLite server/data/hotstep.db
       ├─ /audio → server/data/audio/, /references → server/data/references/ (index.ts:98-121)
       └─ spawns child: ace-server.exe (C++ engine) on 127.0.0.1:8085 (config.ts:102-103)
Dev mode adds Vite on :3000 (HMR); vite.config.ts proxies /api, /audio, /references → 127.0.0.1:3001.
tsx watch auto-restarts the Node server on server-code changes.
```

- Engine spawn args built in `index.ts:166-213`: `--models --host --port`, plus conditional `--adapters`, `--keep-loaded`, `--noise-profile`, `--draft-lm` (deliberately disabled by default — `config.ts:117-122`), `--vae-chunk`, `--vae-overlap`, `--onnx-dir`.
- Crash respawn limiter (`index.ts:284-308`): abnormal exit → restart after 3 s; ≥3 crashes in 30 s → give up, engine marked not-ready ("check logs for missing DLLs").
- Portable-mode first launch downloads cuBLAS/cudart DLLs from HuggingFace before starting the engine (`index.ts:318` onward).
- Config comes from `.env` at repo root, auto-bootstrapped from `.env.example` (`config.ts:20-30`). Settings UI can only edit whitelisted `EXPOSED_ENV_KEYS` (`config.ts:265-286`); keys in `RESTART_REQUIRED_KEYS` (`config.ts:289-294`) never hot-apply; the rest hot-reload via `reloadEnvConfig()` (`config.ts:300`).

## Request path: browser → engine (main generation)

Key term: **`AceRequest`** — the JSON request object sent to the engine, typed at `server/src/services/aceClient.ts:38-151` (caption/lyrics/bpm/seed, LM sampling params, DiT params like `inference_steps`/`guidance_scale`/`scheduler`, model routing, adapter fields, `plugin_params`, streaming fields). **`aceReq`** in code = the instance of this object for the current job.

1. **UI**: `ui/src/components/create/CreatePanel.tsx` + `ui/src/stores/globalParamsStore.ts` (Zustand store holding all generation params) → `ui/src/services/api.ts` (thin fetch wrapper, base `/api`) → `POST /api/generate`.
2. **Orchestrator**: `server/src/routes/generate.ts` (~1550 lines). Jobs live in an in-memory `jobs` Map (`generate.ts:79`) with a serial queue. Inside `runGeneration` (`generate.ts:173`):
   - `translateParams` (`services/generation/translateParams.ts`) converts UI params → `AceRequest`.
   - **LM phase**: `aceClient.submitLm` (`generate.ts:303`); results cached by seed+params in `services/generation/lmCache.ts`.
   - **CRITICAL "sideband" gotcha** (`generate.ts:312-328`): server-only fields (`adapter_runtime_quant`, rebase, `plugin_params`, …) — the "sideband" = params the Node server adds that are not part of the C++ LM struct — do NOT survive the `/lm` round trip. Synth requests are rebuilt from the original `aceReq` plus only the LM-generated fields (`audio_codes`, caption, lyrics, bpm, duration, keyscale, timesignature). Never whitelist-echo the LM response.
   - Source audio/latent loading for cover/repaint/seed tasks (`generate.ts:449-478`, `services/generation/sourceAudio.ts`, `sourceLatentCache.ts`).
   - **Per-track synth**: `aceClient.submitSynth` or `submitSynthMultipart` when binary latents/audio are attached (`generate.ts:823-826`).
   - Optional Whisper lyric transcription (`generate.ts:940+`, `services/whisperTranscribe.ts`); auto-trim silence (`generate.ts:1057-1094`, `services/autoTrim.ts`).
   - **Post-processing chain** `runPostProcessingChain` (`services/generation/postProcessing.ts:70`), fixed order: quality-check(unmastered) → PP-VAE re-encode → Spectral Lifter → Vocal Naturalizer → gain offset → VST chain → Mastering → LUFS normalize → quality-check(mastered) (`postProcessing.ts:105-309`).
3. **Engine client**: `services/aceClient.ts` wraps every engine endpoint. **The engine is single-threaded httplib** — during DiT/VAE/adapter compute it cannot answer HTTP at all. Timeouts are deliberately generous: 15 s quick / 30 s poll / 300 s result fetch (`aceClient.ts:17-19`). A "hung" health check mid-generation is normal.
4. **Progress**: UI polls `GET /api/generate/status/:id`; live log lines via `routes/logs.ts` (`pushLog`/`subscribeLines`, `logs.ts:34,49`).
5. **Watchdogs** in `pollUntilDone` (`generate.ts:102-140`): no progress for 2 min → cancel + fail as stalled; wall-clock timeout clamped 5–120 min, default 45.
6. **Engine job model**: `POST /lm` or `/synth` returns a job id; poll `GET /job?id=`; fetch result via `POST /job`. Node job ids ≠ engine job ids (mapped by the `aceJobId` field, `generate.ts:47`).

**Engine HTTP endpoints** (`engine/tools/hot-step-server.cpp:2704-3535`): `POST /lm /synth /understand /vae /warm /job /models/unload /pp-vae-reencode /supersep/separate /supersep/recombine /spectral-lifter`; `GET /health /props /logs /jobs /job /plugins /vram /models/loaded /supersep/progress /supersep/result /supersep/serve /` (embedded webui).

## Feature → files map (route ⇄ service ⇄ UI ⇄ engine)

API mounts are all registered in `server/src/index.ts:72-95`. Route files in `server/src/routes/`, services in `server/src/services/`, UI folders in `ui/src/components/`.

| Feature / Studio | API mount | Route file | Key service(s) | UI folder / file | Engine piece |
|---|---|---|---|---|---|
| Create (main generation) | `/api/generate` | generate.ts | generation/*, aceClient, autoTrim, latentFormat, whisperTranscribe | create/ + global-bar/ | `/lm` `/synth` → pipeline-lm.cpp, pipeline-synth*.cpp |
| Insta-Gen (genre-first quick gen) | `/api/inspire` | inspire.ts | lireek/llmService (LLM lyrics) | insta-gen/ (`ui/src/services/inspireApi.ts`) | via generate queue |
| Lyric Studio ("Lireek") | `/api/lireek` | lireek.ts | lireek/{llmService, geniusService, profilerService, prompts, slopDetector, exportService, llm/} | lyric-studio/ (`ui/src/services/lireekApi.ts`) | none (external LLMs) |
| Song library | `/api/songs` | songs.ts | audioCrop, disco-analyzer, whisperTranscribe | library/, player/, details/ | — |
| Models dropdown | `/api/models` | models.ts | aceClient (`GET /props`) | global-bar/ModelsDropdown.tsx | model-registry.h, model-store.cpp |
| Model Manager (downloads) | `/api/model-manager` | modelManager.ts | modelDownloadService.ts + `server/src/data/model-registry.json` | model-manager/ | — |
| Adapters (LoRA/LoKr fine-tunes) | `/api/adapters` | adapters.ts | generation/adapterSections.ts | create/AdaptersAccordion.tsx, global-bar/AdaptersDropdown.tsx | adapter-merge.h, adapter-runtime.h, adapter-cancel.h |
| Stem Studio (separation) | `/api/stem-studio` | stemStudio.ts (2 modes: `extract` = DiT per-track `/synth`; `supersep` = ONNX NN, stemStudio.ts:1-8) | aceClient, audioConvert | stem-studio/ (`ui/src/services/stemStudioApi.ts`) | supersep.cpp, supersep-stft.h; `/supersep/*` |
| SuperSep raw proxy | `/api/supersep` | supersep.ts | — | shared/StemMixer.tsx (`ui/src/services/supersepApi.ts`) | `/supersep/*` |
| Stem Builder (layering) | uses stem/gen APIs | — | — | stem-builder/ | — |
| Repaint Studio | `/api/generate` (task_type) | generate.ts | generation/sourceAudio.ts | repaint-studio/ | sampler-repaint.h |
| Song Builder | `/api/builder` | songBuilder.ts | — | song-builder/ | seed-latent path, generate.ts:470-478 |
| Cover Studio (audio covers) | `/api/generate` + `/api/analyze` | generate.ts, analyze.ts | generation/sourceAudio.ts, audioMetadata, disco-analyzer | cover-studio/ | cover fields in request.h |
| Cover Art (images) | `/api/cover-art` | coverArt.ts | coverArt/{coverArtService, coverArtDownloader, promptBuilder} | library/CoverArtPromptModal.tsx, global-bar/CoverArtDropdown.tsx | external sd-cli (not the music engine) |
| Mastering | `/api/mastering` | mastering.ts | generation/postProcessing.ts | create/MasteringSection.tsx, global-bar/MasteringDropdown.tsx | mastering.h, tools/mastering.cpp |
| VST3 chain + monitor | `/api/vst` | vst.ts | spawns vst-host.exe (config.ts:234-250) | global-bar/VstChainDropdown.tsx, `ui/src/stores/vstChainStore.ts` | tools/vst-host.cpp |
| Lua plugins (solvers etc.) | `/api/plugins` | plugins.ts | aceClient (`GET /plugins`) | global-bar/PluginControls.tsx, `ui/src/hooks/usePluginRegistry.ts` | lua-plugin.h, lua-plugin-registry.h, hot-step-sampler.h |
| Assistant (in-app help chat) | `/api/assistant` | assistant.ts | `server/src/data/assistant-knowledge.md`, lireek/llmService | assistant/ (`ui/src/services/assistantApi.ts`) | — |
| Settings (.env editor) | `/api/settings` | settings.ts | config.ts `reloadEnvConfig` | settings/ | — |
| Seeds manager | `/api/seeds` | seeds.ts | — | global-bar/SeedManagerDrawer.tsx *(component name unverified — check global-bar/)* | — |
| Uploads (audio/latent/cover) | `/api/upload` | upload.ts | latentFormat.ts | shared/LatentImport.tsx *(unverified)* | — |
| Downloads / export | `/api/download` | download.ts | audioConvert.ts (ffmpeg) | player/, library/ | tools/mp3-codec.cpp |
| Logs / VRAM / terminal | `/api/logs` (`/`, `/vram`, `/models-loaded`, `/models-unload` — logs.ts:56-133) | logs.ts | logger.ts | terminal/, shared/VramIndicator.tsx *(unverified)* | `/vram`, `/models/loaded` |
| Health / lifecycle | `/api/health`, `/api/shutdown` | health.ts, shutdown.ts | engineState.ts (at `server/src/engineState.ts`, NOT services/) | — | `/health` |
| Auth (local single-user) | `/api/auth` | auth.ts | db/database.ts | — | — |
| Analyze (BPM/key of upload) | `/api/analyze` | analyze.ts | Essentia binary (config.ts:180-182), audioMetadata | cover-studio/ | `/understand` (pipeline-understand.cpp) |

**UI views** (`ui/src/App.tsx:99-114`): default create, plus `insta-gen`, `lyric-studio`, `cover-studio`, `stem-studio`, `stem-builder`, `song-builder`, `repaint`, `library`, `settings`; overlay panels for assistant and terminal. Zustand stores in `ui/src/stores/`: globalParamsStore, playbackStore, audioGenQueueStore, streamingStore, vstChainStore, abCompareStore, discoStore.

**Engine source map**: full per-subsystem table in [reference.md](reference.md). Fast rules: LM phase → `engine/src/pipeline-lm.cpp` / `qwen3-lm.h`; DiT synth → `pipeline-synth*.cpp` / `dit.h`; all solver/scheduler/guidance routing → `hot-step-sampler.h` (upstream `dit-sampler.h` is bypassed); adapters → `adapter-merge.h` / `adapter-runtime.h`; VAE → `vae.h` / `vae-ort.h`; HTTP server → `engine/tools/hot-step-server.cpp`.

**Plugins**: solvers (21) / schedulers (9) / guidance (7) are hot-loadable Lua files in `engine/plugins/{solvers,schedulers,guidance}/`; the registry also scans repo-root `plugins/` as a project overlay (`engine/src/lua-plugin-registry.h:35-50`), which holds `postprocess/` (md_audio_tiled). Drop a `.lua` in the right subdir → appears in UI next launch, no C++ rebuild. Adding a solver/scheduler/guidance = write a Lua plugin, never edit `dit-sampler.h`. Guide: `docs/dev/plugins-authoring.md`. Native headers in `engine/src/solvers|schedulers|guidance/` are legacy — new work goes in Lua.

## Data layer

- **The live data root is `server/data/`, NOT repo-root `data/`.** `DATA_DIR` in `.env` resolves relative to `server/` (`config.ts:192`: `path.resolve(__dirname, '..', DATA_DIR)`). A stale legacy `data/` directory exists at repo root with a plausible-looking but outdated `hotstep.db` — do not trust it, and do not delete it without asking.
- SQLite: `server/data/hotstep.db` (better-sqlite3). Tables (`server/src/db/database.ts:39-191`): users, songs, playlists, playlist_songs, artists, lyrics_sets, profiles, generations, settings, album_presets, audio_generations, builder_projects, builder_sections. **Lireek tables are unified into hotstep.db** (`index.ts:20`) — `config.lireek.dbPath` pointing at `lireek.db` (`config.ts:225-227`) is vestigial; don't trust it.
- Files (all under `server/data/`): `audio/` (outputs), `references/` (mastering refs), `stems/<jobId>/`, `vst/{states,chain.json}`, `lyrics/` (exports). **Never delete generated audio outputs or experiment artifacts on your own judgment — the user verifies results by ear; ask first.**
- Logs: `logs/YYYY-MM-DD_HH-MM-SS/{ace_engine.log, node_console.log, generations/gen_<uuid>_<task>.log}`. Newest folder = current session. Generation failure → matching `gen_*.log` first, then cross-ref the other two.

## Exact commands (PowerShell)

```powershell
.\dev.bat                       # Dev: Vite :3000 HMR + Node :3001 tsx watch
.\LAUNCH.bat                    # Prod
.\dev-rebuild.bat               # C++ rebuild: graceful shutdown -> wait -> build (then relaunch yourself)
npx tsc --noEmit -p server      # Type-check server, from repo root (do NOT npm run build in dev)
npx tsc --noEmit -p ui          # Type-check UI, from repo root
# Stale .obj without the 20-min CUDA recompile:
Remove-Item -Recurse -Force engine\build\acestep-core.dir; Remove-Item -Force engine\build\Release\acestep-core.lib
powershell -File engine\verify-hooks.ps1   # After any upstream sync
```

## Failure signatures (symptom → cause → fix)

| Symptom | Cause | Fix / first file |
|---|---|---|
| `[ace-server] Restarting in 3 seconds... (crash N/3)` then giving up | Missing DLL next to ace-server.exe or bad model | `logs/<newest>/ace_engine.log`; respawn logic `index.ts:284-308` |
| Solver/scheduler/guidance UI options silently do nothing | Upstream sync overwrote the `hot-step-sampler.h` include in `engine/src/pipeline-synth-ops.cpp:9`. **Compiles fine — silent.** | Run `engine\verify-hooks.ps1`; restore the include |
| Compile error in model-store.h / dit.h after sync | Lost `hot-step-params.h` include (`model-store.h:53`) or `adapter-merge.h`+`adapter-runtime.h` (`dit.h:11,13`) | Restore includes; `verify-hooks.ps1` confirms |
| Generation stuck ~2 min then fails "stalled" | `pollUntilDone` stall watchdog (`generate.ts:107,128`) | Check `gen_*.log` for last engine phase; `adapter_precompute` ~17 s LoKr cold-start is normal |
| `/api/health` or engine health flaky mid-generation | Engine is single-threaded httplib; can't answer during compute (`aceClient.ts:6-8`) | Not a bug unless it persists after the job ends |
| Synth params vanish after LM phase (adapter/solver settings ignored) | Sideband fields don't survive the `/lm` round trip | Rebuild synth req from original `aceReq` + LM fields only (`generate.ts:312-328`) |
| Infinite ace-server respawn + file locks during rebuild | You ran `engine\build.cmd` with the app up | Use `.\dev-rebuild.bat` |
| npm install fails / weird dep errors | Node 24+ | Use Node 18–22 |
| Vulkan pinned-memory alloc failure in VAE | VAE chunk too large | Tune `ACESTEPCPP_VAE_CHUNK` / `ACESTEPCPP_VAE_OVERLAP` in `.env` |
| First-launch hang "Downloading CUDA runtime" | Portable CUDA DLL bootstrap (`index.ts:318+`) | Wait or go offline → engine starts CPU-only |
| Accidental CI release build | Pushed a `v*` tag | Use `-CI-Test` suffix for throwaways; see `docs/dev/releasing.md` |

## Institutional knowledge

- **VALIDATED — the real server binary**: `ace-server.exe` builds from `engine/tools/hot-step-server.cpp`; upstream `ace-server.cpp` is uncompiled reference (`engine/CMakeLists.txt:416-419`).
- **VALIDATED — LM echo sideband**: server-only `AceRequest` fields never survive the `/lm` round trip; synth requests must be rebuilt from the original request + LM output fields only, never echoed back (`generate.ts:312-328`).
- **VALIDATED — GGML clobbers input buffers** in the per-section adapter-masking path, so masks must be re-uploaded every step.
- **VALIDATED (by revert) — regional self-attn isolation broke musical continuity** and was reverted in commit `ee041e1`. The `adapter_section_isolation` field still exists in `AceRequest` (`aceClient.ts:97`). Do NOT re-attempt without a design that preserves cross-section musical coherence — naive per-section attention masking is known to break continuity.
- **VALIDATED — draft-LM speculative decoding is deliberately disabled** (`config.ts:117-122`): GGML per-call overhead (~10 ms) negates the speedup.
- **VALIDATED — `--keep-loaded` is spawn-time** (restart required); flips the engine ModelStore to EVICT_NEVER so the ~17 s LoKr precompute happens once. Separate per-request `?keep_loaded=1` co-resident toggle also exists (`config.ts:124-132`). Default OFF (VRAM trade-off).
- **VALIDATED — LM results cached by seed+params** (`lmCache.ts`): changing only synth params reuses cached codes; changing seed re-runs the LM.
- **UNVALIDATED / HYPOTHESIS — LoKr cross-base basin nudge**: `rebase_source`/`rebase_beta` fields exist (`aceClient.ts:112-113`) implementing β·(S−T) toward the adapter's training base; efficacy unproven. Non-turbo XL bases still fail despite ~99% weight identity (basin sensitivity, not drift). **When multiple adapters are stacked, the basin re-base applies once per STACK, never once per adapter** (issue #72; regression class fixed in 168dcb5).
- Repo root has stray build artifacts (`*.obj`, `test_trtllm_*`, `nul`) — untracked experiment leftovers. Do NOT delete them without asking; they may be the user's test artifacts.

## Deeper reading

- [reference.md](reference.md) (this folder) — full engine source map, complete engine endpoint list, `AceRequest` field catalogue, data/config details.
- `FEATURES.md` — full feature catalogue (100+). `engine/docs/ARCHITECTURE.md` — engine internals, CLI, request JSON. `docs/dev/plugins-authoring.md` — Lua plugin authoring. `docs/dev/releasing.md` — release runbook. `server/src/data/assistant-knowledge.md` — in-app assistant KB.
- `docs/plans/` is **gitignored (local-only)** — internal design/investigation docs (perf, adapters, `upstream-sync-workflow.md`); may be absent on a fresh clone.
