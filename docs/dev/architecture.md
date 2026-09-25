# Architecture

HOT-Step CPP is a desktop app for local music generation. It has three tiers: a C++
inference engine, a Node server that owns the database and orchestrates the engine, and a
React UI in the browser. This page shows how they fit together and where each feature lives
in the source tree. Read it before you pick a file to edit.

## The three tiers

| Tier | Stack | Location | Role |
|---|---|---|---|
| Engine | C++17, CUDA/Vulkan/Metal/CPU through GGML | `engine/` | Inference binaries. `ace-server` is the HTTP server the app talks to. Other tools: `ace-train`, `ace-caption`, `ace-midi`, `mastering`, `mp3-codec`, `neural-codec`, `quantize`, `vst-host`, and the upstream CLIs `ace-lm`, `ace-synth`, `ace-understand` |
| Server | Node 18 to 22, TypeScript, Express, better-sqlite3 | `server/src/` | Job queue, SQLite, file storage, spawns and restarts the engine, serves the built UI |
| UI | React 19, Vite, Zustand, Tailwind | `ui/src/` | Browser frontend, one component folder per studio |

```
Browser
  |  dev:  http://localhost:3000  (Vite HMR, proxies /api /audio /references to :3001)
  |  prod: http://localhost:3001
  v
Node server   server/src/index.ts   Express on 0.0.0.0:3001
  |-- /api/*         route files in server/src/routes/
  |-- /audio         <data dir>/audio         (generated songs)
  |-- /references    <data dir>/references    (mastering reference tracks)
  |-- /*             ui/dist, when it exists, with an SPA fallback to index.html
  |-- SQLite         <data dir>/hotstep.db
  |-- runs as needed: ace-train, ace-caption, ace-midi, mastering, mp3-codec,
  |                   vst-host, whisper-cli, Essentia, sd-cli
  v  HTTP, server/src/services/aceClient.ts
ace-server    engine/tools/hot-step-server.cpp   127.0.0.1:8085
  |-- ACE-Step 1.5      /lm /synth /understand /vae /warm /job ...
  |-- MiniMax-Music3    /mm3/*
  |-- YuE2              /yue2/*
  `-- SuperSep, Spectral Lifter, PP-VAE re-encode, SA3 refine
```

| Service | Default port | Set by |
|---|---|---|
| Node server | 3001 | `SERVER_PORT` |
| Vite dev server | 3000 | hard-coded in `ui/vite.config.ts` and `dev.bat` |
| ace-server | 8085 | `ACESTEPCPP_PORT` |

The `<data dir>` is `server/data/` in a git checkout. [config.md](config.md) has every
variable, including how `DATA_DIR` resolves.

## How the server runs the engine

`server/src/services/aceEngineProcess.ts` owns the `ace-server` child process. On startup
`index.ts` calls `startAceServer()`, which passes `--models`, `--host` and `--port`, plus
`--adapters`, `--keep-loaded`, `--noise-profile`, `--draft-lm`, `--vae-chunk`,
`--vae-overlap` and `--onnx-dir` when their settings apply. The full list is in
[config.md](config.md#flags-passed-to-ace-server).

- An abnormal exit respawns the engine after 3 seconds. Three crashes inside 30 seconds
  stop the respawns and mark the engine not ready. The usual cause is a missing DLL next to
  the binary.
- Training jobs stop the engine on purpose to free the GPU (`stopAceServer`) and start it
  again afterwards (`restartAceServer`). A deliberate stop never triggers a respawn.
- MiniMax-Music3 and YuE2 keep their model selection inside the engine process, so a
  restart loses it. `index.ts` registers restore hooks through `onEngineRestarted` that put
  the saved selection back after every respawn.
- In a portable Windows CUDA build, the first launch downloads the cuBLAS and cudart DLLs
  from Hugging Face before it starts the engine. If that fails, the engine starts on CPU.

Because of the respawn logic, killing `ace-server` from outside while Node is running
starts it again straight away. Rebuild the engine with `dev-rebuild.bat`, which asks Node
to shut down first. See [building.md](building.md#engine-rebuild-rules).

## A generation request, end to end

1. The UI builds a camelCase parameter object from the Create panel and the global bar
   (`ui/src/stores/globalParamsStore.ts`, `ui/src/types.ts`) and posts it to
   `POST /api/generate` through `ui/src/services/api.ts`.
2. `server/src/routes/generate.ts` checks that the engine is ready, builds a job envelope
   that names the backend (`services/generation/envelope.ts`), and queues the job. Jobs run
   one at a time on a shared GPU lane (`services/generation/gpuLane.ts`).
3. The route hands the job to the backend's `generate()`. For ACE-Step that is
   `services/backends/ace/generate.ts`:
   - `translateParams()` turns the UI params into a snake_case `AceRequest`
     (`services/generation/translateParams.ts`, type in `services/aceClient.ts`).
   - The LM phase posts to the engine's `/lm`, unless the LM cache
     (`services/generation/lmCache.ts`) already has codes for these LM params.
   - The synth request is rebuilt as `{ ...aceReq, <LM output fields> }`. Fields the Node
     server adds do not survive the `/lm` round trip, so never copy the LM response forward.
     The `generation-request-flow` skill explains this trap in detail.
   - Each track goes to `/synth`, as JSON or multipart when source audio or latents are
     attached.
   - The WAV or MP3 is saved, auto-trim and the post-processing chain run
     (`services/generation/postProcessing.ts`), and a row is inserted into `songs`.
4. The engine accepts each `/lm` or `/synth` as an async job and returns a job id. The
   server polls `GET /job?id=<id>` and fetches the result with `GET /job?id=<id>&result=1`.
   The polling loop is `services/generation/pollUntilDone.ts`, which also holds the stall
   and wall-clock watchdogs.
5. The UI polls `GET /api/generate/status/:id` until the job finishes.

`ace-server` answers HTTP on one thread. While it runs DiT, VAE or adapter work it cannot
answer at all, so a slow `/health` in the middle of a generation is expected. The client
timeouts in `aceClient.ts` are long for that reason.

## Backends

The server can drive three model families through one interface. The registry is
`server/src/services/backends/registry.ts`, and the interface (`EngineBackend`) is in
`services/backends/types.ts`.

| Id | Display name | Server code | Engine code |
|---|---|---|---|
| `ace` (default) | ACE-Step 1.5 | `services/backends/ace/` | `engine/src/` core pipeline |
| `minimax-m3` | MiniMax-Music3 | `services/backends/minimax/` | `engine/src/minimax/`, `/mm3/*` |
| `yue2` | YuE2 | `services/backends/yue2/` | `engine/src/yue2/`, `/yue2/*` |

Each backend reports a capability manifest (duration limits, whether it has an LM, plugins,
adapters and so on). The UI reads it from `GET /api/capabilities` and hides controls a
backend does not support (`ui/src/stores/backendStore.ts`,
`ui/src/components/shared/BackendCapabilityGate.tsx`). The active backend id is stored in
the SQLite `settings` table under `active_backend_id` and switched with
`POST /api/backends/active`. User-facing detail is in [../user/backends.md](../user/backends.md).

## Feature map

Every API mount is registered in `server/src/index.ts`. Route files are in
`server/src/routes/`, services in `server/src/services/`, UI folders in
`ui/src/components/`. The full endpoint list is in [api.md](api.md).

| Feature | API mount | Route file | Main services | UI | Engine side |
|---|---|---|---|---|---|
| Create (main generation) | `/api/generate` | `generate.ts` | `backends/*`, `generation/*`, `aceClient.ts`, `autoTrim.ts` | `create/`, `global-bar/` | `/lm`, `/synth`, `/mm3/synth`, `/yue2/synth` |
| Backend toggle and capabilities | `/api` (`/backends`, `/capabilities`) | `backends.ts` | `backends/registry.ts` | `global-bar/BackendToggle.tsx`, `stores/backendStore.ts` | `/mm3/props`, `/yue2/props` |
| Insta-Gen | `/api/inspire` | `inspire.ts` | `aceClient.ts`, `lireek/llm/registry.ts` | `insta-gen/` | `/lm` in inspire mode |
| Lyric Studio | `/api/lireek` | `lireek.ts`, `lireek/*` | `lireek/` (prompts, slop detector, LLM providers) | `lyric-studio/` | none, uses external LLMs |
| Library | `/api/songs` | `songs.ts` | `audioCrop.ts`, `disco-analyzer.ts`, `library/importTrack.ts`, `generation/rePostProcess.ts` | `library/`, `details/`, `player/` | none |
| Player waveform peaks | `/api/audio` | `audio.ts` | `audio/peaks.ts` | `ui/src/audio/peaksClient.ts` | none |
| Models dropdown | `/api/models` | `models.ts` | `aceClient.ts` (`GET /props`) | `global-bar/ModelsDropdown.tsx` | `model-registry.h`, `model-store.cpp` |
| Model Manager (downloads) | `/api/model-manager` | `modelManager.ts` | `modelDownloadService.ts`, `server/src/data/model-registry.json` | `model-manager/` | none |
| DiT adapters (LoRA, LoKr) | `/api/adapters` | `adapters.ts` | `adapters/stMetadata.ts`, `generation/adapterSections.ts` | `create/AdaptersAccordion.tsx`, `global-bar/AdaptersDropdown.tsx` | `adapter-merge.h`, `adapter-runtime.h` |
| Stem Studio | `/api/stem-studio` | `stemStudio.ts` | `aceClient.ts`, `audioConvert.ts` | `stem-studio/` | `/synth` (extract), `/supersep/*` |
| SuperSep proxy | `/api/supersep` | `supersep.ts` | `audioConvert.ts` | `shared/StemMixer.tsx`, `services/supersepApi.ts` | `supersep.cpp` |
| Stem Builder | uses the generate and stem APIs | none | none | `stem-builder/` | none of its own |
| Repaint Studio | `/api/generate` (task type) | `generate.ts` | `generation/sourceAudio.ts` | `repaint-studio/` | `sampler-repaint.h` |
| Cover Studio | `/api/generate`, `/api/analyze` | `generate.ts`, `analyze.ts` | `generation/sourceAudio.ts`, Essentia | `cover-studio/` | cover task types in `request.h` |
| Song Builder | `/api/builder` | `songBuilder.ts` | `db/database.ts` | `song-builder/` | seed-latent path of `/synth` |
| STORM (continuous stream) | `/api/generate/storm/*` | `generate.ts` | `aceClient.ts` | `storm/` | `/synth` |
| MIDI Studio | `/api/midi-studio` | `midiStudio.ts` | `muscriptor.ts`, `midiParser.ts` | `midi-studio/` | `ace-midi` binary |
| Training Studio | `/api/training` | `training.ts` | `training/*` | `training-studio/` | `ace-train`, `ace-caption`, `/understand` |
| Cover art images | `/api/cover-art` | `coverArt.ts` | `coverArt/*` | `library/CoverArtPromptModal.tsx`, `global-bar/CoverArtDropdown.tsx` | `sd-cli` (stable-diffusion.cpp), not the music engine |
| Mastering | `/api/mastering` | `mastering.ts` | `generation/postProcessing.ts` | `create/MasteringSection.tsx`, `global-bar/MasteringDropdown.tsx` | `mastering`, `mp3-codec` binaries |
| VST3 chain | `/api/vst` | `vst.ts` | none, runs `vst-host` | `global-bar/VstChainDropdown.tsx`, `stores/vstChainStore.ts` | `tools/vst-host.cpp` |
| Lua plugins | `/api/plugins` | `plugins.ts` | `aceClient.ts` (`GET /plugins`) | `global-bar/PluginControls.tsx`, `hooks/usePluginRegistry.ts` | `lua-plugin.h`, `lua-plugin-registry.h`, `hot-step-sampler.h` |
| Assistant | `/api/assistant` | `assistant.ts` | `lireek/llm/registry.ts`, `server/src/data/assistant-knowledge.md` | `assistant/`, `services/assistantApi.ts` | none |
| Settings (.env editor) | `/api/settings` | `settings.ts` | `config.ts` (`reloadEnvConfig`), `gpuDevices.ts` | `settings/` | none |
| Parameter profiles | `/api/profiles` | `profiles.ts` | none | `global-bar/ProfilesModal.tsx` | none |
| Seeds | `/api/seeds` | `seeds.ts` | none | `global-bar/SeedManagerDrawer.tsx` | none |
| Uploads (audio, latents) | `/api/upload` | `upload.ts` | `latentFormat.ts` | `shared/LatentImport.tsx`, studio upload fields | none |
| Downloads and export | `/api/download` | `download.ts` | `audioMetadata.ts` | `player/`, `library/` | `mp3-codec` |
| Logs, VRAM, terminal | `/api/logs` | `logs.ts` | `logger.ts` | `terminal/`, `shared/VramIndicator.tsx` | `/vram`, `/models/loaded`, `/models/unload` |
| Health, shutdown, restart | `/api/health`, `/api/shutdown` | `health.ts`, `shutdown.ts` | `server/src/engineState.ts`, `aceEngineProcess.ts` | none | `/health` |
| Auth (local single user) | `/api/auth` | `auth.ts` | `db/database.ts` | none | none |
| BPM and key analysis | `/api/analyze` | `analyze.ts` | Essentia binary (`ESSENTIA_BIN`) | `cover-studio/` | none |

`POST /api/shutdown/restart` writes a `.restart-requested` marker at the repo root. The loop
in `LAUNCH.bat` and `server/restart-loop.cmd` sees it and starts the server again.

## Engine source layout

The engine is a fork of acestep.cpp. [engine.md](engine.md) covers its internals; this is the
map of top-level pieces.

| Path | What it is |
|---|---|
| `engine/tools/hot-step-server.cpp` | The `ace-server` binary: HTTP routes, request parsing, job queue. `engine/tools/ace-server.cpp` is upstream's version, kept for reference and not compiled |
| `engine/src/pipeline-lm.cpp`, `qwen3-lm.h`, `metadata-fsm.h` | LM phase: caption enrichment, metadata and audio codes |
| `engine/src/pipeline-synth*.cpp`, `dit.h`, `dit-graph.h` | DiT synthesis |
| `engine/src/hot-step-sampler.h` | HOT-Step's sampling loop. Solver, scheduler and guidance dispatch go through here, not upstream's `dit-sampler.h` |
| `engine/src/hot-step-params.h` | The sideband struct (`g_hotstep_params`) that carries HOT-Step-only request fields into the sampler |
| `engine/src/lua-plugin.h`, `lua-plugin-registry.h` | Lua plugin host. Scans `engine/plugins/` and the repo-root `plugins/` overlay for solvers, schedulers, guidance and postprocess plugins |
| `engine/src/adapter-merge.h`, `adapter-runtime.h`, `adapter-cancel.h`, `lokr-*.h` | DiT adapter loading, merge and runtime modes |
| `engine/src/model-store.h`, `model-store.cpp`, `model-registry.h` | Model discovery and VRAM residency |
| `engine/src/vae*.h`, `vae-ort.h` | VAE encode and decode, GGML and ONNX Runtime |
| `engine/src/dit-trt.h`, `lm-trt.h`, `hot-step-sampler-trt.h` | TensorRT paths |
| `engine/src/supersep.cpp`, `spectral-lifter.h`, `sa3-*.h` | Stem separation, the Spectral Lifter, and the Stable Audio 3 refiner |
| `engine/src/hot-step-families.h` | Registry of the extra model families hosted by `ace-server` |
| `engine/src/minimax/` | MiniMax-Music3 port and its `/mm3/*` routes (`mm3-server.h`, `mm3-job.h`) |
| `engine/src/yue2/` | YuE2 port and its `/yue2/*` routes (`yue2-server.h`), plus the SheetSage lead-sheet model |
| `engine/src/moss/` | MOSS-Music audio encoder used by `ace-caption` |
| `engine/src/train/` | Training code behind `ace-train` (ACE-Step DiT and LM, MM3, YuE2) |
| `engine/tools/ace-train.cpp` | Training CLI. Subcommands include `preprocess`, `train-dit`, `train-lm`, `mm3-lm-train`, `mm3-train-dit`, `yue2-joint-train` and others |
| `engine/plugins/` | Built-in Lua solvers, schedulers and guidance modes. See [plugins-authoring.md](plugins-authoring.md) |
| `engine/patches/` | Patches applied to the `engine/ggml` submodule. See [building.md](building.md#the-ggml-patch-stack) |

Three upstream files carry `#include` hooks into HOT-Step code (`pipeline-synth-ops.cpp`,
`model-store.h`, `dit.h`). `engine/verify-hooks.ps1` checks them along with the ggml patches.

## Server source layout

| Path | What it is |
|---|---|
| `server/src/index.ts` | Entry point: logger, DB init, route mounts, static serving, engine bootstrap, shutdown |
| `server/src/config.ts` | Every environment variable and its default. See [config.md](config.md) |
| `server/src/routes/` | One Express router per feature |
| `server/src/services/` | Feature logic. `aceClient.ts` wraps the engine HTTP API; `aceEngineProcess.ts` owns the child process |
| `server/src/services/backends/` | Backend registry and the three backend implementations |
| `server/src/services/generation/` | Queue, envelopes, param translation, LM cache, post-processing |
| `server/src/services/training/` | Dataset Studio and Training Studio runners |
| `server/src/services/lireek/` | Lyric Studio and the LLM provider registry |
| `server/src/db/` | SQLite schema and access (`database.ts`, `lireekDb.ts`) |
| `server/src/data/` | Runtime data shipped with the app, such as `model-registry.json` and `assistant-knowledge.md`. The release workflow copies this folder whole |
| `server/src/engineState.ts` | Engine ready flag and boot status message |

## UI source layout

| Path | What it is |
|---|---|
| `ui/src/App.tsx` | View routing. Views: create (default), `insta-gen`, `lyric-studio`, `cover-studio`, `stem-studio`, `stem-builder`, `song-builder`, `storm`, `midi-studio`, `training-studio`, `repaint`, `library`, `settings` |
| `ui/src/components/<studio>/` | One folder per studio or panel. `global-bar/` is the parameter bar shared by all generation views; `shared/` holds reusable pieces |
| `ui/src/stores/` | Zustand stores. `globalParamsStore.ts` holds generation params; others cover playback, backends, streaming, training, VST chain, post-processing |
| `ui/src/services/` | Fetch wrappers. `api.ts` is the main one (base `/api`); studios with large surfaces have their own (`lireekApi.ts`, `trainingApi.ts`, `stemStudioApi.ts` and so on) |
| `ui/src/hooks/` | Shared hooks such as `useCapabilities.ts` and `usePluginRegistry.ts` |
| `ui/src/i18n/locales/` | UI strings. `en.json` is the source for control labels |

## Data and logs

- SQLite database: `<data dir>/hotstep.db`. Lyric Studio tables live in the same file.
  Tables include `songs`, `playlists`, `artists`, `lyrics_sets`, `profiles`, `generations`,
  `settings`, `album_presets`, `builder_projects`, `training_datasets` and others; the schema
  is in `server/src/db/database.ts`.
- Files under the data dir: `audio/` (outputs), `references/`, `vst/` (`chain.json`,
  `states/`), `lyrics/` (exports), `training/`.
- A git checkout may also have a repo-root `data/` folder. It is not the live data dir.
- Logs: `logs/YYYY-MM-DD_HH-MM-SS/` per session, holding `ace_engine.log`,
  `node_console.log` and `generations/gen_<uuid>_<task>.log`. The newest folder is the
  current session.

## Related

- [building.md](building.md)
- [config.md](config.md)
- [engine.md](engine.md)
- [api.md](api.md)
- [../user/backends.md](../user/backends.md)
- [plugins-authoring.md](plugins-authoring.md)
