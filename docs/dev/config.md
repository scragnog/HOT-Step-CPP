# Configuration

The Node server reads its configuration from environment variables, almost all of them in
`server/src/config.ts`. This page lists every variable the server reads, its default, and
what it changes. None of them are required: with no `.env` the app finds the engine, the
models and its data folder on its own.

## How configuration is loaded

1. `config.ts` sets the project root. If `HOT_STEP_ROOT` is set (the portable launchers set
   it), that is the root and the app runs in portable mode. Otherwise the root is two levels
   above `server/src/`, which is the repo root.
2. If `<root>/.env` does not exist and `<root>/.env.example` does, the example is copied to
   `.env` on first launch, so the Settings page has a file to write to.
3. `dotenv` loads `<root>/.env` into `process.env`. Variables already set in the shell win,
   because `dotenv` does not override them.
4. An empty `CUDA_VISIBLE_DEVICES` is deleted from `process.env`. An empty value hides every
   GPU from child processes, and the Settings GPU picker writes exactly that for "Auto".

The values are read once at startup into the `config` object. Some can be changed while the
app runs; see [Settings page and hot reload](#settings-page-and-hot-reload).

## Where settings live

| Store | What goes there | Edited from |
|---|---|---|
| `<root>/.env` | Paths, ports, engine flags, GPU choice, LLM keys and endpoints, labelling throughput | Settings, Environment and AI Services tabs, or a text editor |
| SQLite `settings` table in `hotstep.db` | Active backend (`active_backend_id`), MiniMax-Music3 and YuE2 model and adapter selections, custom Lyric Studio prompts (`prompt_<name>`), the Insta-Gen system prompt (`instagen_system_prompt`), training defaults (`training_defaults_v1`), active training models (`active_models_v1`) | The matching UI controls |
| Browser `localStorage` | Generation parameters in `ui/src/stores/globalParamsStore.ts` (keys prefixed `hs-`) and other per-browser UI state | The generation controls |

## Settings page and hot reload

`server/src/routes/settings.ts` exposes a whitelist of `.env` keys to the UI:

| Endpoint | Does |
|---|---|
| `GET /api/settings/env` | Returns the value of every exposed key. A key missing from `.env` is filled with the value the server is actually using, so the UI shows real paths. Also returns the list of restart-required keys |
| `POST /api/settings/env` | Body `{ "values": { KEY: "value" } }`. Keys outside the whitelist are dropped. Existing lines are rewritten in place, new keys are appended, comments and line endings are kept. Then calls `reloadEnvConfig()` and returns `{ updated, restartRequired }` |
| `GET /api/settings/gpus` | Lists NVIDIA GPUs from `nvidia-smi` (index, UUID, name, memory). Empty when `nvidia-smi` is not available |

The whitelist is `EXPOSED_ENV_KEYS` in `config.ts`. A key that is not on it can only be
set by editing `.env` or the shell environment. Keys in `RESTART_REQUIRED_KEYS` are written
to `.env` and stored in `config`, but the running engine or server does not pick them up
until the app restarts. Every other exposed key is applied at once by `reloadEnvConfig()`.

In the tables below, the Settings column says where a key appears:

- **Environment** or **AI Services**: a control on that Settings tab.
- **Restart**: on the restart-required list.
- **API only**: exposed by the endpoint but no control in `SettingsPanel.tsx`.
- **env only**: not exposed; edit `.env`.

## Paths

| Variable | Default | Settings | Effect |
|---|---|---|---|
| `ACESTEPCPP_EXE` | First that exists of `engine/ace-server[.exe]`, `engine/build/Release/`, `engine/build/`, `engine/build/Debug/`; else the first | env only | The engine binary to spawn. Other tools (`vst-host`, `ace-train`, `ace-midi`, `ace-caption`, `mastering`) are looked up next to it |
| `ACESTEPCPP_MODELS` | `<root>/models` | Environment, Restart | Passed to the engine as `--models`. Also the base for the default Whisper models folder |
| `ACESTEPCPP_ADAPTERS` | `<root>/adapters` | Environment, Restart | Passed as `--adapters`, only when the folder exists |
| `ACESTEPCPP_ONNX_DIR` | `<root>/models/onnx` | env only | Passed as `--onnx-dir`, only when the folder exists and holds at least one `.onnx` file |
| `ACESTEPCPP_NOISE_PROFILE` | First `.wav` in `<root>/noise_samples/`, or empty | env only | Passed as `--noise-profile` when the file exists |
| `DATA_DIR` | `./data` | Environment, Restart | Data folder: `hotstep.db`, `audio/`, `references/`, `vst/`, `lyrics/`, `training/`. See [How DATA_DIR resolves](#how-data_dir-resolves) |
| `LYRICS_EXPORT_DIR` | `<data dir>/lyrics` | Environment | Where Lyric Studio exports lyrics. Hot-reloaded |
| `MUSCRIPTOR_MODELS_DIR` | `<data dir>/models/muscriptor` | Environment, Restart | Where MIDI Studio keeps its MuScriptor weights. Read in `services/muscriptor.ts` when the module loads |
| `TRAINING_DIR` | `<data dir>/training` | env only | Training Studio working folder |
| `WHISPER_EXE` | `<root>/tools/whisper/whisper-cli[.exe]` | env only | Whisper binary for lyric transcription |
| `WHISPER_MODELS_DIR` | `<models>/whisper` | env only | Whisper model folder |
| `ESSENTIA_BIN` | `<root>/Essentia/essentia_streaming_extractor_music[.exe]` | env only | Essentia extractor used for BPM and key analysis and dataset labelling |

### How DATA_DIR resolves

`DATA_DIR` is resolved against the folder one level above the running server code, not
against the project root:

- In a git checkout the server runs from `server/src/`, so the base is `server/`. The
  default `./data` means `server/data/`.
- In a portable release the server is bundled into `server/server.mjs`, so the base is the
  package root. The shipped `.env.example` sets `DATA_DIR=./server/data`, which means
  `<package>/server/data/`.

Because of this, the repo's `.env.example` value is wrong for a git checkout. If it is
copied to `.env` there, `./server/data` resolves to `server/server/data/` and the app starts
with an empty library. In a checkout, use `DATA_DIR=./data` or an absolute path.

A repo-root `data/` folder is not the live data folder in either layout.

## Ports and hosts

| Variable | Default | Settings | Effect |
|---|---|---|---|
| `SERVER_PORT` | `3001` | Environment, Restart | Port the Node server listens on. The Vite proxy in `ui/vite.config.ts` is hard-coded to `3001`, so dev mode breaks if you change it |
| `SERVER_HOST` | `0.0.0.0` | env only | Interface the Node server binds to |
| `ACESTEPCPP_PORT` | `8085` | Environment, Restart | Engine port. Passed as `--port`; the server connects to `http://<host>:<port>` |
| `ACESTEPCPP_HOST` | `127.0.0.1` | Environment, Restart | Engine bind address, passed as `--host`, and the host the server connects to |

`.env.example` also lists `VITE_PORT` and `VITE_HOST`. Nothing reads them: `dev.bat` starts
Vite with `--port 3000 --host`, and `ui/vite.config.ts` sets port 3000 and host `0.0.0.0`.

## Engine

| Variable | Default | Settings | Effect |
|---|---|---|---|
| `ACESTEPCPP_VAE_CHUNK` | `1024` | Environment, Restart | Passed as `--vae-chunk`. VAE tile size; lower it if the VAE fails to allocate pinned memory on Vulkan |
| `ACESTEPCPP_VAE_OVERLAP` | `64` | Environment, Restart | Passed as `--vae-overlap`. Overlap between VAE tiles |
| `ACESTEPCPP_KEEP_LOADED` | `0` | Environment, Restart | Any value other than `0` passes `--keep-loaded`, which keeps the DiT, adapters and the LoKr precompute resident between requests. Costs VRAM; off by default |
| `YUE2_NO_OVERLAP` | unset | Engine process | Any value keeps every YuE2 render on the work thread instead of the NAR lane (engine/docs/ARCHITECTURE.md) |
| `YUE2_OVERLAP_MIN_FREE_GB` | `4` | Engine process | Free VRAM the NAR lane needs at the handoff; below it the render runs inline |
| `ACESTEPCPP_DRAFT_LM` | empty | env only | Passed as `--draft-lm` when the file exists. Speculative decoding is off on purpose: per-call GGML overhead cancels the gain |
| `CUDA_VISIBLE_DEVICES` | empty (all GPUs) | Environment, Restart | GPU for the engine and for `ace-train`. The Settings picker writes a GPU UUID. A value that is not a UUID is passed through as given, and the server sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match `nvidia-smi`. The child environment is rebuilt from the live `config` value on every spawn, so a training job started after saving uses the new GPU |
| `TENSORRT_LIBS` | `engine/deps/tensorrt_libs` if it contains `nvinfer_10.dll` or `libnvinfer.so.10`, else empty | env only | Prepended to the engine's `PATH` so ONNX Runtime and TensorRT can load. If a `trtllm-libs` folder exists two levels above the engine binary, it is prepended too |

### Warm-up on startup

These pre-load a DiT, VAE and adapter right after the engine starts, so the first
generation skips the cold load. The warm runs only when `ACESTEPCPP_WARM_ON_STARTUP` is on,
`ACESTEPCPP_KEEP_LOADED` is on, and `ACESTEPCPP_WARM_DIT` is set. Without keep-loaded the
engine would evict the models at once, so the server logs that it skipped the warm. The
server waits up to 90 seconds for `/health`, then posts `/warm` and does not wait for it.
All of these are env only.

| Variable | Default | Effect |
|---|---|---|
| `ACESTEPCPP_WARM_ON_STARTUP` | `1` | `0` disables the warm |
| `ACESTEPCPP_WARM_DIT` | empty | DiT file name, resolved against the models folder. Empty disables the warm |
| `ACESTEPCPP_WARM_VAE` | empty | VAE file name |
| `ACESTEPCPP_WARM_ADAPTER` | empty | Adapter file name, resolved against the adapters folder. Empty skips the adapter |
| `ACESTEPCPP_WARM_ADAPTER_SCALE` | `1.0` | Adapter scale for the warm. Match what the UI sends so the render reuses the cached LoKr delta |

### Flags passed to ace-server

`startAceServer()` in `server/src/services/aceEngineProcess.ts` builds the command line:

| Flag | From | When |
|---|---|---|
| `--models <dir>` | `ACESTEPCPP_MODELS` | Always |
| `--host <addr>` | `ACESTEPCPP_HOST` | Always |
| `--port <n>` | `ACESTEPCPP_PORT` | Always |
| `--adapters <dir>` | `ACESTEPCPP_ADAPTERS` | Folder exists |
| `--keep-loaded` | `ACESTEPCPP_KEEP_LOADED` | Not `0` |
| `--noise-profile <wav>` | `ACESTEPCPP_NOISE_PROFILE` | File exists |
| `--draft-lm <gguf>` | `ACESTEPCPP_DRAFT_LM` | File exists |
| `--vae-chunk <n>` | `ACESTEPCPP_VAE_CHUNK` | Non-zero |
| `--vae-overlap <n>` | `ACESTEPCPP_VAE_OVERLAP` | Non-zero |
| `--onnx-dir <dir>` | `ACESTEPCPP_ONNX_DIR` | Folder exists and holds a `.onnx` file |

The engine accepts more flags than these. [engine.md](engine.md) covers the engine's own
command line.

## LLM providers

Lyric Studio, Insta-Gen's LLM path, the Assistant and dataset captioning use these. All of
them are hot-reloaded when saved from Settings.

| Variable | Default | Settings | Effect |
|---|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `gemini` | AI Services | Provider used when a request does not name one |
| `LLM_TIMEOUT_MS` | `300000` | AI Services | Abort limit for one completion call, every provider. Values below 10000 are raised to 10000 |
| `GENIUS_ACCESS_TOKEN` | empty | AI Services | Genius API token for lyric lookup |
| `GEMINI_API_KEY` | empty | AI Services | Google Gemini key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | AI Services | Gemini model |
| `OPENAI_API_KEY` | empty | AI Services | OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | AI Services | OpenAI model |
| `ANTHROPIC_API_KEY` | empty | AI Services | Anthropic key |
| `ANTHROPIC_MODEL` | `claude-3-5-haiku-20241022` | AI Services | Anthropic model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | AI Services | Ollama server |
| `OLLAMA_MODEL` | `llama3` | AI Services | Ollama model |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | AI Services | LM Studio server |
| `LMSTUDIO_MODEL` | empty | AI Services | LM Studio model |
| `LMSTUDIO_API_KEY` | empty | AI Services | Bearer token, for an LM Studio server with authentication on. Empty sends no token |
| `UNSLOTH_BASE_URL` | `http://127.0.0.1:8888` | AI Services | Unsloth server |
| `UNSLOTH_USERNAME` | empty | AI Services | Unsloth login |
| `UNSLOTH_PASSWORD` | empty | AI Services | Unsloth password |
| `UNSLOTH_MODEL` | empty | AI Services | Unsloth model |
| `LLAMACPP_BASE_URL` | `http://127.0.0.1:8080/v1` | AI Services | llama.cpp server |
| `LLAMACPP_MODEL` | empty | AI Services | llama.cpp model |
| `OPENAI_COMPAT_BASE_URL` | empty | AI Services | Any OpenAI-compatible endpoint (vLLM, LocalAI and similar). `.env.example` says the provider only appears in the UI when this is set |
| `OPENAI_COMPAT_API_KEY` | empty | AI Services | Key for that endpoint |
| `OPENAI_COMPAT_MODEL` | empty | AI Services | Model for that endpoint |
| `OPENAI_COMPAT_NAME` | `OpenAI Compatible` | AI Services | Display name for that provider |
| `OPENAI_COMPAT_REASONING_EFFORT` | empty | API only | Default reasoning effort sent to that endpoint (for example `none`, `low`, `medium`, `xhigh`). Empty sends nothing. A per-request `reasoning_effort` or `no_think` wins |

## Labelling concurrency

Throughput limits for Dataset Studio labelling (`server/src/services/training/rateLimit.ts`).
Concurrency is the number of calls in flight; the minimum interval spaces out call starts.
All are on the Environment tab and hot-reloaded. An empty or invalid value falls back to the
default rather than 0, which would stall the limiter.

| Variable | Default | Effect |
|---|---|---|
| `LABEL_ESSENTIA_CONCURRENCY` | `2` | Parallel Essentia runs |
| `LABEL_GENIUS_CONCURRENCY` | `1` | Parallel Genius lookups |
| `LABEL_GENIUS_MIN_INTERVAL_MS` | `400` | Gap between Genius call starts |
| `LABEL_CAPTION_CONCURRENCY` | `1` | Parallel LLM caption calls. The biggest speed lever, and it multiplies requests per minute against your provider quota |
| `LABEL_CAPTION_MIN_INTERVAL_MS` | `250` | Gap between caption call starts |

## Training

Not exposed in Settings on purpose. Edit `.env`.

| Variable | Default | Effect |
|---|---|---|
| `TRAINING_DIR` | `<data dir>/training` | Training Studio working folder |
| `TRAINING_UNDERSTAND_TIMEOUT_MS` | `1200000` (20 min) | Per-file timeout for the engine `/understand` call during labelling |
| `TRAINING_MAX_SCAN_FILES` | `5000` | Scanning a dataset folder with more files than this fails with an error instead of continuing |

## Other variables

| Variable | Default | Read in | Effect |
|---|---|---|---|
| `HOT_STEP_ROOT` | unset | `config.ts` | Project root. Setting it turns on portable mode, which among other things downloads the CUDA runtime DLLs on a Windows CUDA build's first launch. Set by the portable launchers `release/HOT-Step.bat` and `release/HOT-Step.sh` |
| `HOT_STEP_DEV` | unset | `routes/shutdown.ts` | Set by `dev.bat`. Lets `POST /api/shutdown` kill the Vite process on port 3000 after checking it is Node running Vite. Windows only |
| `PP_STEM_CACHE_GB` | `4` | `services/generation/stemCache.ts` | Disk budget for cached vocal and instrumental stems that the post-processing chain reuses when it runs again on the same audio. `0` disables the cache |
| `DOCKER_PATH_MAP` | unset | `services/pathMapper.ts` | JSON object mapping Windows path prefixes to container mount points, so presets saved on Windows work in Docker. Set in `.env.docker` |

## Other env files

- `.env.docker` is copied to `/app/.env` by the `Dockerfile` and used by
  `docker-compose.yml`. It points the engine, models and adapters at container paths.
- `tools/discord-claude/.env.example` configures the Discord bridge tool. The app does not
  read it.

## Adding a variable

1. Read it in `config.ts` with a default, so the app still runs without it.
2. If users should edit it from Settings, add it to `EXPOSED_ENV_KEYS`, add an `apply()` line
   in `reloadEnvConfig()` (or put it in `RESTART_REQUIRED_KEYS`), and add a control in
   `ui/src/components/settings/SettingsPanel.tsx`. A key that is exposed but has no `apply()`
   line saves to `.env` and then does nothing until a restart.
3. If a module caches a value derived from config, register a listener in
   `configReloadListeners` so a Settings save reaches it.
4. Add it to `.env.example` if a user is likely to need it, and to this page.

## Related

- [architecture.md](architecture.md)
- [building.md](building.md)
- [engine.md](engine.md)
- [api.md](api.md)
