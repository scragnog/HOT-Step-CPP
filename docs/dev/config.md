# Configuration

The Node server reads its configuration from environment variables, almost all of them in
`server/src/config.ts`. This page lists every variable the server reads, its default, and
what it changes. None of them are required: with no `.env` the app finds the engine, the
models and its data folder on its own.

## Settings backups

Settings can export application settings and Lyric Studio records as versioned JSON. Import replaces those records in one database transaction after confirmation. It does not change `.env` or machine-specific environment settings. See [Settings and Lyric Studio backup](../user/settings-backup.md).

## How configuration is loaded

1. `config.ts` sets the project root. If `HOT_STEP_ROOT` is set (the portable launchers set
   it), that is the root and the app runs in portable mode. Otherwise the root is two levels
   above `server/src/`, which is the repo root. In portable mode, `dataroot/bootstrap.ts`
   then picks the data folder before anything opens a file; see
   [Release data folder](#release-data-folder).
2. If the `.env` does not exist and `<root>/.env.example` does, the example is copied to
   `.env` on first launch, so the Settings page has a file to write to. The `.env` is
   `<root>/.env` in a checkout and `<data folder>/settings/.env` whenever there is a data
   folder: a release, or a checkout that Settings gave one. That copy comments out the
   example's `DATA_DIR` line.
3. `dotenv` loads that `.env` into `process.env`. Variables already set in the shell win,
   because `dotenv` does not override them.
4. An empty `CUDA_VISIBLE_DEVICES` is deleted from `process.env`. An empty value hides every
   GPU from child processes, and the Settings GPU picker writes exactly that for "Auto".

The values are read once at startup into the `config` object. Some can be changed while the
app runs; see [Settings page and hot reload](#settings-page-and-hot-reload).

## Where settings live

| Store | What goes there | Edited from |
|---|---|---|
| `<root>/.env` (with a data folder: `<data folder>/settings/.env`) | Paths, ports, engine flags, GPU choice, LLM keys and endpoints, labelling throughput | Settings, Environment and AI Services tabs, or a text editor |
| SQLite `settings` table in `hotstep.db` | Active backend (`active_backend_id`), MiniMax-Music3 and YuE2 model and adapter selections, the Stable Audio 3 DiT (`sa3_dit_model`), custom Lyric Studio prompts (`prompt_<name>`), the Insta-Gen system prompt (`instagen_system_prompt`), training defaults (`training_defaults_v1`), active training models (`active_models_v1`) | The matching UI controls |
| Browser `localStorage` | Generation parameters in `ui/src/stores/globalParamsStore.ts` (keys prefixed `hs-`) and other per-browser UI state | The generation controls |

### Update notices

Settings > Updates reads `GET /api/updates/status` (cached, never calls GitHub) and refreshes with `POST /api/updates/check`, optionally `{ channel }` to switch the feed (`release` or `master`). The wire shapes are in `server/src/contracts/updates.ts`; the UI copy is `ui/src/services/updatesApi.ts`, and the panel is `ui/src/components/settings/UpdatesPanel.tsx`. A Windows portable install on updater protocol 1, started by its stable launcher, can also download (`POST /api/updates/prepare`) and install (`POST /api/updates/apply`) a release; every other install, protocol 0 included, gets notices only ([HTTP API index](api.md), [Windows updater layout](releasing.md#windows-updater-layout-protocol-1)). The installed identity comes from `build-info.json` beside `VERSION` (a git checkout reports its own HEAD); a missing file means unknown identity and the UI never guesses a variant.

## Settings page and hot reload

Settings > General > Adapters includes **Update adapter references**. This explicit
command posts to `/api/lireek/presets/update-adapter-references` and clears missing
adapter paths in SQLite's `album_presets`, returning each preset name, adapter field,
and dropped path. It checks ACE files and PEFT/LoKR directories, MM3 paths under
`<adapters>/mm3-lm-adapters`, and absolute YuE2 AR/NAR files. Other preset columns
are preserved. It does not run at startup or change browser generation parameters.
An unavailable adapter root or filesystem access error aborts the command; writes
are transactional.

`server/src/routes/settings.ts` exposes a whitelist of `.env` keys to the UI, along with
the GPU list and the data-folder controls (handlers in `routes/settingsDataFolder.ts`):

| Endpoint | Does |
|---|---|
| `GET /api/settings/env` | Returns the value of every exposed key. A key missing from `.env` is filled with the value the server is actually using, so the UI shows real paths. Also returns the list of restart-required keys |
| `POST /api/settings/env` | Body `{ "values": { KEY: "value" } }`. Keys outside the whitelist are dropped. Existing lines are rewritten in place, new keys are appended, comments and line endings are kept. Then calls `reloadEnvConfig()` and returns `{ updated, restartRequired }` |
| `GET /api/settings/gpus` | Lists NVIDIA GPUs from `nvidia-smi` (index, UUID, name, memory). Empty when `nvidia-smi` is not available |
| `GET /api/settings/engine-workers` | Experimental multi-GPU status: `{ workers, busy }`, one row per engine worker (`id`, `index`, `primary`, `gpuUuid`, `port`, `url`, `logFile`, `pid`, `ready`, `status`, `crashSeq`, `device`, `busy`). `gpuUuid` is null for the primary under an empty opt-in. `busy` is true while any call is bound to a worker |
| `GET /api/settings/data-folder` | Mode, app root, effective data root, any `HOT_STEP_DATA_ROOT` override, the pending operation, absolute path settings that point outside the data folder, and the running work that would block a change |
| `POST /api/settings/data-folder/preview` | Body `{ "kind", "target" }`: `kind` is `move`, `repoint`, `create` or `import`, `target` an absolute folder. Plans the operation and returns its errors without changing anything |
| `POST /api/settings/data-folder/schedule` | Same body. Records the operation in the pointer, then requests a restart, which carries it out before the DB opens. 409 while work is running |
| `POST /api/settings/data-folder/cancel` | Drops a pending operation that has not started |

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
| `ACESTEPCPP_NOISE_PROFILE` | First `.wav` in `<root>/noise_samples/`, or empty | env only | Passed as `--noise-profile` when the file exists |
| `DATA_DIR` | `./data` | Environment, Restart | Data folder: `hotstep.db`, `audio/`, `references/`, `vst/`, `lyrics/`, `training/`. See [How DATA_DIR resolves](#how-data_dir-resolves) |
| `LYRICS_EXPORT_DIR` | `<data dir>/lyrics` | Environment | Where Lyric Studio exports lyrics. Hot-reloaded |
| `DOWNLOAD_DIR` | empty | Environment | Absolute folder that `POST /api/download/:id/save` writes into. Empty keeps browser downloads. Checked (creatable, writable) when saved. Hot-reloaded |
| `MUSCRIPTOR_MODELS_DIR` | `<models>/muscriptor` | Environment, Restart | Where MIDI Studio keeps its MuScriptor and Beat This! weights, one folder per model (`small`, `medium`, `large`, `beat-this`), each holding `model.safetensors` and `config.json`. Both files are needed before a model counts as installed. Setting it overrides the shared and legacy folders below; the value is used as given, so a relative path resolves against the working directory, not the data folder. Read in `services/muscriptor.ts` when the module loads. A download made before this release sits under `<data folder>/library/models/muscriptor` and keeps working there with no network. Copying it to the shared folder is an explicit step that verifies size and SHA-256 and never replaces a different file. Model Manager > Shared downloads into this folder and, for a MuScriptor model, installs, shows and deletes by exact path (`services/modelDownloadService.ts`), not by file name |
| `TRAINING_DIR` | `<data dir>/training` | env only | Training Studio working folder |
| `WHISPER_EXE` | `<root>/tools/whisper/whisper-cli[.exe]` | env only | Whisper binary for lyric transcription. On macOS and Linux, the app also checks for `whisper-cli` on PATH and does not auto-download it. Install with [`brew install whisper.cpp`](https://formulae.brew.sh/formula/whisper.cpp) or set this to an existing executable path. |
| `WHISPER_MODELS_DIR` | `<models>/whisper` | env only | Whisper model folder |
| `ESSENTIA_BIN` | `<root>/Essentia/essentia_streaming_extractor_music[.exe]` | env only | Essentia extractor used for BPM and key analysis and dataset labelling |

In a release the defaults in this table that sit under `<root>` or `<data dir>` move into the
data folder instead: `<data dir>` is `<data folder>/library`, models and adapters are
`<data folder>/models` and `/adapters`, Whisper is `<data folder>/tools/whisper`. A relative
value in the release `.env` resolves against the data folder. Engine, Essentia and noise
samples stay in the app.

### How DATA_DIR resolves

In a git checkout without a data folder, `DATA_DIR` is resolved against `server/`, the
folder one level above `server/src/`. The default `./data` means `server/data/`. The repo's `.env.example` sets
`DATA_DIR=./server/data`, which is wrong for a checkout: it resolves to
`server/server/data/` and the app starts with an empty library. In a checkout, use
`DATA_DIR=./data` or an absolute path.

With a data folder (a release, or a checkout that Settings gave one), an unset `DATA_DIR`
means `<data folder>/library`, and a relative one resolves against the data folder.

Releases before the data folder resolved it against the package root, so the shipped
`DATA_DIR=./server/data` put the library in `<package>/server/data/`, beside the bundled
catalogues, and no `DATA_DIR` meant `<package>/data/`. The first-run migration reads that
install's `.env` to find which one it used.

A repo-root `data/` folder is not the live data folder in either layout.

### Release data folder

A release keeps user data in one folder outside the app, so a new version unzipped
elsewhere finds the same library. Code: `server/src/dataroot/` (`paths.ts` layout and
pointer, `bootstrap.ts` start-up choice, `transfer.ts` moves and path rewrites,
`pointer.ts` the Settings API). A checkout (no `HOT_STEP_ROOT`) uses it only once Settings has given it a data folder; see [Dev checkouts](#dev-checkouts).

| Child | Holds |
|---|---|
| `library/` | What `DATA_DIR` held: `hotstep.db`, audio, references, stems, training, VST state, lyrics, MuScriptor |
| `models/`, `adapters/` | Engine models (with cover art, Whisper models, TRT caches) and adapters |
| `settings/.env` | Settings and API keys |
| `logs/` | Session logs |
| `tools/whisper/`, `tools/runtime/` | Whisper binary; Model Manager runtime DLLs, put first on the library search path of every child process (`PATH` on Windows, `LD_LIBRARY_PATH` elsewhere) and used as the default `TENSORRT_LIBS` |
| `plugins/` | User plugin overlay |
| `hot-step-data.json` | Marks the folder as a HOT-Step data folder |

The chosen folder is recorded in `data-location.json` in the profile folder:
`%LOCALAPPDATA%\HOT-Step\` on Windows, `~/Library/Application Support/HOT-Step/` on
macOS, `$XDG_CONFIG_HOME/hot-step/` or `~/.config/hot-step/` on Linux. The release restart
marker (`restart-requested`) and the transfer lock live there too.

At start-up the folder is, in order: `HOT_STEP_DATA_ROOT`; the folder in
`data-location.json`; if neither exists and this app folder still holds data from an older
version, that data moved to `HOT-Step-Data` beside the app; otherwise a new, empty
`HOT-Step-Data` beside the app. The move renames folders when source and destination are
on the same drive, and copies, checks sizes, then deletes the source when they are not.
Stored absolute paths inside the moved folders (preset adapters and references, dataset
folders, rung checkpoints, settings, JSON request columns, JSON files under the library)
are rewritten after a backup of the DB. Paths outside the moved folders are left alone.
Boundaries are checked on real locations (junctions and links resolved), so a target that
leads back into the app or the folder being moved is refused. Import carries the old
install's settings; a `.env` the data folder already had is kept as
`settings/.env.before-import-<id>`.

Settings schedules a move, repoint, new folder or import from an older install into
`data-location.json`; the next start carries it out before the DB opens. Scheduling is
refused while any busy check registered with `registerBusyCheck` (`dataroot/pointer.ts`)
reports work: generation, post-processing, stem, MIDI, cover-art and Inspire jobs
(`registerJobsBusyCheck` on each job map for queued work, `trackBusy` on each runner until
it settles), live streams with pending recording writes and exports, in-flight audio-queue items and
workflow jobs, model, MuScriptor and cover-art downloads, training and the GPU lane. A new
long-running job type registers its own check. If a saved root is missing or inaccessible,
startup uses the default `HOT-Step-Data` beside the app and logs one warning naming both
paths. It leaves the pointer unchanged, so reconnecting the drive and restarting uses the
saved root again. The fallback may show an empty library; data in the saved root has not
been moved or deleted. An unrelated or shared fallback folder still blocks startup.
`HOT_STEP_DATA_ROOT` keeps precedence and an invalid override still blocks startup.
The read-only `locateData()` inspector and `server/scripts/data-paths.ts` use the same
fallback selection and report `source: "fallback"`, without creating folders or changing
the pointer. Pending transfers remain reported as pending; finish recovery before opening
their database.
An unreadable pointer or a move that stops part way still prints the reason and serves
one explanation page on `SERVER_PORT` instead of the app. Restarting retries.

#### Upgrades, reuse and backup

Two unpacked releases that share one profile folder share one library: the launchers set
only `HOT_STEP_ROOT` (their own folder) and never `HOT_STEP_DATA_ROOT` or
`HOT_STEP_PROFILE_DIR`, so the pointer in the profile decides, not the new app folder's
sibling default. `release/check-layout.mjs` fails a launcher that sets either variable.
The stable launcher of an updater-capable install (`release/updater/launcher.mjs`) is
the exception: every payload lives in `app/<id>/`, so it pins `HOT_STEP_PROFILE_DIR`,
records `<parent of install>/HOT-Step-Data` in the profile pointer on a first start, and
pins `HOT_STEP_DATA_ROOT` only when the saved folder is missing (the server would
otherwise fall back inside `app/`) and while an update's candidate is checked.

- **Reuse** is the upgrade path: the same profile and pointer, nothing copied. Tokens come
  with the folder: service keys in `settings/.env`, the Hugging Face token for MuScriptor in
  `library/muscriptor/hf_token`. **Import previous install** is for an old app folder with no
  pointer; it moves data, it does not merge.
- **Backup** is a copy you keep: the data folder (including both token files) and, for a
  restore on another machine, `data-location.json`. The Settings JSON export is not a data
  backup; see [Settings and Lyric Studio backup](../user/settings-backup.md).
- **Stale root**: with the saved folder missing, every app folder starts on its own
  `HOT-Step-Data` sibling (possibly empty) and the pointer is left alone, so reconnecting the
  drive and restarting reconnects.

Fixtures: `server/src/dataroot/dataroot.test.ts` ("Upgrade fixtures" and the legacy-layout,
override and stale-root tests; platform profile paths, spaces included) and
`release/data-root-launcher.test.mjs` (launcher contract, restart marker per platform).
Run `node --test release/data-root-launcher.test.mjs`.

### Dev checkouts

A checkout keeps its own paths (`.env` at the root, `DATA_DIR` against `server/`, `models/`,
`adapters/`, `logs/`, `tools/whisper/`) until someone confirms a change in the Settings
data-folder panel. Nothing migrates on boot. The change is recorded in the checkout's own
pointer, `data-location.dev-<hash>.json` in the same profile folder, where `<hash>` is the
first 12 hex digits of the SHA-256 of the checkout path (lower-cased on Windows and macOS).
The file also holds a `checkout` field naming that path. A release never reads it, and a
checkout never reads `data-location.json`, so each install and each checkout has its own
pointer. Settings refuses a target that any other pointer in the profile already uses, and
a release's first run refuses a `HOT-Step-Data` sibling that a checkout already uses.
These compare real locations, and start-up checks them again before a transfer and
before opening the folder.

An unavailable saved dev root falls back to this checkout's original folders, with one
warning and no pointer change. Reconnect the drive and restart to resume the saved root.

**Move current data** from a checkout with no data folder moves those checkout paths with
the same rename-or-copy rules as a release migration. Only folders inside the checkout
move. The root `.env` becomes `settings/.env` (left behind as `.env.migrated`) with its
engine binary paths kept, since the checkout is still the app. Planning resolves links and junctions, refuses a source or `.env` that leads outside the
checkout or onto files git tracks, and checks again before anything moves, so
`server/src/data` and other tracked files stay put. Once the checkout has a data folder, **Move**, **Use existing folder**,
**Create empty data folder** and **Import previous install** work as in a release. Runtime
DLLs stay beside the dev engine build, and restart still uses the checkout's
`.restart-requested` marker and `server/restart-loop.cmd` (`dev.sh` on macOS), so it works when the root
changes. A failed transfer serves its explanation page on `:3001`; the Vite UI on `:3000`
shows only failed API calls, so read the server window. Deleting the dev pointer points the
checkout back at its own folders; data that already moved stays in the data folder.

## Ports and hosts

| Variable | Default | Settings | Effect |
|---|---|---|---|
| `SERVER_PORT` | `3001` | Environment, Restart | Port the Node server listens on. The Vite proxy in `ui/vite.config.ts` is hard-coded to `3001`, so dev mode breaks if you change it |
| `SERVER_HOST` | `127.0.0.1` | env only | Interface the Node server binds to. Loopback by default |
| `ALLOWED_HOSTS` | empty | env only | Comma list of extra hostnames accepted in the `Host` header (LAN names, Tailscale MagicDNS) |
| `CORS_ORIGINS` | empty | env only | Comma list of extra browser origins allowed to call the API, for example `http://192.168.1.20:3000` |
| `ACESTEPCPP_PORT` | `8085` | Environment, Restart | Engine port. Passed as `--port`; the server connects to `http://<host>:<port>` |
| `ACESTEPCPP_HOST` | `127.0.0.1` | Environment, Restart | Engine bind address, passed as `--host`, and the host the server connects to |

### LAN and Tailscale access (opt-in)

The server binds `127.0.0.1` and only answers other machines when you opt in:

- `SERVER_HOST=0.0.0.0` to listen beyond loopback. Do not bind a single LAN or Tailscale IP:
  the server then stops listening on `127.0.0.1`, which breaks the local DAW plugin and the
  server's calls to itself. To limit who can connect, use a firewall rule instead.
- Requests whose `Host` is not loopback, an IP literal, `SERVER_HOST`, a `CORS_ORIGINS` host or
  an `ALLOWED_HOSTS` entry get 403. This blocks DNS rebinding. Reaching the server by IP needs
  nothing extra; a name such as `mybox` or `mybox.tail1234.ts.net` must be in `ALLOWED_HOSTS`.
- Loopback browser origins are trusted only on this server's port and the Vite dev port 3000; other local tools (another localhost port) must be listed in `CORS_ORIGINS`.
- Browser pages from a non-loopback origin may only read (GET/HEAD) unless the origin is in
  `CORS_ORIGINS`. A page served by this server and opened by IP counts as same-origin.
- Vite dev from another machine: `ui/vite.config.ts` proxies with `changeOrigin`, so the server
  sees the browser's `http://<lan-ip>:3000` Origin. Set `CORS_ORIGINS=http://<lan-ip>:3000`.
- Using the UI by name (MagicDNS or a LAN hostname): set `CORS_ORIGINS=http://<name>:3001`.
  `ALLOWED_HOSTS` alone lets GET requests through, but writes from that page get 403.
  `CORS_ORIGINS` also adds the name to the `Host` allowlist, so one setting covers both.
- Using the UI by Tailscale or LAN IP needs nothing beyond `SERVER_HOST`.

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
| `CUDA_VISIBLE_DEVICES` | empty (Auto) | Environment, Restart | GPU for the engine and for `ace-train` (`services/gpuDevices.ts`). Auto leaves the choice to the engine when `nvidia-smi` lists fewer than two GPUs; with two or more it passes the UUID of the one with the most VRAM, ties going to the lower index. The Settings picker writes a GPU UUID, passed as given. `all` (any case) leaves `CUDA_VISIBLE_DEVICES` unset for every child, so all GPUs stay visible. An index-only value is translated to UUIDs through `nvidia-smi`; any other value is passed through as given, and the server sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` (unless you set it yourself) so indices match `nvidia-smi`. The child environment is rebuilt from the live `config` value on every spawn, so a training job started after saving uses the new GPU. A YuE2 joint training job can name its own GPU UUID, which overrides this for that job |
| `ACESTEPCPP_WORKER_GPUS` | empty | Environment, Restart | Experimental multi-GPU opt-in (GH #13): comma-separated `GPU-` or `MIG-` UUIDs, one `ace-server` per card. Empty keeps the single engine. UUIDs that `nvidia-smi` does not report, and duplicates, are dropped; if none are left the single engine runs. With the opt-in set, `w0` is pinned to the first UUID on the configured port and `w1`.. take the next free ports, logging to `ace_engine.wN.log`. Each model must fit in one card; nothing is split across cards. With two or more cards listed, jobs run concurrently, one per worker; training, post-processing and previews hold every worker. See [Multi-GPU workers](engine.md#multi-gpu-workers-experimental) |
| `TENSORRT_LIBS` | In a release, `<data folder>/tools/runtime`. Otherwise `engine/deps/tensorrt_libs` if it contains `nvinfer_10.dll` or `libnvinfer.so.10`, else empty | env only | Prepended to the engine's `PATH` so TensorRT can load. If a `trtllm-libs` folder exists two levels above the engine binary, it is prepended too |

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
| `--user-plugins <dir>` | The data folder's `plugins/` | There is a data folder (a release, or a checkout that Settings gave one). The engine scans it after the shipped plugins |
| `--keep-loaded` | `ACESTEPCPP_KEEP_LOADED` | Not `0` |
| `--noise-profile <wav>` | `ACESTEPCPP_NOISE_PROFILE` | File exists |
| `--draft-lm <gguf>` | `ACESTEPCPP_DRAFT_LM` | File exists |
| `--vae-chunk <n>` | `ACESTEPCPP_VAE_CHUNK` | Non-zero |
| `--vae-overlap <n>` | `ACESTEPCPP_VAE_OVERLAP` | Non-zero |

The engine accepts more flags than these. [engine.md](engine.md) covers the engine's own
command line.

## LLM providers

Lyric Studio, Insta-Gen's LLM path, the Assistant and dataset captioning use these. All of
them are hot-reloaded when saved from Settings.

| Variable | Default | Settings | Effect |
|---|---|---|---|
| `DEFAULT_LLM_PROVIDER` | `gemini` | AI Services | Provider used when a request does not name one |
| `LLM_TIMEOUT_MS` | `300000` | AI Services | Completion timeout, every provider. Streaming calls allow this long between response or reasoning tokens; non-streaming calls have a total time limit. The same timer covers headers and body without a separate five-minute HTTP cap. Values below 10000 are raised to 10000 |
| `GENIUS_ACCESS_TOKEN` | empty | AI Services | Genius API token for lyric lookup |
| `GEMINI_API_KEY` | empty | AI Services | Google Gemini key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | AI Services | Gemini model |
| `OPENAI_API_KEY` | empty | AI Services | OpenAI key |
| `OPENAI_MODEL` | `gpt-4o-mini` | AI Services | OpenAI model |
| `ANTHROPIC_API_KEY` | empty | AI Services | Anthropic key |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | AI Services | Anthropic model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | AI Services | Ollama server |
| `OLLAMA_MODEL` | `llama3` | AI Services | Ollama model |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | AI Services | LM Studio server |
| `LMSTUDIO_MODEL` | empty | AI Services | LM Studio model |
| `LMSTUDIO_API_KEY` | empty | AI Services | Bearer token, for an LM Studio server with authentication on. Empty sends no token |
| `UNSLOTH_BASE_URL` | `http://127.0.0.1:8888` | AI Services | Unsloth server |
| `UNSLOTH_USERNAME` | empty | AI Services | Unsloth login |
| `UNSLOTH_PASSWORD` | empty | AI Services | Unsloth password |
| `UNSLOTH_KEYLESS` | empty | AI Services | Set to `1` when a local Studio runs with its "keyless API access" setting on, so no username or password is needed. Any other value is off. The provider stays hidden until this or both credentials are set; credentials win when both are present |
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

## Training workers

Exposed in Settings > Environment > Server and hot-reloaded. Code: `server/src/services/training/trainingWorkers.ts`, routes in `server/src/routes/workers.ts`.

| Variable | Default | Effect |
|---|---|---|
| `TRAINING_WORKERS` | empty | Workers this machine drives: `Name=http://host:3001`, comma-separated. Enables `/api/workers` and the studio's Train on / Run on controls |
| `TRAINING_WORKER_TOKEN` | empty | Sent as `x-hotstep-worker-token` on every call to a worker |
| `WORKER_TOKEN` | empty | Set = this machine is a worker: every `/api` request from a non-loopback address must carry this token (401 otherwise) |

The server binds `127.0.0.1` by default, so a worker must also set `SERVER_HOST=0.0.0.0` or the controller cannot reach it. Set `WORKER_TOKEN` too. The server logs a warning at startup when `WORKER_TOKEN` is set but it is bound to loopback.

A worker keeps pushed datasets under `<TRAINING_DIR>/worker-datasets/<slug>`, with the controller's dataset id and slug. Every machine records the last linked YuE2 pair per slug in `<TRAINING_DIR>/yue2-linked.json`; the controller's pull reads the worker's copy.

The Workers panel calls `GET /api/workers`, which requests each worker's token-gated `GET /api/training/worker/status`. The controller compares the reported startup commit with its own Git HEAD using ancestry, and reports current, behind with commit count, or diverged. `POST /api/workers/:name/update` creates a bundle from the worker commit to local master and uploads it to the worker's token-gated `POST /api/training/worker/update`. The worker checks for queued and running jobs before the reset, then discards tracked checkout changes with submodule recursion disabled. Only when the bundle changes the recorded ggml commit does it sync the submodule URL, check out the committed HOT-ggml fork revision, and require hook verification to pass before the engine build. The checkout replaces tracked changes in `engine/ggml`; files committed in the fork remain in place. It installs dependencies when lockfiles changed, builds the UI, conditionally rebuilds the engine, and writes the restart marker last: `RESTART_MARKER_PATH` (`<root>/.restart-requested` in a checkout), the same file the restart loop (`server/restart-loop.cmd`, `dev.sh` on macOS) consumes. On macOS the updater checks hooks with `pwsh -NoProfile -File engine/verify-hooks.ps1`, runs plain `npm`, and builds with `bash engine/build-mac.sh`; Windows commands are unchanged. `GET` and `DELETE /api/workers/:name/update` read progress or cancel during preparation. A worker without these endpoints needs a one-time bootstrap. Before an engine rebuild the updater stops ace-server, which suspends the engine supervisor so it does not restart the engine mid-build. After the reset, a failed build keeps training writes returning 409 (reads still pass) until a person repairs the checkout; a failure before the reset releases the gate. The first migration to the fork requires a manual bootstrap while the worker is idle and the app and engine supervisor are stopped.

## Other variables

| Variable | Default | Read in | Effect |
|---|---|---|---|
| `HOT_STEP_ROOT` | unset | `config.ts` | Project root. Setting it turns on portable mode, which among other things downloads the CUDA runtime DLLs on a Windows CUDA build's first launch. Set by the portable launchers `release/HOT-Step.bat` and `release/HOT-Step.sh`, and by the stable launcher to the selected `app/<id>` |
| `HOT_STEP_DATA_ROOT` | unset | `dataroot/bootstrap.ts` | Use this data folder for this process (release and dev; it must exist), ignoring `data-location.json`. A scheduled move does not run while it is set |
| `HOT_STEP_PROFILE_DIR` | unset | `dataroot/paths.ts` | Replaces the profile folder that holds `data-location.json` and the dev pointers. Meant for test fixtures; the stable launcher also pins it |
| `HOT_STEP_INSTALL_ROOT` | unset | `routes/updates.ts` | Set by the stable launcher (`release/updater/launcher.mjs`) to its install folder, the parent of `app/<id>/`. Updater state lives in its `updater/` folder |
| `HOT_STEP_UPDATER_DIR` | unset | `services/updates/maintenance.ts` | Set by the stable launcher to `<install>/updater`. Startup reads the maintenance lease and `transaction.json` there before opening SQLite |
| `HOT_STEP_MAINTENANCE_TOKEN` | unset | `services/updates/maintenance.ts` | Set by the stable launcher only for an update's candidate (and for a restart after a crash past commit). Without it a process refuses to start while a lease exists; with it, `/api/health`, `/api/maintenance` and `/api/shutdown` pass the closed gate |
| `HOT_STEP_LAUNCHER_PROTOCOL` | unset | `config.ts` | Set by the stable launcher. Read only with an IPC channel and `HOT_STEP_INSTALL_ROOT`; apply needs it to equal `updaterProtocol` in `build-info.json` |
| `HOT_STEP_DEV` | unset | `routes/shutdown.ts` | Set to `1` by `dev.bat` and `dev.sh`. On Windows it lets `POST /api/shutdown` kill the Vite process on port 3000 after checking it is Node running Vite. It also gates `HOTSTEP_GENERATE_CAPTURE` and `HOTSTEP_YUE2_COVER_FORCE_FALLBACK` |
| `HOTSTEP_GENERATE_CAPTURE` | unset | `services/generation/requestCapture.ts` | Dev only, honoured only when `HOT_STEP_DEV` is set. `record` saves every `POST /api/generate` body as a fixture under `<data dir>/dev-captures/generate/`, then generates as normal; `capture-only` saves it and queues nothing. See [Generation API contracts](api-contracts.md#fixture-capture-dev-only) |
| `HOTSTEP_YUE2_COVER_FORCE_FALLBACK` | unset | `services/training/yue2ArTrainRunner.ts` | Dev only: `1`, with `HOT_STEP_DEV=1`, sends every Cover Studio YuE2 lead-sheet transcription down its fallback path, which waits for the GPU lane and frees the engine for the transcriber. Tests that path on a card where both fit |
| `GENERATION_INTENT_PATH` | `resolved` | `routes/resolve.ts` | `old` is the only rollback, keeping the legacy client builder; unset or any other value sends Create and written-song requests through Node intent resolution. The UI reads it from `GET /api/resolve/path`. Read from the process environment, so a `.env` change needs a restart |
| `PP_STEM_CACHE_GB` | `4` | `services/generation/stemCache.ts` | Disk budget for cached vocal and instrumental stems that the post-processing chain reuses when it runs again on the same audio. `0` disables the cache |
| `HOTSTEP_DAW_KEY` | unset | `services/daw/dawFacade.ts` | Shared secret for the `/api/v1/daw` facade. Set = every caller, loopback and Vite-proxied included, must send it as `X-DAW-Key`. Unset = only demonstrably local callers pass: the socket and every `X-Forwarded-For`, `X-Real-IP` and `Forwarded` hop must be loopback. Uploads and float32 take exports live in `<DATA_DIR>/daw_tmp/<id>/`, swept every 15 min (6 h TTL, 10 min for failed or cancelled jobs) |
| `DOCKER_PATH_MAP` | unset | `services/pathMapper.ts` | JSON object mapping Windows path prefixes to container mount points, so presets saved on Windows work in Docker. Set in `.env.docker` |

## Other env files

- `.env.docker` is copied to `/app/.env` by the Dockerfile's default `runtime` target and used by
  `docker-compose.yml`. It points the engine, models and adapters at container paths.
- `tools/discord-claude/.env.example` configures the Discord bridge tool. The app does not
  read it.

The `runpod` Docker target uses container environment defaults and the persistent
`/workspace` volume instead of the local Compose path mappings. Its entrypoint
uses `HOT_STEP_DATA_ROOT=/workspace` to link models, adapters, data and logs into
that volume. This container guard also works without `HOT_STEP_ROOT`; the
release-only data-root selection described above is a separate server mechanism.
Keep `ACESTEPCPP_MODELS`, `ACESTEPCPP_ADAPTERS` and `DATA_DIR` aligned if you
override the volume path. See the [RunPod guide](runpod.md) for the layout and
the unauthenticated HTTP proxy warning.

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

### Release template and launcher contract

Packagers use `release/.env.example`, leaving managed paths unset. The root `.env.example` is for dev and keeps `DATA_DIR=./server/data` and its lyric-export override. Do not copy that template into a release. `release/profile-path.mjs` mirrors `profileDir()` and `RESTART_MARKER_FILE`; keep that contract synchronized. Relative `HOT_STEP_PROFILE_DIR` overrides resolve against the launch process cwd. Ordinary data discovery uses absolute `HOT_STEP_ROOT` and the profile pointer. See [Data folder and upgrades](../user/data-folder.md).
