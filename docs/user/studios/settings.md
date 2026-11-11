# Settings

Central configuration hub for the engine, the server, LLM providers, downloads, storage cleanup, and performance trade-offs. Reach for it to point HOT-Step at a different models folder, pick a GPU, add an API key for an external LLM provider, change the display language or theme, or clear out generated data. Most values live in the `.env` file (`settings/.env` in the data folder, or the checkout root in a development checkout without one) and this page edits it directly, with a full explanation of each key in [Configuration](../../dev/config.md).

![Settings](../../images/studio-settings.webp)
<!-- screenshot: needed -->

## Where to find it

The "Settings" entry in the sidebar, below Training. The page title shows the app version (`dev` in a source checkout).

A few related controls live outside the Settings page itself, directly in the sidebar or in the right-hand activity column:

- **Theme** (light/dark) is a sun/moon button in the sidebar, just below Settings.
- **Restart** and **Quit** are their own buttons at the bottom of the sidebar.
- The **Terminal** (engine log viewer) is a collapsible section of the right-hand activity column, alongside the Playlist / Recent Songs box and the generation Queue, present on every screen of the wide layout rather than inside Settings.
- The **VST chain** (post-processing plugin chain) opens from the global parameter bar's post-processing area, not from Settings.

See [Terminal](#terminal), [Theme, restart, and shutdown](#theme-restart-and-shutdown), and [VST chain](#vst-chain) below.

## Workflow

1. Open Settings and pick a tab: General, Environment, AI Services, Storage & Data, Performance, Updates, or Report an issue.
2. On the Environment tab, edit any field, then click **Save Environment** (or **Save Changes** on the AI Services tab, which shares the same save bar). Only changed keys are written to `.env`.
3. If a changed key needs a restart to take effect, the save bar shows which one and a **Restart now** button appears. Click it to restart the server and engine; the UI reconnects on its own once they are back.
4. General, Performance, and language/theme changes apply immediately and do not need a restart.

A small reset arrow appears beside a General or Performance setting whenever its value has moved
from the default; click it to snap that one field back. The reset persists like any other change;
it is not a page reload. The Environment and AI Services fields have no reset arrow.

Short controls share a row on wide screens and stack as the Settings area narrows. Paths and credentials keep a full row.

## Settings and Lyric Studio backup

The Storage & Data tab can export a versioned JSON backup of application settings and Lyric Studio data. Import validates the file, shows replacement counts, and asks before overwriting. Media files and environment settings are excluded. See [Settings and Lyric Studio backup](../settings-backup.md).

## Controls

### General tab

| Control | What it does |
|---|---|
| Display Language | Sets the application interface language: English, Russian, Chinese, Japanese or Korean. Applies immediately. |
| Generic artwork | Which set of stand-in cover images a track shows until it has one of its own, Cyberpunk (the default) or Scenic; each track keeps the same image within a set. |
| "Disco Mode": Extract drum stems for visualisations | After generation, extracts kick, snare, and hi-hat stems to drive the beat-reactive visual effects (rainbow borders, snare flash, hi-hat particles). Adds roughly 60 to 90 seconds per song. |
| Default format / MP3 bitrate / Opus bitrate | Preferred audio format for downloads (WAV, FLAC, Opus, or MP3; FLAC by default), and the bitrate used for the two lossy formats: 128 to 320 kbps for MP3 and 96 to 256 kbps for Opus, 192 kbps by default for both. |
| Download version | Whether downloads pull the mastered version when one exists (the default), the original only, or both. |
| Include latent file | Also downloads the `.hslat` latent file alongside each track, which can be reused for covers and remixing. |
| Update adapter references | Checks Lyric Studio album presets and clears references to deleted adapter weights. Lists each changed preset and removed adapter. Existing adapters, scales, and reference tracks are kept. Runs only when clicked. |
| Use filename as trigger word | Auto-injects the adapter's filename (minus the `.safetensors` extension) into the style description at generation time. |
| Trigger word placement | Only shown when the trigger toggle above is on. Prepend, Append, or Replace, controlling where the trigger word lands relative to your style description. |

### Updates tab

The Updates tab holds build information and update checks. A small dot appears on its tab label only when an update is available. General stays in place while the status loads.

| Control | What it does |
|---|---|
| Updates | Shows the installed build (release tag or source commit, plus the build variant), a Release / Master selector, the time of the last successful check and **Check now**. When something newer exists a banner links to its GitHub release notes or commit. Changing the selector only changes which feed is checked, never your installation. It is a notice only: nothing is downloaded or installed. A copy with no build record, a source checkout with uncommitted changes, and a failed check each get a plain message, and a failed check is never shown as "up to date". At startup the app checks again only when the last successful check is more than a day old; **Check now** asks GitHub at most once a minute per feed and shows the saved result in between. |

### Environment tab

This tab is a structured editor for `.env`, in four collapsible groups: Engine, Server, Dataset Labeling and Paths. Fields marked "Restart required" below need the app to restart before the new value takes effect; only some of them carry a "Restart" badge in the UI. The save bar's **Restart now** button (see [Workflow](#workflow)) does that without leaving the page. Every field here has a matching entry in [Configuration](../../dev/config.md), which also lists the keys this tab does not expose.

| Control | What it does |
|---|---|
| Models / Adapters directory | Where the engine looks for model and adapter files. Restart required. |
| Engine port / host | Address `ace-server` binds to. Restart required. Defaults to `8085` / `127.0.0.1`. |
| VAE chunk size / overlap | Tuning knobs for the VAE decoder's chunked decode, useful for reducing memory use on Vulkan or low-VRAM GPUs. Restart required. Defaults to `1024` / `64`. |
| GPU Device | Which GPU the engine uses. Lists detected NVIDIA cards by name and VRAM; "Auto (most VRAM)" picks the card with the most VRAM. Restart required. Stores a stable GPU UUID rather than an index, see [Tips and limits](#tips-and-limits). Falls back to a plain text field when no GPU is detected (AMD, Intel, Apple, or CPU-only). |
| Multi-GPU workers | Experimental, off by default. Shown only when two or more NVIDIA cards are detected. Ticking cards starts one engine per ticked card instead of one engine in total. With two or more cards ticked, jobs run in parallel, one per card; training, post-processing and previews still take every card. Each model must fit in a single card's VRAM; a model is never split across cards, so a 24 GB card still cannot run something that needs 30 GB. CUDA only: Vulkan, CPU and Apple Silicon keep the single engine. Leave all unticked for the normal behaviour, where the GPU Device pick above applies. Restart required. |
| Engine workers | Read-only status, shown once more than one engine is running: worker name, card, port, ready or its status text, and "busy" while a job runs on it. With two or more workers, jobs run in parallel, one per worker. |
| Keep Models in VRAM | Passes `--keep-loaded` to the engine so the DiT, adapter, and VAE stay resident between generations instead of being reloaded each time. Cuts the roughly 17-second adapter precompute paid per render, at the cost of holding around 13 GB of VRAM continuously. Off by default. Restart required. This is a different control from the "Keep DiT & VAE loaded" toggle on the Performance tab, see [Tips and limits](#tips-and-limits). |
| Server port | Port the Node server listens on. Restart required. Default `3001`. |
| Data directory | Where the SQLite database and generated assets are stored. Restart required. Empty uses `library` in the data folder; a development checkout without a data folder uses `server/data`. A relative value resolves against the data folder when there is one. |
| Training workers / Training worker token | Other PCs this one can train YuE2 batches on, as `Name=http://address:3001`, comma-separated, and the token sent to them. Applies live. See [training on another PC](../training/yue2.md#training-on-another-pc). |
| Worker token (this PC as a worker) | Set only on a PC that trains for another. API calls from other machines must then carry this token; the PC itself and a browser on it are not affected. Applies live. |
| Dataset labeling throughput (Essentia / Genius / caption concurrency and intervals) | Rate limits for the background dataset-labeling queue used by Training Studio. Defaults: Essentia concurrency 2, Genius concurrency 1 with a 400 ms minimum interval, AI caption concurrency 1 with a 250 ms minimum interval. Applies live, no restart. |
| Download folder | Absolute folder that downloaded tracks are saved into, with a toast showing where each one went. Empty (the default) keeps normal browser downloads, which also stay the route for the library's zip export. The folder is checked when you save, and an existing file is never overwritten (the new one becomes `name (2).ext`). Applies live. |
| Lyrics export directory | Where Lyric Studio exports lyric files. Applies live. |
| MuScriptor models directory | Where MIDI Studio's transcription models live. Empty keeps the default, `models/muscriptor` under the data directory. Restart required. |

### AI Services tab

| Control | What it does |
|---|---|
| API Keys (Genius, Gemini, OpenAI, Anthropic) | Masked password fields for the access tokens external features use: Genius for lyric lookups, and the three LLM providers for anything that calls an external model. Applies live; a green save confirms the key is in use. |
| Default LLM provider | The chat provider the server uses when a request does not name one: Gemini (the default), OpenAI, Anthropic, Ollama, LM Studio, llama.cpp, Unsloth, or a generic OpenAI-compatible endpoint, listed under its display name. Training Studio's dataset captioning and labeling fall back to it, and so does the scene an LLM writes for automatic cover art. Lyric Studio and Auto-Gen have provider pickers of their own. |
| LLM call timeout (ms) | How long one call to any LLM provider may take before it is abandoned, in milliseconds. Raise it for a local model that shares the GPU with generation. Default 300000 (5 minutes); values below 10000 count as 10000. |
| Provider model overrides | One text field per provider (Gemini, OpenAI, Anthropic, Ollama, LM Studio, Unsloth, llama.cpp, OpenAI-compatible) naming which model to request. Leave blank to use the provider's built-in default. |
| Provider endpoints and credentials | For the self-hosted providers (Ollama, LM Studio, Unsloth, llama.cpp) and the generic OpenAI-compatible slot: a base URL, plus an API key, username/password, or display name where that provider needs one. Ollama and llama.cpp need only a URL; LM Studio and the OpenAI-compatible slot also take an API key; Unsloth takes a username and password, or the **Keyless local Studio** toggle for an Unsloth Studio with keyless API access turned on. Unsloth counts as unavailable, and stays out of the provider pickers, until that toggle is on or both credentials are set. |

With an Anthropic API key set, Claude model choices come from Anthropic's current model list. If that list cannot be reached, the picker shows a built-in list. The default Claude model is Haiku 4.5 unless you set `ANTHROPIC_MODEL`.

### Performance tab

| Control | What it does |
|---|---|
| Parallel Whisper Transcription | Runs lyrics transcription alongside post-processing instead of after it. CPU-only, no VRAM cost. Saves roughly 6 seconds per generation. |
| Parallel Quality Evaluation | Runs audio quality analysis alongside other pipeline stages. CPU-only. Saves roughly 2 seconds. |
| Parallel Cover Art | Starts cover art generation during post-processing instead of waiting for it to finish. Uses the GPU (Flux), so it may need VRAM headroom. Saves roughly 5 seconds. |
| Keep DiT & VAE loaded | Keeps both the DiT diffusion model and VAE decoder in VRAM at once instead of swapping them between stages. Eliminates roughly 13 seconds of load/unload per generation; uses about 8.2 GB more VRAM for an XL model. Applies per generation request, no restart. This is a different control from "Keep Models in VRAM" on the Environment tab, see [Tips and limits](#tips-and-limits). |
| Cache LM audio codes | When a generation repeats the same seed and parameters, reuses the previously computed audio codes instead of re-running the LM. Saves roughly 12 seconds on repeat generations. On by default; every other toggle on this tab starts off. |
| Generation Timeout | Longest the app waits for a single generation before marking it timed out, from 10 minutes up to 6 hours, 30 minutes by default. This budget is per track, not per batch: a 9-track batch spends it nine times over, once per track as the engine picks each one up. Increase it for high step counts or slower hardware. |

### Report an issue tab

| Control | What it does |
|---|---|
| Title / What happened | Your summary and description of the problem. |
| Exact text that will be sent | A live preview of the whole report: your description, system info (app and build version, OS, GPU, compute backend, engine, model file names), the tail of the server and engine logs, and the last failed job. Home folder, user name, IP addresses, email addresses and anything that looks like a key or token are masked in the logs. Read it before you send. |
| Send to GitHub | Opens a prefilled new-issue page in your browser, tagged `in-app`. You submit it from your own GitHub account; nothing is sent from the app. The link is kept under about 7,500 characters, so the oldest log lines are dropped first and a notice says so. |
| Copy full logs | Copies the report with the full log buffers (still masked) so you can paste or attach it to the issue. |

### Storage & Data tab

The **Data folder** panel shows the active data folder and any external locations
outside it. A development checkout keeps its own folders until you schedule a change
here, and can move them to a data folder or use an existing one like a release. A
process `HOT_STEP_DATA_ROOT` override must be cleared before changing the saved location.

Choose an action, enter an absolute folder path, and click **Preview change**:

- **Move current data** transfers managed data to an empty destination. Across
  drives, the copy is verified before the source is removed.
- **Use existing folder** switches to a compatible data folder without copying,
  deleting or merging libraries.
- **Create empty data folder** starts a new library and leaves the current data
  in place.
- **Import previous install** copies data from an older app folder into the current
  empty library, preserving the previous install. Enter the old app folder rather
  than its data subfolder.

The preview shows source and destination paths, transfer size, and free-space needs
for a transfer across drives. External locations remain in place. Running or queued
generation, training and other work, or active downloads, prevent scheduling.
**Confirm and restart** saves the pending operation outside the app folder and
requests a graceful restart. Transfers run at startup before the library opens;
progress appears in the console. The saved location is reused by a fresh unzip.

If the restart request fails, the pending change stays saved. The panel offers
**Restart to apply** and **Cancel scheduled change**. Cancellation is only possible
before the transfer starts. An interrupted transfer resumes on restart; a failed
transfer shows an explanatory page. If only the saved folder is missing or inaccessible,
startup warns in the launch console and uses `HOT-Step-Data` beside the app, or the
original checkout folders in dev. This may show an empty library. The saved location
is kept; reconnect the drive and restart to use it again.

| Control | What it does |
|---|---|
| Extracted Stems | Shows how many stem extractions, individual stems, and total disk space are stored, with a confirm-to-clear button that deletes all of them. |
| Nuke Generations | Permanently deletes all generated audio: songs from the library, audio files on disk, and Lyric Studio's audio-generation entries. Lyrics, profiles, and artist data are untouched. Requires a confirm click. |
| Nuke Written Lyrics | Permanently deletes all generated or written lyrics across every artist. Profiles and source lyrics are untouched. Requires a confirm click. |
| Nuke Profiles | Permanently deletes all artist profiles and the lyrics that depend on them. Source lyrics and artist data are untouched. Requires a confirm click. |

## Theme, restart, and shutdown

The theme toggle in the sidebar switches the whole app between dark and light mode immediately; the choice persists across restarts.

Restart and Quit each open a confirmation dialog before doing anything. Restart stops and relaunches the server and engine, and the UI reconnects on its own once they are back. Quit stops the engine and both servers and shows a screen titled "HOT-Step CPP has shut down" with the message "You may now close this browser tab." Nothing reconnects automatically after Quit.

## Terminal

A live view of `ace-server`'s log output, with a VRAM usage badge, three stream tabs (Logs, Model Output, System) that filter the same feed by prefix, a search box that highlights matches, a pin/jump-to-bottom control for following new lines as they arrive, and a clear button. Hovering the VRAM badge lists the models resident on the GPU, each with an unload button; a model in use cannot be unloaded. It is a section of the right-hand activity column rather than part of the Settings page, see [Where to find it](#where-to-find-it). For reading it during a failed generation, see [Troubleshooting](../troubleshooting.md).

## VST chain

The post-processing VST3 chain is the **VST Chain** section inside the global parameter bar's **Post-Processing** dropdown, not part of the Settings page. **Add Plugin** opens a searchable list of installed VST3 plugins, scanned when the list opens, with a **Rescan** button. The chain lets you remove, reorder, and enable or disable plugins, launch a plugin's native GUI to edit its parameters, save and load named chain presets (stored by the server with your other presets, so every browser that opens the app sees them), and monitor live playback of the track in the player through the chain with **Monitor with VST Chain** before committing it to a render.

## Tips and limits

Two different "keep loaded" controls exist and they are not the same setting. The Environment tab's "Keep Models in VRAM" is a spawn-time engine flag (`--keep-loaded`) that needs a restart. The Performance tab's "Keep DiT & VAE loaded" is a per-request flag the UI sends with each generation and takes effect immediately. Either can be left off if VRAM is tight.

The GPU Device picker stores a GPU UUID, not the index shown next to it. `nvidia-smi` and CUDA can enumerate multiple GPUs in different orders, so an index-based pick has historically selected the wrong card on mixed-GPU machines; UUIDs remove that ambiguity. A `.env` written before this fix, or a hand-edited one, still works: the picker resolves an old index-shaped value onto the matching card and re-saves it as a UUID next time you touch the field.

Warm-on-startup, which pre-loads a configured DiT, VAE, and adapter right after the engine boots so the first generation skips the cold start, is not currently exposed on this page. It is `.env`-only and requires "Keep Models in VRAM" to be on as well; see [Configuration](../../dev/config.md).

There is no password to set. HOT-Step runs as a single local user with automatic sign-in, so nothing under Settings configures authentication.

Saving the Environment or AI Services tab only writes keys you changed; unmodified fields are left alone even if they are showing a resolved default rather than something explicitly set in `.env`.

## Related

- [Model Manager](model-manager.md)
- [Getting started](../getting-started.md)
- [Troubleshooting](../troubleshooting.md)
- [Configuration](../../dev/config.md)
- [Lyric Studio](lyric-studio.md)

For upgrade, layout and recovery instructions, see [Data folder and upgrades](../data-folder.md).
Service keys live in the data folder's `settings/.env` and the MuScriptor token in `library/muscriptor/hf_token`, so a new version that reuses the folder keeps them. The JSON export in the backup panel does not include either; see [Settings and Lyric Studio backup](../settings-backup.md).
