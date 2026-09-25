# Settings

Central configuration hub for the engine, the server, LLM providers, downloads, storage cleanup, and performance trade-offs. Reach for it to point HOT-Step at a different models folder, pick a GPU, add an API key for an external LLM provider, change the display language or theme, or clear out generated data. Most values live in the project's `.env` file and this page edits it directly, with a full explanation of each key in [Configuration](../../dev/config.md).

![Settings](../../images/studio-settings.webp)
<!-- screenshot: needed -->

## Where to find it

The "Settings" entry in the sidebar, below Training Studio.

A few related controls live outside the Settings page itself, directly in the sidebar or in the right-hand activity column:

- **Theme** (light/dark) is a sun/moon button in the sidebar, just below Settings.
- **Restart** and **Quit** are their own buttons at the bottom of the sidebar.
- The **Terminal** (engine log viewer) is a collapsible section of the right-hand activity column, alongside Recent Songs and the generation Queue, present on every screen rather than inside Settings.
- The **VST chain** (post-processing plugin chain) opens from the global parameter bar's post-processing area, not from Settings.

See [Terminal](#terminal), [Theme, restart, and shutdown](#theme-restart-and-shutdown), and [VST chain](#vst-chain) below.

## Workflow

1. Open Settings and pick a tab: General, Environment, AI Services, Performance, or Storage & Data.
2. On the Environment tab, edit any field, then click **Save Environment** (or **Save Changes** on the AI Services tab, which shares the same save bar). Only changed keys are written to `.env`.
3. If a changed key needs a restart to take effect, the save bar shows which one and a **Restart now** button appears. Click it to restart the server and engine; the UI reconnects on its own once they are back.
4. General, Performance, and language/theme changes apply immediately and do not need a restart.

## Controls

### General tab

| Control | What it does |
|---|---|
| Display Language | Sets the application interface language. Applies immediately. |
| Generic artwork | Which set of stand-in cover images a track shows until it has one of its own; each track keeps the same image within a set. |
| "Disco Mode": Extract drum stems for visualisations | After generation, extracts kick, snare, and hi-hat stems to drive the beat-reactive visual effects (rainbow borders, snare flash, hi-hat particles). Adds roughly 60 to 90 seconds per song. |
| Default format / MP3 bitrate / Opus bitrate | Preferred audio format for downloads (WAV, FLAC, Opus, or MP3), and the bitrate used for the two lossy formats. |
| Download version | Whether downloads pull the mastered version when one exists, the original only, or both. |
| Include latent file | Also downloads the `.hslat` latent file alongside each track, which can be reused for covers and remixing. |
| Use filename as trigger word | Auto-injects the adapter's filename (minus the `.safetensors` extension) into the style description at generation time. |
| Trigger word placement | Only shown when the trigger toggle above is on. Prepend, Append, or Replace, controlling where the trigger word lands relative to your style description. |

### Environment tab

This tab is a structured editor for `.env`. Fields marked "Restart" in the UI need the app to restart before the new value takes effect; the save bar's **Restart now** button (see [Workflow](#workflow)) does that without leaving the page. Every field here has a matching entry in [Configuration](../../dev/config.md), which also lists the keys this tab does not expose.

| Control | What it does |
|---|---|
| Models / Adapters directory | Where the engine looks for model and adapter files. Restart required. |
| Engine port / host | Address `ace-server` binds to. Restart required. Defaults to `8085` / `127.0.0.1`. |
| VAE chunk size / overlap | Tuning knobs for the VAE decoder's chunked decode, useful for reducing memory use on Vulkan or low-VRAM GPUs. Restart required. Defaults to `1024` / `64`. |
| GPU Device | Which GPU the engine uses. Lists detected NVIDIA cards by name and VRAM; "Auto" picks the card with the most VRAM. Restart required. Stores a stable GPU UUID rather than an index, see [Tips and limits](#tips-and-limits). Falls back to a plain text field when no GPU is detected (AMD, Intel, Apple, or CPU-only). |
| Keep Models in VRAM | Passes `--keep-loaded` to the engine so the DiT, adapter, and VAE stay resident between generations instead of being reloaded each time. Cuts the roughly 17-second adapter precompute paid per render, at the cost of holding around 13 GB of VRAM continuously. Off by default. Restart required. This is a different control from the "Keep DiT & VAE loaded" toggle on the Performance tab, see [Tips and limits](#tips-and-limits). |
| Server port | Port the Node server listens on. Restart required. Default `3001`. |
| Data directory | Where the SQLite database and generated assets are stored. Restart required. Default `./data`. |
| Dataset labeling throughput (Essentia / Genius / caption concurrency and intervals) | Rate limits for the background dataset-labeling queue used by Training Studio. Applies live, no restart. |
| Lyrics export directory | Where Lyric Studio exports lyric files. Applies live. |
| MuScriptor models directory | Where MIDI Studio's transcription models live. Restart required. |

### AI Services tab

| Control | What it does |
|---|---|
| API Keys (Genius, Gemini, OpenAI, Anthropic) | Masked password fields for the access tokens external features use: Genius for lyric lookups, and the three LLM providers for anything that calls an external model. Applies live; a green save confirms the key is in use. |
| Default Provider | Which LLM backend Lyric Studio and Auto-Gen's "Lyrics + AI" mode use unless you pick a different one in that studio: Gemini, OpenAI, Anthropic, Ollama, LM Studio, llama.cpp, Unsloth, or a generic OpenAI-compatible endpoint. |
| LLM Timeout | How long the server waits for an LLM response before giving up, in milliseconds. Default 300000 (5 minutes). |
| Provider model overrides | One text field per provider (Gemini, OpenAI, Anthropic, Ollama, LM Studio, Unsloth, llama.cpp, OpenAI-compatible) naming which model to request. Leave blank to use the provider's built-in default. |
| Provider endpoints and credentials | For the self-hosted providers (Ollama, LM Studio, Unsloth, llama.cpp) and the generic OpenAI-compatible slot: a base URL, plus an API key, username/password, or display name where that provider needs one. Ollama and llama.cpp need only a URL; LM Studio and the OpenAI-compatible slot also take an API key; Unsloth takes a username and password. |

### Performance tab

| Control | What it does |
|---|---|
| Parallel Whisper Transcription | Runs lyrics transcription alongside post-processing instead of after it. CPU-only, no VRAM cost. Saves roughly 6 seconds per generation. |
| Parallel Quality Evaluation | Runs audio quality analysis alongside other pipeline stages. CPU-only. Saves roughly 2 seconds. |
| Parallel Cover Art | Starts cover art generation during post-processing instead of waiting for it to finish. Uses the GPU (Flux), so it may need VRAM headroom. Saves roughly 5 seconds. |
| Keep DiT & VAE loaded | Keeps both the DiT diffusion model and VAE decoder in VRAM at once instead of swapping them between stages. Eliminates roughly 13 seconds of load/unload per generation; uses about 8.2 GB more VRAM for an XL model. Applies per generation request, no restart. This is a different control from "Keep Models in VRAM" on the Environment tab, see [Tips and limits](#tips-and-limits). |
| Cache LM audio codes | When a generation repeats the same seed and parameters, reuses the previously computed audio codes instead of re-running the LM. Saves roughly 12 seconds on repeat generations. |
| Generation Timeout | Longest the app waits for a single generation before marking it timed out, from 10 minutes up to 6 hours. This budget is per track, not per batch: a 9-track batch spends it nine times over, once per track as the engine picks each one up. Increase it for high step counts or slower hardware. |

### Storage & Data tab

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

A live view of `ace-server`'s log output, with a VRAM usage badge, three stream tabs (Logs, Model Output, System) that filter the same feed by prefix, a search box that highlights matches, and a pin/jump-to-bottom control for following new lines as they arrive. It is a section of the right-hand activity column rather than part of the Settings page, see [Where to find it](#where-to-find-it). For reading it during a failed generation, see [Troubleshooting](../troubleshooting.md).

## VST chain

The post-processing VST3 chain, opened from the global parameter bar, not from the Settings page. It scans for installed VST3 plugins, lets you add, remove, reorder, and enable or disable them, launch a plugin's native GUI to edit its parameters, save and load named chain presets, and monitor live playback of a track through the chain before committing it to a render.

<!-- TODO(verify): confirm the exact label and icon the post-processing area uses to open the VST chain panel, and whether it needs a plugin scan to complete before first use. -->

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
