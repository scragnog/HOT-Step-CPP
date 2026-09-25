# Troubleshooting

Almost every problem in HOT-Step CPP leaves a trace in its log files. This page shows where they
are, then walks through the failures people actually report, each with the log line that
identifies it, the cause, and the fix. If none of it matches, the last section covers what to
put in a bug report.

## Where the logs are

The app writes one folder per session into `logs/`, next to the launcher. Folder names are the
start time, so the last one alphabetically is the newest.

```
logs/2026-09-25_14-03-12/
  node_console.log              everything the server printed, engine lines included
  ace_engine.log                raw output of the engine (ace-server)
  generations/
    gen_<id>_<task>.log         one per generation, e.g. gen_<id>_text2music.log
```

- `node_console.log` is the best place to start. It holds the server's own lines and the
  engine's lines (prefixed `[ace-server]`) in the order they happened.
- `ace_engine.log` has the engine's output only, including a few noisy lines the console
  filters out.
- A `gen_*.log` opens with the full request that went to the engine (models, seed, solver,
  steps) and ends with `GENERATION COMPLETED.` or `GENERATION FAILED:` and the reason. It is
  written when the generation ends, so if the whole app dies mid-render there is no gen log for
  that job. Use `node_console.log` instead.
- Only the gen logs have timestamps. To line up a job with engine output, search
  `node_console.log` for the job id from the gen log file name.
- Every restart starts a new folder. When you look into a failure, take all three files from
  the same folder.

The Terminal in the right-hand column of the app shows the same engine output live.

### Windows exit codes

When the engine stops, the console logs `[ace-server] Process exited with code N`. On Windows
the number says a lot:

| Code | Meaning |
|---|---|
| `3221225781` (0xC0000135) | A DLL the engine needs is missing. See [The engine will not start](#the-engine-will-not-start). |
| `3221226505` (0xC0000409) | The engine crashed. The lines just before it say where. |
| `3221225477` (0xC0000005) | Access violation. Seen right after a failed GPU memory allocation. |
| `3221225786` (0xC000013A) | The console window was closed or Ctrl+C was pressed. |
| `1`, right after `[Server] Shutting down...` | Normal shutdown, not a crash. |

## The engine will not start

Look at the first lines of `node_console.log` for one of these.

`[Server] ace-server not found at: <path>`. The engine binary is not where the server expects
it, usually because the archive was only partly extracted or files were moved out of
`engine/`. Extract the whole archive again into a fresh folder.

`Process exited with code 3221225781`. Windows could not find a DLL. On the CUDA builds this is
almost always the CUDA runtime (`cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll`, or
the `_12` versions on the `cuda12.8` and `cuda12-volta` builds), which the first launch
downloads into `engine/`. See [The GPU is not used](#the-gpu-is-not-used) for how to get them.
To make Windows name the missing file, open a command prompt in `engine/` and run
`ace-server.exe --models ..\models --port 8085` directly; the error dialog names the DLL.

`[Server] ERROR: no models found in <path>`. There are no model files yet. See
[No models found](#no-models-found).

A new release that crashes on startup where the previous one worked has happened once, in
v1.1.3, caused by a broken build rather than anything on the user's side. Go back to the
previous release and [report it](#filing-a-good-issue) with the logs.

## The engine keeps restarting

The server watches the engine. When it exits with an error the server logs
`Restarting in 3 seconds... (crash N/3)` and starts it again. Three crashes within 30 seconds
and it stops trying: `Crashed 3 times within 30s` and generation requests are refused with
`Engine not ready`. Crashes spaced further apart than 30 seconds, such as a crash on every
generation, keep restarting indefinitely.

The restarting is not the problem, the crash is. Read the engine lines just before the first
`Process exited with code`. The usual causes:

- A missing DLL, covered above.
- No models installed, covered [below](#no-models-found).
- A broken model file. `[GGUF] FATAL: '<file>' is truncated or corrupt` means an interrupted
  download; `[GGUF] FATAL: tensor '<name>' not found` means the file is not what its name
  claims. Delete that file in the Model Manager and download it again.

Once the cause is fixed, click **Restart** in the sidebar. Do not end `ace-server` from Task
Manager while the app is running: the server starts it again straight away. Use **Quit** or
**Restart** instead.

## Port already in use

The app uses two ports: 3001 for the web page and server, and 8085 for the engine, which only
listens on 127.0.0.1.

If something else holds port 3001, `node_console.log` shows
`[Server] Uncaught exception: Error: listen EADDRINUSE`, and the browser shows whatever that
other program serves. One user saw an "Upgrade Required" page this way. Close the other program,
or set a different port in `.env` (`SERVER_PORT=3002`) and open that address yourself; the
launcher always opens 3001.

If port 8085 is taken, `ace_engine.log` shows `[Server] FATAL: cannot bind 127.0.0.1:8085` and
the engine exits without the usual restart messages. The most likely holder is an `ace-server`
left over from an earlier session. Quit HOT-Step, end any remaining `ace-server` process, and
start again. To use another port, change **Engine port** on the Environment tab of Settings
(`ACESTEPCPP_PORT` in `.env`) and restart.

## No models found

`[Server] ERROR: no models found in <path>` on a fresh install is expected: download a starter
pack in the [Model Manager](studios/model-manager.md), then click **Restart**. The engine only
scans for models when it starts. See [Getting started](getting-started.md#download-your-first-models).

If you have models and still see it, check **Models directory** on the Environment tab of
Settings points at the right folder.

Other model messages in `ace_engine.log`:

| Message | Cause | Fix |
|---|---|---|
| `WARNING: /synth unavailable, missing: VAE` (or DiT, Text-Enc) | ACE-Step 1.5 needs a DiT, a text encoder and a VAE together. One is missing. | Download the missing one. The Quick Start pack has all three. |
| `[Registry] WARNING: skipping <file> (unknown architecture)` | The file is not an ACE-Step model the engine recognises. | Remove it from the models folder. |
| `No ACE-Step models in <path>`, then `continuing without ACE` | Only MiniMax-Music3 or YuE2 files are installed. | Nothing, if that is what you want. |

A file that shows as installed in the Model Manager but is missing from the model dropdowns
is usually in a subfolder. ACE-Step 1.5 files must sit directly in `models/`; the engine does
not look in subfolders for them, but the Model Manager does. Otherwise the engine has not been
restarted since the download.

## Model downloads fail

The Model Manager tries each file three times, then marks it failed. **Resume** continues from
the partial file. Since v1.3.4 the failure message names the network error and the host it
could not reach, which after a Hugging Face redirect is their file CDN rather than
huggingface.co.

- The downloader does not use `HTTPS_PROXY` or `HTTP_PROXY`, so a proxy that works in your
  browser does not apply to it.
- `Invalid GGUF header` with `got "<!DO"` means Hugging Face sent an error page instead of the
  file. Try again later.
- `Size mismatch` means the file arrived incomplete. Download it again.

If Hugging Face is unreachable from your network, you can download the files another way and
place them by hand. Model files go in `models/` (or `models/mm3/`, `models/yue2/` for those
backends); the CUDA runtime DLLs go in `engine/`, next to `ace-server`. [Model files](models.md)
lists every file and its source repo.

## Out of VRAM

The signs are `CUDA error: out of memory`, a Vulkan `failed to allocate ... buffer` or
`Failed to allocate pinned memory`, or a crash while the stage reads "Decoding audio (VAE)".
Try these in order.

1. Lower **VAE chunk size** on the Environment tab of Settings, then restart. The default is
   1024. 512 fixed decode crashes on 8 GB cards; 256 was needed on a 4 GB card. Decoding gets
   somewhat slower (one report went from about 4 to 7 seconds). The VAE only splits a track
   into chunks when it is longer than the chunk size (1024 is about 41 seconds of audio), so a
   short track can fail at the default while a long one works.
2. Turn off **Keep DiT & VAE loaded** (Performance tab) and **Keep Models in VRAM**
   (Environment tab).
3. Use a smaller quant or pack. See the pack table in
   [Getting started](getting-started.md#which-pack-fits-your-gpu).
4. With two GPUs, check which one the engine picked. The console's `[Server] GPU:` line names
   it. **GPU Device** on the Environment tab chooses; Auto picks the card with the most VRAM.

<!-- TODO(verify): a37a1572 (on master after v1.3.4) makes the engine tile automatically when an untiled VAE decode would not fit. Once released, step 1 can say which version no longer needs the manual setting. -->

If VRAM stays full after one render and the next fails straight away, **Restart** frees it.
Please report it with the logs from that session, since the cause is not yet known.

## The GPU is not used

Generation that is far slower than expected usually means the engine fell back to CPU. The
engine log says which device each model loaded on, for example `[Load] LM backend: CUDA0`.

- Windows CUDA builds, first launch: if the CUDA runtime download failed, the console shows a
  `GPU Runtime Download Failed` box and the engine starts on CPU only. Open the Model
  Manager, go to the **Shared** tab and download the **CUDA 13 Runtime** pack (**CUDA 12
  Runtime** on the `cuda12.8` and `cuda12-volta` builds), or restart with a working internet
  connection. Then restart.
- `[Load] WARNING: ggml-cuda.dll found but CUDA backend did not load.` The lines after it test
  each DLL and mark the missing one with `NOT FOUND in engine dir`. If every DLL loads, the
  engine suggests checking the driver with `nvidia-smi`.
- `[Load] WARNING: ggml-vulkan.dll found but Vulkan backend did not load.` The GPU driver has
  no working Vulkan support. Update the driver.
- Linux `rocm` build: `The ROCm runtime is not installed, or not on the loader path.` The
  engine prints the install command for your distro. If ROCm is installed and it still falls
  back, your card's architecture may not be one this build targets; `rocminfo | grep gfx`
  shows it.
- Two GPUs and the wrong one busy: see step 4 under [Out of VRAM](#out-of-vram).

## Vulkan problems

- Garbled output with the 1.7B language model. That model produces corrupted output on the
  Vulkan build. Use the 4B LM.
- Crashes during VAE decode. Lower **VAE chunk size** to 512 or 256, as described under
  [Out of VRAM](#out-of-vram).
- On Linux, the Vulkan build needs `libvulkan1` and a Vulkan 1.1 driver.

## A generation stalls or times out

Read the reason at the end of the job's gen log.

`Generation stalled`, with the seconds and the last stage. The server cancels a job that shows
no progress for two minutes (fifteen during VAE decode). The quoted stage says where it stopped;
the end of `ace_engine.log` says what the engine was doing. While the engine is computing it
cannot answer web requests, so an unresponsive engine is not necessarily a dead one.

`Generation timed out`, with the limit in minutes. The whole job took longer than
**Generation Timeout** on the Performance tab of Settings (default 30 minutes, up to 6 hours).
Raise it for long songs, high step counts or slow hardware. The first MiniMax-Music3 render
with the TensorRT renderer also builds its engine once, which takes a few minutes.

`Generation failed on ace-server`. The engine rejected or failed the job without saying why to
the server. The reason is in `ace_engine.log`; look for `FATAL` or `failed` near the end.

`Engine not ready`. The engine is still starting, still downloading the CUDA runtime, or has
stopped after repeated crashes. See [The engine keeps restarting](#the-engine-keeps-restarting).

If the queue stays stuck after you cancel, **Restart** clears it. Without restarting, this
request cancels every unfinished job:

```
curl -X POST http://localhost:3001/api/generate/reset-queue
```

## Node.js version

Releases bundle their own Node.js in `runtime/`, so this only affects a git checkout. The
server needs Node 18 to 22; Node 24 and newer are not supported. The usual symptom is an error
from `better_sqlite3.node` saying it `was compiled against a different Node.js version`, or
`npm install` failing. Install Node 22 LTS (with nvm: `nvm install 22` then `nvm use 22`) and
see [Building from source](../dev/building.md).

## Antivirus removes files

Security software can quarantine unsigned programs and scripts, and nothing in HOT-Step is
signed. It has happened to a `.bat` file in the HOT-Step folder under Bitdefender. The signs
are a launcher or DLL that disappears, or the engine exiting with `3221225781`. Check the
antivirus history (on Windows: Windows Security, Virus & threat protection, Protection history),
restore the file, and add the HOT-Step folder as an exclusion. Extracting the archive again also
restores anything that went missing.

## macOS blocks the app

The release binaries are unsigned, so macOS may quarantine them. Remove the quarantine flag
once after extracting:

```bash
xattr -cr /path/to/HOT-Step-CPP-v1.3.4-macOS-arm64/
```

## After upgrading

- The Model Manager shows **Update** on a file when a newer version of it has been published.
  It compares the file on disk against the catalogue's checksum. Download it to replace the old
  copy.
- To reuse models from an older install, point **Models directory** on the Environment tab of
  Settings at the old `models/` folder and restart, or move the folder across.
- On macOS before v1.3.4, **Restart** stopped the app and never brought it back. The launcher
  has a restart loop from v1.3.4.

<!-- TODO(verify): whether copying server/data/ and .env from an old release folder into a new one is the supported way to keep the library and settings. -->

## A sidechain VST does nothing

The VST3 chain feeds each plugin a single input, with no external sidechain bus. Plugins that
need an external key signal (sidechain compressors, keyed gates, duckers) never trigger. Use
them in their internal detection mode instead.

## Filing a good issue

Search the [existing issues](https://github.com/scragnog/HOT-Step-CPP/issues) first, then
[open a new one](https://github.com/scragnog/HOT-Step-CPP/issues/new) with:

- The HOT-Step version (shown next to the title on the Settings page) and the archive you
  downloaded, since the file name says the build, for example `win-x64-cuda13.1`.
- Your OS, GPU and VRAM, and the driver version (`nvidia-smi` shows it on NVIDIA).
- The active backend and the model files selected.
- What you did, what you expected, and what happened instead.
- `node_console.log`, `ace_engine.log` and the failed job's `gen_*.log`, all from the same
  session folder. Logs from a different session cannot show the failure, and this is the most
  common reason a report stalls.

The logs contain file paths from your machine, which can include your user name. Edit them out
if you prefer; the rest of the log is what matters.

For questions rather than bugs, the [Discord](https://discord.gg/ezVtmg9GKX) is where the
maintainer is most active.

## Related

- [Getting started](getting-started.md)
- [Model Manager](studios/model-manager.md)
- [Settings](studios/settings.md)
- [Model files](models.md)
