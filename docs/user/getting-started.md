# Getting started

HOT-Step CPP ships as a portable archive: extract it, run one script, and the app opens in your
browser. Nothing is installed system-wide, and Node.js is bundled. This page takes you from the
download to your first finished song.

## Pick a download

Every release on the [releases page](https://github.com/scragnog/HOT-Step-CPP/releases/latest)
has one archive per platform and GPU. The file name says which: for example
`HOT-Step-CPP-v1.3.4-win-x64-cuda13.1.zip`.

| Your hardware | Windows | Linux | macOS |
|---|---|---|---|
| NVIDIA RTX 20-series or newer, GTX 16-series | `cuda13.1` (recommended) | `cuda13.1` (recommended) | |
| NVIDIA Pascal (GTX 10-series, Tesla P40/P100) | `cuda12.8` | `cuda12.8` | |
| NVIDIA Tesla V100 (Volta) | `cuda12-volta` | `cuda12-volta` | |
| AMD or Intel GPU | `vulkan` | `vulkan` | |
| AMD Radeon RX 6800/6900, RX 7000 or RX 9000 | `vulkan` | `rocm` or `vulkan` | |
| No usable GPU | `cpu` | `cpu` | |
| Apple Silicon (M1 and later) | | | `macOS-arm64` (Metal) |

Some notes on the choice:

- `cuda13.1` covers Turing through Blackwell. `cuda12.8` covers the same cards plus Pascal,
  which CUDA 13 can no longer compile for. On a Turing or newer card, prefer `cuda13.1`.
- `cuda12-volta` exists because the flash-attention and quantized matmul kernels in the
  other CUDA builds have no Volta code. It runs with those disabled.
- Only the Windows `cuda13.1` build includes the TensorRT renderer for the MiniMax-Music3
  flow DiT.
- The Linux `rocm` build does not include the ROCm runtime, which is several GB and tied to
  your kernel driver. Install it yourself first (`sudo apt install rocm-hip-runtime` on
  Ubuntu/Debian). Without it the engine prints the install command for your distro and falls
  back to CPU.
- `cpu` works on any machine, but generation is much slower.
- The Vulkan build produces corrupted output with the 1.7B language model. Use the 4B LM on
  Vulkan. This rules out the Minimal Setup pack below.

Each archive has a matching `.sha256` file if you want to check the download.

### Requirements

- Windows 10 or 11, 64-bit.
- Linux x86_64 with glibc at least as new as Ubuntu 22.04's (the builds are made on 22.04).
  The Vulkan build needs a Vulkan 1.1 driver and `libvulkan1`.
- macOS on Apple Silicon. If macOS refuses to run the unsigned binaries, see
  [Troubleshooting](troubleshooting.md#macos-blocks-the-app).
- An NVIDIA driver new enough for the CUDA version in the file name, for the CUDA builds.
- Disk space for the app (100 to 500 MB depending on the build) plus the models. The
  smallest ACE-Step 1.5 starter pack is about 5 GB, the recommended one about 9 GB.

<!-- TODO(verify): minimum NVIDIA driver per CUDA build. engine/src/backend.h prints "560.xx for CUDA 12.8, 570.xx for CUDA 13.x", which looks lower than NVIDIA's own CUDA 13 requirement. Also the minimum macOS version (the old README said 13+; CI builds on macos-15). -->

## Extract and launch

1. Extract the whole archive into a folder of its own. Keep the folder structure as it is:
   the launcher, `engine/`, `server/`, `ui/` and `runtime/` must stay side by side.
2. Start the app:
   - Windows: double-click `HOT-Step.bat`.
   - Linux and macOS: open a terminal in the folder and run `./HOT-Step.sh`.
3. A console window shows the server's output. Leave it open; closing it stops the app.
4. Your browser opens at `http://localhost:3001`. If it does not, open that address yourself.

On Windows CUDA builds, the first launch downloads the CUDA runtime libraries (about 510 MB
for CUDA 13, about 810 MB for CUDA 12) from Hugging Face into `engine/` before the engine
starts. Progress prints in the console. Generation is unavailable until it finishes. If the
download fails, the engine starts on CPU only; see
[Troubleshooting](troubleshooting.md#the-gpu-is-not-used).

## Download your first models

The archive contains no model weights. On a fresh install the engine finds nothing to load,
logs `no models found`, and stops after three attempts. That is expected at this point.
About eight seconds after the page loads, the [Model Manager](studios/model-manager.md) opens
on its own. You can also open it any time with **Get More Models** at the bottom of the Models
section in the top bar.

1. Stay on the **ACE-Step 1.5** tab. It is the default backend and the only one every studio
   supports.
2. Under **Starter Packs**, click the download button on the **Quick Start** card. It contains the Turbo DiT at
   Q8_0, the 4B language model at Q8_0, the text encoder, and the standard, ScragVAE and
   PP-VAE decoders, about 9 GB in all.
3. Wait for the progress bars to finish. A stalled download can be resumed.
4. Click **Restart** in the sidebar and confirm. The engine only scans the models folder when
   it starts, so this is what makes the new files usable.

The other two music models, MiniMax-Music3 and YuE2, have their own tabs and packs. See
[Backends](backends.md) for what each one does and [Model files](models.md) for every file
in the catalogue.

### Which pack fits your GPU

The registry states VRAM figures for only a few packs. Where it gives none, the table says so
rather than guessing.

| Pack | Backend | Download | VRAM |
|---|---|---|---|
| Minimal Setup | ACE-Step 1.5 | ~5.2 GB | Aimed at low-VRAM cards. Uses the 1.7B LM, so not for Vulkan. |
| Quick Start | ACE-Step 1.5 | ~9.1 GB | Not stated. <!-- TODO(verify): VRAM peak for Quick Start at default settings. --> |
| XL Quality | ACE-Step 1.5 | ~11.9 GB | About 12 GB |
| Blackwell Optimized | ACE-Step 1.5 | ~9.3 GB | RTX 50-series, CUDA builds only. <!-- TODO(verify): VRAM figure. --> |
| MiniMax-Music3 Q4_K_M | MiniMax-Music3 | ~7.6 GB | Described as the option for limited VRAM. |
| MiniMax-Music3 Q8_0 | MiniMax-Music3 | ~13.4 GB | Not stated. |
| MiniMax-Music3 F16 | MiniMax-Music3 | ~23.6 GB | About 22.5 GB of weights plus about 3 GB of working memory. |
| YuE2 Compact (imatrix) | YuE2 | ~3.1 GB | Described as the usual choice for limited VRAM. |
| YuE2 Recommended | YuE2 | ~4.6 GB | Not stated. |

<!-- TODO(verify): measured VRAM peaks for the MM3 Q4/Q8 and YuE2 packs on a named GPU. -->

The CUDA runtime DLLs, which the Windows CUDA packs also list, are left out of the download
sizes above because the first launch already fetched them.

Two settings trade VRAM for speed, and both are off by default: **Keep DiT & VAE loaded** on
the Performance tab (about 8.2 GB more for an XL model) and **Keep Models in VRAM** on the
Environment tab (holds around 13 GB). Leave them off on a smaller card. On an 8 GB card or
smaller, lowering **VAE chunk size** fixes the most common out-of-memory crash; see
[Troubleshooting](troubleshooting.md#out-of-vram).

For which quant sounds best, see [Getting higher quality output](quality.md).

## Your first song

1. Click **Custom-Gen** in the sidebar.
2. Type a **Style Description**: genre, instruments, vocal style, tempo. "Warm 90s trip-hop,
   dusty breakbeat, upright bass, breathy female vocals, 88 BPM" works better than "chill song".
3. Write **Lyrics** with section tags such as `[Verse]` and `[Chorus]`, each on its own line,
   or turn on **Instrumental**.
4. Click **Generate**. Progress shows in the queue, and the Terminal in the right-hand column
   shows the engine's live output.
5. The finished song appears in the [Library](studios/library.md).

If you would rather not write anything, **Auto-Gen** starts from a genre and writes the caption,
lyrics and title for you.
[Generation](generation.md) explains the global bar controls that apply to every render:
models, solver, steps and guidance.

## Where things live on disk

Everything stays inside the folder you extracted.

| Path | Contents |
|---|---|
| `models/` | Model weights. ACE-Step 1.5 files sit directly in this folder; MiniMax-Music3 and YuE2 files go in `models/mm3/` and `models/yue2/`. |
| `adapters/` | LoRA and LoKr adapters. |
| `engine/` | The engine binaries, plus the CUDA runtime DLLs the first launch downloads. |
| `server/data/` | The song library database (`hotstep.db`), the Lyric Studio database (`lireek.db`) and generated audio (`audio/`). |
| `logs/` | One folder per app session. See [Troubleshooting](troubleshooting.md#where-the-logs-are). |
| `.env` | Settings, created from `.env.example` on first launch. The Environment tab of Settings edits it. |

The models and adapters folders can be moved elsewhere with **Models directory** and
**Adapters directory** on the Environment tab of Settings. Both need a restart.

<!-- TODO(verify): the supported way to move to a new release while keeping the library: copy server/data/ and .env into the new folder, or keep one folder and overwrite? Database migrations run on startup, but no upgrade procedure is documented anywhere. -->

## Running from source

`HOT-Step.bat` and `HOT-Step.sh` are the release launchers; a git checkout uses `LAUNCH.bat`,
`launch.sh` or the hot-reload `dev.bat` instead, all covered in [Building from source](../dev/building.md).

## Related

- [Model Manager](studios/model-manager.md)
- [Backends](backends.md)
- [Generation](generation.md)
- [Getting higher quality output](quality.md)
- [Troubleshooting](troubleshooting.md)
