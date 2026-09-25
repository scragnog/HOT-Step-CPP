# Model Manager

Model Manager is where you browse, download, and remove the model files HOT-Step generates with: the DiT, the language model, the VAE, the text encoder, and the extra files specific backends and post-processing features need. Reach for it the first time you launch the app with nothing installed, when you want to try a different quantisation or a bigger model, or when a studio tells you a required file is missing.

![Model Manager](../../images/hot-step-model-manager.webp)

## Where to find it

The **Get More Models** button at the bottom of the Models section in the global param bar opens it. When the active backend is MiniMax-Music3 or YuE2, that section shows the backend's own model pickers instead of the ACE-Step 1.5 dropdowns; its equivalent button (**Get Models** when nothing is installed yet, **Get More Models** otherwise) opens the same modal. A few other spots open it too when a file they need is missing, such as the MiniMax-Music3 renderer row and the YuE2 training cards in Training Studio.

Nothing needs to be running first. Model Manager talks to the Node server, not the engine, so it works even while the engine is still starting or is down entirely. On first launch, if the server still finds no DiT, LM, or VAE files after giving the engine 8 seconds to scan, it opens the modal automatically; closing it (the X button, the backdrop does not close it) keeps it from popping up again for the rest of that session.

## Workflow

1. Open Model Manager from the Models section of the global bar, or let it open on its own on first launch.
2. Pick a family tab: **ACE-Step 1.5**, **MiniMax-Music3**, **YuE2**, or **Shared**. This also filters the starter packs above the catalogue.
3. For a quick setup, download a **Starter Pack**: one click installs every file it lists, and the card shows how much is left to fetch.
4. Or scroll down to **All Models** and pick files yourself. Families with more than one role show role sub-tabs underneath (DiT Models, Language Models, Text Encoder, VAE, PP-VAE for ACE-Step 1.5; StableStep, Stem Separation, Whisper, Captioning for Shared); MiniMax-Music3 and YuE2 list their files directly, grouped by quant on the YuE2 tab.
5. Click **Download** on a row. Progress shows inline on that row and in the Active Downloads banner at the top, with percent, speed, and ETA.
6. If a download stalls or fails, click **Resume** to pick it up where it left off, or **Cancel** to stop it.
7. Click the trash icon on an installed file, then confirm, to remove it and free disk space.
8. Close with the X button. The footer shows the models folder path, how many files are installed, and total disk used.

See [the model list](../models.md) for every file in the catalogue: exact filenames, sizes, and source repos.

## Controls

| Control | What it does |
|---|---|
| Family tabs | ACE-Step 1.5 / MiniMax-Music3 / YuE2 / Shared. Filters the Starter Packs section above and the catalogue below to the same family; the choice is remembered between visits. |
| Starter Packs | Pre-bundled sets of files for a complete pipeline, for example Quick Start (standard Turbo, full pipeline) or Minimal Setup (smallest viable, low VRAM). Download installs every missing file in the pack; a fully-installed pack shows All Installed. |
| Role sub-tabs | Shown under families with more than one role. Each label carries an installed/total count. |
| Info button | Opens a short explanation of the category, what the role is for and how its variants differ, inline under the group header. |
| Model row | One file: display name, quant badge, size, and a Download / Update / Installed control. Hover an installed row to reveal Delete, which asks for confirmation once. |
| Download progress | Percent, transfer speed, and ETA while a file is downloading; Resume for a paused or failed job, Cancel for one still queued or in progress. |
| License acceptance (StableStep tab) | A checkbox gating every download in that tab until you accept the Stability AI Community License. |
| Hugging Face token (StableStep tab) | Optional, only needed if a source repository is gated. Stored locally and sent only to huggingface.co. |

## Tips and limits

**Roles.** DiT is the model that denoises audio latents; LM (language model) turns your caption and lyrics into the audio-code sequence the DiT conditions on; the text encoder embeds the caption for the DiT and is architecturally fixed, you need exactly one; VAE decodes latents to audio and, for cover/repaint/extend, encodes audio back into latents; PP-VAE is an optional polish pass. MiniMax-Music3 and YuE2 are separate backends with their own LM, DiT/flow, and decoder roles. Stem separation, Whisper, and MOSS captioning are shared utility models used by Cover Studio, lyric transcription, and Training Studio respectively, not by generation itself.

**Quant levels.** BF16 / F32 / F16 are reference precision, largest and best quality. Q8_0 is near-lossless, about half the size. Q6_K and Q5_K_M stay close to source quality with a real size saving, Q5_K_M smaller again than Q6_K. Q4_K_M / Q4_K_S are the compact end, noticeably smaller for lower-VRAM setups; on the YuE2 ladder these and the ranks below them are imatrix-guided, which keeps quality up further down the scale than a plain quant of the same size would. MXFP4 and NVFP4 are FP4 formats: MXFP4 gets native Tensor Core acceleration on RTX 5000-series (Blackwell) GPUs, with the same quality and compression as a software fallback on older cards; NVFP4 is experimental and scores below the k-quants above it in testing so far. The registry only states VRAM numbers for a few entries directly: the XL Quality starter pack needs about 12 GB, Minimal Setup targets low-VRAM cards, and MiniMax-Music3 needs about 24 GB to hold its language model and flow stack loaded together. Elsewhere, quant descriptions speak to size and quality, not an exact VRAM figure.

**Manual installs.** A file you add by hand must sit directly in the models folder (default `models/` next to the app, shown in the footer here; changeable at Settings > Models directory, which needs an app restart to take effect), not in a subfolder. The engine only scans that top level for loose GGUF/safetensors files, but Model Manager's own installed check also looks one folder down, so a file left in a subfolder can show Installed here while staying invisible to generation. Restart the app after adding a file so the engine picks it up.

**Installed means the filename matched.** The installed check looks for the right filename at the right size class of extension, not file size or contents, so a partial or corrupted file placed there by hand can still show as Installed. Anything downloaded through Model Manager itself is checked: file size against the catalogue's expected size, plus a header check for `.gguf` and `.dll` files, before it is kept.

**ONNX VAEs are decode-only.** If the only VAE installed is an ONNX one, cover, repaint, and extend have no VAE available for the encode step. Keep a GGUF or safetensors VAE installed alongside it.

**Downloads run concurrently.** Start as many files as you like at once; each gets its own row in the Active Downloads banner and its own progress bar, cancel, and resume control.

**Runtime files install elsewhere.** CUDA/cuDNN DLLs and TensorRT builder resources install next to the engine binary, not into the models folder, even though they appear as rows and packs here. TensorRT builder resources are GPU-specific; the MiniMax-Music3 TensorRT group marks the one that matches your GPU when it can detect it.

## Related

- [Getting started](../getting-started.md)
- [Backends](../backends.md)
- [Settings](settings.md)
- [Models](../models.md)
- [Troubleshooting](../troubleshooting.md)
