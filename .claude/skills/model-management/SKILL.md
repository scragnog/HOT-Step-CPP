---
name: model-management
description: Explains how model files, checkpoints, GGUF quantization, and the Model Manager work in HOT-Step CPP. Use when adding/converting/quantizing models, debugging "no GGUF models found" or missing-model/wrong-model failures, working on the model download service or Model Manager UI, or answering which component (LM/DiT/VAE/text-encoder) needs which file.
---

# Model Management — Models, Checkpoints & Quantization

All paths are repo-relative to the repo root (`d:\Ace-Step-Latest\hot-step-cpp`). All commands are Windows PowerShell (use `;` to chain, never `&&`).

## Terminology (read first)

- **LM** — the language model (Qwen3-based) that turns caption+lyrics text into audio codes. Files: `acestep-5Hz-lm-{0.6B,1.7B,4B}-*.gguf`.
- **DiT** — Diffusion Transformer, the denoising model that generates audio latents. Files: `acestep-v15-*.gguf`. One DiT GGUF is self-contained (also carries the condition encoder, FSQ tokenizer/detokenizer, `silence_latent`, `null_condition_emb`).
- **Text encoder** — Qwen3 embedding model that encodes the caption for the DiT. Files: `Qwen3-Embedding-0.6B-*.gguf`.
- **VAE** — decodes latents to 48 kHz audio (and encodes audio to latents for cover/repaint/extend). Files: `vae*.gguf` / `scragvae*.gguf`.
- **PP-VAE** — optional post-processing VAE ("polish" re-encode pass). Files: `pp-vae-*.gguf`.
- **GGUF** — the GGML binary weight format the C++ engine loads (mmap'd). Each GGUF declares its role in the header key `general.architecture`.
- **Quant** — reduced-precision weight encoding (Q4_K_M, Q8_0, MXFP4, ...) to shrink VRAM/disk. Produced from BF16 GGUFs by `engine/tools/quantize.cpp`.
- **Adapter** — LoRA/LoKr fine-tune delta (`.safetensors`), lives in `adapters/`, not `models/`.
- **Model Manager** — the "Get More Models" modal in the UI + Node download service that fetches curated files from HuggingFace.

## When to use this skill

- Installing, converting, or quantizing model files; deciding where a file must live.
- Debugging: engine won't start, `/synth unavailable`, model missing from dropdowns, crash-loops, corrupt downloads.
- Modifying the Model Manager (route `server/src/routes/modelManager.ts`, service `server/src/services/modelDownloadService.ts`, UI `ui/src/components/model-manager/`), or the catalogue `server/src/data/model-registry.json`.

## Golden rules

1. **Two independent registries exist — don't confuse them.** The Node-side curated catalogue (`server/src/data/model-registry.json`) controls what is *downloadable*; the C++ engine's startup scan (`engine/src/model-registry.h`) controls what is *usable*. A file can show "installed" in the Model Manager yet be invisible to generation dropdowns (and vice versa). WHY: they use different detection logic (filename presence vs GGUF-header architecture) and different directory depths.
2. **GGUF files must sit in the models root, not a subfolder.** The engine scans only root-level `.gguf` files (`engine/src/model-registry.h:234-267`), but the Node installed-check also scans one subdir level (`modelDownloadService.ts:177-208`). A GGUF in a subfolder = "installed" in the UI, dead to the engine. WHY: silent classic confusion — no error anywhere.
3. **Never hand-edit or hand-build a GGUF's tensor set for a recognized architecture.** A recognized-arch GGUF with a missing tensor kills the whole ace-server process: `gf_load_tensor()` prints `[GGUF] FATAL: tensor 'x' not found` and calls `exit(1)` (`engine/src/gguf-weights.h:164-168`). Node respawns it, so a bad model file mid-request looks like a random engine crash-loop.
4. **Quantize from BF16 sources only, and never quantize the VAE.** `quantize.exe` reads BF16 GGUF input; VAE-arch tensors and small/critical tensors (`silence_latent`, `scale_shift_table`, `null_condition_emb`, 1-D tensors, text-enc `embed_tokens`) are deliberately never quantized (`engine/tools/quantize.cpp:89-109`). WHY: quantizing these destroys audio quality or breaks generation outright.
5. **Rebuild rules apply here too:** any change to `engine/src/model-registry.h`, `model-store.h`, `gguf-weights.h`, etc. means rebuild via `dev-rebuild.bat` at repo root — never `engine/build.cmd` directly (you cannot reliably tell whether the app is running; Node auto-respawns ace-server, and killing it uncleanly causes an infinite respawn + file-lock loop). Never `cmake --clean-first` (20+ min CUDA recompile).
6. **Runtime DLLs do not go in `models/`.** Catalogue entries with `role: "runtime"` (cuBLAS, cudart, ONNX Runtime, cuDNN) install **next to `ace-server.exe`** (`modelDownloadService.ts:110-118`). WHY: missing DLLs there are the #1 cause of the "crashed 3 times within 30s — giving up" loop.
7. **A model that only exists on this machine does not exist.** Adding a model the app resolves at runtime is a three-part act: upload the weights to Hugging Face, add the entry to `server/src/data/model-registry.json`, and put it in a pack if a feature requires it. Do all three or none — a feature gated on a file that was never uploaded looks fine here and is dead for every user. Verify with `node server/scripts/check-release-prereqs.mjs`. WHY: MM3 training shipped in v1.3 demanding `mm3-rvq-*.gguf` and `mm3-enc-*.gguf` that had never left the dev box (#137, fixed 96d442fb).
8. **Model/adapters directory paths are restart-required config.** `ACESTEPCPP_MODELS` / `ACESTEPCPP_ADAPTERS` env vars override defaults `<repo>/models` and `<repo>/adapters` (`server/src/config.ts:53-54,100-101`); the engine gets them as spawn-time `--models`/`--adapters` flags.

## Directory layout (expected)

```
models/                                  # config.aceServer.models (default <repo>/models)
  acestep-v15-*.gguf                     # DiT (arch "acestep-dit"): base/sft/turbo/merge x BF16/Q8_0/Q6_K/Q5_K_M/Q4_K_M/MXFP4/NVFP4...
  acestep-5Hz-lm-{0.6B,1.7B,4B}-*.gguf   # LM (arch "acestep-lm")
  Qwen3-Embedding-0.6B-*.gguf            # Text encoder (arch "acestep-text-enc")
  vae-*.gguf, scragvae-*.gguf            # VAE (arch "acestep-vae"); ScragVAE/Regrind = drop-in decoder variants
  pp-vae-*.gguf                          # PP-VAE (arch "pp-vae")
  vae-*.safetensors                      # safetensors VAE, classified by filename prefix only
  vae-*.onnx                             # ONNX VAE — DECODER-ONLY (see failure table)
  <name>/                               # HF safetensors checkpoint dir: config.json + model.safetensors
                                        #   (or model.safetensors.index.json for sharded) — classified by config.json content
  onnx/                                 # ONNX Runtime / TensorRT model dirs (config.aceServer.onnxDir)
  supersep/*.onnx                       # stem-separation nets (Cover/Stem Studio)
  whisper/ggml-*.bin                    # whisper.cpp models (config.ts:253-255)
adapters/                                # config.aceServer.adapters
  <name>.safetensors                    # ComfyUI single-file LoRA (alpha baked in)
  <name>/adapter_model.safetensors      # PEFT directory format
```

## Which component needs which model

Verified in `engine/src/model-store.h:68-81` (ModelKind comments):

| Component | Model file | Notes |
|---|---|---|
| LM (`MODEL_LM`) | `acestep-5Hz-lm-*.gguf` | ONE shared instance for generate + ace-understand — enforced by identical ModelKey (`model-store.h:17-22`) |
| Text encoder (`MODEL_TEXT_ENC`) | `Qwen3-Embedding-*.gguf` | |
| Cond-enc + DiT + FSQ tok/detok | the **same** `acestep-v15-*.gguf` | Self-contained; also holds `silence_latent` + `null_condition_emb` (`model-store.h:112-118`) |
| VAE encode + decode | `vae*.gguf` (has `encoder.*` and `decoder.*`) | |
| PP-VAE polish | `pp-vae-*.gguf` | Request flag `pp_vae_reencode` (`engine/src/request.h:146`); availability = `GET /api/models/pp-vae` scans for `pp-vae*.gguf` (`server/src/routes/models.ts:50-65`) |
| ORT/TRT acceleration | `models/onnx/` subdirs | `MODEL_*_ORT` kinds |
| SuperSep stems | `models/supersep/*.onnx` | + ONNX Runtime/cuDNN DLLs beside ace-server.exe |
| Whisper transcription | `models/whisper/ggml-*.bin` | + `tools/whisper/whisper-cli.exe` (`config.ts:254-255`) |

**Synth pipeline needs DiT + Text-Enc + VAE simultaneously.** Missing any one → `/synth unavailable` warning (server stays up LM-only if an LM exists), or exit 1 if no LM either (`engine/tools/hot-step-server.cpp:2600-2616` — the compiled server; the same block exists in the UNCOMPILED reference copy ace-server.cpp:1686-1711). Request-level selection: `synth_model`, `lm_model`, `vae` are **filenames** resolved against the engine's scanned registry; empty string = first matching entry (`engine/src/request.h:125-142`).

## Procedure: quantize a BF16 GGUF

```powershell
# From repo root. Binary lives at engine\build\Release\quantize.exe
.\engine\build\Release\quantize.exe <input-BF16.gguf> <output.gguf> <TYPE>
# Example:
.\engine\build\Release\quantize.exe models\acestep-v15-turbo-BF16.gguf models\acestep-v15-turbo-Q4_K_M.gguf Q4_K_M
```

- Valid TYPEs (case-insensitive, `quantize.cpp:8`): `Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0 NVFP4 MXFP4`. IQ3/IQ4 quants seen on disk are **not** producible by this tool.
- Mixed-precision policy mirrors llama-quantize: "important" tensors (`v_proj`, `down_proj`; L variants add `o_proj`) bumped one tier; `embed_tokens` always Q6_K (Q8_0 for Q8_0/NVFP4/MXFP4) (`quantize.cpp:40-54,74-86`).
- Streaming write, low memory. Prints `Quantized N/M tensors` + compression ratio.
- Output goes straight into `models\` root → picked up on next engine restart.

## Procedure: convert HF safetensors → BF16 GGUF (`engine/convert.py`)

- **No CLI args.** Hardcoded: reads checkpoint dirs from `engine/checkpoints/`, writes GGUFs to `engine/models/` (`convert.py:14-16`). **Neither directory exists in this working tree** — create `engine\checkpoints\`, put the HF checkpoint dir inside, run it, then move the output GGUF to repo-root `models\`.
- Classification is by checkpoint **directory name**: `acestep-5Hz-lm*` → LM, `acestep-v15*` → DiT, `Qwen3-Embedding*` → text-enc, and exactly `vae` → VAE (`convert.py:55-64`). Skips outputs that already exist.
- Alternative: the engine loads safetensors checkpoint dirs **directly** (drop `<name>/` with `config.json` + `model.safetensors` into `models/`) — conversion is optional. Sharded (`model.safetensors.index.json`) and diffusers (`diffusion_pytorch_model.safetensors`) layouts supported (`engine/src/weight-source.h`, `engine/src/model-registry.h:328-375`).

## Procedure: convert ComfyUI int8 DiT safetensors → Q8_0 GGUF (`engine/convert-comfy-int8.py`)

For ComfyUI `comfy_quant` int8 DiT checkpoints (int8 `.weight` + F32 `.weight_scale` scalar or per-row + `.comfy_quant` JSON tensor per layer), including **ConvRot** files (`"convrot": true` + `convrot_groupsize`). Per-tensor and per-row int8 grids are exactly representable in Q8_0 (block scale = tensor/row scale), so weights are **repacked bit-faithfully** — no dequant/requant round trip. Needs a **donor GGUF** of the same architecture (any convert.py-produced `acestep-v15-*.gguf`) to supply `silence_latent` and the `acestep.*` config KVs, which ComfyUI files lack. Aborts on any tensor-shape mismatch vs the donor.

```powershell
python engine\convert-comfy-int8.py <comfy.safetensors> models\<matching-donor>-BF16.gguf models\<out>-Q8_0.gguf --name <general.name>
```

ConvRot handling: rotated `decoder.*` weights stay rotated and are recorded in GGUF KV `acestep.convrot_map` (`name:group;...`); the engine applies the matching group-wise Hadamard rotation to that linear's activations at inference (`dit.h` load + `dit-graph.h`/`dit-alignment-graph.h`, commit 182faef). Rotated encoder/tokenizer/detokenizer weights are dequantized + **unrotated** to BF16 offline (run once per generation — not worth graph wiring). `--no-runtime-rotation` builds an all-BF16 unrotated reference GGUF of the same quantized model, used for same-seed A/B validation of the engine rotation path. **Adapter merge mode is refused on ConvRot models** (deltas are unrotated); runtime adapter mode works (deltas consume raw activations).

**ConvRot cardinal rule — any code reading ConvRot base weights for unrotated-space math must unrotate them first** (`convrot_transform_rows` in `engine/src/convrot.h`, fast radix-4, self-inverse). Violation signature: generation "succeeds" but output is garbled full-band noise. First instance: the runtime basin re-base nudged deltas with β·(S−T) using rotated T — fixed in `adapter_runtime_rebase` (commit 22820ae) by unrotating T for `acestep.convrot_map` tensors. Audit any future weight-reader (TRT export, distills, external merge scripts) against this.

Producing ConvRot files from a local checkpoint: `pip install convert_to_quant` (needs torch+CUDA, triton-windows) then
`ctq -i <model.safetensors> -o <out.safetensors> --int8 --scaling_mode row --convrot --dynamic_convrot --comfy_quant --save-quant-metadata` (~35 min for a 5B XL on an RTX 5090, learned rounding included).

First applied 2026-07-15: hrktxz xl_sft_turbo (plain int8) → `acestep-v15-xl-sft-turbo-comfy-int8-Q8_0.gguf`; merge-base-sft-turbo-xl-thirds self-quantized with real ConvRot → `...-convrot-Q8_0.gguf` (+ `...-convrot-ref-BF16.gguf` reference). Numerical parity: rotated-path error 0.9% vs original F32 weights; skipping rotation → ~140% (i.e. rotation is load-bearing).

## Procedure: add a model manually

1. Copy the `.gguf` into `models\` **root** (not a subfolder — golden rule 2).
2. Restart the engine (`dev-rebuild.bat` restarts everything, or restart the app). The scan runs only at ace-server startup.
3. Check the newest `logs\<session>\ace_engine.log` for `[Registry] <file> -> DiT` (or LM/VAE/...). A `WARNING: skipping X (unknown architecture)` means the GGUF header lacks a recognized `general.architecture` (`acestep-lm|acestep-dit|acestep-text-enc|acestep-vae|pp-vae`, `model-registry.h:99-130`).
4. The file now appears in `GET /api/models` (Node proxies engine `/props`; buckets `lm`, `embedding`, `dit`, `vae` — `hot-step-server.cpp:2326-2329` — the compiled server, not the uncompiled ace-server.cpp).

## Procedure: publish a new model so users can get it

Local conversion/quantization is only half the job. Until these steps are done
the model does not exist for anyone but you.

1. **Upload the weights.** `huggingface_hub` is installed; the token lives in
   `~/.cache/huggingface/token` (account `scragnog`). **Ask the user before
   pushing to a public repo** — it is outward-facing and hard to walk back.

   ```python
   from huggingface_hub import HfApi
   HfApi().upload_file(path_or_fileobj='models/mm3/<file>.gguf', path_in_repo='<file>.gguf',
                       repo_id='scragnog/<repo>', repo_type='model',
                       commit_message='Add <file>')
   ```

2. **Add the registry entry** to `server/src/data/model-registry.json` — `id`,
   `filename`, `role`, `subdir`, `displayName`, `quant`, exact `sizeBytes`
   (the downloader validates size ±5%), `repo`, `description`, `tags`, and a
   `companions` LICENSE entry if the weights carry one. The JSON round-trips
   exactly under `json.dumps(indent=2, ensure_ascii=False)`, so it can be edited
   programmatically without reformatting the whole file.

3. **Add it to a pack** if a feature needs it, and check the reverse: a feature
   that resolves files by *prefix scan* rather than by registry id (e.g.
   `resolveMm3TrainModels` takes the newest `mm3-rvq-*.gguf` on disk) will not
   be satisfied unless a published filename matches the prefix.

4. **Credit the author** in the HF model card if the weights are not ours, and
   keep the upstream licence. Community encoders and adapters are other
   people's work.

5. **Verify**: `node server/scripts/check-release-prereqs.mjs` — checks every
   entry resolves on HF at the claimed size, packs reference real ids, and
   runtime data files are packaged. Exit 1 = do not ship.

## Procedure: drive the Model Manager via API

Routes in `server/src/routes/modelManager.ts`, mounted at `/api/model-manager`:

```powershell
Invoke-RestMethod http://localhost:3001/api/model-manager/registry            # catalogue + installed flags
Invoke-RestMethod -Method Post -Uri http://localhost:3001/api/model-manager/download -ContentType 'application/json' -Body '{"fileId":"<id>"}'
# GET /downloads = SSE progress stream; POST /download/<jobId>/cancel | /resume; DELETE /files/<filename>
```

Download mechanics: HuggingFace URL `https://huggingface.co/{repo}/resolve/main/{repoPath || filename}`, resume via HTTP Range + `.part` file, 3 attempts (0/2s/5s), validation before rename (size ±5%, `MZ` header for `.dll`, `GGUF` magic for `.gguf`) — `modelDownloadService.ts:352-473`. Details and data shapes: [reference.md](reference.md).

## Key files

| Path | Role |
|---|---|
| `engine/src/model-registry.h` | Engine startup scan/classification of `--models` and `--adapters` dirs |
| `engine/src/model-store.h` | Refcounted VRAM ownership; EVICT_STRICT (default) vs EVICT_NEVER (`--keep-loaded`); ModelKey caching incl. adapter extras |
| `engine/src/gguf-weights.h` | mmap GGUF loader; truncation guard; FATAL exit on missing tensor |
| `engine/src/safetensors.h`, `engine/src/weight-source.h` | safetensors parser + format-agnostic layer (GGUF/safetensors) |
| `engine/tools/quantize.cpp` → `engine/build/Release/quantize.exe` | BF16 GGUF → K-quant/FP4 GGUF |
| `engine/convert.py` | HF safetensors checkpoint dir → BF16 GGUF (hardcoded dirs) |
| `engine/tools/ace-server.cpp` | Startup validation, `/props` endpoint |
| `server/src/config.ts` | `aceServer.models/adapters/onnxDir`, `keepLoaded`, warm-on-startup, whisper paths |
| `server/src/services/modelDownloadService.ts` | Download jobs, resume, validation, installed-check, variant filtering |
| `server/src/routes/modelManager.ts` | `/api/model-manager/*` REST + SSE |
| `server/src/routes/models.ts` | `/api/models` (proxies engine `/props`), `/api/models/pp-vae` |
| `server/src/data/model-registry.json` | Curated catalogue: 152 files, 9 packs |
| `server/src/index.ts` | ace-server spawn/respawn limiter (152-156, 284-308); first-launch CUDA DLL bootstrap (318-380) |
| `ui/src/components/model-manager/` | Modal UI: `ModelManagerModal.tsx`, `ModelCatalogueTab.tsx` (7 tabs), `ModelRow.tsx`, `StarterPackCard.tsx`, `DownloadProgressBar.tsx`, `useModelRegistry.ts`, `useDownloadStream.ts` |

## Failure signatures

| Symptom | Cause | Fix |
|---|---|---|
| `[Server] ERROR: no models found` + engine exit 1; Node retries 3x then gives up | Empty/wrong models dir (`ACESTEPCPP_MODELS`), or nothing classifiable — and no MM3 weights either. Since the issue-#118 fix, MM3 weights (`mm3-*.gguf` at root or in `mm3/`) keep the server alive MM3-only ("No ACE-Step models … continuing MM3-only") | Point at the right dir / install models; restart |
| `[Registry] WARNING: skipping X (unknown architecture)` | GGUF header lacks a recognized `general.architecture` | Convert via `convert.py`, or it's not an ACE-Step GGUF |
| `[Server] WARNING: /synth unavailable, missing: VAE` (etc.) | Partial install — synth needs DiT+Text-Enc+VAE together | Download the missing role (Model Manager quick-start pack) |
| `[GGUF] FATAL: '<f>' is truncated or corrupt ... file is only N bytes` | Interrupted download / prematurely renamed `.part` | Delete and re-download |
| `[GGUF] FATAL: tensor 'x' not found` then process death | Recognized arch, wrong/incomplete tensor set — kills ace-server mid-request | Remove the bad GGUF |
| Download "completes" then `Invalid GGUF header — got "<!DO"` | HuggingFace served an HTML error page (auth/rate-limit/404) | Retry; check the repo/path in the catalogue entry |
| `Size mismatch: expected X MB, got Y MB` | Catalogue `sizeBytes` drift vs repo file, or corrupt transfer | Re-download; fix `sizeBytes` in `model-registry.json` if repo file changed |
| Crash-loop "3 times within 30s" + missing-DLL hint | cuBLAS/cudart DLLs absent beside ace-server.exe (CUDA variant) | Model Manager "CUDA Runtime" pack; first-launch bootstrap normally handles it |
| Model shows "installed" in Model Manager but absent from generation dropdowns | GGUF in a subdir (Node scans subdirs, engine scans root only), unknown arch, or engine down (`aceServerDown: true`) | Move to models root / check engine log |
| Cover/repaint fails while text2music works, ONNX VAE selected | ONNX VAEs are **decoder-only**; VAE encode requires a non-ONNX VAE (`registry_find_non_onnx`, `model-registry.h:62-85`) | Install a GGUF/safetensors VAE alongside |
| Download of the DreamVAE entry fails instantly | That catalogue entry has no `repo` field → URL contains `undefined` | Treat as local/display-only (see below). Regrind entries were fixed 2026-07-29 (now download from `mdmachine/ACEStep-XL-Regrind-V1`) |

## Institutional knowledge

- **VALIDATED (production incident, 2026-08-31):** the registry and Hugging Face
  are the distribution boundary, not `models/`. MM3 training in v1.3 required
  `mm3-rvq-*.gguf` + `mm3-enc-*.gguf`, which existed only on the dev machine and
  in neither the catalogue nor any HF repo, so the Training Studio asked every
  user for files that could not be obtained (#137). Fixed 96d442fb by publishing
  both and adding a `minimax-music3-training` pack (which also carries
  `mm3-depth-f16` — the trainer needs f16 depth while every generation pack ships
  a quantised one). `node server/scripts/check-release-prereqs.mjs` now gates it.
- **The RVQ encoder is not ours.** `mm3-rvq-53kpooled-f32.gguf` is PurpleOrc's
  open-rvq (SimpleTuner v4 architecture, 53k-track corpus), mirrored to our repo
  under the same MiniMax-Music3 community terms with credit in the model card.
  Codes are encoder-specific: an adapter trained on these codes must keep using
  this encoder, so replacing it means re-exporting every code cache.

- **VALIDATED — subdir-GGUF blind spot:** engine scans root-only for `.gguf`; Node installed-check scans one subdir level. This mismatch is a real, recurring "installed but not selectable" source (code cited in golden rule 2).
- **VALIDATED — truncation guard exists for a reason:** the byte-range check in `gf_load()` (`gguf-weights.h:132-150`) was added specifically because truncated downloads used to segfault deep in `cuMemcpyHtoDAsync`. Keep it if touching the loader.
- **CORRECTED 2026-08-31 — the `repo`-less entry is fixed.** `vae-dreamvae-onnx` used to have no `repo` field, so `startDownload` built `https://huggingface.co/undefined/...` and failed; it now carries `repo: daydreamlive/DreamVAE` + `repoPath: onnx/model.onnx` and resolves (verified by `check-release-prereqs.mjs`). The failure mode is still worth knowing: an entry missing `repo` is display-only, visible in the catalogue and undownloadable, which is exactly the class of fault the prereq check now catches. (The Regrind V9b entries had the same gap until 2026-07-29 — all 8 Regrind VAEs now carry `repo: mdmachine/ACEStep-XL-Regrind-V1` + `repoPath: vae/...`; that repo keeps files in `vae/`/`dit/`/`lora/` subfolders, so `repoPath` is mandatory for anything added from it.)
- **VALIDATED — installed-check is name-only:** filename presence, no size/hash (`modelDownloadService.ts:177-208`). A stale/partial file with the right name shows "installed".
- **VALIDATED — `--keep-loaded` trade-off:** default EVICT_STRICT reloads models per request; `ACESTEPCPP_KEEP_LOADED=1` → `--keep-loaded` (EVICT_NEVER) avoids ~17 s LoKr adapter precompute per request but pins ~13 GB VRAM (`config.ts:124-132`, `index.ts:177-184`). Warm-on-startup env vars (`ACESTEPCPP_WARM_DIT`/`_VAE`/`_ADAPTER`) only fire when keepLoaded is on.
- **VALIDATED — DiT instances cache per adapter combo:** ModelKey includes adapter path/scale, per-group scales, basin re-base (`rebase_source`/`rebase_beta`), and multi-adapter `adapter_stack` signature (`model-store.h:83-103`) — different values = distinct cached DiTs, each costing VRAM under keep-loaded.
- **VALIDATED — `/api/models` fallback shape is not a faithful mirror:** when the engine is down, Node returns buckets `{ dit, lm, vae, understand }` + `aceServerDown: true` (`models.ts:28-36`), but the live engine `/props` sends `{ lm, embedding, dit, vae }` (`ace-server.cpp:1512-1515`) — no `understand` bucket exists, and the fallback lacks `embedding`.
- **VALIDATED — `cleanupJobs()` doc-drift:** its comment says "older than 60s" but it deletes all terminal jobs immediately, and no route calls it (`modelDownloadService.ts:341-348`).
- **VALIDATED — speculative-decoding draft LM is DISABLED:** `ACESTEPCPP_DRAFT_LM` plumbing remains (`config.ts:117-122`) but GGML per-call overhead negated the speedup; auto-detect commented out.
- **VALIDATED — "convrot" HF uploads may be mislabeled plain int8:** all files in `hrktxz/ACE_Step_1.5_ComfyUI_int8_convrot` (checked 2026-07-15) contain only `{"format": "int8_tensorwise"}` per-tensor-scale layers — no Hadamard rotation metadata anywhere, despite the repo name. Real ConvRot (QuaRot-style group-wise Hadamard, ComfyUI ≥0.27) would need `"convrot": true` + `convrot_groupsize` in the `.comfy_quant` JSON and runtime activation rotation. Our vendored GGML already ships FWHT kernels (CUDA/Vulkan/CPU) + `GGML_HINT_SRC0_IS_HADAMARD` (`ggml.h:444`, unused by engine code) if we ever want native support. Check the safetensors header before believing a quant-format claim.
- **UNVALIDATED:** provenance of on-disk IQ3/IQ4 quant files (not producible by quantize.cpp — presumably made with external tooling); whether `neural-codec`/`mp3-codec` binaries take model paths (not inspected).

## Deeper reading

- [reference.md](reference.md) (this folder) — engine scan order/classification details, ModelKey/VRAM policy, download-service internals, catalogue statistics, API data shapes, quantize policy table.
- `engine/docs/ARCHITECTURE.md` — engine internals, request JSON, generation modes (committed).
- `README.md` — install/build; states safetensors DiT/LM/Text-Enc/VAE all loadable and BF16 safetensors bit-perfect vs BF16 GGUF.
- `docs/plans/2026-05-03-model-manager-design.md` — original Model Manager design (**gitignored, local-only, may be absent**; code has drifted from it — trust code).
