# Model Management — Reference

Resolve runtime paths first using [Finding the data folder](../../../AGENTS.md#finding-the-data-folder).
`<data root>/...` paths below describe the managed layout; environment overrides
can place individual children elsewhere.

Deep detail backing [SKILL.md](SKILL.md). All line numbers verified against `master` (HEAD `dda5f6b6`) on 2026-10-23; re-verify after large refactors.

## 1. Engine startup scan — exact order and rules (`engine/src/model-registry.h`)

`registry_scan()` (line 235) runs once at ace-server startup on the `--models` dir, and again on `<models>/onnx`:

1. **Root `.gguf` files**, classified by GGUF header key `general.architecture` (`registry_classify_gguf`, lines 100-131):
   `acestep-lm` → LM, `acestep-dit` → DiT, `acestep-text-enc` → Text-Enc, `acestep-vae` → VAE, `pp-vae` → PP-VAE.
   Unknown arch → `[Registry] WARNING: skipping <f> (unknown architecture)` and ignored, except files named `sa3-*`, `mm3-*` or `yue2-*`, which log `-> <family> (loaded by its own backend, not an ACE-Step model)` (lines 248-265, #151).
2. **Root `.safetensors` files**, classified by lowercase filename **prefix only** (lines 285-307): `pp-vae`/`pp_vae` → PP-VAE; else `vae` or `scragvae` → VAE. No other single-file safetensors roles exist: a safetensors DiT/LM must be a checkpoint **directory** (step 4).
3. **Root `.onnx` files** (lines 309-339): prefix `dit_`/`dit-` → DiT, `lm_`/`lm-` → LM (TensorRT). ONNX VAEs and text encoders are skipped with `(ONNX VAE: no runtime for it)` / `(ONNX text encoder: ...)` since ONNX Runtime was removed; anything else is skipped with `(unrecognized ONNX prefix)`.
4. **Subdirectories with `config.json` + `model.safetensors` (or `model.safetensors.index.json` for sharded)** (lines 341-388), classified by **`config.json` content** via `config_json_classify()`, not by directory name. `ModelEntry.path` is the directory; the loader finds files inside.
5. **Subdirectories containing at least one `.onnx`** (lines 390-483), classified by the ONNX filename prefix, falling back to the directory-name prefix (`lm*`, `dit*`, `vae*`, `text*`/`enc*`). Only DiT and LM entries are kept (TensorRT); `onnx/`, `sa3/`, `mm3/` and `yue2/` are recognized as other backends' folders and logged as such. Dirs already claimed as safetensors checkpoints are skipped.

Returns false only if the total classified count is zero; the server then exits with `[Server] ERROR: no models found in <dir>` unless another family's weights are present (section 2).

`registry_scan_adapters()` (lines 492-529): root `.safetensors` files logged as `(ComfyUI)`; subdirs containing `adapter_model.safetensors` logged as `(PEFT)`, or `lokr_weights.safetensors` as `(LoKR)`. `registry_scan_lm_adapters()` (lines 534-570) does the same for `<adapters>/lm/` into a separate bucket, so LM adapters never appear in DiT dropdowns.

`registry_find_non_onnx()` (lines 66-86): returns the first (or named) non-`.onnx` entry in a bucket. It dates from ONNX VAEs being decoder-only; with ONNX VAEs no longer registered it is a guard for old state.

## 2. Startup validation (`engine/tools/hot-step-server.cpp`)

The compiled server is `hot-step-server.cpp`; `engine/tools/ace-server.cpp` is upstream's, kept as an uncompiled reference (`engine/CMakeLists.txt:482-484`).

- `--models <dir>` required (lines 3235-3239).
- Scan finds nothing → if `hot_step_any_family_weights_present()` (`engine/src/hot-step-families.h`) sees MM3, YuE2 or SA3 weights: `[Server] No ACE-Step models in <dir> — <families> weights found, continuing without ACE`; else `[Server] ERROR: no models found in <dir>`, exit 1 (lines 3245-3263).
- Pipeline check (lines 3305-3350): `have_synth = DiT && Text-Enc && VAE`. Incomplete + LM present → `WARNING: /synth unavailable, missing: <list>`, server stays up LM-only. Incomplete + no LM + another family present → `WARNING: ACE pipeline unusable (missing: <list>) — continuing for <families>`. Otherwise `ERROR: no usable pipeline, synth missing: <list>`, exit 1.
- `/props` (`handle_props`, lines 3028-3100): `models` object has buckets **`lm`, `embedding`, `dit`, `vae`** (the text-enc bucket is named `embedding` here); plus `adapters` and `lm_adapters` arrays, `cli` (`max_batch`, `mp3_bitrate`), `default` (full AceRequest defaults), and `presets` (`turbo`: 8 steps / guidance 1.0 / shift 3.0; `sft`: 50 steps / guidance 1.0 / shift 1.0).

Node side: `server/src/services/aceEngineProcess.ts` spawns ace-server; on abnormal exit it respawns after 3 s, at most 3 crashes per 30 s window (`MAX_CRASHES`/`CRASH_WINDOW_MS`, lines 80-84), then logs `[ace-server] Crashed 3 times within 30s — giving up.` (line 321) and marks the engine not ready with the missing-DLL hint. With the engine down, `GET /api/models` returns the fallback `{ models: { dit: [], lm: [], vae: [], understand: [] }, adapters: [], lmAdapters: [], config: {}, defaults: {}, aceServerDown: true, error }` (`routes/models.ts:29-37`); note the shape mismatch vs live `/props` (no `embedding`, spurious `understand`).

First-launch bootstrap (`server/src/index.ts:218-383`): before spawning ace-server, checks the 3 required cuBLAS/cudart DLLs (ids `cuda-rt-cublas[,-lt,-cudart]`, or `-12` variants when `.cuda-version` beside the exe says CUDA ≤ 12) in the runtime dir and beside the exe, and auto-downloads any missing via the download service.

## 3. GGUF loading behavior (`engine/src/gguf-weights.h`)

- `gf_load()` (line 74) mmaps the file and verifies **every tensor's byte range fits inside the file** before use (lines 132-150). Truncated download → `[GGUF] FATAL: '<path>' is truncated or corrupt ... Re-download the file` and a clean load failure, added to avoid a segfault deep in `cuMemcpyHtoDAsync`.
- `gf_load_tensor()` (line 159) on a missing tensor name → `[GGUF] FATAL: tensor '<n>' not found` then **`exit(1)`** (lines 166-167; a second differently-worded FATAL for meta-context misses is at :173-174). The whole ace-server process dies; Node respawns it.
- Small tensors (norms, biases, scale_shift_table, etc.) are converted to F32 at load time (`gf_load_tensor_f32`).

## 4. Safetensors handling

- `engine/src/safetensors.h`: minimal read-only mmap parser (8-byte LE header length + JSON header). Dtypes F32/BF16/F16.
- `engine/src/weight-source.h`: format-agnostic layer over GGUF/safetensors. GGUF tensor names are canonical; safetensors may need a `name_prefix` (e.g. `model.` for the text encoder). **Shape order differs**: GGUF stores ggml order (ne[0] innermost), safetensors stores PyTorch order; `ws_shape()` normalizes to ggml order. `st_multi_open()` (line 63) tries `model.safetensors`, then sharded `model.safetensors.index.json`, then diffusers `diffusion_pytorch_model.safetensors` (mirrored by `convert.py:find_sf_files`).

## 5. ModelStore / VRAM policy (`engine/src/model-store.h`)

- `EVICT_STRICT` (default): at most one GPU module resident. `require`/`release` refcounting; a conflicting `require` while another module has refcount > 0 **asserts** (documented at lines 37-41; the EvictPolicy enum is at :99-102): that is a caller programming error, not a recoverable state.
- `EVICT_NEVER` (`--keep-loaded`): nothing evicted, everything accumulates. Node passes the flag when `ACESTEPCPP_KEEP_LOADED=1` (`config.ts:246`, `aceEngineProcess.ts:179-181`); a request's `?keep_loaded=1` flips a running engine one way.
- Invariant under both policies: exactly ONE LM instance process-wide, shared between generate and understand via identical ModelKey (lines 17-22, 30-35).
- `ModelKind` (lines 65-75): LM, text-enc, cond-enc, DiT, VAE enc/dec, FSQ tok/detok, and `MODEL_SA3_GGML` (the StableStep refiner).
- `ModelKey` (lines 77-97): `(kind, path)` plus LM extras (`max_seq`, `n_kv_sets`) and DiT extras: `adapter_path`, `adapter_scale`, `adapter_group_scales`, basin re-base `rebase_source`/`rebase_beta`, and the `adapter_stack` string. `pipeline-synth.cpp:177-215` builds that string from the stack signature (paths + scales) plus suffixes for per-section masking, runtime DoRA, runtime quant, low-VRAM merge and the adapter mode. Any difference = a distinct cached DiT.
- `DiTMeta` (lines 106-112): CPU-cached config + `silence_full` (from `silence_latent`) + `null_cond_cpu` + `is_turbo`/`is_merge` flags, available before the DiT hits the GPU.
- Manual-unload hooks: `store_evict_lm` (line 146, used by Song Builder), `store_list_loaded` (line 152), `store_evict_label` (line 158, labels like `"LM"`, `"DiT"`, `"VAE-Dec"`; skips in-use modules), `store_evict_all` (line 165, the pass that makes restoring EVICT_STRICT safe).
- Warm-on-startup: `ACESTEPCPP_WARM_DIT`/`_VAE`/`_ADAPTER`(+`_SCALE`) post `/warm` after boot, gated on keepLoaded and `ACESTEPCPP_WARM_ON_STARTUP` (default on) (`config.ts:247-263`); under EVICT_STRICT the warm would be dropped instantly.

## 6. quantize.cpp policy detail (`engine/tools/quantize.cpp`)

Usage: `quantize <input.gguf> <output.gguf> <type> [--imatrix <file.gguf>]` (lines 58-60). Input is BF16/F16 (F32 tensors are also converted on non-MM3 files). Variant table (lines 115-146):

| Type | Base | Bump type | embed | Bump mode |
|---|---|---|---|---|
| Q2_K | Q2_K | Q4_K | Q6_K | first 4 layers |
| Q3_K_S | Q3_K | — | Q6_K | none |
| Q3_K_M | Q3_K | Q5_K | Q6_K | first + last + every 3rd |
| Q3_K_L | Q3_K | Q5_K | Q6_K | all important |
| Q4_K_S | Q4_K | Q5_K | Q6_K | first 4 layers |
| Q4_K_M | Q4_K | Q6_K | Q6_K | first + last + every 3rd |
| Q5_K_S | Q5_K | — | Q6_K | none |
| Q5_K_M | Q5_K | Q6_K | Q6_K | first + last + every 3rd |
| Q6_K | Q6_K | — | Q6_K | none |
| Q8_0 / NVFP4 / MXFP4 | as named | — | Q8_0 | none |
| IQ1_S | IQ1_S | IQ2_XXS | Q2_K | first 4 layers |
| IQ1_M | IQ1_M | IQ2_XXS | Q4_K | first 4 layers |
| IQ2_XXS / IQ2_XS | as named | IQ3_XXS | Q4_K | first 4 layers |
| IQ3_XXS | IQ3_XXS | IQ4_XS | Q5_K | first 4 layers |
| IQ4_XS | IQ4_XS | — | Q6_K | none |

- Three tensor-name conventions are matched: ACE-Step HF names, MM3/YuE2 llama.cpp names, and MM3's prefixed synth modules (`dit.blk.N.*`, `depth.blk.N.*`); layer bumps count each stack's own depth (`split_layer`).
- "Important" tensors: `v_proj`/`attn_v` + `down_proj`/`ffn_down` (+ MM3 DiT `ffn_out`) for S/M; L adds `o_proj`/`attn_output` (lines 199-208). Embeddings (`embed_tokens`, `token_embd`, MM3's `output.weight` and `depth.audio_embd`) take the embed type (lines 210-215); MM3 boundary tensors never drop below it (lines 221-227).
- Never quantized (`should_quantize`, lines 230-269): any tensor in a `vae` arch, 1-D tensors (promoted to F32 instead), text-enc `embed_tokens`, `silence_latent`, `scale_shift_table`, `null_condition_emb`, SA3's `model.transformer.memory_tokens` and `conditioner.*`, YuE2's `latent_pos_embed`, `vae2llm`, `llm2vae` and `time_embd.*`. On MM3 files (`mm3.model` key) only F16 tensors are converted (lines 669-676).
- Importance matrix: IQ1_S/IQ2_XXS/IQ2_XS refuse to run without one (lines 496-501) and fail up front if a needed tensor has no entry (lines 680-689). The output records `quantize.imatrix*` provenance keys.
- Streaming one-tensor-at-a-time write; low memory. Non-block-aligned tensors kept as-is.

## 7. Node download service internals (`server/src/services/modelDownloadService.ts`)

Data shapes live in `server/src/contracts/modelManager.ts`, shared with the UI:

```ts
DownloadStatus    = 'queued'|'downloading'|'paused'|'completed'|'failed'|'cancelled'
DownloadJob       = { jobId, fileId, filename, status, bytesDownloaded, totalBytes, speed, error? }
RegistryFileEntry = { id, filename, role, subdir?, repoPath?, displayName, scale?, variant?, quant,
                      sizeBytes, sha256?, repo, description, tags[], companions?, sm?, family? }
RegistryFile      = RegistryFileEntry & { installed, outdated }
```

- Target dir (lines 140-155): `role === 'runtime'` → the runtime dir (`<data root>/tools/runtime/` in a release, beside ace-server.exe in dev); else `modelsDir[/subdir]`.
- Installed check (lines 254-324): filename presence with extensions `.gguf|.onnx|.safetensors|.bin` in models root + one subdir level, plus everything in `onnx/sa3/`, `.json` in `mm3/`, `.json`/`.i32` in `yue2/`, and `.dll` in the runtime dir and beside the exe. **Name-only, no size check.**
- Staleness (`_isKnownStale`, `_matchesRegistry`): entries declaring `sha256` are reported `outdated` when the installed hash differs; a download request re-downloads the file. A `<file>.sha256` sidecar stamped with size and mtime preserves completed verification across restarts.
- Background verification (`_drainHashQueue`) reads one file at a time, smallest first. It defers new reads while registered jobs are busy and rechecks sidecars before reading queued files.
- Refresh hashes with `node server/scripts/fill-model-hashes.mjs --write`, or omit `--write` to preview. The script fetches only HF LFS metadata, supports `HF_TOKEN`, preserves existing hashes, and reports conflicts and every entry still without a hash.
- `startDownload` (line 343) dedupes: an existing queued/downloading job for the same fileId returns its jobId. An optional `hfToken` is sent as `Authorization: Bearer` for gated repos.
- URL: `https://huggingface.co/{repo}/resolve/main/{repoPath || filename}` (line 535); an entry with no `repo` fails with an explicit "not published yet" error (lines 529-534). Relative redirects resolved against the original URL; max 5 redirects. HTTP 416 on a Range request treated as "already complete".
- Retries: 3 attempts with 0 / 2 s / 5 s delays (lines 476-477). Resume via `.part` file size → `Range: bytes=N-`. Failure text comes from `describeDownloadError` (line 58), which unwraps Node's AggregateError and names the host (#171).
- Validation before rename (`_validateDownload`, `_executeDownload`): size within ±5% of `sizeBytes`; `MZ` PE header for `.dll`; `GGUF` magic for `.gguf`; SHA-256 content verification for every entry declaring a hash, including `.onnx`, `.bin` and `.safetensors`. Failure deletes the file. Optional companions are fetched best-effort afterwards; required companions are validated and staged before the weights are downloaded.
- Deletion (`deleteFile`, lines 419-463): extension allowlist `.gguf|.onnx|.safetensors|.bin|.dll|.json|.i32`, path-traversal guards; DLLs deleted from the runtime dir, models from root + one subdir level + `onnx/sa3/`.
- Variant filtering (`getRegistry`, lines 159-250): `.variant` and `.cuda-version` marker files beside ace-server.exe hide CUDA-only files/packs on vulkan/cpu builds; Windows-only `.dll` entries are hidden on other platforms; CUDA ≤ 12 shows the `cuda12-runtime` pack and remaps `cuda-rt-*` ids to `-12` variants inside packs; no pack is left referencing a hidden file.
- `cleanupJobs()` (lines 465-472): comment says "older than 60s" but it deletes ALL terminal jobs immediately, and nothing in `server/src` calls it.

## 8. Catalogue statistics (`server/src/data/model-registry.json`, verified 2026-11-11)

- **463 files** by role: 123 heartmula, 86 dit, 67 mm3, 64 stablestep, 54 yue2, 25 runtime, 12 vae, 8 lm, 6 moss, 5 supersep, 4 whisper, 4 muscriptor, 3 pp-vae, 2 embedding. 459 declare a `sha256`. The non-LFS JSON entries `stablestep-tokenizer-config`, `stablestep-special-tokens` and `yue2-minted-manifest` have no HF content hash. The fourth unhashed entry, mm3-lm-q8_0-pissa-r128, is deferred until HOTSTEP-1137 prevents replacement of user-generated residuals required by existing adapters.
- **38 packs** by family: 4 `as1.5`, 9 `mm3`, 11 `yue2`, 8 `shared`, 2 `sa3`, 4 `heartmula`. The files span 26 HF repositories.
- **16 repos**: `ACE-Step/Ace-Step1.5`, `Serveurperso/ACE-Step-1.5-GGUF`, `ggerganov/whisper.cpp`, `mdmachine/ACEStep-XL-Regrind-V1` (Regrind VAEs, `vae/` subfolder), `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`, and the `scragnog/` repos `Ace-Step-1.5-MXFP4-Quants`, `Ace-Step-1.5-ScragVAE`, `HOT-Step-CPP-PP-VAE`, `HOT-Step-CPP-StableStep`, `HOT-Step-CPP-SuperSep` (also hosts runtime DLLs), `HOT-Step-CPP-TensorRT`, `MOSS-Music-8B-Instruct-GGUF`, `MiniMax-Music3-GGUF`, `Stable-Audio-3-Medium-Base-GGUF`, `YuE2-GGUF`, `ace-step-1.5-gguf-merge-models`.
- No entry lacks a `repo` field (the last one, `vae-dreamvae-onnx`, left with ONNX Runtime in 885b504d).

## 9. Model Manager API summary (`server/src/routes/modelManager.ts`, mount `/api/model-manager`)

| Route | Behavior |
|---|---|
| `GET /registry` | `{ packs, files: (RegistryFileEntry & {installed, outdated})[], modelsDir, variant, cudaMajor }` |
| `POST /download` `{fileId, hfToken?}` | `{ jobId }` (dedupes active jobs) |
| `GET /downloads` | SSE: `data: {"jobs": DownloadJob[]}` on every progress event + 1 s interval |
| `POST /download/:jobId/cancel` | `{ ok }` |
| `POST /download/:jobId/resume` | `{ jobId }` |
| `DELETE /files/:filename` | `{ ok }` (extension allowlist, traversal guard) |

UI: `ui/src/components/model-manager/ModelCatalogueTab.tsx` has four family tabs (`FAMILY_TABS`: ACE-Step 1.5, MiniMax-Music3, YuE2, Shared), each with role sections (`FAMILY_SECTIONS`: dit/lm/embedding/vae/pp-vae under ACE-Step, stablestep/supersep/whisper/moss under Shared); a file's family comes from its `family` field, else a role fallback. Entry points are "Get More Models" in `ui/src/components/global-bar/ModelsDropdown.tsx` and `BackendModelsDropdown.tsx`; when the engine reports no DiT, LM or VAE the Model Manager opens by itself after 8 s, unless it was dismissed earlier in the browser session (`GlobalParamBar.tsx:262-275`).

## 10. convert.py specifics (`engine/convert.py`)

- Hardcoded I/O: `engine/checkpoints/` → `engine/models/` (lines 14-16); no CLI args. `engine/checkpoints/` does not exist in this working tree; create it before running, then move the output to `<data root>/models/`.
- Classification by checkpoint directory name (lines 55-64): `acestep-5Hz-lm*` → `acestep-lm`, `acestep-v15*` → `acestep-dit`, `Qwen3-Embedding*` → `acestep-text-enc`, and the directory named **exactly `vae`** → `acestep-vae`. Anything else is skipped.
- Handles single-file, sharded-index, and diffusers safetensors layouts (`find_sf_files`, lines 31-45). Output GGUFs are self-contained (weights + config + tokenizer + silence_latent). Skips outputs that already exist.

## 11. Design-doc drift (code wins)

The original design doc (`docs/plans/2026-05-03-model-manager-design.md`, gitignored, may be absent) describes 5 roles / 5 repos / ~104 files / 5 UI tabs, pause-resume ETA fields, and `recommendedSettings` per entry. The shipped catalogue has 14 roles, 26 repos and 463 files, plus `role: runtime`, `subdir`, `repoPath`, `sha256` and `family` fields the doc lacks. There is no `recommendedSettings` in actual entries. Always trust the code and the JSON over that doc.
