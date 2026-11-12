# HeartMuLa publication

The model card is [heartmula-model-card.md](heartmula-model-card.md).
[heartmula-hf-manifest.json](heartmula-hf-manifest.json) inventories every file
in the active `models/heartmula/` folder with its exact size and SHA-256.
The full inventory is published in [HeartMuLa-GGUF](https://huggingface.co/scragnog/HeartMuLa-GGUF). The [verification receipt](heartmula-hf-verification.json) records the revision, public URL, exact size and SHA-256 for every hosted file.

## Exact upload set

The 126 source entries are uploaded at root-level repository paths: 123 GGUFs totalling 150,260,222,528 bytes, plus three quant provenance manifests. The source inventory totals 150,260,271,695 bytes. The provenance manifests retain their original build paths for reproducibility; those paths are not installation requirements. README, LICENSE, NOTICE and manifest.json bring the hosted total to 150,260,314,195 bytes in 130 files. The MuLaCover cover model (`heartmula-mulacover-global-BF16.gguf`, `heartmula-mulacover-local-BF16.gguf` and `MODEL_LICENSE-mulacover`) was added to the repository later under CC BY-NC 4.0 and its own model terms. It is not in this manifest; its sizes and SHA-256 are in `server/src/data/model-registry.json`.

The publication includes these metadata files:

| Local file under `docs/dev/` | Repository path |
|---|---|
| `heartmula-model-card.md` | `README.md` |
| `heartmula-hf-manifest.json` | `manifest.json` |
| `heartmula-LICENSE` | `LICENSE` |
| `heartmula-NOTICE` | `NOTICE` |

## Publication commands

Publication was authorized on 2026-10-26. The repository was created public and the source files were sent through `HfApi.upload_large_folder` with eight workers and an allow-list of the 126 inventory paths. Metadata was uploaded with `HfApi.upload_file`. The equivalent CLI commands below require the models directory resolved through `cd server; npx tsx scripts/data-paths.ts`. Any later publication needs its own authorization.

```powershell
hf upload scragnog/HeartMuLa-GGUF "$env:ACESTEPCPP_MODELS/heartmula" . --include "*.gguf" "*-quant-manifest.json"
hf upload scragnog/HeartMuLa-GGUF docs/dev/heartmula-model-card.md README.md
hf upload scragnog/HeartMuLa-GGUF docs/dev/heartmula-LICENSE LICENSE
hf upload scragnog/HeartMuLa-GGUF docs/dev/heartmula-NOTICE NOTICE
hf upload scragnog/HeartMuLa-GGUF docs/dev/heartmula-hf-manifest.json manifest.json
```

Every source hash was checked before upload. Every hosted size and SHA-256 was then checked against the publication manifest. Public resolve URLs were checked anonymously; GGUF hashes were compared with HF LFS object SHA-256, and small metadata/provenance files were downloaded and hashed. The receipt records which method was used.

## Component gaps

The scalar decoder remains F16 and RVQ remains F32 because the current
quantizer has no 2-D matrices to reduce in either, as documented in
[the component contract](../../engine/docs/HEARTMULA_CONTRACT.md).
A duplicate file labelled Q8_0 would save no space. The four-file encoder
pack is available at F16. Q8_0 encoder files and a separate scalar encoder
are absent locally, so no invented sizes or registry entries are supplied.

## Other backends

ACE-Step 1.5, MiniMax-Music3 and YuE2 retain their existing source repositories
and download paths. Stable Audio 3 already has its GGUFs in
[the StableStep repository](https://huggingface.co/scragnog/HOT-Step-CPP-StableStep)
and its training base in
[the Medium Base repository](https://huggingface.co/scragnog/Stable-Audio-3-Medium-Base-GGUF).
The registry now groups both under Stable Audio 3. These are already hosted;
this change requires no SA3 re-hosting. Their model cards declare the
Stability AI Community License. License acceptance remains required in the
manager, including for starter packs.
