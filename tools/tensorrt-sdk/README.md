# tools/tensorrt-sdk

Dev-only tooling that produces the TensorRT assets `release.yml` downloads
for the Windows `cuda13.1` build (the only variant that compiles the MM3
TensorRT DiT renderer, `HOT_STEP_TRT`). See `docs/dev/releasing.md` → "TensorRT
SDK for CI" for the operational runbook, and
`docs/plans/2026-09-11-mm3-trt-dit-shipping.md` for the wider shipping plan.

## Why two scripts, not one SDK zip

NVIDIA's TensorRT SLA makes the runtime DLLs (`nvinfer_10.dll`,
`nvonnxparser_10.dll`, ...) distributable, and the public headers
(`github.com/NVIDIA/TensorRT`, `include/`) are Apache-2.0 — but the Windows
SDK download's `.lib` import libraries are not named as distributable. So
nothing derived from those `.lib` files is hosted. Instead:

- **`make-sdk-zip.ps1`** packages only `include/*.h` + a copy of the
  Apache-2.0 license text into a small zip. This is what gets hosted.
- **`make-import-libs.ps1`** regenerates an equivalent `.lib` at build time,
  directly from a DLL's own export table (`dumpbin /exports` → a `.def` file
  → `lib.exe /def`) — a standard technique for linking against a DLL that
  didn't ship its own import lib. Nothing is copied out of NVIDIA's SDK
  archive; the symbols already live in the (distributable) DLL. Verified
  locally against the vendor SDK's own `.lib` files: `nvonnxparser_10.lib`
  matches all 9 public symbols exactly; `nvinfer_10.lib` matches 4370/4371
  real API symbols (the one difference is a decorated-vs-plain spelling of
  the same two internal build-watermark constants, not an API gap).

`release.yml`'s Windows cuda13.1 job runs both: unpack the hosted headers
zip, download the two DLLs, run `make-import-libs.ps1` against them, and only
then configure CMake.

## make-sdk-zip.ps1

```powershell
./make-sdk-zip.ps1 [-SdkDir <path>] [-OutDir <path>]
```

Reads the version from `<SdkDir>/include/NvInferVersion.h` and writes
`tensorrt-<version>-headers.zip` (contents: `include/`,
`LICENSE-APACHE-2.0.txt`) to `-OutDir`. Defaults point at the vendored dev
copy (`engine/deps/tensorrt`) and the gitignored scratch dir
`_experiments/2026-09-11-mm3-speed/trt-sdk/`. Prints the zip's sha256 and
size — both go into `release.yml`'s `TRT_SDK_ZIP` / `TRT_SDK_SHA256` env vars.

## make-import-libs.ps1

```powershell
./make-import-libs.ps1 -Dll <dll-path>[,<dll-path>...] -OutDir <path>
```

Runs `dumpbin /exports` on each DLL, writes a `.def` listing every exported
name, and calls `lib.exe /def:... /machine:x64 /out:<name>.lib`. Locates
`dumpbin.exe`/`lib.exe` via PATH first (a Developer shell / after
`ilammy/msvc-dev-cmd`, which is how `release.yml` uses it), falling back to
`vswhere.exe` to find the latest Visual Studio's MSVC toolchain otherwise.

## Where the files live

- Headers zip + the two runtime DLLs: Hugging Face
  `scragnog/HOT-Step-CPP-TensorRT`.
- Filenames and sha256 values CI expects: `release.yml`'s top-level `env:`
  block (`TRT_HF_REPO`, `TRT_SDK_ZIP`, `TRT_SDK_SHA256`, `TRT_NVINFER_DLL`,
  `TRT_NVINFER_DLL_SHA256`, `TRT_NVONNXPARSER_DLL`,
  `TRT_NVONNXPARSER_DLL_SHA256`).
- `server/scripts/check-release-prereqs.mjs` verifies all three files exist
  on HF before a release tag.

## License note

The headers zip's `LICENSE-APACHE-2.0.txt` is the standard Apache License
2.0 text. **[Rob: fill in the final license note / attribution wording here
before the HF repo goes public — this placeholder should not stand as the
final word on redistribution terms.]**
