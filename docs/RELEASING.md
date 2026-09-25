# Releasing HOT-Step CPP — agent runbook

How to cut and publish a release, plus the non-obvious gotchas. Written for an
agent (or human) driving the process with the `gh` CLI on Windows/Git-Bash.

## TL;DR

Releases are cut by **pushing a `vX.Y.Z` tag**. The `Release` workflow builds
every platform and creates a **draft** GitHub Release; you review and publish it.
A separate `Cache Warm` workflow keeps the engine build cache on `master` so
release builds take ~10–15 min instead of ~1.5h for the CUDA jobs.

## Prerequisites

- `gh` authenticated (as `scragnog`).
- On `master`, working tree clean, everything committed **and pushed**.
- Pick a semver version **without a hyphen**: `vX.Y.Z` (hyphens are reserved for
  test/pre-release tags — see gotchas).
- **Prerequisites check passes** (below). This one is not optional.

## 0. Check that users can actually get everything the build needs

```bash
node server/scripts/check-release-prereqs.mjs
```

Exit 0 or do not tag. It verifies that every model in the catalogue really
exists in its Hugging Face repo at the size claimed, that the repos are public,
that no pack points at a missing file id, and that every runtime data file is
packaged by `release.yml`.

**Why this is a hard gate.** A packaged build only ships code. Everything else —
weights, catalogues, corpora — has to be either inside the archive or
downloadable, and both routes are easy to forget because the failure is
invisible here: the file is on the dev machine, so the feature works, tsc is
clean, and CI is green. v1.3 shipped MM3 training gated on two GGUFs that had
never been uploaded (#137) and an MM3 caption corpus that CI never copied into
the archives (#139). Neither was catchable by any other check.

If you added a model this cycle, uploading the weights is a separate deliberate
step — see the `model-management` skill — and it must happen **before** the tag,
not after the release goes live.

## 0b. Run the release gate on a packaged build

```bash
node tools/release-gate/run.mjs --zip release/out/HOT-Step-CPP-vX.Y.Z-win-x64-cuda.zip
```

Exit 0 or do not tag. The gate boots the extracted zip against this checkout's
models, then walks the app through its API: every ACE generation mode, a check
that solvers, schedulers, guidance modes and adapters actually change the
output, MM3 and YuE2 renders, the audio tools, a few steps of every trainer,
and fixed-seed fingerprints. Budget about an hour on the 5090 with training
included, or `--tiers 0-3` for the half-hour subset. The run ends by staging
every render in `_experiments/_LISTENING/<stamp>-release-gate/` with a score
sheet: listen before you tag, because nothing mechanical judges quality.
Details and flags: [tools/release-gate/README.md](../tools/release-gate/README.md).

## 1. (Optional) Compile-test before releasing

To verify CI compiles without cutting a real release, push a throwaway
**hyphenated** tag — it triggers the same build pipeline but is ignored by the
changelog logic:

```bash
git tag -a vX.Y.Z-CI-Test -m "compile test" && git push origin vX.Y.Z-CI-Test
# ...watch it (section 3)... then delete when done:
gh release delete vX.Y.Z-CI-Test --cleanup-tag --yes   # removes draft + remote tag
git tag -d vX.Y.Z-CI-Test
```

Re-pushing the **same** `-CI-Test` name is free (delete remote+local, recreate,
push). Tags cannot be renamed.

## 2. Cut the release

```bash
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z
```

This triggers `Release` → builds Windows (cuda13.1 / cuda12.8 / vulkan / cpu),
Linux (same four), macOS (Metal) → creates a **draft** release with **22 assets**
(11 archives + 11 `.sha256`).

To change the commit or re-run: delete + re-push the tag (it rebuilds).

## 3. Monitor

```bash
gh run list --limit 5
gh run view <run-id>                 # per-job status + timings
```

To read a **failed/cancelled job's** log while the run is still in progress
(`gh run view --log` won't show it yet), pull it from the API:

```bash
MSYS_NO_PATHCONV=1 gh api repos/scragnog/HOT-Step-CPP/actions/jobs/<job-id>/logs > log.txt
```

`MSYS_NO_PATHCONV=1` stops Git-Bash rewriting the leading-slash API path into a
filesystem path.

## 4. Publish

The workflow leaves the release as a **draft**. Verify the asset count (18) and
the `What's Changed` notes, then:

```bash
gh release view vX.Y.Z --json assets --jq '.assets | length'   # expect 22
gh release edit vX.Y.Z --draft=false --latest
```

## 5. Cleanup

Delete any leftover test tags and their drafts (see section 1).

## Build caching — why releases are fast (and how it breaks)

- **GitHub Actions caches are ref-scoped.** A cache saved by one tag run is NOT
  visible to a different tag run — only **default-branch (`master`) caches** are
  visible to every run, including release tags. So releases can only reuse a
  cache that was created on `master`.
- **`.github/workflows/cache-warm.yml`** builds the engine on `master` (when
  `engine/ggml` or `engine/CMakeLists.txt` change, or via manual dispatch) and
  saves the build dir under the **same cache keys** `release.yml` uses. Release
  runs restore it and skip the CUDA compile (the long part).
- **Timings:** cold (no master cache) CUDA jobs ≈ 1.5h each; warm ≈ 7–13 min.
- **If CUDA suddenly rebuilds slow:** the master cache is missing/stale. Re-warm
  it: GitHub → Actions → **Cache Warm** → *Run workflow* (on `master`). It also
  auto-runs when `engine/ggml`/`CMakeLists.txt` change.
- Cache reuse depends on git-restored source mtimes (incl. the **ggml submodule**
  — its `.cu` files live in the submodule's own history, not the superproject).

## TensorRT SDK for CI

The Windows `cuda13.1` variant is the only build that compiles the TensorRT
DiT renderer (`HOT_STEP_TRT`, MM3's TensorRT DiT path). CMake only enables it
when `engine/deps/tensorrt/{include,lib}` exist at configure time, and that
directory is gitignored and never checked out — so `release.yml` fetches it
itself, for that one matrix entry only.

- **Headers + license zip** (`tensorrt-<version>-headers.zip`, hosted at HF
  `scragnog/HOT-Step-CPP-TensorRT`) unpacks to `engine/deps/tensorrt/include`.
  Public TensorRT headers are Apache-2.0, so this zip is small and freely
  redistributable — built by `tools/tensorrt-sdk/make-sdk-zip.ps1`.
- **Import libraries** (`nvinfer_10.lib`, `nvonnxparser_10.lib`) are **not**
  hosted anywhere — NVIDIA's SLA doesn't name the SDK's own `.lib` files as
  distributable, only the runtime DLLs and headers. `release.yml` instead
  downloads the (distributable) `nvinfer_10.dll` / `nvonnxparser_10.dll` from
  the same HF repo and regenerates equivalent import libs from their own
  export tables at build time, via `tools/tensorrt-sdk/make-import-libs.ps1`
  (`dumpbin /exports` → a `.def` → `lib.exe /def`). This is purely mechanical
  — nothing is copied out of NVIDIA's SDK archive.
- Every filename and sha256 the workflow fetches lives in `release.yml`'s
  top-level `env:` block (`TRT_HF_REPO`, `TRT_SDK_ZIP`, `TRT_SDK_SHA256`,
  `TRT_NVINFER_DLL`, `TRT_NVINFER_DLL_SHA256`, `TRT_NVONNXPARSER_DLL`,
  `TRT_NVONNXPARSER_DLL_SHA256`). The download step verifies each sha256
  before use and fails the job on a mismatch.
- The assembled `engine/deps/tensorrt/` is cached (`actions/cache`, keyed on
  `TRT_SDK_ZIP` + the nvinfer DLL sha256) so a normal release re-run doesn't
  re-download or re-derive anything.
- A step right after the CMake configure asserts the configure log contains
  `[TRT] Found vendored SDK` — if TRT silently failed to enable, the job fails
  loudly instead of shipping a cuda13.1 build with a stubbed-out renderer.
- The runtime DLLs themselves are **never** packaged into the release archive
  — a user gets them through the Model Manager, same as any other large model
  file (see `docs/plans/2026-09-11-mm3-trt-dit-shipping.md`). CI downloads
  them only to derive import libs; `check-release-prereqs.mjs` checks that all
  three files (headers zip + both DLLs) actually exist on HF before a tag.

**Bumping the TensorRT version:**

1. Update the vendored dev copy at `engine/deps/tensorrt/` (used for local
   builds) to the new SDK.
2. `./tools/tensorrt-sdk/make-sdk-zip.ps1` — regenerates the headers zip and
   prints its name, size, and sha256.
3. Upload the new zip to `scragnog/HOT-Step-CPP-TensorRT` on Hugging Face,
   alongside the new `nvinfer_10.dll` / `nvonnxparser_10.dll` (get their
   sha256 with `Get-FileHash`).
4. Update `release.yml`'s `env:` block: `TRT_SDK_ZIP`, `TRT_SDK_SHA256`, and
   the DLL sha256 values (the DLL filenames themselves rarely change).
5. `node server/scripts/check-release-prereqs.mjs` — confirms all three files
   are reachable on HF before you tag.

## Gotchas / lessons learned

- **Windows runner is pinned to `windows-2022`.** Do NOT switch to
  `windows-latest` — that's windows-2025, whose MSVC (`_MSC_VER >= 1950`) is
  rejected by CUDA 12.8/13.1 `nvcc` (`host_config.h`: VS 2017–2022 only).
- **Any pushed `v*` tag triggers the Release pipeline.** Use `vX.Y.Z` for
  releases and `-CI-Test` (or other hyphenated) tags for throwaway checks; delete
  them afterward. Don't push local feature tags that match `v*`.
- **Changelog range** = commits since the previous **non-hyphenated** tag
  (`git describe ... --exclude '*-*'`). This is why a stray `vX-CI-Test` tag must
  not be treated as a release; the exclude guard handles it, but still clean up.
- **The release is a draft** — it does not auto-publish. Review before going live.
- **Cache key** = `cmake-<runner>-<variant>-<hash(engine/ggml, CMakeLists)>`.
  Changing the runner image invalidates it (compiler abs-paths bake into
  `CMakeCache.txt`); the key includes the runner image to prevent stale restores.

## Reference

- Workflows: [`.github/workflows/release.yml`](../.github/workflows/release.yml),
  [`.github/workflows/cache-warm.yml`](../.github/workflows/cache-warm.yml)
