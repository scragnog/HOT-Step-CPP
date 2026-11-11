# Releasing HOT-Step CPP — agent runbook

How to cut and publish a release, plus the non-obvious gotchas. Written for an
agent (or human) driving the process with the `gh` CLI on Windows/Git-Bash.

## TL;DR

Releases are cut by **pushing a `vX.Y.Z` tag**. The `Release` workflow builds
every platform and creates a **draft** GitHub Release; you review and publish it.
A separate `Cache Warm` workflow keeps the engine build cache on `master` so
release builds take ~10–15 min instead of ~1.5h for the CUDA jobs.

The separate `RunPod image` workflow (`.github/workflows/runpod-image.yml`) runs
when you **publish** the GitHub Release
(not on tag push, so the draft stays private) and publishes
`ghcr.io/scragnog/hot-step-cpp:<release-tag>`, plus `:latest` unless the release is
a prerelease or the tag is hyphenated (`-CI-Test`). It uses `GITHUB_TOKEN`
with `packages: write`, builds the `runpod` Docker target for Linux x86_64 and
has its own cache and concurrency group. It runs independently of the portable
release jobs, so inspect its result separately; neither workflow gates the other.
Ordinary `master` pushes do not build images. See the [RunPod guide](runpod.md)
for pulling the image or building your own.

Manual `RunPod image` dispatch must select an existing `v*` release tag; branch
dispatches fail before checkout or publishing, and the same `:latest` rule applies
(a manual run of a plain `vX.Y.Z` tag does update `latest`, including reruns of older
tags). Publishing a throwaway `-CI-Test` release builds and pushes its own image tag,
so delete drafts instead of publishing them. On the first GHCR
publication, check the package visibility and make it public for anonymous pulls,
or configure registry credentials in RunPod for a private package.

## Prerequisites

- `gh` authenticated (as `scragnog`).
- On `master`, working tree clean, everything committed **and pushed**.
- Pick a semver version **without a hyphen**: `vX.Y.Z` (hyphens are reserved for
  test/pre-release tags — see gotchas). Use exactly three numbers. The in-app
  update check (`server/src/services/updates/checker.ts`) reads GitHub's latest
  release and accepts only `vX.Y.Z`, so while a tag like `v1.4.1.1` or `v1.3` is
  the latest release, every install's check fails with "latest release is not a
  stable version tag".
- **Prerequisites check passes** (below). This one is not optional.

## 0. Check that users can actually get everything the build needs

```bash
node server/scripts/check-release-prereqs.mjs
```

Exit 0 or do not tag. It verifies that every model in the catalogue really
exists in its Hugging Face repo at the size claimed (and the sha256, where an
entry declares one), that the repos are public, that no pack points at a missing
file id, and that every runtime data file is packaged by `release.yml`. It also
fails when an engine binary the server spawns is missing from `release.yml`'s
package lists, when the portable layout checks below fail, when the TensorRT
files CI downloads are missing from Hugging Face, when the latest Cache Warm run
on `master` is still running or did not succeed (it asks `gh`), and when
`tools/docs/check-docs.mjs` fails. `--offline` skips only the Hugging Face and
TensorRT checks.

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
`models/` and `adapters/` folders, or the ones you pass with `--models` and
`--adapters` (take them from `cd server && npx tsx scripts/data-paths.ts` when
your data folder has moved), then walks the app through its API: every ACE
generation mode, a check that solvers, schedulers, guidance modes and adapters
actually change the output, MM3 and YuE2 renders, the audio tools, a few steps
of every trainer, and fixed-seed fingerprints. Budget about an hour on the 5090
with training included, or `--tiers 0-3` for the half-hour subset. The run ends
by staging every render in `_experiments/_LISTENING/<stamp>-release-gate/` with
a score sheet: listen before you tag, because nothing mechanical judges quality.
Details and flags: [tools/release-gate/README.md](../../tools/release-gate/README.md).

<!-- TODO(verify): the timings above are estimates; no full gate run is recorded under logs/release-gate/ in this checkout. -->

The example zip name is what `release/build-release.ps1 -Variant cuda` writes. A
CI archive, such as a preflight artifact, is named by its exact variant
(`win-x64-cuda13.1.zip`).

Every assembler (the local `release/build-release.ps1` and `package-release.sh`, and each CI platform job) also runs `release/smoke-boot.mjs` on the freshly bundled `server.mjs`, once the server and its native deps are staged, before anything is packaged. It boots the bundle from the staged release root, with a throwaway data folder and profile, no engine, on a free port, and fails the build if it exits or never answers `/api/health` (60 s). Run it with the Node the release ships, because the staged `better-sqlite3` is built for that version. A bundle that dies at startup cannot be zipped.

The server bundle is `server/server.mjs` plus `server/chunks/`. esbuild code splitting is on deliberately: without it, the top-level await in `dataroot/bootstrap.ts` plus the server's import cycles deadlock the bundle at startup (exit 13, the 1.4.1 bug).

## 1. Windows preflight before releasing

After the intended changes are committed and pushed to `master`, manually dispatch the
Release workflow. Both the push and dispatch require the maintainer's approval:

```bash
gh workflow run release.yml --ref master
```

Manual dispatch uses the same five Windows build jobs and fresh `windows-2022` archive
test jobs as a tag release. It derives `0.0.0-preflight-<12-character commit SHA>` from
the dispatched commit for both the packaged `VERSION` and archive names. There is no
version input or branch name in the archive path. It uploads artifacts only, creates no
tag or release, and skips Linux, macOS, Essentia and all release notes/creation work.

Inspect all five `test-windows-artifacts` results and their `windows-smoke-<variant>`
evidence artifacts. Each job verifies the shipping zip's SHA256, extracts it under
`HOT Step smoke` outside the checkout, boots the packaged server with its bundled Node,
checks engine endpoints with stub models, and runs the model-free backend diagnostic.
The fresh test VM installs no toolkit and receives no build tree. Child process `PATH`
contains Windows system directories plus the isolated registry-provisioned CUDA runtime.
Missing backend and runtime DLL negative checks must fail as expected. CUDA runtime
downloads are cached by their registry entries; the logs record bytes and download time.
See [Windows archive gate](building.md#windows-archive-gate) for acceptance and limits.

A tag release depends on every Windows archive test passing before creating its draft.
It collects the original checksummed `release-*` artifacts, excluding smoke evidence.
The Windows preflight proves startup and backend loading on hosted runners; the local
release gate and listening check above still cover model execution and audio quality.

### Optional CPU generation measurement

For a measured CPU trial, obtain approval for both pushing the reviewed commits and
dispatching the workflow. Push first: `--ref master` uses the remote branch, so local
commits are not included until pushed. After approval and a successful push, run:

```bash
gh workflow run release.yml --ref master -f cpu_generation_trial=true
```

The input defaults to `false`. This still builds and tests all five Windows variants;
only the added generation job uses the CPU archive. It downloads approximately
3.3 GB of models without caching, with a ten-minute download budget, then attempts
one ten-second LM generation and one-step synthesis within five minutes total,
including engine startup. It starts the archive's `ace-server.exe`, submits `/lm`,
polls `/job`, then submits `/synth?format=wav16` and fetches the completed WAV,
using the same HTTP sequence as the app. The trial does not need standalone
LM or synthesis executables.
It creates no tag or release and does not gate release creation.

Inspect the `cpu-generation-trial` job and its evidence artifact directly. The job
uses `continue-on-error`, so a green workflow alone does not prove generation passed.
Report the run link, `download.json` download seconds, `inference.json` inference
seconds and peak working set bytes, and whether WAV validation passed. If it times
out or fails, report that outcome rather than a successful generation time.
The artifact includes logs and any generated audio for inspection.
If the packaged server cannot start or become healthy, the step fails and records
null inference time and peak RAM, rather than reporting a measurement.

The reduced-step trial establishes execution only, not audio quality or normal
generation speed. Keep it non-gating until hosted measurements support a decision.
See [trial details and validation limits](building.md#optional-hosted-cpu-generation-trial).

### Optional full-platform compile test

To verify CI compiles without cutting a real release, push a throwaway
**hyphenated** tag — it triggers the same build pipeline but is ignored by the
changelog logic:

```bash
git tag -a vX.Y.Z-CI-Test -m "compile test" && git push origin vX.Y.Z-CI-Test
# ...watch it (section 3)... then delete when done:
gh release delete vX.Y.Z-CI-Test --cleanup-tag --yes   # removes draft + remote tag
git tag -d vX.Y.Z-CI-Test
```

A failed run creates no draft, so `gh release delete` reports that the release
was not found. Delete the remote tag with
`git push origin --delete vX.Y.Z-CI-Test` instead.

Re-pushing the **same** `-CI-Test` name is free (delete remote+local, recreate,
push). Tags cannot be renamed.

## 2. Cut the release

```bash
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z
```

This triggers `Release`: builds Windows (`cuda13.1`, `cuda12.8`, `cuda12-volta`,
`vulkan`, `cpu`), Linux (those five plus `rocm`) and macOS (Metal), then tests all
five Windows archives before creating a **draft** release with **24 assets**
(12 archives + 12 `.sha256`).

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

The workflow leaves the release as a **draft**. Before publishing, arrange the
release body in this order:

1. Curated release notes under a `Highlights` heading.
2. GitHub's generated `What's Changed` list and `Full Changelog` link, generated
   against the previous release tag. Keep both; never replace them with the
   curated notes.
3. The Downloads table, unchanged.

The draft's own `What's Changed` section is not GitHub's generated list. The
workflow builds it from commit subjects since the previous non-hyphenated tag,
bucketed into Features (`feat`), Fixes (`fix`) and Other, then appends the
Downloads table, a ROCm install note when a ROCm archive is present, and a line
about the `.sha256` files.

Verify the asset count and notes, then leave the draft for the maintainer to
publish. Publishing starts the `RunPod image` workflow (see the TL;DR), and once
the release is the latest one, the in-app update check reports it as an update
to older installs on the release channel (see
[Update notices](config.md#update-notices)). The maintainer uses:

```bash
gh release view vX.Y.Z --json assets --jq '.assets | length'   # expect 24
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
  `engine/ggml`, `engine/CMakeLists.txt` or the workflow file change, or via
  manual dispatch) and saves the build dir under the **same cache keys**
  `release.yml` uses. Release runs restore it and skip the CUDA compile (the long
  part). It warms the five Windows and six Linux variants only; there is no macOS
  job, so `build-macos` never gets a `master` cache.
- **Timings:** cold (no master cache) CUDA jobs ≈ 1.5h each; warm ≈ 7–13 min.
- **Tag only after Cache Warm has finished green.** A release job restores the
  cache once, at its start; a warm run still in flight is a miss and that job
  compiles cold. `check-release-prereqs.mjs` fails while the latest Cache Warm
  on `master` is running or failed (v1.3.5 hit both: Cache Warm had been
  failing for four days, then the Windows CUDA 13.1 cache landed seven minutes
  after the release job looked for it).
- **If CUDA suddenly rebuilds slow:** the master cache is missing/stale. Re-warm
  it: GitHub → Actions → **Cache Warm** → *Run workflow* (on `master`). It also
  auto-runs when `engine/ggml`/`CMakeLists.txt` change.
- Cache reuse depends on git-restored source mtimes (incl. the **ggml submodule**
  — its `.cu` files live in the submodule's own history, not the superproject).
- Commit times do not follow build order, so every engine build also writes
  `engine/build/.built-commit`. After a restore, the timestamp step touches every
  `engine/` file changed since that commit (all of `engine/src` and
  `engine/tools` when the stamp is missing or unknown) and every `engine/ggml`
  file changed between the cached and current ggml pins (all of `engine/ggml`
  when the cached pin cannot be resolved). This guard exists because of the
  v1.1.3 mixed-ABI crash (#82, #83).

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
- The step after the engine build asserts the configure log
  (`engine/build/configure.log`) contains
  `[TRT] Found vendored SDK` — if TRT silently failed to enable, the job fails
  loudly instead of shipping a cuda13.1 build with a stubbed-out renderer.
- Cache Warm has no TensorRT step, so the `master` cache for `cuda13.1` is
  configured without `HOT_STEP_TRT`. The release job's configure turns it on, and
  because `acestep-core` defines it `PUBLIC`, the engine targets recompile on
  every release while the ggml objects stay cached.
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
- **The Windows Vulkan SDK is pinned (1.4.357.0), not `latest`,** in both
  `release.yml` and `cache-warm.yml`. LunarG can list a new
  version before its `config.json` exists, and `latest` then 404s in
  `setup-vulkan-sdk`. To bump it, check
  `https://vulkan.lunarg.com/sdk/config/<ver>/windows/config.json` returns 200.
  Linux installs LunarG's `vulkan-sdk` apt package, which is not pinned.
- **Any pushed `v*` tag triggers the Release pipeline.** Use `vX.Y.Z` for
  releases and `-CI-Test` (or other hyphenated) tags for throwaway checks; delete
  them afterward. Don't push local feature tags that match `v*`.
- **Changelog range** = commits since the previous **non-hyphenated** tag
  (`git describe ... --exclude '*-*'`). This is why a stray `vX-CI-Test` tag must
  not be treated as a release; the exclude guard handles it, but still clean up.
- **The release is a draft** — it does not auto-publish. Review before going live.
- **Cache keys** hash `engine/ggml/**` and `engine/CMakeLists.txt`:
  `cmake-windows-2022-<variant>-<hash>`, `cmake-linux-<variant>-<hash>` and
  `cmake-macos-<hash>`. The Windows key names the runner image because compiler
  absolute paths bake into `CMakeCache.txt`; changing the image starts a new
  key. With no exact match, `restore-keys` restores the newest cache with the
  same prefix, so a changed hash still builds incrementally on the older cache.
  CMake flags are not in the key.

## Reference

- Workflows: [`.github/workflows/release.yml`](../../.github/workflows/release.yml),
  [`.github/workflows/cache-warm.yml`](../../.github/workflows/cache-warm.yml),
  [`.github/workflows/essentia.yml`](../../.github/workflows/essentia.yml) (called
  by `release.yml` for the Linux and macOS Essentia bundles),
  [`.github/workflows/runpod-image.yml`](../../.github/workflows/runpod-image.yml)

## Build identity

Every package carries `build-info.json` beside `VERSION`. All five assemblers (the three CI platform jobs, `release/build-release.ps1` and `package-release.sh`) write it with `release/build-info.mjs`, then run `node release/check-layout.mjs --package <root> --os <os> --variant <variant>`, which fails the build when the file is missing, breaks a rule in the table below, disagrees with the asset, or names a tag that `VERSION` does not match.

| Field | Value |
|---|---|
| `format` | `1` |
| `tag` | The release tag, only when it names the packaged commit; otherwise `null` |
| `sha` | Full source commit (HEAD when packaged) |
| `channel` | `release` with a tag, `master` without |
| `os`, `arch` | `windows`/`linux` with `x64`, or `macos` with `arm64` |
| `variant` | Exact asset variant: `cuda13.1`, `cuda12.8`, `cuda12-volta`, `vulkan`, `cpu`, `rocm` (Linux), `metal` (macOS) |
| `engineSha` | Git tree id of `engine/` at that commit; it changes with any engine source or ggml pin change |
| `updaterProtocol` | Windows uses `UPDATER_PROTOCOL` in `release/build-info.mjs`, currently `1`. Linux and macOS default to `0` regardless of that constant. Explicit `--protocol` overrides the default for one build |

The package check also matches the ggml backend library in `engine/` to the variant, and `engine/.variant` and `.cuda-version` when present. `engine/.variant` keeps its coarse meaning (`cuda` for every CUDA asset) for the code that reads it. The server reads the file in `config.ts` (`BUILD_INFO`) and reports `build.tag`, `build.sha`, `build.variant` and `build.channel` in `/api/health`. A package without the file (an older release), a package whose file breaks those rules (the server logs a warning and ignores it), or a git checkout has unknown identity: `sha` falls back to the checkout's HEAD and the rest is `null`. Nothing guesses a variant. The update notices read the same file for the installed identity (`server/src/services/updates/install.ts`).

## Windows updater layout (protocol 1)

Windows releases ship protocol 1. `UPDATER_PROTOCOL` sets the Windows default only; Linux and macOS stay notify-only at protocol 0. A Windows package built with `updaterProtocol` 1 is the versioned layout the in-app updater installs into; `.\release\build-release.ps1 ... -UpdaterProtocol 0` still builds the flat package.

The first protocol-1 release updates nobody: a flat install refuses apply, so every Windows user downloads that release by hand once. Its release notes must say so, and must say that the zip can go over the old folder or into a new one, and that an install from before v1.4.1 (data still inside the app folder) put into a new folder needs **Import previous install** before anything is created in the new library ([Data folder](../user/data-folder.md)). Unzipped over the old folder, the first start moves that data out by itself. Both Windows assemblers run `node release/updater/package-layout.mjs <staging>` and then the package check just before zipping, so a local build and CI produce the same archive. For protocol 0 the script changes nothing.

```
HOT-Step.bat        runs runtime\node.exe launcher.mjs (release/updater/HOT-Step.bat)
launcher.mjs        the stable launcher (release/updater/launcher.mjs), never replaced by an update
runtime/node.exe    the Node that runs the launcher
current.json        {"format":1,"payload":"<id>"}, one line, written atomically
README.txt
app/<id>/           the old flat package, minus its HOT-Step.bat, plus payload-manifest.json
updater/            created at run time: staging, transaction.json, lease, snapshots, last-result.json
```

`<id>` is `<tag>-<first 12 of sha>-<variant>`, for example `v1.6.0-0123456789ab-cuda13.1`. `payload-manifest.json` holds the SHA-256 of every file in the payload; apply refuses a payload with added, edited or missing files, because an update would strand them in an old folder. The archive check (`checkPackageBuildInfo`) accepts both layouts and, for the versioned one, requires exactly one payload named for its build. Update staging (`server/src/services/updates/download.ts`) checks the same layout.

The Windows archive smoke gate and optional CPU generation trial resolve `current.json` with the launcher pointer reader, check the whole archive with `checkPackageBuildInfo`, and run binaries from the selected payload. Flat archives still run from the extraction root. `node release/smoke-artifact.mjs payload-root <extracted-root>` exposes that same check to the CPU trial. The updater harness explicitly builds its flat fixture with `-UpdaterProtocol 0`, independent of the release default.

The launcher is the only thing that starts a server on the root. It refuses a second launcher (a named pipe the OS frees on exit), settles an interrupted update before anything opens SQLite, clears a stale `updater/prepare.lock`, pins the profile (and the data folder only where the server would otherwise derive it from `app/<id>`), and starts `app/<id>/runtime/node.exe app/<id>/server/server.mjs` with an IPC channel. An update runs like this:

1. The running server's `POST /api/updates/apply` opens the maintenance transaction and sends the launcher `apply` over IPC. The launcher checks it (staged payload, lease and token, snapshot list) and acknowledges.
2. The server stops through `exitForLauncher()` in `routes/shutdown.ts` and reports `stopped` with `clean`. Without a clean report and exit code 0 the candidate is not started; the old version restarts and the result is `failed`.
3. The old payload's `server/updater-tool.mjs` snapshots the databases and files, `transaction.json` is written `pending`, the staged `app/<id>` moves into `app/`, and `current.json` switches.
4. The candidate starts under the lease token with the data folder pinned. Within 300 s `GET /api/maintenance/status` must report the gate closed under this lease, the exact build, the DB open, the UI files and an engine answer. The engine part is `engine: { ready, noModels, failed, status }`. `ready` is true only after the primary engine's own `/health` has answered; the server sets it the same way at every start, not just during updates. `noModels` means the binary ran and reported an empty models folder, and it also passes, so a user can update before downloading a model. `failed` (binary missing, spawn failed, or the crash limiter gave up) ends the wait at once. Otherwise the launcher waits out the bound, which allows for a slow first start of a new engine binary. The failure message carries the engine `status`.
5. Pass: `committed`, `POST /api/maintenance/release`, lease and record removed, result `installed`, browser opened. The old payload and the snapshot stay until a later plain start or restart answers `/api/health` and deletes them (`updater/retained.json`). A payload that a setting points into is kept: process env and `settings/.env` are read with dotenv's rules and relative values resolve against the data folder, as `config.ts` does. If the settings cannot be read, no payload is deleted.
6. Fail: the candidate is stopped, `current.json` goes back, the old payload's tool restores the snapshot, and the old version starts (`rolled-back`). If the restore fails, the lease and record stay, nothing starts and the result is `recovery-required`. The window then prints the message, the first step (close what locks files, check disk space, start again) and the full URL of [Recovery after an update](../user/update-recovery.md) (`failureText` and `RECOVERY_DOC` in `launcher.mjs`). The launcher is never replaced by an update, so that text stays general and the page carries the detail; keep the page at that path.

At start-up a `pending` record means step 6. A `committed` record means the new version is kept and never restored: it is started under a fresh lease and step 5 is finished. That check skips the engine: the engine already answered before the commit, and a broken engine must not stop a working server and database from starting. The app then shows the engine's status. Malformed pointer, record or lease files stop the launcher with nothing started. Locked or full targets get three attempts within about 15 s; a lock on an old payload only delays its deletion.

**Maintainer note: a user stuck at `recovery-required` with a damaged snapshot.** The user page tells them to copy `updater/` and ask for help, and that is all it should say. One way out exists, for the maintainer to judge case by case. With the launcher stopped and `updater/` copied aside, change only `"state": "pending"` to `"committed"` in `updater/transaction.json`. The next start then takes the committed path: `current.json` switches to the new payload, it starts on the data as it is, nothing is restored and the result is `installed`. The record and both leases are removed; `retained.json` keeps the old payload, the snapshot and the staging folder. This has been run once, on a packaged fixture where the candidate had never started (the launcher was killed before the pointer write, and the snapshot was damaged afterwards). It says nothing about a candidate that failed its health check or changed the data, and the committed path skips the engine check. Suggest it only when the message shows the restore stopped before it touched anything and the new version is known to be good.

A protocol-1 fixture for updater tests: tag the commit locally (`git tag v1.6.0`, never pushed), build with `-UpdaterProtocol 1`, and unzip it to a temporary folder outside the checkout with its own `HOT_STEP_PROFILE_DIR`. The candidate needs a second build with a newer tag. Fault cases: `node --test release/updater/launcher.test.mjs`. The acceptance run on real packages (clean update, failed engine, power loss at each boundary, busy refusal, damaged snapshot) is `tools/update-gate/`; its README has the commands.

## Data layout acceptance

Archives contain app binaries and immutable `server/data/` resources. User data is initialized in sibling `HOT-Step-Data` or reused through the profile pointer. Package `release/.env.example`, `profile-path.mjs`, the platform launcher and `README.txt`, plus `open-browser-if-needed.mjs` on Windows (a Node helper; the launcher must never start PowerShell, which antivirus quarantines); do not create mutable app-root models or adapters. Windows uses `runtime/node.exe`, Linux and macOS use `runtime/bin/node`. The CI archives also carry `presets/yue2-plan-play/`, copied from `docs/user/presets/yue2-plan-play/`, and each platform job fails when `plan-play-sections.json` is missing from it. Local packagers follow the same layout.

The prereq script checks three workflow platform sections and both local assemblers, required engine targets and wholesale resource copies, unset managed defaults, profile marker agreement and disposable release/dev fixtures. Run it without `--offline` to retain Hugging Face checks. Actual native platform startup remains a release smoke-test requirement. See [Data folder and upgrades](../user/data-folder.md).
