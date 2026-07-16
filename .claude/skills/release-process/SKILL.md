---
name: release-process
description: Runbook for cutting and publishing a HOT-Step CPP release via a v* git tag that triggers the multi-platform CI build and drafts a GitHub Release. Use when asked to cut a release, publish a release, bump the version, push a version tag, run a CI compile test, verify release assets, or debug a failed Release/Cache Warm workflow run.
---

# Cutting & publishing a release — HOT-Step CPP

HOT-Step CPP ships as portable archives for Windows (CUDA/Vulkan/CPU), Linux, and
macOS. A release is produced entirely by CI: you push a git tag matching `v*`,
the `Release` workflow (`.github/workflows/release.yml`) builds every platform
variant and creates a **draft** GitHub Release on `scragnog/HOT-Step-CPP`. You
then verify the draft and publish it with the `gh` CLI. There are **no version
numbers to edit in any file** — the git tag is the sole version source.

All commands below are Windows PowerShell (use `;` to chain, never `&&` in
older shells — this repo's convention). `gh` is already authenticated as
`scragnog`.

## When to use this skill

- The user asks to cut, tag, or publish a release (`vX.Y.Z`).
- The user wants a throwaway CI compile check of the full build matrix.
- A `Release` or `Cache Warm` workflow run failed and needs diagnosing.
- You are about to push ANY tag starting with `v` (read Golden rules first).
- Someone asks "where do I bump the version?" (answer: nowhere — see step 2).

## Golden rules (hard constraints)

1. **ANY pushed tag matching `v*` triggers the full multi-platform CI build.**
   The trigger is `on: push: tags: ['v*']` (release.yml:14-17) with no other
   filter. That is ~11 GitHub-hosted jobs including multiple CUDA toolkit
   installs. Never push a local marker/feature tag that starts with `v`.
   WHY: historical junk tags like `v1.5-pre-ggml-migration` exist locally;
   pushing one fires the whole pipeline and can pollute the release list.
2. **Real releases are plain semver `vX.Y.Z` — no hyphen. Throwaway builds
   MUST be hyphenated**, conventionally `vX.Y.Z-CI-Test`. WHY: release-notes
   anchoring finds the previous release with
   `git describe --tags --abbrev=0 --exclude '*-*' "${TAG}^"` (release.yml:1138),
   so hyphenated tags are skipped by the changelog. A stray *non-hyphenated*
   tag between releases would silently truncate the next release's notes.
3. **Pushing a tag requires explicit user approval** (repo git rule: every push
   needs approval). Ask before `git push origin vX.Y.Z`.
4. **Do not bump any `package.json` version.** `server/package.json` and
   `ui/package.json` both sit at `"version": "1.0.2"` while v1.1.2 has shipped —
   these fields are stale and read by nothing in the release pipeline. There is
   no root `package.json`. Editing them achieves nothing.
5. **Everything must be committed AND pushed to `origin/master` before
   tagging.** CI checks out the tag's commit, not your working tree —
   UNCOMMITTED work will not be in the release. (Committed-but-unpushed work
   *would* technically ship via the tag push itself, but push master first
   anyway so `origin/master` is never behind a published release tag.) Stage
   explicit paths only — NEVER `git add -A` or `git add -f` (untracked dirs
   like `data/`, `models/`, `toinstall/` sit in the tree right now and a
   blanket add would sweep them in).
6. **The release is created as a DRAFT** (`gh release create --draft`,
   release.yml:1220-1228). Nothing auto-publishes. **Publishing
   (`--draft=false`) requires explicit user approval, same as a push** —
   present the asset count and notes to the user and wait for their
   go-ahead. Shipped binaries deserve at least the scrutiny of a push.
7. **Never change the Windows runner from `windows-2022`** (release.yml:34).
   `windows-latest` is windows-2025 whose MSVC (`_MSC_VER >= 1950`) is rejected
   by CUDA 12.8/13.1 `nvcc` (`host_config.h`: "Only the versions between 2017
   and 2022 are supported"). The runner image is also baked into the build
   cache key on purpose.
8. **Clean up hyphenated test tags after use** (release draft + remote tag +
   local tag). WHY: leftover drafts clutter the release page, and tag hygiene
   protects the changelog logic.

## Procedure

### 0. Preconditions

- On `master` (the only branch used in this repo), working tree clean.
- `git status` clean; `git push` of master already done (with user approval).
- **If master recently absorbed an upstream acestep.cpp sync, run
  `engine/verify-hooks.ps1` before tagging** — the sampler hook can be lost
  silently (compiles, but all solvers/schedulers/guidance go dead) and
  nothing in CI or the asset check catches it.
- Pick the version: look at the latest release
  (`gh release list --limit 3`) and bump semver appropriately.

### 1. Optional: throwaway compile test first

Use this to verify CI compiles across all platforms without cutting a real
release. Hyphenated tags build everything but are excluded from changelog
anchoring.

```powershell
git tag -a vX.Y.Z-CI-Test -m "compile test"
git push origin vX.Y.Z-CI-Test        # ask the user first — this is a push
# ... monitor (step 3) ...
# cleanup when done:
gh release delete vX.Y.Z-CI-Test --cleanup-tag --yes   # deletes draft + remote tag
git tag -d vX.Y.Z-CI-Test
```

If the compile test FAILED, no draft exists (the release job needs all three
build jobs green), so `gh release delete` errors with "release not found".
Delete the tag directly instead:

```powershell
git push origin --delete vX.Y.Z-CI-Test; git tag -d vX.Y.Z-CI-Test
```

Re-pushing the same `-CI-Test` name after deletion is fine. Tags cannot be
renamed — delete and recreate.

### 2. Version bump — there is none

This is the biggest doc/reality trap. "Cutting a release" is: commit + push
master, create tag, push tag. **No file edits.**

- The tag is the version. `release.yml` derives the version string from
  `${{ github.ref_name }}` (e.g. release.yml:335, 515) purely for archive
  naming: `HOT-Step-CPP-vX.Y.Z-win-x64-<variant>.zip`.
- Engine binaries embed the **git short hash + commit date**, not semver:
  `engine/tools/version.cmake:29` writes
  `#define ACE_VERSION "<short-hash> (<date>)"` into a generated `version.h`,
  wired as an always-run CMake target in `engine/CMakeLists.txt:9-15`.
  Automatic at build time — never edit it.
- Nothing in `server/src` checks GitHub for updates, so no in-app version
  string needs touching either.

### 3. Cut the release

```powershell
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z                # requires explicit user approval
```

To rebuild against a different commit: delete the tag remotely and locally,
re-create it on the new commit, re-push (the workflow re-runs).

### 4. Monitor the build

```powershell
gh run list --limit 5                 # find the Release run for your tag
gh run view <run-id>                  # per-job status + timings
```

Expected jobs: `build-windows` × 5 variants (`cuda13.1`, `cuda12.8`,
`cuda12-volta`, `vulkan`, `cpu`), `build-linux` × the same 5, `build-macos`
(Apple Silicon Metal, `macos-15`), then a final `release` job that collects
artifacts, generates notes, and creates the draft. Warm-cache CUDA jobs run
roughly 7–13 min; a cold cache means ~1.5 h per CUDA job (timings from workflow
comments, not re-measured — see Failure signatures for the fix).

To read a failed/cancelled job's log **while the run is still in progress**
(`gh run view --log` won't show it yet), pull it from the API:

```powershell
gh api repos/scragnog/HOT-Step-CPP/actions/jobs/<job-id>/logs > log.txt
```

(If you happen to be in Git-Bash instead of PowerShell, prefix with
`MSYS_NO_PATHCONV=1` so the leading-slash API path isn't mangled.)

Note `fail-fast: true` on the release build matrices: one variant failing
cancels its siblings. Cancelled siblings are not the root cause — find the one
that *failed*.

### 5. Verify the draft, then publish

```powershell
gh release view vX.Y.Z --json assets --jq '.assets | length'   # expect 22
gh release view vX.Y.Z --json body --jq .body                  # eyeball notes + download table
gh release edit vX.Y.Z --draft=false --latest                  # publish — ONLY after explicit user approval (Golden rule 6)
```

**Expect 22 assets** (11 archives + 11 `.sha256`) — verified live against
v1.1.2. `docs/RELEASING.md` says 18; that count is **stale** (predates the
`cuda12-volta` variant). Asset names:

- `HOT-Step-CPP-vX.Y.Z-win-x64-{cuda13.1,cuda12.8,cuda12-volta,vulkan,cpu}.zip`
- `HOT-Step-CPP-vX.Y.Z-linux-x64-{same 5}.tar.gz`
- `HOT-Step-CPP-vX.Y.Z-macOS-arm64.tar.gz`
- one `.sha256` per archive

If the count is short, do NOT publish — a build job failed or an artifact
upload was missed; go back to step 4.

### 6. Cleanup

Delete any leftover hyphenated test tags and their draft releases (step 1
cleanup commands). Leave nothing matching `v*-*` on the remote.

## Release notes — shaped by commit messages

The `release` job (release.yml:1132-1218) buckets commits since the previous
non-hyphenated tag by conventional-commit prefix on the first line:
`feat*` → "🎵 Features", `fix*` → "🔧 Fixes", everything else → "📝 Other",
then appends a per-file Downloads table and a SHA256 note. **Commit message
discipline on master directly becomes the release notes** — write
`feat(...)`/`fix(...)` first lines that read well in a changelog.

## Key files

| Path | Role |
|---|---|
| `.github/workflows/release.yml` | The entire pipeline: 5 Windows + 5 Linux + 1 macOS builds, notes generation, draft creation. Only workflow triggered by `v*` tags. |
| `.github/workflows/cache-warm.yml` | Builds the engine on master under the same cache keys release.yml uses, so tag runs can restore it (GitHub caches are ref-scoped; only master caches are visible to tag runs). Triggers: manual dispatch, or master push touching `engine/ggml` (submodule gitlink), `engine/CMakeLists.txt`, or itself. |
| `docs/RELEASING.md` | Human runbook. Mostly accurate; asset count (18) and variant list (4/OS) are stale — reality is 22 assets, 5 variants per OS. |
| `engine/tools/version.cmake` | Generates `version.h` with `ACE_VERSION "<git-hash> (<date>)"` at build time. Never hand-edit versions. |
| `engine/CMakeLists.txt:9-15` | `version` custom target wiring for the above. |
| `release/` | Packaging inputs used by CI: `esbuild.config.mjs` (bundles server to `server.mjs`), `HOT-Step.bat`/`HOT-Step.sh` launchers, `README.txt`. |
| `server/package.json`, `ui/package.json` | `version` fields are stale (1.0.2) and unused by the pipeline. Do not bump. |

## Failure signatures

| Symptom | Cause → fix |
|---|---|
| CUDA jobs take ~1.5 h instead of ~7–13 min | Master build cache missing/stale. Re-warm: `gh workflow run cache-warm.yml` (or GitHub → Actions → Cache Warm → Run workflow on master), wait for it, then re-run the release (delete + re-push the tag). |
| `nvcc` fatal / `host_config.h` "Only the versions between 2017 and 2022 are supported" | Someone switched the runner to `windows-latest` (= windows-2025). Restore `runs-on: windows-2022` (release.yml:34). |
| CMake configure fails after cache restore (stale `cl.exe`/ninja path) | Cache from a different runner image or dead ephemeral tool path. The build step's self-repair deletes every `CMakeCache.txt` (release.yml:250-252); if it recurs, check the runner-image part of the cache key wasn't changed (release.yml:209). |
| ninja loops `build.ninja still dirty ... system time is not set` | Future-dated mtimes from a cache written by a clock-ahead runner; handled by the mtime clamp (release.yml:254-261). |
| Vulkan job loops `build.ninja still dirty after 100 tries` | Restored `vulkan-shaders-gen-prefix` ExternalProject; handled by the prefix nuke (release.yml:263-268). |
| Release notes show only 2–3 commits | Changelog anchored to a stray **non-hyphenated** tag pushed between releases (the `--exclude '*-*'` guard only protects against hyphenated ones). Delete the stray tag, delete the draft, re-push the release tag. |
| Draft has fewer than 22 assets | A build job failed (fail-fast cancelled siblings) or an artifact upload was missed. Don't publish; `gh run view <run-id>` and pull the failed job's log via the API command in step 4. |
| Whole matrix cancels when one variant fails | `fail-fast: true` on the release matrices — expected. Diagnose the variant that actually failed. |
| `throw "Build failed: ace-server.exe not found"` in a Windows job | The engine build produced no binary — a real compile failure earlier in that job's log. To reproduce/fix locally, follow CLAUDE.md's build rules: `dev-rebuild.bat` (never `engine/build.cmd` directly) and never `cmake --clean-first` (20+ min CUDA recompile). |
| Released binary crashes instantly (Windows `0xC0000409` / Linux SIGSEGV) right after `[Server] Models: ...`, but builds green and local build is fine | **Stale-object mixed-ABI binary from the build cache** (bit v1.1.3, 2026-07-16, issues #82/#83). The git-mtime-restore stamps sources with their *commit* time; any commit authored while the last Cache Warm was still running (or otherwise not in the warm build but committed before its cache-save time) gets an mtime OLDER than the cached `.obj` files, so ninja never rebuilds its dependents. If that commit changed a struct (e.g. `AceRequest` in `request.h`), TUs disagree on layout → memory corruption on first use (`/props` is the first endpoint to touch `AceRequest`, hit by the UI on load — hence "crashes on startup"). Only variants whose warm job SAVED after the stray commit's timestamp are affected (slow CUDA jobs), which is why cpu/vulkan variants work — check warm job `completedAt` vs `git log --pretty=%cI` of struct-touching commits. Fix: force a cold build (bump the cache key by touching `engine/CMakeLists.txt`) or re-warm from current master, then re-release. Smoke-test releases with `ace-server.exe --models <dir>` + `curl /props`, not just job success. |

## Institutional knowledge

Facts from the departing lead engineer, verified against the workflows on
2026-07-02 unless noted:

- **VALIDATED — the tag is the only version.** No root `package.json`; the
  server/ui `version` fields are dead (stale at 1.0.2 vs v1.1.2 shipped, both
  checked). Engine binaries self-version from git via `version.cmake`.
- **VALIDATED — cache scoping is why cache-warm exists.** GitHub Actions caches
  are ref-scoped: a cache saved by tag run A is invisible to tag run B; only
  default-branch (master) caches are visible to all runs. `cache-warm.yml`
  builds on master under the same keys
  (`cmake-windows-2022-<variant>-<hashFiles('engine/ggml/**','engine/CMakeLists.txt')>`
  and Linux/macOS equivalents) so tag runs get warm restores.
- **VALIDATED — timestamp restore is load-bearing.** `actions/checkout` sets
  all mtimes to "now", which makes Ninja rebuild everything despite a cache
  hit. The workflow restores git commit mtimes for `engine/` AND separately
  for the ggml **submodule from its own history** (release.yml:213-237) — the
  CUDA kernels, the long pole, live in the submodule, and the superproject only
  tracks the gitlink.
- **VALIDATED — selective submodule init is intentional.** `submodules: false`
  on checkout; jobs manually init `engine/ggml` and `engine/vendor/vst3sdk`
  with only the `base cmake pluginterfaces public.sdk` sub-submodules
  (release.yml:87-96) — vstgui4 is "fragile + unneeded".
- **VALIDATED — cuda12-volta exists for Tesla V100 (sm_70).** ggml's mma
  flash-attention and MMQ have no sm_70 device code, so that variant builds
  with `-DGGML_CUDA_FORCE_CUBLAS=ON -DHOT_STEP_DISABLE_FA=ON
  -DCMAKE_CUDA_ARCHITECTURES=70-real` (release.yml:56-65).
- **VALIDATED — minor warm/release flag mismatch:** `cache-warm.yml`'s Windows
  `cpu` variant omits `-DGGML_BACKEND_DL=OFF` which `release.yml`'s cpu has
  (cache-warm.yml:78-82 vs release.yml:73-78). Harmless if OFF is the CMake
  default; note the cache *key* hashes only ggml + CMakeLists, not the flags.
- **VALIDATED — releasing never touches the local dev build.** No
  `dev-rebuild.bat` involvement; CI builds from scratch/cache on runners.
- **VALIDATED — auth:** `permissions: contents: write` (release.yml:19-20) +
  `GH_TOKEN: ${{ github.token }}` is all `gh release create` needs; no PAT.
- **UNVALIDATED — exact timings.** The 7–13 min warm / ~1.5 h cold CUDA figures
  come from workflow and doc comments, not re-measurement.
- **UNVALIDATED — Essentia dir in CI.** `release.yml` copies an `Essentia/`
  dir if present (guarded `if exists`); whether it exists in a CI checkout was
  not verified. Harmless either way.

Deeper detail (packaging contents, cache keys per OS, pinned tool versions,
recent test-tag naming history): see [reference.md](reference.md).

## Deeper reading

- `docs/RELEASING.md` — committed human runbook. Trust it EXCEPT the asset
  count (says 18, reality 22) and the variant list (missing `cuda12-volta`).
- `.github/workflows/release.yml` and `cache-warm.yml` — ground truth; when
  the doc and the workflow disagree, the workflow wins.
- `docs/plans/2026-05-11-release-automation-design.md` — design doc referenced
  at release.yml:10. `docs/plans/` is **gitignored/local-only and may be
  absent** on your checkout.
- `CLAUDE.md` — repo-wide git rules (master only, explicit-path staging, push
  needs approval) that still apply during releasing.
