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
