# release-process — reference

Depth beyond the runbook in [SKILL.md](SKILL.md). All line references are into
`.github/workflows/release.yml` unless stated. Verified against the workflow
files and the live v1.1.2 release on 2026-07-02. Only two workflows exist in
this repo: `release.yml` and `cache-warm.yml`.

## Build matrix — exact variants and flags

Pinned tool versions (release.yml:22-24): Node `22.16.0`, ONNX Runtime `1.25.1`.

### build-windows (release.yml:30-78) — runner `windows-2022` (pinned)

| Variant | CUDA toolkit | Extra CMake flags beyond the common set |
|---|---|---|
| `cuda13.1` | 13.1.0 | `-DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` |
| `cuda12.8` | 12.8.1 | same as cuda13.1 (compatibility build) |
| `cuda12-volta` | 12.8.1 | cuda flags + `-DGGML_CUDA_FORCE_CUBLAS=ON -DHOT_STEP_DISABLE_FA=ON -DCMAKE_CUDA_ARCHITECTURES=70-real` (Tesla V100 sm_70; ggml mma flash-attn/MMQ have no sm_70 device code) |
| `vulkan` | — | `-DGGML_VULKAN=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON` |
| `cpu` | — | `-DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BACKEND_DL=OFF` |

All Windows variants also set `-DHOT_STEP_STATIC_RUNTIME=ON`.

### build-linux (release.yml:763-803) — runner `ubuntu-22.04`

Pinned to 22.04 for wider glibc compatibility across distros. Same 5 variants;
same flags minus `-DHOT_STEP_STATIC_RUNTIME=ON`.

### build-macos (release.yml:549-550) — runner `macos-15`

Single Apple Silicon / Metal build. `macos-15` because Xcode 16+ is needed for
`std::from_chars<float>`.

### release job (release.yml:1110-1228) — runner `ubuntu-latest`

`needs: [build-windows, build-macos, build-linux]`. Downloads all artifacts,
collects `*.zip` / `*.tar.gz` / `*.sha256` into `release-assets/`, generates
notes, then `gh release create "<tag>" --draft --title "<tag>"
--notes-file release-notes.md release-assets/*`.

`fail-fast: true` on both release build matrices (release.yml:36, 766) — one
variant failure cancels its siblings. `cache-warm.yml` uses `fail-fast: false`
so a warm run finishes as many caches as it can.

## What's inside each archive

Assembled in "Assemble release" (Windows: release.yml:332-510; Linux/macOS have
mirrored steps). Each portable package bundles:

- Portable Node 22.16.0 runtime (`runtime/node.exe` on Windows, release.yml:345). **Never bump `NODE_VERSION` past 22.x** — Node 24+ breaks dependencies (repo `engines` field enforces `<24`); this pinned Node ships inside every archive, so a bump would publish broken packages.
- Engine binaries + Lua plugins (`engine/plugins` copied in, release.yml:400-405)
- Variant marker files (release.yml:495-504):
  - `engine/.variant` — `cuda` / `vulkan` / `cpu` (both CUDA builds are `cuda`
    from the server's perspective)
  - `engine/.cuda-version` — CUDA major version, only for cuda variants
    (runtime DLL version selection)
- esbuild-bundled server: `node release/esbuild.config.mjs` producing
  `server.mjs`; the job throws if it's missing (release.yml:437-441)
- `better-sqlite3` rebuilt/prebuilt against the portable Node
  (release.yml:307-313 Windows; `prebuild-install --target 22.16.0 --arch
  arm64` on macOS, release.yml:626)
- ffmpeg from `ffmpeg-static`'s install script (release.yml:294-303, 461-464)
- Prebuilt `ui/dist`, empty `models/` and `adapters/` dirs (release.yml:491-493)
- `.env.example`, `release/HOT-Step.bat` (or `.sh`), `release/README.txt`,
  `open-browser-if-needed.ps1` (release.yml:506-510)
- Guarded optional copies: `noise_samples/`, `Essentia/` (`if exists` — whether
  Essentia is present in a CI checkout was not verified)

Windows packages are `.zip` (compressed via .NET `ZipFile`, release.yml:522-527)
with a sidecar `.sha256` (release.yml:533-535). Linux/macOS are `.tar.gz` with
the same sidecar scheme.

## Cache machinery (why releases take ~10-15 min, not ~1.5 h)

- **Ref scoping:** GitHub Actions caches saved on `refs/tags/vX` are invisible
  to other tag runs. Only caches saved on the default branch (master) are
  visible to every run. Hence `cache-warm.yml`: it builds the engine on master
  under the exact keys release.yml restores.
- **Cache keys** (release.yml:209, 581, 894):
  - `cmake-windows-2022-<variant>-<hashFiles('engine/ggml/**','engine/CMakeLists.txt')>`
  - `cmake-linux-<variant>-<same hash>`
  - `cmake-macos-<same hash>`
  The runner image is baked into the Windows key deliberately — compiler
  absolute paths live in `CMakeCache.txt`, so a cache from a different image is
  poison. Note the key hashes only ggml + `engine/CMakeLists.txt`, **not** the
  CMake flags: flag edits do NOT invalidate the cache, while any
  `engine/CMakeLists.txt` change invalidates everything.
- **cache-warm.yml triggers** (cache-warm.yml:21-28): `workflow_dispatch`, or a
  master push touching `engine/ggml` (the submodule **gitlink**, i.e. a
  submodule pointer bump), `engine/CMakeLists.txt`, or the workflow file
  itself. It stops after the engine build — no packaging, no release.
- **Timestamp restore** (release.yml:213-237, mirrored per-OS and in
  cache-warm): `actions/checkout` sets every mtime to "now", which makes Ninja
  treat all cached objects as stale. The step rewrites mtimes from git commit
  history for `engine/`, and separately for the ggml submodule **from the
  submodule's own history** — the superproject only tracks the gitlink, and the
  CUDA kernels (the long pole) live inside the submodule.
- **Cache-restore self-repair** in the build step (release.yml:239-268):
  1. Delete every `CMakeCache.txt` recursively (top-level and nested
     ExternalProject ones) — dead ephemeral ninja / Vulkan-SDK absolute paths.
  2. Clamp future-dated mtimes to now — a cache written by a clock-ahead
     runner otherwise makes ninja loop `build.ninja still dirty ... system
     time is not set`.
  3. Nuke `vulkan-shaders-gen-prefix` — that ExternalProject loops
     `build.ninja still dirty after 100 tries` when restored; it rebuilds in
     seconds.
- **Known warm/release drift:** cache-warm's Windows `cpu` variant omits
  `-DGGML_BACKEND_DL=OFF` which release.yml's cpu variant sets
  (cache-warm.yml:78-82 vs release.yml:73-78). Harmless if OFF is the default,
  but keep the matrices in sync when editing either file — the file headers
  say "must match".

## Release notes internals (release.yml:1132-1218)

- Previous tag: `git describe --tags --abbrev=0 --exclude '*-*' "${TAG}^"`.
  Empty result (first release) falls back to the full `git log --oneline`.
- Bucketing is a case-insensitive grep on the first line of each commit:
  `^<hash> feat` → Features, `^<hash> fix` → Fixes, everything else → Other.
  Scopes like `feat(adapter):` match fine; a body-only `feat:` does not.
- Downloads table rows are matched by filename regex (`win.*cuda13`,
  `win.*cuda12-volta` before `win.*cuda12`, etc.) — renaming the archive
  pattern breaks table classification silently.

## Tag hygiene and history

- Real releases: `v1.1.2` (latest, published, 22 assets — verified live),
  `v1.1.1`, `v1.1.0`, ...
- Throwaway convention: any hyphenated `v*` tag. `docs/dev/releasing.md` uses
  `vX.Y.Z-CI-Test`; recent actual throwaways were lowercase
  `v0.0.0-ci-test-N`. Case/format of the suffix doesn't matter to the
  changelog guard — **only the hyphen does**.
- Local-only junk tags exist (e.g. `v1.5-pre-ggml-migration`,
  `v1.0.7a-pre-docker`). They are harmless while local; pushing any of them
  fires the full pipeline.
- Tags cannot be renamed; delete (remote + local) and recreate:
  `gh release delete <tag> --cleanup-tag --yes; git tag -d <tag>`.

## gh CLI cheat sheet (PowerShell)

```powershell
gh release list --limit 5
gh run list --limit 5
gh run view <run-id>
gh run view <run-id> --log                                   # finished runs only
gh api repos/scragnog/HOT-Step-CPP/actions/jobs/<job-id>/logs > log.txt   # in-progress runs
gh release view vX.Y.Z --json assets --jq '.assets | length'
gh release view vX.Y.Z --json body --jq .body
gh release edit vX.Y.Z --draft=false --latest
gh release delete vX.Y.Z-CI-Test --cleanup-tag --yes
gh workflow run cache-warm.yml                                # re-warm master cache
```

Quoting note: the `--jq` expressions above are single-quoted; in PowerShell
keep them single-quoted so `$` and `|` inside are passed literally.

## Known stale spots in docs/dev/releasing.md

- Line 43-45: says 4 Windows + 4 Linux variants and "18 assets (9 archives +
  9 .sha256)". Reality: 5 variants per OS (`cuda12-volta` added later), 22
  assets (11 + 11). Lines 68-72 repeat the 18 count.
- Everything else in the doc matched the workflows when checked (2026-07-02).
- The design doc `docs/plans/2026-05-11-release-automation-design.md`
  (referenced at release.yml:10) lives in gitignored `docs/plans/` and may not
  exist on a given checkout.
