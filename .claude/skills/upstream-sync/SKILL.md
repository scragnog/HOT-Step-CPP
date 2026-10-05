---
name: upstream-sync
description: Safely pulls upstream acestep.cpp changes into the HOT-Step engine fork without destroying its integration hooks. Use when syncing/merging upstream acestep.cpp updates, verifying or repairing hook files, or diagnosing "unresolved external symbol hotstep_sampler_linked_" or dead solvers/schedulers/guidance after a sync.
---

# Syncing with upstream acestep.cpp

## When to use this skill

- Pulling new commits from upstream `acestep.cpp` (https://github.com/ServeurpersoCom/acestep.cpp) into this fork's `engine/`.
- Checking whether a sync is pending, or repairing a botched one.
- Diagnosing: linker error `unresolved external symbol hotstep_sampler_linked_`, or a build that compiles clean but all solvers/schedulers/guidance modes stopped working.

**Background you need (no prior context assumed):** The C++ engine in `engine/` is a fork of acestep.cpp, a GGML port of the ACE-Step music model. The generation pipeline is LM → DiT → VAE, where **DiT** = Diffusion Transformer, the denoising model. HOT-Step replaces upstream's DiT sampling loop (`dit-sampler.h`) with its own (`hot-step-sampler.h`) that adds 17 solvers, 9 schedulers, guidance modes, and Lua plugins. The fork is **not** a git merge — it is file-copy based, using three zones:

| Zone | What | Sync strategy |
|---|---|---|
| **Upstream Zone** | Most of `engine/src/*.h` / `*.cpp` — kept byte-identical to upstream | Direct `Copy-Item` from a vanilla clone |
| **HOT-Step Zone** | `engine/src/hot-step-*`, `engine/src/solvers/`, `engine/src/schedulers/`, `engine/src/guidance/`, `engine/src/mastering.h`, `engine/src/adapter-runtime.h`, `engine/src/adapter-merge.h`, `engine/src/model-registry.h`, `engine/tools/hot-step-server.cpp` — never exists upstream | Never conflicts; never copy |
| **Modified Upstream Zone** | Exactly 3 files: `pipeline-synth-ops.cpp`, `model-store.h`, `dit.h` — upstream files carrying HOT-Step `#include` hooks | **Manual merge only, NEVER direct-copy** |

> **⚠ Zone model drift (verified 2026-07-14):** two of this table's assumptions no longer hold.
> 1. **`src/solvers/` now EXISTS upstream** (added 2026-05-06, commit `8974778` — modular Euler/SDE/DPM++3M/STORK4 dispatch). Its 7 filenames collide with the fork's own `engine/src/solvers/` (created independently 2026-04-18, commit `889c250`, different interface). The Phase 2 copy loop MUST also exclude everything under `src/solvers/` — a direct copy clobbers HOT-Step's native solvers with incompatible upstream versions.
> 2. **The Upstream Zone has eroded.** As of 2026-07-14, 11 of the 13 upstream-touched files since `31cc9ea` are fork-modified (e.g. `qwen3-lm.h` carries safetensors loading via fork-only `weight-source.h`/`config-json.h`; `backend.h`, `dit-graph.h`, `vae.h`, `request.*`, `wav.h`, `qwen3-enc.h`, `pipeline-synth.*` also differ). Do NOT assume direct-copy safety from this table — before copying any file, verify pristineness: diff the fork's copy against `git show <lastSynced>:src/<file>` from the upstream clone (with `--strip-trailing-cr`). Pristine → copy; modified → manual merge.

`engine/tools/ace-server.cpp` is upstream's server kept as a direct-copy **reference only — not compiled**. The shipped `ace-server` binary is built from `engine/tools/hot-step-server.cpp` (see `engine/CMakeLists.txt:416-418`).

The full workflow doc lives at `docs/plans/upstream-sync-workflow.md`, which is **gitignored (local-only)** — other machines will not have it. This skill is the portable distillation, with two known errors in that doc corrected (see [reference.md](reference.md)).

## ggml is not part of this sync

acestep.cpp pins its own ggml (ServeurpersoCom/ggml). HOT-Step does not follow that pin:
`engine/ggml` is pinned to HOT-ggml's `hot-step-neutral` branch, which is ggml-org master plus
HOT-Step's ggml changes as commits ([docs/dev/ggml-fork.md](../../../docs/dev/ggml-fork.md)).
Copying acestep.cpp files never touches `engine/ggml`. If a synced file needs a newer ggml
op or fix, land it on HOT-ggml first (cherry-pick from ServeurpersoCom/ggml or ggml-org),
then move the gitlink, then rerun `engine/verify-hooks.ps1` (hooks 7-17 cover the fork).

## Golden rules (hard constraints)

1. **NEVER direct-copy the 3 Modified Upstream Zone files** (`pipeline-synth-ops.cpp`, `model-store.h`, `dit.h`). WHY: copying `pipeline-synth-ops.cpp` wholesale is the *silent* failure — it compiles clean but kills every HOT-Step sampler feature (see hook table below). This actually happened once; the whole defense system exists because of it.
2. **ALWAYS run `engine/verify-hooks.ps1` after copying files and BEFORE building.** WHY: it catches hook loss in seconds; a build wastes minutes and a silent regression wastes hours.
3. **Commit the fork locally before starting.** WHY: file-copy sync has no git-merge undo; the commit is your only rollback point.
4. **NEVER push the `v1.5-upstream-sync-*` tag** (or anything without explicit user approval). WHY: any pushed `v*` tag triggers the full multi-platform CI release build. The tag is fine locally.
5. **Rebuild via `dev-rebuild.bat` at repo root, NEVER `engine/build.cmd` directly.** WHY: the app may be running without you knowing (you cannot reliably tell); the Node server auto-respawns ace-server on crash → infinite respawn + file-lock loop. `dev-rebuild.bat` handles clean shutdown and is a safe no-op when nothing is running. It does not relaunch the app afterwards: start it again with `dev.bat` (detached) and check `http://localhost:3000`.
6. **NEVER `cmake --build . --clean-first`.** WHY: CUDA kernel recompile is 20+ min. Stale `.obj`? Delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
7. **Never add HOT-Step code to Upstream Zone files** (new behavior goes in a `hot-step-*` file), and **never add `hot-step-*` files to the upstream clone.** WHY: keeps future syncs a trivial copy.
8. Git hygiene: all work on `master`, stage explicit paths only, never `git add -A`.

## The three hook files (verified line numbers)

| File | Hook | If clobbered by direct copy |
|---|---|---|
| `engine/src/pipeline-synth-ops.cpp` | `#include "hot-step-sampler.h"` at line 9 (replaces upstream's `dit-sampler.h`). Lines 10–12 are also fork-only: `hot-step-sampler-trt.h`, `adapter-trt.h`, `stream-pipeline.h` | **SILENT at compile time** — all solvers/schedulers/guidance/custom timesteps/DCW/APG go dead. Caught only by the linker sentinel |
| `engine/src/model-store.h` | `#include "hot-step-params.h"` at line 53 (provides `AdapterGroupScales`) | Compile error (self-catching) |
| `engine/src/dit.h` | `#include "adapter-merge.h"` (line 11) + `#include "adapter-runtime.h"` (line 13); line 12 `hot-step-build-flags.h` is also fork-only | Compile error (self-catching) |

**Why the sampler hook failure is silent:** the hook is a *substitution*, not an addition. Upstream `dit-sampler.h` and fork `hot-step-sampler.h` both provide a function named `dit_ggml_generate`. A whole-file copy of upstream `pipeline-synth-ops.cpp` replaces both the include AND the call site (`pipeline-synth-ops.cpp:1503`) with mutually-consistent upstream versions — everything compiles, HOT-Step's sampler simply stops being called.

**The defense — linker sentinel:** `engine/src/hot-step-sampler.h:1312` defines external-linkage symbol `hotstep_sampler_linked_`; `engine/tools/hot-step-server.cpp:51-52` references it. If the include reverts to `dit-sampler.h`, the symbol vanishes and the link fails with `unresolved external symbol hotstep_sampler_linked_`. That exact error means: re-hook `hot-step-sampler.h` in `pipeline-synth-ops.cpp`. Mechanism detail: [reference.md](reference.md).

## verify-hooks.ps1

```powershell
powershell -File "D:\Ace-Step-Latest\hot-step-cpp\engine\verify-hooks.ps1"
```

Exit 0 = intact, exit 1 = broken. It runs **5 checks** (not 3 — the script is authoritative over the docs): the three hook files above, plus `hot-step-server.cpp` → `hot-step-params.h` include, plus the sentinel string in `hot-step-sampler.h` (`engine/verify-hooks.ps1:15-68`).

## The sync procedure

Prerequisite: fork committed locally (Rule 3). The vanilla upstream clone lives at `D:\Ace-Step-Latest\acestepcpp\acestep.cpp` (a separate repo; if this machine lacks it, `git clone https://github.com/ServeurpersoCom/acestep.cpp` there first).

### Phase 1 — Fetch and assess

```powershell
git -C D:\Ace-Step-Latest\acestepcpp\acestep.cpp pull origin master
```

Read the last-synced commit from the marker. The marker uses `KEY=VALUE` lines — parse the `UPSTREAM_COMMIT=` line (the workflow doc's "first non-comment line" snippet is WRONG for this format):

```powershell
$lastSynced = ((Get-Content "D:\Ace-Step-Latest\hot-step-cpp\engine\UPSTREAM_SYNC" | Where-Object { $_ -match '^UPSTREAM_COMMIT=' }) -replace '^UPSTREAM_COMMIT=','').Trim()
git -C D:\Ace-Step-Latest\acestepcpp\acestep.cpp log --oneline "$lastSynced..HEAD" -- src/ tools/ace-server.cpp
git -C D:\Ace-Step-Latest\acestepcpp\acestep.cpp diff --stat "$lastSynced..HEAD" -- src/ tools/ace-server.cpp
```

Zero commits → already up to date, stop.

### Phase 2 — Direct-copy Upstream Zone files (with hard exclusions)

```powershell
cd D:\Ace-Step-Latest\acestepcpp\acestep.cpp
$excludeFiles = @("pipeline-synth-ops.cpp", "model-store.h", "dit.h")   # NEVER direct-copy
$changedFiles = git diff --name-only "$lastSynced..HEAD" -- src/
foreach ($f in $changedFiles) {
    $basename = Split-Path $f -Leaf
    if ($excludeFiles -contains $basename) {
        Write-Host "SKIP (modified upstream): $f — manual merge in Phase 3" -ForegroundColor Yellow
        continue
    }
    Copy-Item "D:\Ace-Step-Latest\acestepcpp\acestep.cpp\$f" "D:\Ace-Step-Latest\hot-step-cpp\engine\$f" -Force
    Write-Output "Copied: $f"
}
# New upstream files — any new .cpp needs an engine/CMakeLists.txt entry:
git diff --diff-filter=A --name-only "$lastSynced..HEAD" -- src/
```

Do NOT copy upstream `src/dit-sampler.h` expecting it to matter — the fork's `engine/src/dit-sampler.h` is **dead code** (nothing compiled includes it; the active sampler is `hot-step-sampler.h`). Copying it is harmless but see Phase 4.

### Phase 3 — Manual-merge the Modified Upstream Zone

```powershell
cd D:\Ace-Step-Latest\acestepcpp\acestep.cpp
foreach ($f in @("src/pipeline-synth-ops.cpp", "src/model-store.h", "src/dit.h")) {
    if (git log --oneline "$lastSynced..HEAD" -- $f) { Write-Host "NEEDS MERGE: $f" -ForegroundColor Yellow }
}
```

For each flagged file: review `git diff $lastSynced..HEAD -- src/<file>` in the upstream clone, then hand-port those changes into the fork's copy **without removing the hooks**:

- `pipeline-synth-ops.cpp`: keep the fork-only includes (lines 9–12) and the fork's extended `dit_ggml_generate` call site (line 1503). This file has diverged well beyond the include line — treat the fork's copy as the base and port upstream's diff into it, never the reverse.
- `dit.h`: keep the includes at lines 11–13 (`adapter-merge.h`, `hot-step-build-flags.h`, `adapter-runtime.h`).
- `model-store.h`: keep the `hot-step-params.h` include (line 53).

### Phase 4 — Port fixes from upstream dit-sampler.h

```powershell
git -C D:\Ace-Step-Latest\acestepcpp\acestep.cpp log --oneline "$lastSynced..HEAD" -- src/dit-sampler.h
```

If changed: review the diff and hand-port relevant **bug fixes** into `engine/src/hot-step-sampler.h`. Do NOT overwrite `hot-step-sampler.h` with upstream's file. If upstream changed the `dit_ggml_generate` signature, update both the fork's signature in `hot-step-sampler.h` (a superset with extra params) and the call site in `pipeline-synth-ops.cpp:1503` to stay compatible.

### Phase 5 — Verify hooks, build, validate

```powershell
powershell -File "D:\Ace-Step-Latest\hot-step-cpp\engine\verify-hooks.ps1"   # MUST pass BEFORE building
cmd /c "D:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat"                      # never engine\build.cmd directly
```

Post-sync validation checklist:
- [ ] Generate a text2music track
- [ ] Generate with an adapter in merge mode (adapter = LoRA-style fine-tune applied to the DiT)
- [ ] Generate with mastering enabled
- [ ] Switch models mid-session
- [ ] All solvers/schedulers/guidance modes selectable in the UI — **ask the user to eyeball this; do not use a browser agent**
- [ ] **Keep every generated validation track** — do NOT delete audio outputs on your own quality judgment; the user verifies by ear. Report output paths/song IDs so the user can A/B them against pre-sync generations (a subtle sampler/VAE regression is exactly what these exist to catch).
- [ ] Newest `logs/YYYY-MM-DD_HH-MM-SS/ace_engine.log` shows a **`[DiT] Guidance:` line**. This is the definitive runtime proof the HOT-Step sampler is active — upstream's sampler never prints it. (A bare `[DiT] Solver:` line is NOT proof: upstream also prints one at the upstream clone's *own* `src/dit-sampler.h:397` — NOT the fork's stale copy at `engine/src/dit-sampler.h`, which predates that feature and contains no such line. The HOT-Step version additionally includes the scheduler name — format `Solver: <name> (<scheduler>, N NFE/step, order N)` from `hot-step-sampler.h:503`.)

### Phase 6 — Record marker, commit, tag (do NOT push)

Write the marker in its real `KEY=VALUE` format (the workflow doc's Phase 6 heredoc writes an obsolete bare-hash format — do not use it):

```powershell
$newHead = git -C D:\Ace-Step-Latest\acestepcpp\acestep.cpp rev-parse HEAD
@"
UPSTREAM_REPO=acestep.cpp
UPSTREAM_COMMIT=$newHead
SYNC_DATE=$(Get-Date -Format "yyyy-MM-dd")
SYNC_NOTES=<one-line summary of what came in>
"@ | Set-Content "D:\Ace-Step-Latest\hot-step-cpp\engine\UPSTREAM_SYNC"

cd D:\Ace-Step-Latest\hot-step-cpp
$shortHash = $newHead.Substring(0, 7)
git add engine/UPSTREAM_SYNC engine/src/
git commit -m "sync: upstream acestep.cpp at $shortHash"
git tag -a "v1.5-upstream-sync-$shortHash" -m "sync: upstream acestep.cpp at $shortHash"
```

**Do not push the tag** (Golden rule 4). If also updating `engine/CMakeLists.txt` or `engine/tools/`, stage those paths explicitly too.

## Key files

| Path | Role |
|---|---|
| `engine/UPSTREAM_SYNC` | Sync marker: last-synced upstream commit, `KEY=VALUE` format |
| `engine/verify-hooks.ps1` | 5-check hook verifier; run after copy, before build |
| `engine/src/pipeline-synth-ops.cpp` | Modified-upstream: sampler hook (line 9) + extended call site (line 1503) |
| `engine/src/model-store.h` | Modified-upstream: `hot-step-params.h` hook (line 53) |
| `engine/src/dit.h` | Modified-upstream: adapter hooks (lines 11–13) |
| `engine/src/hot-step-sampler.h` | The fork's active DiT sampler; linker sentinel at lines 1311–1315 |
| `engine/src/dit-sampler.h` | Upstream's sampler — **dead code in the fork**, nothing includes it |
| `engine/tools/hot-step-server.cpp` | Compiled server binary source; sentinel consumer at lines 51–52 |
| `engine/tools/ace-server.cpp` | Upstream server, direct-copy reference only, NOT compiled |
| `engine/src/_backup_pre_sync/` | Git-tracked snapshot of `engine/src/` taken before the 2026-04-26 sync — read-only rollback reference |
| `D:\Ace-Step-Latest\acestepcpp\acestep.cpp\` | Vanilla upstream clone (separate repo, machine-specific path) |
| `docs/plans/upstream-sync-workflow.md` | Original workflow doc — **gitignored, local-only, has 2 stale snippets** |

## Failure signatures

| Symptom | Cause | Fix |
|---|---|---|
| Linker: `unresolved external symbol hotstep_sampler_linked_` | `pipeline-synth-ops.cpp` direct-copied from upstream; include reverted to `dit-sampler.h` | Restore line 9 `#include "hot-step-sampler.h"` and re-merge the fork's call site (line 1503) plus fork-only includes (lines 10–12); re-run verify-hooks |
| Compiles clean but solver/scheduler/guidance selections dead or ignored; engine log missing `[DiT] Guidance:` | Sampler hook clobbered AND sentinel lost (e.g. `hot-step-sampler.h` itself damaged) — the fully silent regression | verify-hooks checks 1 and 5; restore from git history; rebuild |
| Compile error: `AdapterGroupScales` undefined in `model-store.h` | `model-store.h` direct-copied; lost line-53 include | Re-add `#include "hot-step-params.h"`; re-merge upstream diff |
| Compile errors for adapter merge/runtime symbols via `dit.h` | `dit.h` direct-copied; lost lines 11–13 includes | Re-add includes; re-merge upstream diff |
| Infinite ace-server respawn + file locks during rebuild | Built via `engine/build.cmd` while the app runs | Use `dev-rebuild.bat` from repo root |
| CI release build unexpectedly kicked off | Pushed a `v1.5-upstream-sync-*` (or any `v*`) tag | Never push sync tags; delete the remote tag immediately if it happens |
| `$lastSynced` resolves to `UPSTREAM_REPO=acestep.cpp` and git errors on the range | Used the workflow doc's stale marker-parse snippet | Parse the `UPSTREAM_COMMIT=` line (Phase 1 above) |

## Institutional knowledge

- **VALIDATED (it happened):** a whole-file direct copy of upstream `pipeline-synth-ops.cpp` once silently disabled every HOT-Step sampler feature with a clean build. The linker sentinel + `verify-hooks.ps1` were built in response (design doc: `docs/plans/2026-05-05-harden-upstream-sync-implementation.md`, local-only).
- **VALIDATED:** last completed sync is upstream commit `9d38f00` on 2026-07-14 (fork commit `a178b57`, local tag `v1.5-upstream-sync-9d38f00`), done via per-file 3-way merge (`git merge-file` with base = previous sync point) rather than direct copy — required by the zone-model drift above. Previous sync: `31cc9ea` on 2026-04-26.
- **VALIDATED (2026-07-14 assessment):** upstream HEAD `9d38f00`; ~60 commits pending. Headline items: LM perf overhaul (`qwen3-lm.h` +552 lines — persistent graph arenas, static batched decode graph replay, `set_rows` KV writes; new headers `graph-arena.h`/`static-graph.h`, no CMake entry needed), snake autofuse pass + fused Vulkan/CUDA snake, K/V→F16 cast before `flash_attn_ext` (touches hook file `dit.h`!), WAV PCM24 fixes, `GGML_CUDA_GRAPHS` default ON, SYCL/Intel Arc cmake support. LM/snake work requires bumping `engine/ggml` (fork at `e705c5fe` 2026-05-25; upstream submodule at `b677b63c` 2026-07-08). The turbo-CFG-clobber fix (`b8ba253`) is ALREADY in the fork. Upstream's `dit-sampler.h` refactor + `src/solvers/` are dead code for the fork (see zone-drift warning) — harvest math fixes only.
- **VALIDATED:** docs saying "3 hook files" undercount the script — `verify-hooks.ps1` runs 5 checks across 4 files. The script wins.
- **VALIDATED:** the workflow doc's Phase 1 marker-parse and Phase 6 marker-write snippets are stale versus the real `KEY=VALUE` marker format. Use this skill's corrected versions.
- **UNVALIDATED / HYPOTHESIS:** swapping only the include line (not the whole file) back to `dit-sampler.h` would *probably* be a compile error rather than silent, because the fork's call site passes an extended argument list upstream's signature lacks. Only the whole-file copy is believed truly silent. Not empirically tested.
- Rejected hardening ideas (explored, deliberately NOT implemented — don't "restore" them): `hot-step-hooks-verify.h` compile guard, `static_assert` in the clobbered file, making `dit-sampler.h` a 2-line shim. Details in [reference.md](reference.md).

## Deeper reading

- [reference.md](reference.md) (this folder) — sentinel mechanism with code, `_backup_pre_sync/` details, DONE/NOT-DONE hardening ledger, marker file contents.
- `docs/plans/upstream-sync-workflow.md` — original workflow (**gitignored/local-only; may be absent on other machines; two snippets stale**).
- `docs/plans/2026-05-05-harden-upstream-sync-implementation.md` — hardening design history (**gitignored/local-only**).
- `CLAUDE.md` § "Upstream sync" — the short committed summary.
- `engine/docs/ARCHITECTURE.md` — engine internals (committed).
