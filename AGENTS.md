# AGENTS.md — HOT-Step CPP

Orientation map for agents, and the **single source of truth** for project rules. Claude Code reads it through `CLAUDE.md` (a one-line `@AGENTS.md` import); Codex reads it directly. Edit this file only. Keep it short and navigational — point at the deep docs, don't duplicate them.

## Shared skills

Project skills live in `.claude/skills/`. On this Windows checkout,
`.agents/skills` is a directory junction to that folder, so Codex discovers the
same files Claude Code uses. Edit or add skills in `.claude/skills/`; do not
create a separate copy for another agent. Each skill needs a `SKILL.md` with
`name` and `description` frontmatter for discovery. Keep mandatory project rules
in this file, since skill selection depends on the task.

The skill library under `.claude/skills/` is committed (`.gitignore` whitelists
it); the `.agents/skills` junction is local and needs recreating on a new
checkout. Refresh the client's skill list after changes, or restart it if the
changes do not appear.

## What this is

A desktop app for **local AI music generation** — a heavily-extended superset of [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) (a C++/GGML port of ACE-Step 1.5). Caption + lyrics in → stereo 48 kHz audio out, fully local. Ships as portable releases (Windows CUDA/Vulkan/CPU, Linux, macOS Metal). GitHub: `scragnog/HOT-Step-CPP`.

## Architecture (3 tiers)

| Tier | Stack | Location | Role |
|------|-------|----------|------|
| **Engine** | C++17 / CUDA / GGML | [engine/](engine/) | Inference binaries: `ace-lm`, `ace-synth`, `ace-server`, `ace-understand`, `neural-codec`, `mp3-codec`, `quantize`. Pipeline: LM → DiT → VAE. Also hosts the **MiniMax-Music3 backend** ([engine/src/minimax/](engine/src/minimax/), `/mm3/*` endpoints) |
| **Server** | Node / TypeScript / Express / better-sqlite3 | [server/src/](server/src/) | Orchestrates the engine, manages songs/jobs/SQLite, serves UI. Per-feature [routes/](server/src/routes/) + [services/](server/src/services/) |
| **UI** | React 19 / Vite / Zustand / Tailwind | [ui/src/](ui/src/) | Browser frontend. Component folder per "studio" |

```
LAUNCH.bat → Node server (Express :3001)
  ├── serves React frontend (prebuilt ui/dist/)
  ├── /api/* → SQLite
  └── spawns child: ace-server.exe (C++ engine) on :8085
```

| Service | Port |
|---------|------|
| Node server | 3001 (prod) |
| Vite dev server | 3000 (dev, HMR) |
| ace-server (C++ engine) | 8085 (default, `config.ts`) |

## Environment

- **Windows 11 + PowerShell.** This repo's primary dev environment is Windows. Your harness (Claude Code, Codex) may also give you a Bash (POSIX) tool — each takes its own syntax. In PowerShell use `;` not `&&`.
- **Node 18–22 LTS only.** Node 24+ breaks dependencies (`engines` field enforces `<24`).
- **Call `.bat`/`.cmd` by absolute path.** Some agent shells run with `NoDefaultCurrentDirectoryInExePath=1`, so `cmd.exe` will not resolve `build.cmd` from the working directory — you get `'build.cmd' is not recognized` even though it is right there. Worse, `cmd.exe /c "script.bat"` can return **exit 0 having run nothing**, so never take a batch exit code as proof it ran: check the output for the script's own first line.

## Build & run rules (IMPORTANT — learned the hard way)

- **C++ engine changes → `dev-rebuild.bat`, NEVER `engine/build.cmd` directly.** The Node server auto-respawns ace-server on crash; killing it without clean shutdown causes an infinite respawn + file-lock loop. `dev-rebuild.bat` handles clean shutdown + rebuild — it does **not** relaunch; start the app again yourself with `dev.bat`/`LAUNCH.bat`.
  - Recompile **immediately** after editing any `engine/src/` or `engine/tools/` file — don't wait to be asked.
- **NEVER `cmake --build . --clean-first`** unless the GGML/CUDA layer itself changed — CUDA kernel recompilation is **20+ min**. For stale `.obj` issues, delete only `engine/build/acestep-core.dir/` and `engine/build/Release/acestep-core.lib`.
- **Don't `npm run build` during dev.** Only build before user testing. Type-check with:
  - `server/` → `npx tsc --noEmit`
  - `ui/` → **`npx tsc --noEmit -p tsconfig.app.json`** (or `npx tsc -b`). A bare `npx tsc --noEmit` in `ui/` **silently checks nothing and exits 0** — `ui/tsconfig.json` is `{"files": [], "references": [...]}`, so the root project has no inputs. It is not a passing check, it is no check.
- **`dev.bat`** = dev mode (Vite :3000 HMR + Node :3001, tsx watch auto-restart). **`LAUNCH.bat`** = prod. Use `dev.bat` for development.

## Git rules

- **All work on `master`. No feature branches, ever.**
- **Never `git add -A`** (re-adds gitignored dirs: `.agents/`, `checkpoints/`, `node_modules/`, etc.). **Never `git add -f`** on gitignored paths. Stage explicit paths.
- **Never `git reset --hard`.** This checkout sets `submodule.recurse=true`, so a hard reset in the superproject **also resets `engine/ggml`** and silently wipes the whole `engine/patches/` stack out of its tracked files. The next build looks like a broken engine change, not a git accident: CMake re-applies the stack, but the two patches that *create* files (`flash-attn-train`, `zz-yue2-convrot8`) fail with "already exists in working directory" — their `.cu`/`.cuh` are untracked, so the reset left them behind. You then get hundreds of CUDA errors about undefined `GGML_OP_CONVROT8` / `ggml_flash_attn_train_*`, because the orphaned kernels reference ops that are no longer declared. To undo an unwanted working-tree change, use `git checkout -- <path>` or `git restore <path>` on explicit paths. To recover from a hard reset: delete `engine/ggml/src/ggml-cuda/{convrot8,fattn-train}.{cu,cuh}`, re-apply those two patches, then run `engine/verify-hooks.ps1`.
- **Push requires explicit user approval — always ask first.**
- Commit to local git **often** (data has been lost before to uncommitted files).
- **Releases:** push a `vX.Y.Z` tag → the `Release` workflow builds all platforms and drafts a GitHub Release. **Any pushed `v*` tag triggers a build** — use a `-CI-Test` suffix for throwaway compile checks, and don't push local feature tags matching `v*`. Full process + gotchas: [docs/dev/releasing.md](docs/dev/releasing.md).
- Use `gh` CLI for GitHub ops (authenticated as `scragnog`).

## Shipping to users (works here ≠ works for them)

This is a public app. A feature can pass every local check because **this machine**
holds a file that was never part of the distribution — model weights sitting in
`models/`, a data file that CI never copies into the archive. Nothing catches it:
paths are resolved at runtime, so tsc is clean and the build is green, and the
feature is simply dead for everyone who downloads it. It has shipped twice
(MM3 training encoders, #137; the MM3 caption corpus, #139).

- **Any new file the app resolves at runtime must be reachable by a user** —
  weights uploaded to Hugging Face **and** listed in
  [`server/src/data/model-registry.json`](server/src/data/model-registry.json);
  runtime data files packaged by [`release.yml`](.github/workflows/release.yml)
  (it copies `server/src/data/` wholesale, so put them there).
- **Before pushing anything to `master`, and always before a release tag:**

  ```
  node server/scripts/check-release-prereqs.mjs
  ```

  It verifies every registry entry exists on HF at the claimed size, that packs
  reference real files, and that every runtime data file gets packaged. Exit 1 =
  do not ship. Details: [.claude/skills/validating-changes/SKILL.md](.claude/skills/validating-changes/SKILL.md) (Tier 6).

## Upstream sync (fork hooks that break silently)

The C++ engine is a patched fork of acestep.cpp. Three upstream files carry HOT-Step `#include` hooks that break if overwritten during a sync:

| Upstream file | Hook | If lost |
|---|---|---|
| `pipeline-synth-ops.cpp` | `hot-step-sampler.h` (replaces `dit-sampler.h`) | **SILENT** — compiles, but all solvers/guidance/schedulers go dead |
| `model-store.h` | `hot-step-params.h` | compile error |
| `dit.h` | `adapter-merge.h` + `adapter-runtime.h` | compile error |

After any sync: run `engine/verify-hooks.ps1`. `build.cmd` also runs it before every compile and stops the build if a hook or ggml patch is missing — a missing op costs ten seconds to spot there and twenty minutes of CUDA compile to spot the other way.

**The ggml patch stack has no reliable "already applied" check.** CMake tests it with `git apply --reverse --check`, which cannot work for patches that touch the same lines: `flash-attn-train` and `zz-yue2-convrot8` both add to the same enum in `ggml.h`, so once both are applied neither reverses in isolation. CMake logs "neither applies nor reverses" for `flash-attn-train` on **every healthy build** — that warning is expected and is not evidence of anything. `verify-hooks.ps1` greps for the symbols themselves and is the only trustworthy answer.

## UI / browser verification

- **Don't use the built-in browser agent to visually verify UI** — too slow/unreliable here. **Ask the user to check**; they provide screenshots/feedback. Browser agent is fine for non-visual tasks (hitting API endpoints).

## Debugging — logs

App writes per-session logs to `logs/` at repo root:

```
logs/YYYY-MM-DD_HH-MM-SS/        ← one folder per session (name-sorted = time-sorted)
  ├── ace_engine.log              ← C++ engine output
  ├── node_console.log            ← Node server output
  └── generations/gen_<uuid>_<task>.log
```

Start with the newest session folder. Generation failures → matching `gen_*.log` first, then cross-ref `ace_engine.log` + `node_console.log`. Startup/crash → `node_console.log` + `ace_engine.log`.

## Plugin system

Solvers, schedulers, guidance modes and postprocess steps are **hot-loadable Lua plugins** in [engine/plugins/](engine/plugins/) — drop a `.lua` in the right subdir, appears in the UI next launch, no C++ rebuild. Each plugin can declare its own UI params. Native C++ bridge via `apg()`; advanced plugins use `post_step()` for extra forward passes. **Adding a solver/scheduler/guidance = write a `.lua` plugin** (the old approach of editing `dit-sampler.h` is obsolete — the engine now routes through `hot-step-sampler.h`). Authoring guide: [docs/dev/plugins-authoring.md](docs/dev/plugins-authoring.md).

## Documentation is part of the change

Public docs live in `docs/user/` (users) and `docs/dev/` (contributors and agents), indexed
from [docs/README.md](docs/README.md) and [FEATURES.md](FEATURES.md). A change that alters
behaviour a user or developer can see is not done until the owning page says so. Rules and the
page template: [docs/dev/docs-contributing.md](docs/dev/docs-contributing.md).

| Change in | Update |
|---|---|
| `ui/src/components/<studio>/` | `docs/user/studios/<studio>.md` (map in `tools/docs/check-docs.mjs`), the one-liner in `FEATURES.md` |
| `ui/src/components/global-bar/`, generation params, post-processing | `docs/user/generation.md` |
| A new studio or UI folder | new page + a row in `STUDIO_PAGES` in `tools/docs/check-docs.mjs` + `FEATURES.md` |
| `server/src/routes/`, `server/src/index.ts` mounts | `node tools/docs/build-docs.mjs` (regenerates `docs/dev/api.md`) |
| `engine/plugins/`, `plugins/` | `node tools/docs/build-docs.mjs` (regenerates `docs/user/plugins.md` tables) |
| `server/src/data/model-registry.json` | `node tools/docs/build-docs.mjs` (regenerates `docs/user/models.md`) + `check-release-prereqs.mjs` |
| `engine/src/`, `engine/tools/` | `docs/dev/engine.md`; `engine/docs/ARCHITECTURE.md` if request JSON or CLI flags changed |
| `server/src/config.ts`, env vars, Settings UI | `docs/dev/config.md`, `docs/user/studios/settings.md` |
| Backend capabilities (`server/src/services/backends/*/index.ts`) | `docs/user/backends.md` capability matrix |
| Adapter loading, stacking, masking | `docs/user/adapters.md`, `adapter-system` skill |
| Training flags, defaults, recipes | `docs/user/training/<backend>.md`, `docs/dev/training-internals.md` |
| Build scripts, CI workflows, release steps | `docs/dev/building.md`, `docs/dev/releasing.md` |
| Anything the in-app assistant should know | `docs/user/` first, then `node tools/docs/build-docs.mjs` refreshes the generated blocks in `server/src/data/assistant-knowledge.md`; hand-written blocks there are for behaviour rules only |
| A new `.claude/skills/<skill>` | row in `.claude/skills/README.md` |

Then run `node tools/docs/check-docs.mjs`. It fails on a studio with no page, a stale generated
table, a broken link, or an unindexed skill, and runs in CI (`docs.yml`) and inside
`check-release-prereqs.mjs`. Claude Code also gets a Stop hook (`.claude/settings.json`) that
blocks once when code changed with no doc change; Codex has no hooks, so this table is the rule.

The assistant knowledge base is read once at server start, so a regenerated
`assistant-knowledge.md` is picked up on the next restart, not live.

## Agent work coordination

**Only when the user explicitly requests Lyric Studio MCP or the HOT-Step Work
channel.** A request for sub-agents, parallel work, or coordination does not
authorize this connector. Do not join a work channel, reserve resources, or
call any Lyric Studio `work_*` / `collab_*` tool otherwise. When the user does
explicitly request it, the channel is `HOT-Step` and the rules (catch-up,
reservations, `work-run.ps1`, context corrections) live in
[work channel usage](tools/mcp-lyricstudio/README.md#work-channels).

## Discord transcripts

The MM3 working group lives in Discord, and a lot of project-relevant decisions
happen there. [tools/discord-claude/](tools/discord-claude/) bridges that thread to
Claude *and* logs every message to `logs/discord/<channelId>.jsonl` (gitignored —
it is other people's chat). Read it with:

```
node tools/discord-claude/read-log.mjs --list                     # channels + message counts
node tools/discord-claude/read-log.mjs --since 12h                # busiest channel, recent
node tools/discord-claude/read-log.mjs --channel all-for-one --last 200
node tools/discord-claude/read-log.mjs --all --grep "encoder|NVFP4" --context 3
```

`backfill.mjs` pulls history from the Discord API (safe to re-run; dedupes by message id).
**Do not** reconstruct the thread by scraping `~/.claude/projects/*.jsonl` — those sessions
only ever saw a rolling window and are lossy.

## Read-Y-for-X index

| For… | Read |
|------|------|
| **Any maintenance task — start here** (per-domain procedures, gotchas, distilled institutional knowledge) | [.claude/skills/README.md](.claude/skills/README.md) — one skill per maintenance domain |
| **MiniMax-Music3 backend** (second generation backend: engine port, /mm3 endpoints, backend registry/toggle, trap list) | [.claude/skills/mm3-backend/SKILL.md](.claude/skills/mm3-backend/SKILL.md) |
| MM3 caption/prompt format (genre adherence) | [.claude/skills/mm3-captioning/SKILL.md](.claude/skills/mm3-captioning/SKILL.md) |
| **Training an MM3 LM adapter** (album/artist clone: rank, steps, which checkpoint to ship, likeness-vs-coherence) | [.claude/skills/mm3-lm-adapter-training/SKILL.md](.claude/skills/mm3-lm-adapter-training/SKILL.md) |
| **Any listening test** (checkpoint ladders, A/B renders, "listen and tell me") — the local HTML score sheet Rob scores in | [.claude/skills/ear-test-scoresheet/SKILL.md](.claude/skills/ear-test-scoresheet/SKILL.md) |
| **What the Discord working group said** (MM3 group: bghira, Serveurperso, testerf, Shaz…) — searchable transcripts of every channel | `node tools/discord-claude/read-log.mjs --list` — see [Discord transcripts](#discord-transcripts) |
| **Writing anything a human reads** (issue replies, commits, PR bodies, release notes, docs) | [docs/dev/writing-style.md](docs/dev/writing-style.md) — no emojis, no AI tells, honest confidence |
| **Every doc page, both audiences** | [docs/README.md](docs/README.md) |
| Feature catalogue, one line each, linking to the page | [FEATURES.md](FEATURES.md) |
| **Updating docs when you change code** (ownership table above, page template, generators) | [docs/dev/docs-contributing.md](docs/dev/docs-contributing.md) |
| System architecture, feature → route/service/UI/engine map | [docs/dev/architecture.md](docs/dev/architecture.md) |
| Engine guide (binaries, per-backend pipeline, hooks, plugins, TRT, training) | [docs/dev/engine.md](docs/dev/engine.md) |
| Engine request JSON and CLI flag reference | [engine/docs/ARCHITECTURE.md](engine/docs/ARCHITECTURE.md) |
| HTTP route index (generated) | [docs/dev/api.md](docs/dev/api.md) |
| Env vars and settings reference | [docs/dev/config.md](docs/dev/config.md) |
| **Training system** (dataset→preprocess→LM/DiT training→audition; ace-train, FSQ, ggml training gotchas) | [docs/dev/training-internals.md](docs/dev/training-internals.md) |
| **Flash-attention training** (fused attention backward: porting `--attn flash` to a trainer, VRAM-model branches, measurement traps) | [.claude/skills/flash-attn-training/SKILL.md](.claude/skills/flash-attn-training/SKILL.md) |
| Writing a Lua plugin | [docs/dev/plugins-authoring.md](docs/dev/plugins-authoring.md) |
| Build, dev loop, type-checks, portable packaging | [docs/dev/building.md](docs/dev/building.md) |
| Cutting & publishing a release (agent runbook) | [docs/dev/releasing.md](docs/dev/releasing.md) |
| **Pre-release go/no-go** (`node tools/release-gate/run.mjs`: API-driven regression tiers, few-step training, renders staged for the ear test) | [tools/release-gate/README.md](tools/release-gate/README.md) |
| Internal design/investigation docs (perf, adapters, upstream sync, feature designs) | `docs/plans/` *(gitignored, local-only)* |
| In-app assistant behaviour/KB | [server/src/data/assistant-knowledge.md](server/src/data/assistant-knowledge.md) |

> **Doc convention:** committed docs = `README.md`, `FEATURES.md`, everything under `docs/user/` and `docs/dev/`, `engine/docs/ARCHITECTURE.md`, and the skills. Internal planning/investigation docs live in `docs/plans/`, which is **gitignored** (local only); distil lasting decisions into `docs/dev/`. This file (`AGENTS.md`) is committed; `CLAUDE.md` only imports it.
