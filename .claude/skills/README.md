# HOT-Step CPP Skill Library

Thirteen skills that encode how this project is actually maintained — including
institutional knowledge that exists nowhere else in committed form (the local
`docs/plans/` design docs are gitignored; their essential content is distilled
into these skills). Written for engineers and AI sessions with zero prior
context beyond `CLAUDE.md`.

Every skill was adversarially fact-checked against the repo (paths, line
numbers, commands, behavioral claims) as of 2026-07-02, HEAD `168dcb5`.
Line-number citations drift as code changes — trust the named
file/function/string over the exact line.

## Picking a skill

| Task | Skill |
|---|---|
| "Where does X live?" / triage which tier owns a bug | [project-map](project-map/SKILL.md) |
| Edit / rebuild / smoke-test C++ engine code | [engine-dev-loop](engine-dev-loop/SKILL.md) |
| A generation failed, crashed, hung, or won't start | [debugging-runtime](debugging-runtime/SKILL.md) |
| Add/modify a solver, scheduler, guidance mode, postprocess | [lua-plugin-authoring](lua-plugin-authoring/SKILL.md) |
| Add/modify a Node route, service, SQLite schema | [server-feature-dev](server-feature-dev/SKILL.md) |
| Add/modify a React studio, store, UI control | [ui-feature-dev](ui-feature-dev/SKILL.md) |
| Plumb a generation param end-to-end; "param does nothing" | [generation-request-flow](generation-request-flow/SKILL.md) |
| Anything LoRA/LoKr: loading, stacking, masking, conversion | [adapter-system](adapter-system/SKILL.md) |
| Pull upstream acestep.cpp changes into the fork | [upstream-sync](upstream-sync/SKILL.md) |
| Cut, verify, publish a release | [release-process](release-process/SKILL.md) |
| Know when a change is actually "done" | [validating-changes](validating-changes/SKILL.md) |
| Ship a new model/data file so **users** can get it, not just this machine | [model-management](model-management/SKILL.md) + `node server/scripts/check-release-prereqs.mjs` |
| Model files, GGUF, quantization, model manager | [model-management](model-management/SKILL.md) |
| Speed: TensorRT paths, profiling, quality/speed knobs | [engine-performance](engine-performance/SKILL.md) |
| Anything MiniMax-Music3: engine port, /mm3 endpoints, backend toggle, MM3 debugging | [mm3-backend](mm3-backend/SKILL.md)* |
| Format a MiniMax-Music3 caption; MM3 genre drift/adherence | [mm3-captioning](mm3-captioning/SKILL.md)* |
| Train an MM3 LM style/album adapter; pick rank, steps, checkpoint | [mm3-lm-adapter-training](mm3-lm-adapter-training/SKILL.md)* |
| Port `--attn flash` (the fused attention backward) to a trainer; VRAM-model branches; interpret any flash-vs-exact number | [flash-attn-training](flash-attn-training/SKILL.md) |

Feature-specific work (Stem Studio, Lyric Studio/lireek, mastering, cover art,
VST, whisper, i18n, auth, …) starts at **project-map**'s feature table, which
maps each feature to its route, service, UI folder, and engine piece.

\* `mm3-lm-adapter-training` (added 2026-08-24) records the 6-album,
30,000-step sweep that produced the current training recipe.

\* `mm3-captioning` (added 2026-08-13) is a later addition outside the
original thirteen's fact-check pass; its `upstream/` reference library
(MiniMax's official caption-rewriter content) is **gitignored** — local
installs fetch it per the skill's PROVENANCE.md instructions.

## The five rules that prevent the expensive disasters

Each skill restates the ones relevant to its domain; they are absolute:

1. C++ rebuilds go through `dev-rebuild.bat` — never `engine/build.cmd`
   directly, never `cmake --clean-first` (20+ min CUDA recompile).
2. Any pushed `v*` tag triggers a full multi-platform CI release build; every
   push needs explicit user approval.
3. After an upstream sync, run `engine/verify-hooks.ps1` — sampler-hook loss
   is silent (compiles clean, all solvers/schedulers/guidance dead).
4. Never delete generated audio or experiment artifacts on your own quality
   judgment — the user verifies by ear.
5. Never rebuild post-LM synth requests via a field whitelist — spread the
   original `aceReq` and take only the 7 LM output fields.
