# HOT-Step CPP documentation

Everything in one place. New here? Start with [Getting started](user/getting-started.md).
Looking for a specific feature? [FEATURES.md](../FEATURES.md) lists them one line each with a
link to the page that explains it.

## For users

| Page | What it covers |
|---|---|
| [Getting started](user/getting-started.md) | Download, first launch, first model pack, first song, where files live |
| [Data folder and upgrades](user/data-folder.md) | Shared data across versions, move, repoint, import and recovery |
| [Running on Google Colab](user/colab.md) | The Colab notebook: install, models, Drive persistence, opening the UI |
| [Backends](user/backends.md) | ACE-Step 1.5, MiniMax-Music3, YuE2 and Stable Audio 3: what each is for, how to switch, capability matrix |
| [Generation](user/generation.md) | The global parameter bar: models, steps, seeds, guidance, post-processing chain, queue, task modes |
| [Getting higher quality output](user/quality.md) | Model precision, prompts, steps and solvers, post-processing, per-backend cheat sheet |
| [Adapters](user/adapters.md) | Using LoRA, LoKr and DoRA adapters per backend: loading, stacking, section masking, trigger words |
| [Plugins](user/plugins.md) | Solvers, schedulers and guidance modes, with the generated list of every plugin |
| [YuE2 Plan Play presets](user/presets/yue2-plan-play/README.md) | Four importable YuE2 profiles that change how closely the renderer follows the plan |
| [Model files](user/models.md) | Every downloadable pack and file, generated from the registry |
| [Driving HOT-Step with an agent (MCP)](user/mcp.md) | The `lyricstudio` and `hotstep` MCP servers: setup, security posture, a generation and training tour |
| [Troubleshooting](user/troubleshooting.md) | Logs, common failures and fixes, how to file a good issue |

### Studios

In sidebar order. The sidebar lists only the studios the active backend can run.

| Page | Sidebar name |
|---|---|
| [Auto-Gen](user/studios/insta-gen.md) | Auto-Gen |
| [Custom-Gen](user/studios/create.md) | Custom-Gen |
| [Library](user/studios/library.md) | Library, player, playlists, song details |
| [Lyric Studio](user/studios/lyric-studio.md) | Lyric Studio |
| [Cover Studio](user/studios/cover-studio.md) | Cover Studio |
| [Repaint Studio](user/studios/repaint-studio.md) | Repaint |
| [Stem Separator](user/studios/stem-studio.md) | Stem Separator |
| [Compare](user/studios/compare-studio.md) | Compare |
| [Stem Builder](user/studios/stem-builder.md) | Stem Builder |
| [Song Builder](user/studios/song-builder.md) | Song Builder |
| [STORM](user/studios/storm.md) | STORM |
| [MIDI Studio](user/studios/midi-studio.md) | MIDI Studio |
| [Training Studio](user/studios/training-studio.md) | Training |
| [Settings](user/studios/settings.md) | Settings, terminal, VST chain |
| [AI Assistant](user/studios/assistant.md) | Assistant |
| [Model Manager](user/studios/model-manager.md) | Not in the sidebar: Get More Models in the global bar's Models section |

### Training

| Page | What it covers |
|---|---|
| [Training Studio](user/studios/training-studio.md) | The UI: datasets, captioning, preprocessing, jobs, previews, auditions |
| [ACE-Step 1.5](user/training/ace-step.md) | LM and DiT adapters: dataset, settings, checkpoints |
| [MiniMax-Music3](user/training/minimax-music3.md) | Planner LM adapters: rank, optimiser, steps, which checkpoint to ship |
| [YuE2](user/training/yue2.md) | AR and NAR adapters: the staged chain, refinement, review |
| [HeartMuLa](user/training/heartmula.md) | Backbone LoRA: Prepare, Train, checkpoint previews, memory and step time |
| [Stable Audio 3](user/training/sa3.md) | Instrumental Medium DoRA-rows: vocal stripping, recipe, resume and audition |

## For contributors and agents

| Page | What it covers |
|---|---|
| [Architecture](dev/architecture.md) | Three tiers, ports, request path, feature to file map |
| [Building](dev/building.md) | Prerequisites, full build, dev loop, engine rebuild rules, type checks, packaging |
| [Configuration](dev/config.md) | Every environment variable and setting, with defaults |
| [Engine](dev/engine.md) | Binaries, per-backend pipeline, hook files, plugin host, adapters, TensorRT, training, the ggml fork |
| [The HOT-ggml fork](dev/ggml-fork.md) | The `engine/ggml` submodule: HOT-Step's ggml changes as fork commits, and how to move the pinned commit |
| [Engine request and CLI reference](../engine/docs/ARCHITECTURE.md) | Request JSON fields, generation modes, binary flags, engine endpoints |
| [HTTP API index](dev/api.md) | Every Node route, generated |
| [Generation API contracts](dev/api-contracts.md) | Versioning and the versioned `/api/v1/daw` facade, local auth, the generate request and response, errors, jobs, YuE2 score preview, media, stream sessions, the capability manifest, caption resolution, the durable audio queue, workflow jobs, the UI transport client and shared contracts for older routes |
| [Building a replacement frontend](dev/build-a-frontend.md) | Entry point for a client built on the HTTP contracts alone: origin/auth/media, discovery, errors/versioning, job polling and resume, Create's queue ownership, and the full workflow catalogue |
| [Create's control dictionary](dev/frontend-create-controls.md) | The generation parameter dictionary for a frontend client: submission shapes, task modes, defaults |
| [Studio workflows for frontend clients](dev/frontend-studios.md) | Auto-Gen, Cover (with the YuE2 score and Build routes), Repaint, Stem Builder's layer render, Lyric Studio batch and stem separation wire contracts |
| [Training for frontend clients](dev/frontend-training.md) | ACE/MM3/YuE2 recipes, SA3 starts, following a job, restarts, review |
| [Library, playlist, drafts and Song Builder](dev/frontend-library.md) | Song, playlist, studio draft and Song Builder wire contracts |
| [Preference presets](dev/frontend-presets.md) | Preference families, import, YuE2 joint preset resolution |
| [Media for frontend clients](dev/frontend-media.md) | Import, export and the two live audio streams |
| [Plugin authoring](dev/plugins-authoring.md) | Writing a Lua solver, scheduler, guidance or postprocess plugin |
| [Training internals](dev/training-internals.md) | Trainer architecture, measured numbers, open decisions |
| [Releasing](dev/releasing.md) | Cutting and publishing a release, CI caching, gotchas |
| [RunPod Pod template](dev/runpod.md) | The `runpod` Docker target on a RunPod GPU Pod, its HTTP proxy and the missing login; the published image arrives with the next release |
| [Writing and maintaining the docs](dev/docs-contributing.md) | Layout, page template, generated tables, the checker |
| [Writing style](dev/writing-style.md) | Rules for prose a human reads |
| [UI design rules](dev/ui-design.md) | The shared controls every screen uses (dropdown, toggle, hover explanation), accent colours, and the check that enforces them |
| [UI server state](dev/ui-server-state.md) | The contract for server snapshots in the UI: TanStack Query owns them, Zustand keeps client state, event streams drive the cache |
| [TanStack Query audit and roadmap](dev/tanstack-query.md) | Every TanStack integration audited, plus the P8 roadmap for `queryClient.query()` call sites, duplicate fetch owners, and the move to reactive reads |
| [Skill library](../.claude/skills/README.md) | Procedures per maintenance domain for coding agents |
| `AGENTS.md` (local only) | Project rules for agents, including the documentation ownership table |
