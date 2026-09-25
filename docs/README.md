# HOT-Step CPP documentation

Everything in one place. New here? Start with [Getting started](user/getting-started.md).
Looking for a specific feature? [FEATURES.md](../FEATURES.md) lists them one line each with a
link to the page that explains it.

## For users

| Page | What it covers |
|---|---|
| [Getting started](user/getting-started.md) | Download, first launch, first model pack, first song, where files live |
| [Backends](user/backends.md) | ACE-Step 1.5, MiniMax-Music3 and YuE2: what each is for, how to switch, capability matrix |
| [Generation](user/generation.md) | The global parameter bar: models, steps, seeds, guidance, post-processing chain, queue, task modes |
| [Getting higher quality output](user/quality.md) | Model precision, prompts, steps and solvers, post-processing, per-backend cheat sheet |
| [Adapters](user/adapters.md) | Using LoRA and LoKr adapters: loading, stacking, section masking, trigger words |
| [Plugins](user/plugins.md) | Solvers, schedulers and guidance modes, with the generated list of every plugin |
| [Model files](user/models.md) | Every downloadable pack and file, generated from the registry |
| [Troubleshooting](user/troubleshooting.md) | Logs, common failures and fixes, how to file a good issue |

### Studios

In sidebar order.

| Page | Sidebar name |
|---|---|
| [Auto-Gen](user/studios/insta-gen.md) | Auto-Gen |
| [Custom-Gen](user/studios/create.md) | Custom-Gen (Create) |
| [Library](user/studios/library.md) | Library, player, playlists, song details |
| [Lyric Studio](user/studios/lyric-studio.md) | Lyric Studio |
| [Cover Studio](user/studios/cover-studio.md) | Cover Studio |
| [Repaint Studio](user/studios/repaint-studio.md) | Repaint |
| [Stem Separator](user/studios/stem-studio.md) | Stem Separator |
| [Stem Builder](user/studios/stem-builder.md) | Stem Builder |
| [Song Builder](user/studios/song-builder.md) | Song Builder |
| [STORM](user/studios/storm.md) | STORM |
| [MIDI Studio](user/studios/midi-studio.md) | MIDI Studio |
| [Training Studio](user/studios/training-studio.md) | Training Studio |
| [Settings](user/studios/settings.md) | Settings, terminal, VST chain |
| [Model Manager](user/studios/model-manager.md) | Model Manager |
| [AI Assistant](user/studios/assistant.md) | Assistant |

### Training

| Page | What it covers |
|---|---|
| [Training Studio](user/studios/training-studio.md) | The UI: datasets, captioning, preprocessing, jobs, previews, auditions |
| [ACE-Step 1.5](user/training/ace-step.md) | LM and DiT adapters: dataset, settings, checkpoints |
| [MiniMax-Music3](user/training/minimax-music3.md) | Planner LM adapters: rank, optimiser, steps, which checkpoint to ship |
| [YuE2](user/training/yue2.md) | AR and NAR adapters: the staged chain, refinement, review |

## For contributors and agents

| Page | What it covers |
|---|---|
| [Architecture](dev/architecture.md) | Three tiers, ports, request path, feature to file map |
| [Building](dev/building.md) | Prerequisites, full build, dev loop, engine rebuild rules, type checks, packaging |
| [Configuration](dev/config.md) | Every environment variable and setting, with defaults |
| [Engine](dev/engine.md) | Binaries, per-backend pipeline, hook files, plugin host, adapters, TensorRT, training, ggml patches |
| [Engine request and CLI reference](../engine/docs/ARCHITECTURE.md) | Request JSON fields, generation modes, binary flags, engine endpoints |
| [HTTP API index](dev/api.md) | Every Node route, generated |
| [Plugin authoring](dev/plugins-authoring.md) | Writing a Lua solver, scheduler, guidance or postprocess plugin |
| [Training internals](dev/training-internals.md) | Trainer architecture, measured numbers, open decisions |
| [Releasing](dev/releasing.md) | Cutting and publishing a release, CI caching, gotchas |
| [Writing and maintaining the docs](dev/docs-contributing.md) | Layout, page template, generated tables, the checker |
| [Writing style](dev/writing-style.md) | Rules for prose a human reads |
| [Skill library](../.claude/skills/README.md) | Procedures per maintenance domain for coding agents |
| [AGENTS.md](../AGENTS.md) | Project rules for agents, including the documentation ownership table |
