# Training

Builds a labeled dataset from your own audio and trains a style adapter on it, entirely on your own GPU. Reach for it when you want HOT-Step to sound like a specific artist, album, or sound rather than the base model: point it at a folder of songs, label them, and train a planner (LM) adapter, a sound (DiT) adapter, or both, depending on which backend is active.

![Training](../../images/studio-training-studio.webp)
<!-- screenshot: needed -->

## Where to find it

The "Training" entry in the sidebar (`/training-studio`). Labeling with local BPM/key analysis works even if the engine is down; AI captioning needs either the local MOSS captioner or a configured cloud LLM provider (Settings > AI Services); preprocessing and training need the `ace-train` binary bundled with the app.

Training follows whichever generation backend is active (ACE-Step, MiniMax-Music3, or YuE2). A pill next to the title names it when more than one backend is available, and the phases below change shape with it. Switch backends from the pill in the top bar, not from inside this studio.

## Workflow

1. Click **New dataset**, pick a source folder (recursively scanned by default), and confirm the auto-filled name, trigger word, and lyric language. HOT-Step scans the folder and shows a preview (file count, extensions, whether a `dataset.json` already exists) before anything is created.
2. On the **Label** step, run local analysis: Essentia for BPM/key, Genius for lyrics, and an AI caption (local MOSS, or a configured cloud provider) for style and genre. Scope the run to unlabeled tracks, all tracks, or a grid selection, and choose a merge policy for tracks that already have partial data.
3. On the **Review** step, check and hand-edit every track in a spreadsheet-style grid: caption, genre, BPM, key, time signature, language, instrumental flag, lyrics, per-track trigger tag, and an exclude checkbox. Select rows for bulk edits or a scoped re-label.
4. On the **Build** step, set the trigger word's position in the caption (prepend, append, replace) and a genre-mix ratio, then build. This writes `dataset.json` next to the audio, in the same format Side-Step's own dataset builder produces.
5. Move to **Preprocess** (ACE) or **Codes** (MiniMax-Music3): pick the base DiT/VAE/text-encoder models and how much of each track to encode, then start. This runs in the standalone `ace-train` binary, which needs the GPU to itself, so the app stops the local engine for the duration and restarts it afterward. YuE2 has no separate Preprocess phase; its own latent-cache stage lives on the Train page instead.
6. Move to **Train** and start a run. ACE offers two adapter cards, a planner LoRA (LM) and a sound LoRA (DiT); MiniMax-Music3 trains a single planner LoRA from RVQ codes; YuE2 runs a five-stage chain (latent cache, semantic codes, lyric-timing spans, NAR adapter, AR adapter) that a **Perform all stages** button drives end to end, with Joint Training as the default method for the NAR/AR pair. Each form shows only the handful of settings that change what gets trained; everything else sits behind an Advanced drawer.
7. Watch the run: a live loss chart and stats update as it trains, and periodic milestone checkpoints appear as badges once the run has enough epochs behind it.
8. Use **Audition** to A/B the trained planner against the base model on identical caption, lyrics, and seed, decoded straight through without the DiT. Optionally render both sides through the DiT too for a fuller comparison. Click a milestone badge to audition that checkpoint directly.
9. For YuE2 only, use **Refine** to push a finished planner adapter up the KL-divergence rungs and render a preview at each step, then **Review** to see every dataset with an unscored refinement ladder and score them by ear.
10. Use **Import multiple…** from the dataset list to queue several source folders through label, build, and (for ACE) preprocess and train in one unattended run; the **Monitor** phase shows its progress as a queue of datasets, each with a chip per pipeline stage. This is separate from YuE2's own **Train multiple…**, which runs preparation, joint training and refinement over several already-created datasets instead.

## Controls

| Control | What it does |
|---|---|
| New dataset | Opens the import wizard: source folder, recursive toggle, name, trigger word, and default lyric language. |
| Import multiple… | Bulk-imports several sibling folders as separate datasets and queues label/build/preprocess/train stages for each; watched from the Monitor phase. |
| Train multiple… (YuE2 only) | Runs preparation, joint training and refinement over several already-created datasets, one after another, with one recipe. The batch runs on the server, so it survives a page reload and a server restart (resume it from the batch panel after a restart). The panel sits above every Training Studio phase with a "Go to the running training" link; "Follow" opens each dataset as the batch moves to it and never pulls you off a dataset you opened yourself. |
| Phase tabs (Dataset · Preprocess/Codes · Train · Refine · Review · Monitor) | Switches between the stages above. Preprocess/Codes and Monitor are hidden under YuE2; Refine and Review only appear under YuE2. |
| Label step: scope, sources, merge policy | Restricts labeling to unlabeled/all/selected tracks, picks which of Essentia/Genius/caption to run, and how new data merges with what a track already has. |
| Review grid | Per-track editable table with bulk actions (exclude, mark instrumental, set genre/tag, scoped re-label). Opens a full-detail drawer per track. |
| Build: trigger position, genre ratio, Build | Where the trigger word sits in the built caption, and the mix ratio between genre-style and literal captions, then writes `dataset.json`. |
| Preprocess / Codes form | Base model pickers plus an Advanced drawer for encoding settings; MiniMax-Music3 shows an RVQ codes export instead, including an optional cover-laundering pass for dense mixes. |
| LM card / DiT card (Train) | Adapter name, a handful of always-visible settings, an Advanced drawer, and a Start button. Running shows a live chart, stats, and a done-state summary with the adapter's output path once finished. |
| Milestone badges | Appear once a run has saved checkpoints; click one to audition that checkpoint. |
| Audition card | Same-seed A/B between the base planner and the trained adapter, with an optional DiT render pass. |
| Send to Lyric Studio (Build step) | Exports the dataset's artist/album metadata into Lyric Studio, optionally linking the dataset's trained adapters and a reference track as that album's generation preset. |
| Monitor queue | Read-only view of the batch pipeline: one row per dataset, a status chip per stage, live progress on whichever stage is currently running. |

## Tips and limits

Trained adapters are written under the app's adapters folder, split by base architecture and size (for example a 4B ACE-Step planner LoRA lands under `lm-4b/<name>/<run>/`, a DiT LoRA under `dit-<base>/<name>/<run>/`), with a timestamped subfolder per run so retraining never overwrites an earlier adapter. They show up in the adapter pickers used elsewhere in the app (Create, Custom-Gen, the audition card here) without any extra step; see [Adapters](../adapters.md) for how adapters are selected, scaled, and stacked at generation time.

The Preprocess and Train phases stop the local engine while `ace-train` runs and restart it afterward. Expect generation to be unavailable in the app for the duration of a preprocess or training job. Labeling does not require the engine unless MOSS captioning is selected.

What settings to actually use for a given backend (rank, steps, optimizer, crop length, and the rest of the Advanced drawer) is not this page's concern; see the backend-specific guide: [ACE-Step training](../training/ace-step.md), [MiniMax-Music3 training](../training/minimax-music3.md), or [YuE2 training](../training/yue2.md).

Side-Step is a separate, more comprehensive Python training suite by the same author. This system reaches parity with it, and its LoRA/LoKr adapters are interchangeable with this studio's own. A link to it appears under the page title.

<!-- TODO(verify): confirm whether a dataset's audio files are required to stay in their original source folder after Build/Preprocess/Train, or whether the folder can be moved once dataset.json and any tensor caches exist. -->
<!-- TODO(verify): confirm whether the generation-side adapter picker reads the adapter folder live or needs a refresh/restart after a training run finishes. -->

## Related

- [ACE-Step training](../training/ace-step.md)
- [MiniMax-Music3 training](../training/minimax-music3.md)
- [YuE2 training](../training/yue2.md)
- [Adapters](../adapters.md)
- [Model Manager](model-manager.md)
- [Backends](../backends.md)
- [Training system architecture](../../dev/training-internals.md)
