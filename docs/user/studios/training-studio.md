# Training

Builds a labeled dataset from your own audio and trains a style adapter on it, entirely on your own GPU. Reach for it when you want HOT-Step to sound like a specific artist, album, or sound rather than the base model: point it at a folder of songs, label them, and train a planner (LM) adapter, a sound (DiT) adapter, or both, depending on the training target.

![Training](../../images/studio-training-studio.webp)
<!-- screenshot: needed -->

## Where to find it

The "Training" entry in the sidebar (`/training-studio`). Labeling with local BPM/key analysis works even if the engine is down; AI captioning needs a local captioner (MOSS or Music Flamingo) or a configured cloud LLM provider (Settings > AI Services); preprocessing and training need the `ace-train` binary bundled with the app.

Training follows the active generation backend: ACE-Step, MiniMax-Music3, YuE2, HeartMuLa or Stable Audio 3. Switch it with the engine picker at the top of the sidebar, including for SA3 training. When more than one backend is available, a pill next to the title names that backend. The target drives the dataset chips, phase tabs, Preprocess page, Train page and checkpoint previews. While the open dataset has a preparation or training job queued or running, the studio keeps that job's target. The phase tabs, dataset chips and checkpoint-preview view come from one shared shell, with YuE2 as the reference, and each target shows only the tabs and settings it supports.

An external agent (Claude Code, Codex) can also drive the dataset and training pipeline through the `hotstep` MCP server. See [Driving HOT-Step with an agent](../mcp.md).

HeartMuLa uses YuE2's preparation cards, live progress panel and training form, including recipe cards and saved presets. Train multiple queues datasets on the server, and Perform all stages prepares stale audio before training. Its cache-clear card deletes only HeartMuLa preparation. Its checkpoint ladder lives on Review. Its Prepare and Train pages show its own preparation status and training readiness for the open dataset. They remain available without ACE-Step capabilities or a built ACE-Step manifest; missing HeartMuLa requirements are shown on the cards.

## Preparing your files

The folder you point the studio at is the source of truth: audio files live there, and each track's metadata lives in a small text **sidecar** next to it. Nothing is cached — edit or delete a sidecar on disk and the studio reflects that change. On first scan a track with no usable caption shows as **unlabeled**; the Label step fills in captions, lyrics and BPM/key for you, but you can also author sidecars by hand to skip it.

**Supported audio:** `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`. When several files share a base name (e.g. `song.flac` and `song.mp3`), only one is kept, in this priority order: `.flac` > `.wav` > `.m4a` > `.aac` > `.ogg` > `.opus` > `.mp3`.

For a track at `…/MyArtist/song.wav`, the studio looks for sidecars beside it and uses the first convention that matches, in this order:

1. **Single sidecar** — `song.txt` with one `key: value` per line. The six recognised keys are `caption`, `genre`, `bpm`, `key`, `signature`, `is_instrumental`; everything after a `lyrics:` line is taken verbatim as the lyric. A track counts as labeled as soon as this file has a `caption:` line:
   ```
   caption: dreamy lo-fi beat, soft vinyl crackle, warm bass
   genre: lofi hip hop
   bpm: 82
   key: F# minor
   signature: 4/4
   is_instrumental: true
   lyrics: [verse]
   ...
   ```
   Only `caption` and `lyrics` matter for labeling; the rest are optional.
2. **Split files** — `song.caption.txt` holds the caption (its whole contents) and `song.lyrics.txt` holds the lyrics. Used when there is no usable single sidecar.
3. **Lyrics-only** — a bare `song.txt` with no `key: value` lines is read as plain lyrics, so the track has lyrics but still no caption and stays unlabeled until one is added.

**How a caption is found:** a track's caption is the `caption:` line from its single sidecar, or the whole contents of a `song.caption.txt` file — whichever exists first. If neither is present the track has no caption and shows as **unlabeled**; that is what the Label step generates (an AI caption for style/genre, Genius for lyrics, Essentia for BPM/key).

**File names with dots:** a sidecar's base name is the audio file minus its final extension, so `feat. jo.smith.wav` pairs with `feat. jo.smith.txt` — only the last dot (the extension) is stripped.

Some backends also read a model-specific caption sidecar beside the generic ones above: `<name>.mm3.txt` for MiniMax-Music3's structured caption and `<name>.yue2.txt` for YuE2's planner sentence.

## Workflow

1. Click **New dataset**, pick a source folder (recursively scanned by default), and confirm the auto-filled name, trigger word, and lyric language. HOT-Step scans the folder and shows a preview (file count, extensions, whether a `dataset.json` already exists) before anything is created.
   If another dataset already uses that trigger, HOT-Step appends this dataset's slug suffix; the resulting trigger appears on its card so the two datasets stay distinct.
2. On the **Label** step, run local analysis: Essentia for BPM/key, Genius for lyrics, and an AI caption (local MOSS or Music Flamingo, or a configured cloud provider) for style and genre. **Transcribe missing lyrics locally** (on by default) fills every track with no lyrics with a Music Flamingo transcription from the audio, with `[Verse]`/`[Chorus]` section tags, written as each track finishes; tracks marked instrumental are left alone, and lyrics already in a sidecar are kept unless the merge policy overwrites lyrics, which re-transcribes them. Genius is off by default. A transcript that falls into a repetition loop is redone once with a light repetition penalty. It runs on the GPU, about 5 seconds a track on an RTX 5090, and the toggle is disabled with the reason shown when the Music Flamingo files are not in `models/flamingo/`. Lyrics from every source are written with one set of section tags, unnumbered: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Breakdown]`, `[Interlude]`, `[Instrumental]` and `[Outro]`. Genius names, speaker tags and stage directions are dropped and the words kept; `[Hook]` becomes `[Chorus]`, or `[Post-Chorus]` in a song that also has a chorus. For lyrics already in a dataset, **Normalise lyric tags** in the Enhance panel rewrites them the same way, backing up each changed sidecar; with **Recover missing sections from the audio** on, lyrics that have no section tags get them from a Music Flamingo transcript lined up word by word with the existing lyrics, whose words are kept. Optional minimum and maximum tempo fields narrow Essentia's search range for this run; blank fields keep its defaults. If a file's embedded BPM tag is half or double Essentia's estimate, the tag resolves that tempo ambiguity; otherwise Essentia's estimate is kept. Scope the run to unlabeled tracks, all tracks, or a grid selection, and choose a merge policy for tracks that already have partial data.
3. On the **Review** step, check and hand-edit every track in a spreadsheet-style grid: caption, genre, BPM, key, time signature, language, instrumental flag, lyrics, per-track trigger tag, and an exclude checkbox. The BPM cell has ½ and 2× buttons to correct a detected half or double tempo; the changed value saves like a typed edit before you build the dataset. Select rows for bulk edits or a scoped re-label.
4. On the **Build** step, set the trigger word's position in the caption (prepend, append, replace) and a genre-mix ratio, then build. This writes `dataset.json` next to the audio, in the same format Side-Step's own dataset builder produces.
5. Move to the second phase. On **Preprocess** (ACE-Step), pick the base DiT, VAE and text encoder and how much of each track to encode, then start. On **Codes** (MiniMax-Music3), export the RVQ codes. Both run in the standalone `ace-train` binary, which needs the GPU to itself, so the app stops the local engine for the duration and restarts it afterward. Stable Audio 3 has no such tab, because its Train page strips vocals and the trainer encodes the audio itself. Under HeartMuLa this phase is also **Prepare**: it encodes each track to HeartCodec codes and builds its prompt, see [HeartMuLa training](../training/heartmula.md). Under YuE2 this phase is **Prepare**: the latent cache, semantic codes, lead sheets and (with lyric timing on) the vocal stems and alignment, with a **Perform all stages** button that runs whichever are missing.
6. YuE2 only: **Optimise** takes optional measurements of the prepared album before training, saved in the dataset folder as `_hotstep-optimisation.json` (see [YuE2 training](../training/yue2.md#optimise-optional)).
7. Move to **Train** and start a run. ACE offers two adapter cards, a planner LoRA (LM) and a sound LoRA (DiT); MiniMax-Music3 trains a single planner LoRA from RVQ codes; YuE2 trains the planner and decoder adapters together; its **Perform all stages** button runs any missing preparation first, then training. HeartMuLa trains one backbone LoRA, with checkpoint previews on its own **Review** tab. Stable Audio 3 trains one DoRA-rows adapter from the Train page, preparing the audio first. Each form shows only the handful of settings that change what gets trained; everything else sits behind an Advanced drawer.
8. Watch the run: a live loss chart and stats update as it trains, and saved milestone checkpoints appear as badges. YuE2 shows separate AR and NAR charts by step, side by side or stacked in a narrow window. Its checkpoint loss labels show the mean of up to 20 logged steps through that checkpoint.
9. ACE-Step: use **Codes audition** below the two cards to A/B the trained planner against the base model on identical caption, lyrics, and seed, decoded straight through without the DiT. Optionally render both sides through the DiT too for a fuller comparison. Click a planner milestone badge to audition that checkpoint directly. Stable Audio 3 has its own **Adapter audition** card (see [Stable Audio 3 training](../training/sa3.md#audition)); MiniMax-Music3 and YuE2 runs are judged from their checkpoint previews.
10. Checkpoint previews use one shared view on the Train page: a collapsible settings block with an on/off switch and that target's preview settings, and one card per checkpoint with a player for each take. YuE2 renders a scored ladder from it (see [YuE2 training](../training/yue2.md#picking-the-adapter)), MiniMax-Music3 renders a preview every set number of steps or minutes, and Stable Audio 3 pauses training at each cadence step to render every prompt and seed (see [Stable Audio 3 training](../training/sa3.md#checkpoint-previews)). ACE-Step has no checkpoint previews: use **Codes audition** instead. **Review** lists every YuE2 dataset with a ladder still to score. The older **Refine** phase (KL-rung refinement of the earlier recipe) is hidden; Review still opens its ladders.
11. Use **Import multiple…** from the dataset list to queue several source folders through label, build, (for ACE) preprocess and train, and Send to Lyric Studio in one unattended run; the **Monitor** phase (ACE-Step and MiniMax-Music3 targets) shows its progress as a queue of datasets, each with a chip per pipeline stage. This is separate from YuE2's own **Train multiple…**, which runs preparation and joint training over several already-created datasets instead.

A small reset arrow appears beside a label whenever its value has moved from the default; click
it to snap that one field back. A blank optional field (shown greyed out with its default as a
placeholder) resets to blank rather than to a number, so it goes back to letting the server decide.
The DiT training method, quality dial and LM base-size pickers are the exception: their arrow
reapplies that control's whole preset bundle, the same as clicking the option itself, rather than
snapping back a single field.

On the MiniMax-Music3 and YuE2 trainer cards, the arrow returns a field to the server's *current*
default rather than a fixed number baked into the page — so if a later update changes the shipped
recipe, an untouched field keeps following it and only fields you actually changed stay put.

YuE2 Joint Training keeps its loss graph and step count visible with **Blind rungs** enabled. The **Review** tab lists unscored checkpoint ladders even when previews are disabled or have not finished rendering. Open a run to render previews and score its rungs.
The ladder shows the dataset name from the Review list as a static heading above its rungs.

## Controls

| Control | What it does |
|---|---|
| New dataset | Opens the import wizard: source folder, recursive toggle, name, trigger word, and default lyric language. The language list has every language the models know; **Auto / mixed** leaves each file's detected language alone, for datasets that mix languages. Correct a single file in the Review grid's language column. |
| Import multiple… | Bulk-imports several sibling folders as separate datasets and queues label, build, preprocess, DiT training, Send to Lyric Studio and LM training stages for each (LM training is off by default); watched from the Monitor phase. |
| Dataset filter | Filters the dataset grid by dataset name, artist (from the linked Lyric Studio album) or album. Under YuE2 a second menu shows only datasets whose AR and NAR adapters are both done, or only the ones still to do. The AR and NAR chips on each card read joint-training runs as well as the older separate trainers. |
| Base model (YuE2 Joint Training) | The frozen model the adapters train against: the ConvRot int8 checkpoint (the default on a CUDA build, and CUDA only) or any installed `yue2-lm-*.gguf` (any build). Beside it, **Device** names the GPU; empty means this build's first one. See [YuE2 training](../training/yue2.md#requirements). |
| Train multiple… (YuE2 only) | Runs preparation and joint training over several already-created datasets, one after another, with one recipe. While a batch is running, the same button adds more datasets to the end of its queue, trained with the running batch's recipe. The batch runs on the server, so it survives a page reload and a server restart (resume it from the batch panel after a restart). The panel sits above every Training Studio phase while a batch is running or paused, and disappears once it has finished; it has a "Go to the running training" link, and "Follow" opens each dataset as the batch moves to it and never pulls you off a dataset you opened yourself. |
| Train on | Shown when Settings lists a training worker. Points job control at that PC: its datasets, batches, live progress and charts, while its GPU does the training. Each checkpoint and preview it produces is copied to this PC as it lands, so ladders, previews, scores and cleanup always work on this PC's copy. **Fetch finished adapters** copies each adapter pair the worker has linked back to this PC and links it to the album preset here. See [YuE2 training](../training/yue2.md#training-on-another-pc). |
| Run on (Train multiple, YuE2) | Sends the batch to a training worker instead of this PC. Each dataset is captioned here (Gemini), its files are sent to the worker, and it joins the worker's batch; the Train on bar shows each dataset's progress through those steps. |
| Workers (beside Train on) | Shows each configured worker's Git commit and local-change flag, engine version and build time, GPU memory and use, and its active training job or idle state. Current, Behind by N commits, and Diverged compare the worker with this PC's commit. Update is available only for an idle worker behind this PC. |
| Mirror status (Review, with Train on set to a worker) | One line under the Review heading: whether the worker is online, the time of the last sync, and how many files (and how much data) are still to copy. A failed sync shows its error. **Sync now** runs a sync straight away and reloads the list. Syncs also run on their own, every 15 seconds while the worker is training and every 2 minutes otherwise. |
| Dataset chips | Every dataset card shows Labeled and Built, then pills that follow the target. ACE-Step: Tensors, DiT, LM. MiniMax-Music3: Codes, LM adapter. YuE2: Latents, Codes, Cursor spans, NAR and AR adapters. Stable Audio 3: Audio, Encoded, Snapshots, Adapter. A pill is on only when the file exists on disk. |
| Phase tabs | Switches between the stages above. Which tabs show depends on the target. ACE-Step: Dataset, Preprocess, Train, Monitor. MiniMax-Music3: the same, with Preprocess named Codes. YuE2: Dataset, Prepare, Optimise, Train, Review. Stable Audio 3: Dataset and Train. The earlier Refine phase is hidden since 2026-09-27; Review still opens its ladders. With a dataset open, **All datasets** at the left of the tabs goes straight back to the grid. |
| YuE2 ladder previews | The player streams a 192 kbps MP3 copy of each take, made the first time it is played (a few seconds) and kept beside the WAV. If ffmpeg is unavailable or conversion fails, it streams the WAV instead. The **download** link still saves the WAV as `<dataset>_step<N>_take<M>_seed<S>.wav`, with `rung<letter>` in place of `step<N>` while Blind rungs is on. |
| Delete run (YuE2 Review) | Removes a joint run after confirmation. Deletes the run folder, including its checkpoints, previews and MP3 cache. Archived logs and any checkpoint saved outside the run folder stay. Live runs are refused by the server, and the error appears on Review. Legacy standalone NAR/AR runs have no delete control. |
| Label step: scope, sources, merge policy | Restricts labeling to unlabeled/all/selected tracks, picks which of Essentia/Genius/caption to run, and how new data merges with what a track already has. |
| Review grid | Per-track editable table with bulk actions (exclude, mark instrumental, set genre/tag, scoped re-label). Opens a full-detail drawer per track. |
| Build: trigger position, genre ratio, Build | Where the trigger word sits in the built caption, and the mix ratio between genre-style and literal captions, then writes `dataset.json`. |
| Preprocess / Codes form | Base model pickers plus an Advanced drawer for encoding settings; MiniMax-Music3 shows an RVQ codes export instead, including an optional cover-laundering pass for dense mixes. Either way it is a collapsible stage card whose run button stays visible when folded. ACE-Step lists its tensor caches below the card, one per base model, each with **Delete cache**. |
| LM card / DiT card (Train, ACE-Step) | Collapsible stage cards: adapter name, a handful of always-visible settings, an Advanced drawer, and a **Start training** button that stays visible when the card is folded. Running shows a live chart, stats, and a done-state summary with the adapter's output path once finished. |
| Training GPU (YuE2 Joint Training) | On NVIDIA builds, pick a card for this run or use the GPU selected in Settings. The choice applies to the training child; see [YuE2 training](../training/yue2.md#recommended-settings). |
| AR (stage 1) / NAR (stage 2) charts (YuE2) | "Stage 1 writes the song's coarse token plan from lyrics and tags." "Stage 2 turns stage 1's tokens into fine audio detail." Joint runs show AR CE and KL vs base on the AR side, Decoder loss and Reconstruction on the NAR side. See [Reading the training charts](../training/yue2.md#reading-the-training-charts) for the metrics and stop readings. |
| More (YuE2 charts) | Reveals extra series controls: Cursor CE and Learning rate on the joint AR side, Drift and Learning rate on the joint NAR side. Grad norm appears once below the pair for joint runs, alongside step count, Pace and VRAM when recorded. |
| Milestone badges (ACE-Step) | Appear in the done summary once a run has saved milestones (Milestone loss step above 0). Click a planner (LM) badge to audition that checkpoint; the sound (DiT) badges are labels only. |
| Codes audition (ACE-Step) | Same-seed A/B between the base planner and the trained adapter, with an optional DiT render pass. |
| Send to Lyric Studio (Build step) | Exports the dataset's artist/album metadata into Lyric Studio, optionally linking the dataset's trained adapters and a reference track as that album's generation preset. |
| Monitor queue | Read-only view of the batch pipeline: one row per dataset, a status chip per stage, live progress on whichever stage is currently running. |

## Tips and limits

### Updating a training worker

Open **Workers** beside **Train on**. An idle worker marked **Behind** can receive an update from this PC. The update sends a Git bundle through the token-protected worker API, discards the worker checkout's tracked local changes while preserving its patched ggml submodule, installs server or UI dependencies only when their lockfiles changed, builds the UI, and rebuilds the engine only when files under `engine/` changed. If the bundle changes the recorded ggml commit, the update checks out that commit, replays every training patch in name order, and checks the engine hooks before building. Progress and build output appear in the worker row. The launcher restarts the worker when all steps finish. Updates stop if a job is running or queued. **Diverged** means the worker has commits this PC cannot advance from; resolve that history before updating. A worker running an older version without the update endpoint needs a one-time bootstrap.

YuE2 lead sheets are also saved as plain-text `<stem>.abc` files next to the
dataset audio and its metadata sidecar. A later lead-sheet stage reuses these
scores instead of transcribing those tracks again; a forced transcription
replaces a score only when it produces usable ABC. Clearing prepared caches or
starting a fresh batch does not remove these source files.

Trained adapters are written under the app's adapters folder, split by base architecture and size (for example a 4B ACE-Step planner LoRA lands under `lm-4b/<name>/<run>/`, a DiT LoRA under `dit-<base>/<name>/<run>/`), with a timestamped subfolder per run so retraining never overwrites an earlier adapter. They show up in the adapter pickers used elsewhere in the app (Create, Custom-Gen, the audition card here) without any extra step; see [Adapters](../adapters.md) for how adapters are selected, scaled, and stacked at generation time.

ACE-Step preprocessing and training, MiniMax-Music3 codes export and training, the YuE2 preparation stages and Stable Audio 3 training stop the local engine by default while `ace-train` runs and restart it afterward, so expect generation to be unavailable for the duration. MiniMax-Music3 and Stable Audio 3 pause training to bring the engine back for each checkpoint preview. YuE2 Joint Training leaves the engine running by default so its previews render in parallel; see [YuE2 training](../training/yue2.md#recommended-settings). Labeling does not require the engine unless a local captioner (MOSS or Music Flamingo) or local lyric transcription is selected. The local captioners are told each file's own genre tag, because without it they recognise a cover from its lyrics and describe the original song's genre instead of the arrangement; they also write the YuE2 caption themselves, so a local caption run makes no call to a chat provider. A dataset with no genre tags gets no hint.

What settings to actually use for a given backend (rank, steps, optimizer, crop length, and the rest of the Advanced drawer) is not this page's concern; see the backend-specific guide: [ACE-Step training](../training/ace-step.md), [MiniMax-Music3 training](../training/minimax-music3.md), [YuE2 training](../training/yue2.md), or [Stable Audio 3 training](../training/sa3.md).

Side-Step is a separate, more comprehensive Python training suite. This system reaches parity with it, and its LoRA/LoKr adapters are interchangeable with this studio's own. A link to it appears under the page title.

<!-- TODO(verify): confirm whether a dataset's audio files are required to stay in their original source folder after Build/Preprocess/Train, or whether the folder can be moved once dataset.json and any tensor caches exist. -->
<!-- TODO(verify): confirm whether the generation-side adapter picker reads the adapter folder live or needs a refresh/restart after a training run finishes. -->

Dataset and training-status views keep results separate for each **Train on**
target. Switching workers does not cancel the job on the previous worker.
Run indexes, mirrored previews and review records remain local to this PC.
Active job status refreshes while work runs and stops at a terminal state;
returning to Train, Review or Refine does not repeat a start or audition command.
The YuE2 Train tab keeps a finished run's chart and stops replaying its history
once the complete chart is saved. An interrupted first replay resumes on the
next visit. After a server restart, joint runs recover their chart from the
run folder's training logs. If those logs are unavailable, Train says the
history is no longer available and does not retry or save an empty chart.
Finished and approved runs leave Review's "Scored or finished" list.
Review opens faster by reading the run catalogue once per request for all datasets.

## Related

- [ACE-Step training](../training/ace-step.md)
- [MiniMax-Music3 training](../training/minimax-music3.md)
- [YuE2 training](../training/yue2.md)
- [Stable Audio 3 training](../training/sa3.md)
- [Adapters](../adapters.md)
- [Model Manager](model-manager.md)
- [Backends](../backends.md)
- [Training system architecture](../../dev/training-internals.md)
