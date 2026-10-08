# YuE2 training

This guide covers training YuE2 adapters in the [Training Studio](../studios/training-studio.md) with Joint Training, the default method: what gets trained, what the dataset needs, which preset to start from, and how to refine and pick the finished adapter by ear. YuE2 must be the active backend, because the Training Studio trains for whichever backend is selected in the top bar. For how the trainer works inside, see [training internals](../../dev/training-internals.md).

## What you can train

YuE2 has two halves, and Joint Training trains an adapter for each in every step.

| Half | Also called | What its adapter changes |
|---|---|---|
| AR | The planner | Structure, diction and late-song behaviour: what gets sung where, and whether the song holds together to the end. |
| NAR | The decoder | Timbre. Most of the likeness to the artist lives here. |

The two adapters are saved together in each checkpoint and applied together at generation time.

The older Legacy seven-stage trainer was removed from the Training Studio on 2026-09-27; Joint Training is the only method offered.

## Requirements

- A GPU the app's build supports: CUDA (NVIDIA) or Vulkan (AMD, Intel, NVIDIA). On a Vulkan build Joint Training trains on a GGUF base and runs several times slower than on CUDA; roughly 4 to 5 minutes per update of 4 songs on an RTX 4090 over Vulkan, against about 1 minute over CUDA.
- A base model, picked with **Base model** on the Joint Training card:
  - **ConvRot int8** (`yue2_3b_int8_convrot.safetensors`) is the checkpoint the recipe was tuned and ear-tested on. It needs a CUDA build, and it is the default there.
  - **GGUF** is any installed `yue2-lm-*.gguf`, on any build. bf16 is full precision; the quantized files hold the base in less memory (Q4_K_M is about 2 GB) and trained within about 1% of bf16's first-step loss. On a build without CUDA the default is bf16 when it is installed.
  The adapter a run produces works with every base at generation time.
- The YuE2 Joint Training Pack, installed from the Model Manager. The card warns if model paths are missing.
- YuE2 selected as the active backend. Under YuE2 the Training Studio shows Dataset, Train, Refine and Review; there is no separate Preprocess phase, because the caches are built on the Train page.

## What the dataset needs

Create and label the dataset in the Dataset phase as for any backend (see [Training Studio](../studios/training-studio.md)). Lyrics matter more here than for the other backends: the AR half trains on each track's caption and lyrics, and lyric timing supervision aligns against those lyrics. Tracks without lyrics are skipped by those stages.

Captions. The planner is prompted with one sentence in a fixed order (language, genre, vocal, instruments, mood, production, BPM). A labelling pass writes that sentence to `<stem>.yue2.txt` beside each track, using a chat provider to rewrite the existing label; MOSS has no mode for it. For tracks labelled before this existed, the "YuE2 caption" button under Enhance labels backfills them. A re-run overwrites the sidecar in place; no `.prev` backup is kept (MM3 captions still keep one).

The latent cache's "Clip captions" setting decides what the trainer reads. When the dataset has sidecars it defaults to "Sidecar lyrics, with the one-sentence YuE2 caption (.yue2.txt) as the style", which falls back to the ACE-Step caption for tracks without a `.yue2.txt`. The other modes (trigger word only, one typed caption for every clip, or the raw sidecar text) throw the lyrics away or feed field names in as style. If `.yue2.txt` files are written or changed after the cache was cut, the card warns. For Joint Training you do not need to re-cut: preparation notices the stale captions and re-reads the sidecars without re-encoding any audio.

Loudness. The cache brings every track to -14 LUFS before encoding. A cache cut before that existed trains at each album's own mastering level, and loud albums then render clipped; the card warns and a re-run fixes it.

The latent cache's "Rewrite the manifest" option is on by default: a re-run rewrites the manifest with the caption mode chosen above and re-cuts at the clip length, and the cached latents are reused, so it costs no GPU time. "Perform all stages", "Train multiple" and Start training cut the cache with the YuE2 caption mode, and re-cut a cache that was cut with the ACE captions while `.yue2.txt` files sit beside the audio; the card warns about such a cache until it is re-run.

**Caption tracks that have no YuE2 caption** (on by default, on the Joint Training card and for Train multiple): before training, every track without a `.yue2.txt` is re-captioned from the audio, the same as the Label step with only the caption box ticked. It rewrites the ACE caption and writes the MM3 and YuE2 captions; lyrics and BPM are left alone. Pick Gemini (with any model from the live list) or MOSS; MOSS still needs a chat provider such as Gemini for the one-sentence YuE2 caption. It does nothing when every track already has a `.yue2.txt`, and training stops rather than fall back to the long ACE captions if any track is still missing one afterwards. In a batch it is the `captions` stage, before the latent cache; if tracks are still uncaptioned afterwards (a network outage fails every call at once), the batch waits ten minutes and runs the stage again, up to three times, before giving up on that album.

Files named `*.engine.wav` in a dataset folder are ignored. Older versions left them beside the source as conversion caches; they are copies of real tracks and, if trained, teach the adapter instrumental versions with no lyrics. Delete any you find.

The cache stages are latent cache, codes, lead sheets, vocal stems and lyric cursor spans. The last two are only needed for lyric timing supervision, and appear once its toggle (its own card, between lead sheets and the stems) is on. "Perform all stages" runs them in order; beside it, the clear card deletes the vocal stems and the dataset's MM3 and ACE caches. It keeps the YuE2 latents, codes, lead sheets and prepared set (about 40 MiB an album, but minutes of GPU to rebuild; they are rebuilt anyway when the audio, captions or loudness change) unless you turn on **Also delete the YuE2 latents, codes, lead sheets and prepared set**. Source files, labels and adapters are always kept. Each stage card opens collapsed: its run button stays visible (it reads "... again" once the stage is complete), and clicking the title shows the explanation, status and settings. Lead sheets can be previewed as each source finishes, not only when the stage ends. You do not have to prepare anything by hand: Start training prepares the dataset automatically and reuses prepared data that has not changed.

"Perform all stages" sends one accepted preparation command. The server captures the dataset and source revisions, selected worker, stage list and (for the joint card) the visible form before preparation begins; preparation-only ends without starting a trainer. It keeps running after you close the Training Studio — reopen the dataset's Prepare or Train page to see the latest command and each stage's status, with Pause, Resume, Cancel and explicit Retry for a failed or interrupted command. The preflight skip list above the button is advisory; the server checks current artifacts before each stage, and a changed dataset or source stops the accepted command rather than running against stale data. For joint training, a form edit after pressing "Perform all stages" does not change that command; direct **Start training** still uses the card's current form. If the server restarts, the run waits for an explicit retry; it does not start another training job on its own.

How many tracks: the app warns below 10 files.
<!-- TODO(verify): no track-count guidance specific to YuE2 joint training was found in code or skills. -->

## Optimise (optional)

The YuE2 phases run **Dataset → Prepare → Optimise → Train → Review**. Prepare holds the latent cache, codes, lead sheets and the optional lyric-timing stages; Train keeps the joint training card, **Train multiple** and the ladder, and its **Perform all stages** still runs any missing preparation first.

Optimise measures the prepared album before training. Nothing on it changes training by itself. Every result is saved in the dataset's own folder, beside the audio, as `_hotstep-optimisation.json`, one section per measurement, so it can be read again later without the app.

**Base model loss** measures how far the album is from what YuE2 already knows: the base model's loss on every song, with no adapter. **Planner CE** is how surprised the planner is by the album's songs, structure and lead sheets; **Decoder MSE** is how surprised the decoder is by its sound, averaged over three noise levels on the same 60 s window training uses. Higher means further from the base, but dense, busy music also reads higher, so compare albums of a similar style. The page lists every song, most unfamiliar planner first: a song far out of line with the rest is worth checking for a wrong caption or lyrics before training. The same album measures the same every time (fixed crops and noise). It runs the same preparation as training (reused when unchanged) and takes about a minute for a 20-song album on an RTX 5090, with the engine left running. Whether these numbers predict how an album trains is not yet known; they are collected so Dataset-Calibrated Training can find out.

## Training method

Since 2026-09-27 Joint Training uses one recipe, taken from how the YuE2 technical report says the base model itself was trained, as far as an adapter can follow it: the planner's loss weighted at a quarter of the decoder's, AdamW with betas 0.9 / 0.95 and weight decay 0.1, the decoder trained on whole songs (the presets crop it to 60 s, which a blind test scored level with whole songs), several songs averaged into each update, the style text or lyrics dropped from the prompt on some steps, and the audio cache cut without loudness normalisation. It has no KL anchor and no early stop: the run goes to the update count, saves a checkpoint every 10 updates, and you pick the rung by ear.

It replaced the earlier recipe (Prodigy, a KL anchor and KL stop, lyric timing, a 60 s decoder window, then a refinement pass) after a blind ear test on a full album: every rung of the new recipe scored 5 on diction, audio and coherence, with no looping, bad endings or late-song decay at any point, and likeness reached 5 by 100 updates. It was judged better in every way. The old recipe still exists in the server and engine so that its runs can be resumed, but the card no longer offers it, nor the "Refine a finished adapter" box.

Two of the recipe's numbers are agreed guesses rather than report values: the learning rate (1e-4; the report only gives full-model rates) and the three prompt-dropout rates (0.1 each; the report says the drops happen but not how often). The rest are the report's.

Things to know:

- The first run on a dataset re-cuts its audio cache with loudness normalisation off (the cache key changes, so the codes and lead-sheet stages run again) and re-prepares it once with the prompt variants the recipe trains on. Both are done for you by "Start training" through the stage chain and by "Train multiple".
- Songs per update multiplies the time per update: 4 songs with the presets' 60 s decoder crop is about 17 s an update on an RTX 5090, 8 songs about twice that; 4 whole songs is about 28 s, 8 about 60 s. Whole-song decoder training takes about 12 GB of VRAM. If a song and its prompt do not fit the model's context, the decoder trains on the longest window that does.
- Lyric timing supervision is not used by this recipe; the toggle only decides whether the stem and alignment stages run, and it is off by default.

## Recommended settings

On an NVIDIA build, Joint Training can choose a GPU for each run. Leave **Training GPU** on **Settings GPU** to use the card selected in Settings. A chosen card is pinned by UUID and appears as CUDA0 inside the trainer.

Pick a preset. Fast, Balanced and Thorough are the same recipe and all train a dim-128 LoKr; they differ in how many updates run, how many songs each update averages, and how often a checkpoint is saved. All three train the decoder on 60 s crops. Times are estimates for a 15-track album on an RTX 5090, from measured per-update times.

| Preset | Updates | Songs per update | Decoder | Checkpoint every | Time (card shows it relative to Balanced) |
|---|---|---|---|---|---|
| Legacy | up to 500 (stops at planner KL 1.0) | 1 | 60 s crops | 25 updates | ~20-25 min, varies |
| Fast | 100 | 4 | 60 s crops | 10 updates | ~30 min |
| Balanced (default) | 200 | 4 | 60 s crops | 20 updates | ~1 h |
| Thorough | 300 | 8 | 60 s crops | 30 updates | ~3 h |

**Legacy** runs the recipe the card used before the current presets: Prodigy with cautious updates, one song per update, a dim-256 LoKr (alpha 256), caption dropout 0.5, and a stop once the planner's KL to the base model reaches 1.0. It can finish sooner than Fast because it stops early, but its adapters scored lower by ear than the new presets, which fixed its weak endings, structure problems and late-run degradation. It is close to the old recipe, not exact: lyric timing is off (it needs vocal stems and alignment first), and an audio cache already cut for the new presets is not re-cut for it. With checkpoint previews on, it renders a preview at every saved checkpoint, as the other presets do.

**Save every** (Legacy, KL target only) can count in **Steps** (default 25) or **AR KL** (default 0.15). In AR KL mode a rung is saved, and previewed, each time the planner's KL passes the next multiple of the value, up to the KL target, plus one at the target itself. Steps saves still happen every 25 updates as resume points, but they get no preview. Once the planner freezes, its KL stops moving, so the decoder phase saves and previews every 25 updates instead. If previews pause training (cards under 28 GB), the run also pauses every 25 updates during the planner phase, so you get extra previews there.

The optimizer is a setting under **Advanced**. Fast, Balanced and Thorough default to AdamW (graph), which runs the recipe's warmup and cosine decay; Legacy defaults to Prodigy. Plain AdamW uses a flat rate after warmup, with no decay.

**Advanced** also has a **Legacy recipe** section with the knobs only Legacy reads: what ends the run (AR KL target, step count or target loss), the KL target and how it is read, planner and decoder learning-rate scales, the KL anchor, caption dropout, the spike guard and its stop, and the learning-rate schedule. Under the other presets the section is greyed out, because they fix their own values. A blank field uses the recipe's own value.

**Train the decoder on after the KL stop** (Legacy, KL target only; on by default when you pick Legacy): at the KL target the planner freezes instead of the run ending, and the decoder trains alone until its reconstruction meter flattens. That is the plateau test the Refine page used: a line fitted through the last 10 checkpoints must gain at least 0.5% across them, or the run stops. A decoder checkpoint is kept only if its reconstruction beats the best so far by 0.003, so the ladder shows real gains. The decoder budget (250 steps by default) and **Updates** both cap it, so raise Updates if the KL stop comes late. An optional reconstruction target also stops it.

Every joint run, whatever the preset, logs the planner's KL to the base model at every update and the decoder's reconstruction and drift meters at every checkpoint, so runs can be compared. The KL reading costs about 7% of each update under the new presets, which do not otherwise read it.

The new presets save ten checkpoints each, so the ladder to listen through is ten rungs long; see Picking the adapter below.

How the presets were chosen: the first ear test (8 songs x 200 updates, whole songs, dim-64 LoKr) reached full likeness and quality by update 90. A blind test with three takes per condition then scored 60 s decoder crops level with whole songs, at about 40% less time per update. The current values, 200 updates and dim 128 for Balanced, come from an overnight batch of albums at those settings that sounded right by ear; whole songs gave no audible gain over crops, so every preset crops.

Defaults the card ships:

| Setting | Default | Notes |
|---|---|---|
| Adapter type | LoKr, dim 128, factor 4, alpha 256 | About 213 MB for the AR and NAR pair (dim 64 is about 106 MB); the card shows the expected size for whatever dim, factor or rank you enter. |
| Learning rate | 1e-4 | Linear warmup over 3% of the updates, then cosine decay to 0.1x at the end. |
| Weight decay | 0.1 | The report's value. |
| Adam beta2 | 0.95 | The report's value; the old recipe used 0.999. |
| Planner loss weight | 0.25 | The report's value. |
| Prompt dropout | 0.1 text, 0.1 lyrics, 0.1 both | Agreed guesses; the report gives no rates. |
| ABC dropout | 0.5 | Matches the report's balanced mix of tasks with and without a score. |
| Decoder crop | 1500 (60 s) on every preset | The report packs whole songs; 0 trains on them. In a blind test the crop scored level with whole-song runs and trains about 40% faster. |
| Planner crop | 0 (whole song) | Off by default and not yet tested by ear. The planner trains on only the first this-many frames of each song (25 per second). It is the biggest speed lever, since the planner's backward pass is the largest cost of an update, but the planner never learns the rest of the song or its ending, and whole songs are what fixed endings and structure in this recipe. Only applies while the decoder crop is on. |
| Save every | 10 / 20 / 30 updates (Fast / Balanced / Thorough) | Ten rungs per preset. |
| Lyric timing supervision | Off | Not used by the recipe. |
| Checkpoint previews | On, in parallel, two 300 s draft takes per checkpoint (fixed seed and random seed). On a card under 28 GB, or one the app cannot measure, training pauses for each preview instead | The run's ladder; see Picking the adapter. |
| Stop the engine during training | Off | In the Checkpoint previews section. The parallel previews need the engine up, so turning it on hides "In parallel with training". Turn it on (and previews off) on a smaller card. |

When to move off them:

- Turn "Stop the engine during training" on, and previews off, on a card without room for the trainer (about 14 GB) plus the engine and a render (10-12 GB) at once; then render the ladder after the run.
- The form is saved per dataset in this browser, so your settings survive a restart. "Changed from defaults" under the presets lists every setting you have moved off the recipe; reset one with its arrow, or all of them with **Reset all to defaults**.
- **Your presets**, under the Fast / Balanced / Thorough buttons, saves your current settings under a name in this browser. It never saves dataset, checkpoint or output paths. Click a preset's name to load it; it is highlighted while the form matches it. The save icon overwrites it with the current settings and the cross deletes it, each after asking; saving under an existing name also replaces that preset after asking. The download icon on a preset saves it as a `.json` file to share or back up, and **Import** adds presets from such a file (any paths in it are dropped).
- Every YuE2 training stage is stopped as hung only when it prints nothing for an hour. There is no total time limit, so a slow GPU or Apple Silicon can run a long job to the end.

**Dataset-Calibrated Training** (off by default, experimental) sizes the run to the album instead of running the preset as shown. The preset's updates and save interval are both multiplied by the album's minutes of audio divided by 45, kept between 0.6 and 2, so the ladder keeps its ten rungs: a 32-minute album on Balanced trains for 140 updates and saves every 14, a 74-minute album for 330 and every 33. The rule comes from the ear-scored ladders of the earlier recipe, where longer albums took longer to reach full likeness. It has not yet been confirmed on this recipe, and runs with it on are that test. It applies to new runs, single and in "Train multiple" (each album is sized on its own), never to a resume. The training log gives the numbers it chose and the run records them. Off, training is exactly as before.

"Train multiple" runs the whole chain (cache, codes, lead sheets, preparation, joint training) for several datasets in sequence with shared settings and a separate output folder each. Its dataset picker hides datasets that already have a linked YuE2 adapter pair; turn on **Show trained** to list them again for a retrain. The batch runs on the server: it survives a page reload, and after a server restart it is listed as paused so you can resume it with every finished stage kept. The batch panel above the Training Studio phases, shown only while a batch is running or paused, lists the queue, the running stage's step count, and a link back to the running training from anywhere in the studio.

## Training on another PC

A second PC with its own GPU can train YuE2 batches while this one does something else, managed from this PC's Training Studio.

1. On the other PC, install HOT-Step at the same version, copy the YuE2 models it needs (the files under `models/yue2` and `models/supersep`), and set **Worker token** in Settings → Server to a long random string. Allow inbound TCP 3001 through its firewall, and start the app.
2. On this PC, set **Training workers** to `Name=http://<its address>:3001` and **Training worker token** to the same string.
3. Start **Train multiple** here and pick the worker under **Run on**. For each dataset, this PC writes any missing YuE2 captions (always with Gemini, so the worker needs no API keys), sends the audio, sidecars, labels and the preview lyrics Lyric Studio would pick, and adds the dataset to the worker's batch. The first dataset starts training while the rest are still being sent. A second send of the same dataset copies only files that changed.
4. The studio switches **Train on** to the worker so you can follow the run: job control (start, cancel, status) comes from there. This PC copies each checkpoint's AR and NAR weights and each finished preview into its own `yue2-joint-adapters/<trigger>_<date>_<time>` folder as they land on the worker (the date and time are when the run started there). From then on the run is an ordinary run on this PC: the ladder, previews, rung and album scores, Use this rung, cleanup and Finish scored all read and write local files. Until the first sync lands a new run, the Train page says it is waiting for it. Once the worker's run has ended, its previews are all rendered and every file has been copied, the worker's copy is deleted. A run stopped part-way (interrupted) stays on the worker so it can resume. Deleting a copied run here also stops it coming back on the next sync.
5. **Fetch finished adapters** copies every adapter pair the worker has linked (by a finished ladder, "Use this rung", or the end of a run) into this PC's adapters folder, under the same relative path, and links it to the album preset here.

Syncs run in the background whether or not the studio is open: every 15 seconds while the worker is training or has files still to copy, every 2 minutes otherwise. The Review page shows the last sync, what is still to copy, and a **Sync now** button. Training weights for resuming (`optimizer.resume`) are never copied, so a run cannot be resumed on this PC.

Limits: only the Training Studio's routes reach the worker; generation still runs on this PC. The queue of datasets waiting to be captioned and sent lives in memory, so a restart of this PC's server drops it (datasets already on the worker's batch keep going); start the rest again and nothing already sent is sent twice. A dataset's files are not replaced while a job is running for it on the worker.

## Picking the adapter

Every saved checkpoint is a rung of the run's ladder, shown under the run on the Train page as soon as training starts. With previews on (the default) each rung gets two 300 s draft takes (12 decoder steps, the caption of the dataset's first sung track, and by default the newest lyrics generated for this artist in Lyric Studio, so each rung sings a song it never trained on; **Preview lyrics** switches to the dataset's own lyrics, and a dataset with no generations uses its own lyrics anyway. Every rung of a run, including ones rendered later with **Render**, sings the same words) rendered in parallel while training continues (on a card under 28 GB training pauses for each preview, since both together do not fit), so the ladder is ready to listen to when the run ends; a rung without a take yet says so, and **Render** adds takes.

Take 1 uses seed 424242 on every rung. With the same seed, prompt and lyrics, neighbouring rungs write similar lead sheets: tempo, key, intro and section layout tend to stay put while the melody and chords change. That makes take 1 the one to compare rungs on, since most of what changes between them comes from training. Take 2 uses a new random seed on every rung, so it is a song no other rung made. It shows how the checkpoint does on a fresh draw and catches failures (a garbled line, a runaway ending) that the fixed seed happens to miss, but it is not comparable rung to rung. Once a rung has its fixed-seed take, **Render** adds random-seed takes only. Rendering in parallel keeps the engine running during training, which is why "Stop the engine during training" is off by default; on a smaller card turn previews off and render the ladder afterwards.

Each preview re-plans until its lead sheet passes the judge with no plan flags (one chord for the whole song, a looped riff, a melody on one or two pitches, too few sections), up to 10 tries, and every take lists its flags. When a checkpoint cannot produce a clean plan in 10 tries, the least-flagged plan is rendered anyway and the rung shows a red banner saying the checkpoint is not writing good plans.

Joint ladders start with **Blind rungs** on. The cards appear as Rung A, B, C in letter order, with the step and training meters hidden while you listen. Turn it off in the ladder header to see the usual step order and details. The label stays with its checkpoint if you leave and return or more checkpoints arrive. Score each rung 1-5 on likeness and corruption and leave notes; the saved score records whether you rated blind and which letter you saw, alongside the real step. The floating scoreboard marks the best by an overall score (likeness and inverted corruption averaged, minus a small penalty for re-plans). Press **Use this rung** on your pick: the checkpoint becomes the dataset's adapter (linked to its Lyric Studio album preset) and a cleanup dialog offers to delete the other checkpoints, the previews of the other rungs, the run's resume file, other joint runs for the dataset, and the vocal stems and other backends' caches, with sizes. **Keep everything** also finishes the ladder. After cleanup succeeds, the app reveals the step behind your chosen letter. The YuE2 latents, codes, lead sheets and prepared set are kept; so is the batch's **clear cache** option. The chosen rung's adapter files, the source audio and your scores are never touched, and the run stays where it is, marked finished. Blind rungs do not apply to the older Refine ladders.

Above the rungs, **How well did this album train?** takes one 1-5 score for the whole run, with notes, and for **Instruments** and **Vocals** which way each missed at the rung you would pick: **Under** (the band's sound or the singer never really arrived), **Right**, or **Over** (overcooked: memorised parts, the same riff or phrase in every song, garbled words). The score says how well the album trained; the directions say whether its next run should train more or less. Score it against your other albums, not against this run's rungs: most rungs end up at likeness 5, so the rung scores show how fast an album trains but not how well, and this score is what Dataset-Calibrated Training learns the difference from. The cleanup keeps what that learning needs: the run's loss log is copied into the dataset's `train-logs` folder when a rung is chosen or a run is deleted, and the lead sheets are saved as `yue2-sheets.json` before a full clear deletes them.

Batches score the same way: a finished run's ladder appears on the **Review** page as soon as it has previews, with how many rungs are still unscored. Click a row to open that run's ladder on the Train page. When a dataset has more than one run (a cancelled one and a retrain, say), a **Run** list above the ladder switches between them. Once a ladder has a scored rung, **Finish scored** captures the best-scored rung for every ladder you tick and queues those exact checkpoints in one server-side batch. The server checks the run and score revision when you press Finish; if the ladder changed, refresh it and choose again. A missing human score never becomes a scored rung from metrics alone. Manual **Use this rung** keeps its separate cleanup choice.

The ear test that set the recipe scored 5 on every quality criterion at every rung, so the pick there was the last rung; on other albums, listen for the point where likeness stops improving.

## The Refine phase (earlier recipe)

The Refine tab is hidden since 2026-09-27; its code is kept. It belongs to the earlier recipe and its KL rungs, and the Review page still opens those ladders on it. What follows applies only to runs made before that date and to refinements started from them.

Refinement continues the finished run with the planner live, saves a checkpoint each time the KL crosses the next rung, and renders previews per rung so you can hear where it starts to fall apart late in the song. Its defaults:

| Setting | Default | Notes |
|---|---|---|
| KL ceiling | 2.0 | A search range, not a target. Late-song decay has shown from about 1.7 on some albums and not at all by 1.8 on others. |
| Rung size | 0.1 KL | Smaller rungs mean more previews to listen to. |
| Learning rate | 0.1 times the source run's | |
| Tracks per rung | 2, 300 s each | Take 1 is **this rung's plan**: the rung's own planner writes the lead sheet. Take 2 is the **shared sheet**: the lead sheet the first rung wrote (the earliest rung whose plan passed the judge), so across the ladder take 2 is the same song and only the rung's composer and decoder change it. Every rung uses the same seeds (take 1 424242, take 2 424243) and the same sequence of re-plan seeds. Both takes render as one engine batch. Render more adds own-plan takes. The scoreboard's Overall subtracts 0.25 per replan per take and 0.1 per plan flag per take the rung planned itself, capped at 1 together. |
| In parallel with training | On | Needs VRAM for both, about 22 GB measured. Untick it on a smaller card. |
| Draft quality | On | Previews only: 12 decoder steps instead of 32, about a third of the decoder time. Timbre is a little softer; structure, diction and late-song behaviour are unchanged. |
| Keep a checkpoint per recon drop | 0.003 | Further decoder training only: a checkpoint is kept when the reconstruction meter has dropped by at least this since the last kept one; the rest are deleted once a newer one lands, so the ladder shows progress rather than every 10 steps. The newest checkpoint is always kept. |

The rung that crosses the KL ceiling is rendered too, like every rung before it. Each take shows how many planner and composer re-plans the render needed, and each rung sums them; rising counts are an early sign of over-training, softened by the app's own auto re-plan at generation time.

Each take's lead sheet is also read for legibility, since a plan can pass the health check and still be a bad song. A take is flagged when a voice repeats the same one-to-four-bar cell for 32 bars or more without variation, the vocal melody uses fewer than four pitches, the whole sheet has fewer than three chords, or the plan has fewer sections than the lyric has section tags. The rung shows "plan flags n/takes" in red, with the reasons on hover, and each flagged take names its worst finding. The thresholds were set against a handful of base and adapter plans, not against ear scores, so a flag is a prompt to listen closely, not a verdict, and it does not move the scoreboard. Cleanup now keeps every take's record and lead sheet (only the audio goes), so the flags can be checked against your scores later.

Pick the last good rung by ear. No automatic measure has been able to tell a good rung from a decaying one, so the ladder is a listening test. Score each rung's likeness (higher is better) and corruption (lower is better); a floating scoreboard ranks the rungs by an overall figure, (likeness + (6 − corruption)) / 2 minus 0.25 per re-plan per take (capped at 1), and marks the best one. Ties go to the earlier rung, which has drifted less. The scoreboard is a suggestion; the pick is yours.

When a ladder finishes, the panel says so and waits for you to press "Use this rung" on your choice. With "Further training for NAR" on, the decoder then trains on from that rung with the planner frozen, until its reconstruction target (0.25), the 250-step budget, or the plateau, whichever comes first. The plateau is read from a line fitted through the last 10 checkpoints (100 steps): the run stops when that line gains less than 0.5% across them, so one noisy reading no longer ends it, and it cannot stop in its first 100 steps. Turn **Stop at the plateau** off to train to the budget or the target regardless. At about 1.5 s a step, 250 steps is roughly 6 minutes. **Decoder rate (× run)** sets the decoder pass's rate as a fraction of the source run's, default 0.2, twice the 0.1 the ladder itself ran at; the ladder's pacing (a 30-step warm-up and any halving from skipped rungs) still applies on top. The plain run rate threw a converged decoder out of its basin within six steps, so raise it in small steps and watch the reconstruction meter. Pressing Stop during the decoder pass counts as finishing it: the last complete checkpoint on disk is linked as the adapter and the cleanup dialog opens, the same as when it stops on its own. The final checkpoint becomes the adapter; pressing "Use this rung" on a decoder run's checkpoint links it directly rather than starting another decoder run. The app then offers to clean up the other checkpoints and caches, and either way marks the run finished: it leaves the Review page's Finish scored list and is never deleted by a later cleanup. The ladder selection survives leaving the tab, and a refinement that starts on its own is shown as soon as it begins.

The Review phase lists every refinement ladder that still has unscored previews, across datasets, so a batch that ran overnight is one list in the morning.

If you have found the winner without scoring every rung, press **Reviewing complete** next to the ladder picker on the Refine tab: the Review phase then counts the ladder as scored, and Finish scored still uses the best-scored rung. Press it again to undo.

Once ladders are scored, **Finish scored (N)** on the Review phase finishes them on the server, one after another: for each ladder it takes the best rung by the scoreboard's overall score (ties go to the earlier step), runs Further training for NAR from it at the Refine tab's defaults (recon target 0.25, 250-step budget, decoder rate 0.2, keep-delta 0.003, plateau stop on unless you untick **Stop NAR at the plateau**), links the last checkpoint to the album preset, and cleans up with every option ticked (other runs, other checkpoints, other previews, the optimizer resume file, and the dataset's vocal stems and other backends' caches). The list shows the rung each ladder will use; untick any you want to finish by hand. It joins the end of a running batch if there is one, and shows in the batch panel like any other batch.

Checkpoint previews during the main run are optional. They and the Refine rung previews use the first track in the dataset that has lyrics and is not marked instrumental (so an instrumental intro track is skipped), with seed 424242 for take 1 (take 2 of a main-run ladder rung uses a random seed). A preview that reaches its length cap stopped at the preview limit, not at a natural ending.

## Using the result

Adapters are saved in your adapters folder under `yue2-joint-adapters/<trigger>_<date>_<time>` and stay there after cleanup. Runs parked in the older `refined/` subfolder are still found and count as finished. In "Audition a joint checkpoint", pick a checkpoint and press "Use for generation" to apply its AR and NAR adapters together for the next generation; YuE2 must be the active backend. The two strengths are independent and default to 1 and 1. "Use in Lyric Studio album preset" links the checkpoint to the album preset instead.

The joint adapters folder decides whether a dataset counts as trained in "Train multiple". Copying a complete run folder from another computer makes its dataset disappear when **Show trained** is off; deleting that folder makes the dataset available again. New runs include `run.json` inside the folder, so their run details travel with them.

Checkpoint loss labels for native AR, native NAR and joint runs show the mean of up to 20 logged steps ending at that checkpoint. A single step spike therefore has less weight in the checkpoint list. When an AR eval landed on the checkpoint's step, the adapter card in the top bar also shows its held-out val (minted_val), which should stay flat. Both are training diagnostics, not listening scores.

The Caption source control, in Lyric Studio and in Create, picks the style caption from the training dataset: "Automatic from dataset" uses the track nearest in tempo, "From dataset track" a specific one, and "Custom" the caption you typed. It is keyed by the dataset, not the adapter, so it survives a moved run folder and a cleared cache: Lyric Studio resolves the dataset from the album, Send to Custom-Gen carries it over, and Create has a Dataset dropdown for renders with no album behind them. Automatic is the default only while an adapter is in force; a base-model render keeps the caption you wrote. Every training caption is an in-distribution prompt for the adapter, so picking one steers the render towards that track's character. The trigger opener is not added twice. See [Backends](../backends.md) for YuE2 generation settings and [Adapters](../adapters.md) for adapter handling in general.

## Known limits

- Joint Training is new. Native and reference calculations differ by documented rounding, and audio quality from it is still being qualified by ear.
- A planner pushed too far decays late in the song: looping outros, lost diction. The point where that starts varies by album, which is why refinement is a listening ladder.
- The decoder carries likeness and the planner carries structure. A run that stops the planner early can still sound like the artist and fall apart late in the song, and the reverse.
- On Windows, the Stop button ends the trainer process at once. Only checkpoints already saved survive it.

## Related

- [Training Studio](../studios/training-studio.md)
- [Adapters](../adapters.md)
- [Backends](../backends.md)
- [Getting higher quality output](../quality.md)
- [Training internals](../../dev/training-internals.md)
