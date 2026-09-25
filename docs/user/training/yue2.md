# YuE2 training

This guide covers training YuE2 adapters in the [Training Studio](../studios/training-studio.md) with Joint Training, the default method: what gets trained, what the dataset needs, which preset to start from, and how to refine and pick the finished adapter by ear. YuE2 must be the active backend, because the Training Studio trains for whichever backend is selected in the top bar. For how the trainer works inside, see [training internals](../../dev/training-internals.md).

## What you can train

YuE2 has two halves, and Joint Training trains an adapter for each in every step.

| Half | Also called | What its adapter changes |
|---|---|---|
| AR | The planner | Structure, diction and late-song behaviour: what gets sung where, and whether the song holds together to the end. |
| NAR | The decoder | Timbre. Most of the likeness to the artist lives here. |

The two adapters are saved together in each checkpoint and applied together at generation time.

The older Legacy seven-stage trainer, which trains the NAR and AR adapters in separate stages, is still available under Training method. This guide covers Joint Training.

## Requirements

- A CUDA build of the app and an NVIDIA GPU with BF16 support (Ampere or newer). Joint Training does not run on other hardware.
- The YuE2 Joint Training Pack, installed from the Model Manager. The card warns if model paths are missing.
- YuE2 selected as the active backend. Under YuE2 the Training Studio shows Dataset, Train, Refine and Review; there is no separate Preprocess phase, because the caches are built on the Train page.

## What the dataset needs

Create and label the dataset in the Dataset phase as for any backend (see [Training Studio](../studios/training-studio.md)). Lyrics matter more here than for the other backends: the AR half trains on each track's caption and lyrics, and lyric timing supervision aligns against those lyrics. Tracks without lyrics are skipped by those stages.

Captions. The planner is prompted with one sentence in a fixed order (language, genre, vocal, instruments, mood, production, BPM). A labelling pass writes that sentence to `<stem>.yue2.txt` beside each track, using a chat provider to rewrite the existing label; MOSS has no mode for it. For tracks labelled before this existed, the "YuE2 caption" button under Enhance labels backfills them.

The latent cache's "Clip captions" setting decides what the trainer reads. When the dataset has sidecars it defaults to "Sidecar lyrics, with the one-sentence YuE2 caption (.yue2.txt) as the style", which falls back to the ACE-Step caption for tracks without a `.yue2.txt`. The other modes (trigger word only, one typed caption for every clip, or the raw sidecar text) throw the lyrics away or feed field names in as style. If `.yue2.txt` files are written or changed after the cache was cut, the card warns. For Joint Training you do not need to re-cut: preparation notices the stale captions and re-reads the sidecars without re-encoding any audio.

Loudness. The cache brings every track to -14 LUFS before encoding. A cache cut before that existed trains at each album's own mastering level, and loud albums then render clipped; the card warns and a re-run fixes it.

The latent cache's "Rewrite the manifest" option is on by default: a re-run rewrites the manifest with the caption mode chosen above and re-cuts at the clip length, and the cached latents are reused, so it costs no GPU time. "Perform all stages", "Train multiple" and Start training cut the cache with the YuE2 caption mode, and re-cut a cache that was cut with the ACE captions while `.yue2.txt` files sit beside the audio; the card warns about such a cache until it is re-run.

The cache stages are latent cache, codes, lead sheets, vocal stems and lyric cursor spans. The last two are only needed for lyric timing supervision. "Perform all stages" runs them in order. Lead sheets can be previewed as each source finishes, not only when the stage ends. You do not have to prepare anything by hand: Start training prepares the dataset automatically and reuses prepared data that has not changed.

How many tracks: the app warns below 10 files.
<!-- TODO(verify): no track-count guidance specific to YuE2 joint training was found in code or skills. -->

## Recommended settings

Pick a preset. Each stops the planner when its KL reading (how far the planner has moved from the base model) reaches a target; the step count is a cap.

| Preset | AR KL target | Step cap |
|---|---|---|
| Fast | 0.8 | 300 |
| Balanced (default) | 1.0 | 500 |
| Thorough | 1.6 | 700 |

KL means the same thing for every artist, which is why the run stops on it rather than on loss. The presets are deliberately conservative; the Refine phase (below) is where you push the planner further and listen for where it breaks.

Defaults the card ships:

| Setting | Default | Notes |
|---|---|---|
| Adapter type | LoKr, dim 64, factor 4, alpha 256 | About 106 MB for the AR and NAR pair, against 279 MB for a rank-64 LoRA. LoKr is marked experimental in the UI but is the tested default. |
| Optimizer | Prodigy, cautious updates on | Prodigy sets its own step size, so the learning rate field is ignored. |
| Planner / decoder learning-rate scale | 0.6 / 1.0 | |
| Learning-rate schedule | Warmup, flat, triggered decay (wsd) | Flat, then a short decay that ends on the stop, so the kept weights are annealed. With a KL target the decay starts when the KL trend says the target is about a decay away, so the run lands on the target rather than 40 steps past it. If the KL still passes the target by the overshoot margin (0.1) during the decay, the stop acts at once. |
| KL reading | 30-step trend line | The 20-step mean lags by about 10 steps. |
| Save every | 25 steps | |
| Lyric timing supervision | On, timing loss weight 0.08 | Uses vocal stems and forced alignment. |
| Caption dropout | 0.5 | So a new caption lands on the artist rather than beside one memorised track. |
| Spike guard | Skip updates above 5x the recent median; stop after 3 in 20 steps | Ends the run on the last good weights if the gradients blow up. |
| Decoder stop | Under 0.5% gain over 3 checkpoints | |
| Automatically proceed to refinement | On | |
| Checkpoint previews | Off | 90 s when on. Previews belong to the Refine phase, which renders one per rung. |
| Stop the engine during training | On | |

When to move off them:

- If you switch to LoRA, the card sets the KL target to 1.4 and the planner scale to 0.3. For LoRA, likeness starts near a KL of 1.25 and planner damage (looping outros) near 1.9. LoKr moves further per unit of KL, which is why it ships at 1.0.
- Turn lyric timing supervision off for a dataset without usable lyrics. It skips the stem and alignment stages.
- Turn "Stop the engine during training" off if you want to keep generating or scoring while it trains. Both then share the GPU and run slower, and if VRAM runs out Windows spills into system memory.
- Leave the schedule on wsd. Plain cosine leaves an early-stopped run mid-decay at a high rate, and cosine restarts were expected to shake the planner.
- Training presets saves your current settings under a name in this browser. It never saves dataset, checkpoint or output paths.

"Train multiple" runs the whole chain (caches, preparation, joint training) for several datasets in sequence with shared settings and a separate output folder each. The batch runs on the server.

How long it takes and how much VRAM it needs at these settings have not been recorded in the code or skills yet.
<!-- TODO(verify): joint training wall-clock time and peak VRAM at the Balanced preset. -->

## Refining and picking the adapter

A run saves a checkpoint every 25 steps and one at the KL stop. With "Automatically proceed to refinement" on, it then starts a refinement and switches to the Refine phase.

Refinement continues the finished run with the planner live, saves a checkpoint each time the KL crosses the next rung, and renders previews per rung so you can hear where it starts to fall apart late in the song. Its defaults:

| Setting | Default | Notes |
|---|---|---|
| KL ceiling | 2.0 | A search range, not a target. Late-song decay has shown from about 1.7 on some albums and not at all by 1.8 on others. |
| Rung size | 0.1 KL | Smaller rungs mean more previews to listen to. |
| Learning rate | 0.1 times the source run's | |
| Tracks per rung | 2, 300 s each | Renders are not deterministic, so two takes is the minimum to trust a rung. |
| In parallel with training | On | Needs VRAM for both, about 22 GB measured. Untick it on a smaller card. |
| Draft quality | On | Previews only: 12 decoder steps instead of 32, about a third of the decoder time. Timbre is a little softer; structure, diction and late-song behaviour are unchanged. |
| Keep a checkpoint per recon drop | 0.003 | Further decoder training only: a checkpoint is kept when the reconstruction meter has dropped by at least this since the last kept one; the rest are deleted once a newer one lands, so the ladder shows progress rather than every 10 steps. The newest checkpoint is always kept. |

The rung that crosses the KL ceiling is rendered too, like every rung before it. Each take shows how many planner and composer re-plans the render needed, and each rung sums them; rising counts are an early sign of over-training, softened by the app's own auto re-plan at generation time.

Each take's lead sheet is also read for legibility, since a plan can pass the health check and still be a bad song. A take is flagged when a voice repeats the same one-to-four-bar cell for 32 bars or more without variation, the vocal melody uses fewer than four pitches, the whole sheet has fewer than three chords, or the plan has fewer sections than the lyric has section tags. The rung shows "plan flags n/takes" in red, with the reasons on hover, and each flagged take names its worst finding. The thresholds were set against a handful of base and adapter plans, not against ear scores, so a flag is a prompt to listen closely, not a verdict, and it does not move the scoreboard. Cleanup now keeps every take's record and lead sheet (only the audio goes), so the flags can be checked against your scores later.

Pick the last good rung by ear. No automatic measure has been able to tell a good rung from a decaying one, so the ladder is a listening test. Score each rung's likeness (higher is better) and corruption (lower is better); a floating scoreboard ranks the rungs by an overall figure, (likeness + (6 − corruption)) / 2 minus 0.25 per re-plan per take (capped at 1), and marks the best one. Ties go to the earlier rung. The scoreboard is a suggestion; the pick is yours.

When a ladder finishes, the panel says so and waits for you to press "Use this rung" on your choice. With "Further training for NAR" on, the decoder then trains on from that rung with the planner frozen, until its reconstruction target (0.25), the 500-step budget, or the point where the reconstruction meter stops improving. At about 1.5 s a step, 500 steps is roughly 13 minutes. The final checkpoint becomes the adapter; pressing "Use this rung" on a decoder run's checkpoint links it directly rather than starting another decoder run. The app then offers to clean up the other checkpoints and caches, and either way moves the run into `yue2-joint-adapters/refined/`, repointing the album preset and the engine's adapter pick. Runs under `refined/` are never deleted by a later cleanup. The ladder selection survives leaving the tab, and a refinement that starts on its own is shown as soon as it begins.

The Review phase lists every refinement ladder that still has unscored previews, across datasets, so a batch that ran overnight is one list in the morning.

Checkpoint previews during the main run are optional. They use the first track in the dataset with seed 424242. A preview that reaches its length cap stopped at the preview limit, not at a natural ending.

## Using the result

Adapters are saved in your adapters folder under `yue2-joint-adapters/<trigger>_<date>_<time>`, and a refined run moves to `yue2-joint-adapters/refined/` after its cleanup. A run folder moved into `refined/` by hand is found again on the next read. In "Audition a joint checkpoint", pick a checkpoint and press "Use for generation" to apply its AR and NAR adapters together for the next generation; YuE2 must be the active backend. The two strengths are independent and default to 1 and 1. "Use in Lyric Studio album preset" links the checkpoint to the album preset instead.

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
