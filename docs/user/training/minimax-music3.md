# MiniMax-Music3 training

This guide covers training a MiniMax-Music3 (MM3) planner adapter in the [Training Studio](../studios/training-studio.md): what it changes, what the dataset needs, which recipe to pick and how to judge the checkpoints. MM3 must be the active backend, because the Training Studio trains for whichever backend is selected in the top bar. For how the trainer works inside, see [training internals](../../dev/training-internals.md).

## What you can train

The Training Studio trains one thing for MM3: a LoRA on the planner LM ("LM LoRA training" on the Train page). The planner writes the whole song's plan, so the adapter moves structure, arrangement, vocal delivery and vocal identity. Roughly, its attention layers carry the plan and genre and its MLP layers carry the voice. The audio model that turns the plan into sound is not trained here.

The recipe is tuned for an album clone: a render that could pass for another track off the record. Memorising the album and some style bleed into unrelated prompts are accepted costs. If you want an adapter that leaves the base model untouched except when triggered, this recipe is not aimed at that.

## What the dataset needs

Create and label the dataset in the Dataset phase as for any backend (see [Training Studio](../studios/training-studio.md)). MM3 adds three requirements.

Per-track MM3 captions. The trainer reads a Structured Caption from a `<stem>.mm3.txt` file beside each audio file, and refuses a run with no captions at all. These files are written by a caption pass with MOSS (local) or Gemini as the provider, from either Label or Enhance labels. Both listen to the track; BPM and key are then taken from the local analysis, because the captioners get them wrong. Do not rename the ACE-Step `.txt` sidecars to `.mm3.txt`: an ACE caption trains the wrong genre, and the rename puts the lyrics in the prompt twice. Lyrics come from the normal sidecar.
<!-- TODO(verify): that mm3-condition takes lyrics from the <stem>.txt sidecar (inferred from the skill's "lyrics twice" note, not traced in code). -->

RVQ codes. The trainer learns from codes, not audio. On the Codes page (the Preprocess slot, relabelled for MM3), press Export codes once per dataset. It took about a minute for 12 whole tracks in testing. The engine pauses while it runs.

Track length under six minutes. The recipe trains every track as one whole sequence, up to 9000 frames (360 s). Longer tracks are left out and named in the log. The server refuses a run that this would cut by more than half.

How many tracks: the app warns below 10. Between 10 and 20 tracks, size showed no reliable effect on the result. Hold-out evaluation is skipped below 6 songs.

### Cover-laundered codes for dense mixes

If the vocal sits buried in a dense mix, tick "Cover-launder the codes (dense mixes)" on the Codes card. It renders each track back through the model before encoding, which puts the vocal where the codes can carry it. The result goes to a separate cache, and the train form then offers "Train on cover-laundered codes". It runs at about 1.4 times real time per track, once per dataset, and needs the rec7 encoder model installed. On one dense-mix album the laundered adapter won an A/B on every axis. It is unmeasured on sparse or acoustic material, so do not use it by default.

## Recommended settings

Pick a recipe on the Recipe row. All three train each track as a whole song and differ only in depth:

| Recipe | Steps | Time per album on a 32 GB card | Status |
|---|---|---|---|
| Fast | 300 | 30 to 60 minutes | Heard: about half of renders ended naturally across three albums. |
| Balanced (default) | 600 | 1 to 2 hours | Fast at twice the depth. Not yet judged by ear. |
| Thorough | 900 | 1.5 to 3 hours | Whether the extra steps buy anything is not yet measured. |

The form opens with 500 steps, which the Recipe row shows as Custom. Click Balanced to get the 600-step recipe.
<!-- TODO(verify): MM3_LM_DEFAULTS.steps is 500 while the Balanced preset is 600, so the card opens on Custom. -->

The rest of the recipe is shared:

| Setting | Default | Notes |
|---|---|---|
| Adapter method | HOT-PiZZA, rank 128, alpha 128 | PiSSA with principal-subspace dropout. It won a blind test against plain LoRA, LoKr, DoRA and corrected PiSSA on one album. |
| Optimizer | AdamW, learning rate 8e-5 | |
| Base precision | q8_0 | |
| Flash attention | On | What makes whole-song training fit. |
| Acoustic loss weight | 1.0 | Keeps vocal timbre from drifting. |
| Leave out tracks longer than the window | On | |
| Train the trigger | On | Puts the trigger into the training captions. |
| Checkpoint every | 100 steps | |
| Previews | Every 100 steps, 40 s, MLP scale 1.0 | One preview per checkpoint. |
| Hold-out fraction | 0.15 | Evaluated every 250 steps. |
| Prior preservation | Off | |
| Train until | Step count | |
| Keep resume state after completion | Off | |

When to move off them:

- Leave the learning rate alone. Doubling it to shorten the run scored lower every time it was tried, and one such run planned a song with no vocals.
- Leave the acoustic loss on. Turning it off was the lowest-scored variant tested.
- Do not pick Muon. Its learning-rate scale was tuned on a different model, and on MM3 it produced silent renders at rank 64. Prodigy is selectable; it tied AdamW by ear and uses about 2.3 GB more memory.
- Stop on step count. HOT-PiZZA's training loss reads high by construction, so a target-loss stop would run to the cap anyway. Target loss is there for plain LoRA runs, where 1.0 on the training mean was the measured stopping point.
- Rank 64 is not a free saving. In an earlier sweep rank 64 turned lyrics into gibberish while keeping the likeness; rank 128 was enough to avoid it.
- LoKr, DoRA, HiRA, LoHa, HRA, PiSSA, the artist token and trainable prefix columns are selectable but have not been heard on MM3.

<!-- TODO(verify): the Optimizer hint in the MM3 Advanced drawer still says Muon is the default; the shipped default is AdamW. -->

## VRAM and time

Measured on an RTX 5090 at rank 128: whole-song training peaks at about 29 GB and runs at 6 to 12 seconds per step. Peak memory grows with the longest track in the album: about 25.4 GB at 2000 frames, 26.8 GB at 4000 and 29.1 GB at 9000, so an album of short songs costs less. As shipped this is a 32 GB recipe.

The form estimates the peak for your card and tells you whether it fits, is tight, or is over. Close to the limit, Windows spills into shared memory and steps take several times longer. On a 24 GB card, the levers are a lower rank or turning the acoustic loss off, and neither has been heard under the whole-song recipe. The app's own floor is roughly 11 GB at the smallest usable settings.

Smaller base precisions trade fidelity for memory. Step time barely moves between them. The form marks bases that are too lossy to train against.

Flash attention training needs a CUDA build. On Vulkan or Metal the checkbox is greyed out and the trainer uses exact attention, whose longest practical window on a 32 GB card was about 4300 frames (172 s).
<!-- TODO(verify): whether whole-song MM3 training is practical at all on a non-CUDA build. -->

The engine is paused while training runs, so you cannot generate until it finishes.

## Picking a checkpoint

Checkpoints land straight in the MM3 adapter folder, so each one shows up in the LM Adapter dropdown as soon as it is saved. Each checkpoint gets a 40-second preview in the preview strip.

Do not pick by held-out loss. It bottoms out early and then rises, while the checkpoint that sounds right came 1 to 8 times later in the sweeps behind this recipe. The held-out loss measures how well the adapter generalises to unseen songs, and the goal here is a clone of the seen ones. Treat it as an alarm for a run going badly wrong, nothing more.

Listen for two things that move in opposite directions as training goes on. Likeness (does it sound like the band) rises with steps. Coherence (does the song hold together, are the vocals articulate) falls; the symptom is jumbled, slurred vocals. The MLP slider at generation time recovers coherence at a late checkpoint without giving back much likeness, so try MLP 0.5 before retraining.

A preview is one sample at one seed. A single garbled preview between two good ones may be sampling luck, not a broken checkpoint. Render that checkpoint at two other seeds before believing it.

Albums differ. One album had a rhythm defect at an early checkpoint that went away with more training, and a dense-lyric album never got coherent at any rung. Do not apply a blanket stop-early rule.

To train further, open Previous runs, pick the run and press Continue with the extra steps. That needs the run's resume state, which is kept when a run stops early or when "Keep resume state after completion" was on (about 4 GB). Continuing restarts the tail of the learning-rate schedule, so a continued run is not the same as having asked for more steps at the start.

## Using the result

With MM3 active, pick the adapter in the LM Adapter dropdown in the top bar. The defaults are strength, attention and MLP all at 1.0, which is right for adapters trained with the acoustic loss. Leave the depth thirds at 1.0; lowering the late third made songs fade or never end. Recommended resets the sliders to the adapter's own suggested scales.
<!-- TODO(verify): the MLP info text in the LM Adapter dropdown still calls attention 1.0 / MLP 0.5 the tested dial; the default is now 1.0. -->

The trigger is added to the caption for you ("Add trigger to caption"), in the same place the trainer put it. If your caption already starts with it, nothing is doubled.

Your caption should look like a training caption. When you send a song from Lyric Studio to Create for an album whose training tracks have MM3 captions, Create shows a Caption source control above the caption box. "Automatic from dataset" copies the caption of the dataset track nearest in tempo; "From dataset track" lets you pick one; Dismiss goes back to your own caption. A training caption used word for word with new lyrics is a strong style prompt. See [Backends](../backends.md) for the MM3 caption format.

Write a full song's worth of lyrics even for a short render: MM3 plans the whole track from the whole lyric, and the duration only truncates. Irregular line lengths that match the artist's own writing work far better than tidy four-line verses.

## Known limits

- Endings are not solved. Even with per-track captions, each plan ends naturally only about 20 to 50 percent of the time, and the rate depends on the caption and lyrics pair. Whole-song training raised it a lot compared with the older 30-second-crop recipe, and no training-side change has raised it further without losing likeness.
- Pitch and tempo can drift over the course of an adapter render. The cause is still open.
- The acoustic loss reduced the "chipmunk" or "goblin" voice shift that early adapters had, but in one test on a later album the shift was still audible. Do not assume timbre is always safe.
- Adapters learn timbre and style faster than long-range song structure.
- Dense rap lyrics and stylistically scattered albums train worse than focused ones.
- Tracks over six minutes are excluded unless you untick the option, which trains them in window-sized crops instead.
- Adapters trained from the old shared-caption option, or before the acoustic loss existed, should be retrained rather than re-dialled.

## Related

- [Training Studio](../studios/training-studio.md)
- [Adapters](../adapters.md)
- [Backends](../backends.md)
- [Getting higher quality output](../quality.md)
- [Training internals](../../dev/training-internals.md)
