# ACE-Step training

This guide covers training your own ACE-Step 1.5 adapters in the [Training Studio](../studios/training-studio.md): what each adapter type changes, how to prepare a dataset, which settings to start from and how to tell whether the result is any good. It assumes ACE-Step is the active backend, because the Training Studio trains for whichever backend is selected in the top bar. For how the trainers work inside, see [training internals](../../dev/training-internals.md).

## What you can train

ACE-Step generates in two stages, and you can train an adapter for each.

| Adapter | Where it appears in the Train phase | What it changes |
|---|---|---|
| Planner LoRA (LM) | "Planner LoRA (LM)" | The 5 Hz planner that writes the song's plan from your caption and lyrics: structure, section layout, how the caption is read. |
| Sound LoRA (DiT) | "Sound LoRA (DiT)" | The audio model itself: tone, production and instrumentation. Most of a record's timbre lives here. |

They are independent. You can train one, the other or both from the same dataset, and load them separately at generation time. If you want "sounds like this record", start with the sound adapter. The planner adapter moves arrangement and structure, and on its own it will not make a generic render sound like your dataset.

## Preparing a dataset

1. In the Dataset phase, create a new dataset from a folder of audio. The scanner accepts `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, `.m4a` and `.aac`, and can include sub-folders.
2. Give it a trigger word. It is added to every caption so you can call the style up later. Pick something that is not an ordinary word.
3. Set the lyric language yourself. It overrides the automatic guess on every file.
4. Run Label. One pass per track detects BPM and key locally (Essentia), fetches lyrics from Genius if you have added a Genius token in Settings, and writes a caption and genre with an AI captioner.
5. Check the labels in Review, then Build. Build writes `dataset.json` next to your audio.

How many tracks: the app warns below 10 files, and 10 or more gives much better results. A single album is the usual unit.

Labels live in a `<stem>.txt` sidecar beside each audio file (caption, genre, lyrics, BPM, key and so on). Side-Step reads the same format, so a folder labelled there imports with its labels kept.

Which captioner to use: MOSS (local, needs its model installed) and Gemini both listen to the track, so their captions describe what is actually there. Other LLM providers write from the title, artist and lyrics only. A MOSS or Gemini caption pass also writes the MiniMax-Music3 and YuE2 caption files beside the audio, so one labelling pass serves all three backends.

Two Build options matter for ACE-Step. The trigger position puts the trigger before the caption, after it, or instead of it; "Instead of the caption" trains an adapter that only ever saw the bare trigger, so the descriptive caption and the genre-only share are discarded. "Genre-only captions" sets the share of samples that train on the genre tags alone instead of the full caption.

Changing the trigger word or its position after preprocessing makes Preprocess re-encode every song, because the trigger is part of the encoded caption.

## Preprocessing

Preprocess encodes the dataset into training tensors against one DiT base. Pick a BF16 base: the app warns on quantized bases, and results from them may be degraded. The sound adapter can only be trained against the base you preprocessed with, so choose the base you intend to generate with.

Defaults: max duration 600 s per track, peak normalisation to -1 dB, max caption tokens 512, max lyric tokens 2048. The engine stops for the duration of the job and restarts afterwards. Measured on an RTX 5090, preprocessing uses about 6 GB of VRAM and takes about 3 seconds per song.

## Planner LoRA (LM)

The planner trainer first extracts 5 Hz codes for every song (the first run encodes everything, later runs only new or changed songs), then trains.

Defaults the form ships:

| Setting | Default | Notes |
|---|---|---|
| Base size | 4B | 4B trains in low-VRAM mode, around 12 GB. 0.6B and 1.7B are faster per step. |
| Adapter type | LoRA, rank 16, alpha 32 | LoKr is available and marked experimental. |
| Optimizer | AdamW, learning rate 1e-4 | Gradient accumulation 2. |
| Target loss | 4.0 | Training stops when an epoch's average loss reaches it. |
| Max epochs | 150 | Only a backstop. |
| Artist token | On, 32 vectors, token learning rate 0.005 | Named after the adapter. |
| Prefix columns | 8 | Trained with the LoRA and shipped in the same file. |
| Caption dropout | 0.3 | Trains some steps on the trigger word alone. Needs a trigger word; the server drops it with a warning if the dataset has none. |
| Prior preservation | Off | |
| Flash attention | On | |
| Continue from this adapter's newest run | On | A name with no previous run trains from scratch. |
| Calibrate after training | Off | |

Why the target is 4.0 and not lower: the soft prompt (artist token and prefix) learns much faster than the LoRA. With it on, a lower target was reached in a handful of epochs and the adapter started replaying training codes verbatim. The run that stopped near 4.0 had no replay.

When to move off the defaults:

- For a LoRA-only run (untick the artist token and set prefix columns to 0), a lower target is reasonable. Everything a planner adapter measurably did in testing was in place by a loss of about 2.0, and pushing much further mostly bought looping plans.
- For LoRA-only runs, prior preservation every 3rd step together with caption dropout 0.3 measured best in the one experiment behind it. Prior preservation needs at least one other artist preprocessed at the 600 s cap to act as the reference.
- Pick a smaller base size if 4B does not fit your card. The form shows its VRAM estimate and the longest sequence it can fit; songs longer than that are skipped and reported.

How long it takes: in the measurements behind the trainer, a 4B run that stopped at target 4.0 took about 12 minutes on an RTX 5090. That was an earlier recipe without the soft prompt.
<!-- TODO(verify): wall-clock time for the current 4B default recipe (artist token + prefix on). -->

## Sound LoRA (DiT)

Defaults the form ships:

| Setting | Default | Notes |
|---|---|---|
| Adapter type | LoKR, dim 512, alpha 512, factor 6 | The recommended default. LoRA (rank 128, alpha 256) is one click away. |
| Quality | Balanced | Fast, Balanced and Thorough set max epochs to 150, 500 and 900. |
| Target loss | 0.3 (0.2 on Thorough) | Measured on the 5-epoch average. Max epochs is only a backstop. |
| Optimizer | Prodigy (automatic step size) | AdamW and Muon are selectable. |
| Learning rate | 0.002 for LoKR, 5e-4 for LoRA | |
| Gradient accumulation | 4 | |
| Also train the MLP projections | On | Most of a record's timbre lives there. It costs VRAM, so the audio window gets shorter to pay for it. |
| Flash attention (full-song crops) | On | |
| Audio window | Auto | Fitted to free VRAM. |
| Mirror precision | BF16 store, F32 maths | |
| Batch size | 1 | |
| Continue from this adapter's newest run | On | |

When to move off the defaults:

- Leave the audio window on Auto. The auto-fit shortens the window before it gives up depth. Pinning a long window on a small card forces the trainer to train fewer layers instead, and the panel marks such a run "part-depth". Whether a part-depth adapter sounds good is not established.
- Keep the mirror on "BF16 store, F32 maths". The faster "BF16" option rounds the arithmetic as well as the storage, and adapters trained with it were judged coarse by ear.
- Batch sizes above 1 were measured about 2.5 times slower at full depth on a 32 GB card. They only help shallow, partial-depth runs.
- DoRA, HiRA, LoHa and HRA are LoRA variants. In a blind listening test none of them beat plain LoRA, so treat them as experiments.
- If you change the learning rate, change gradient accumulation with it. For LoKR the two were tuned together, and one without the other is untested.

VRAM: the trainer refuses to start only when its cheapest configuration does not fit your free VRAM, and the refusal names what it tried. On the tested XL base, a card with 12 GB free trained all 32 layers at a short window with a rank-16 LoRA, and the frozen base alone sets a floor of about 8.3 GB. The larger LoKR default with MLP projections needs more, and bigger cards get longer windows. On a 32 GB card, flash attention let the auto-fit choose full-song windows.
<!-- TODO(verify): the Sound LoRA panel still says "Needs a 16 GB GPU (24 GB for full depth)"; the trainer no longer refuses on that rule. -->

How long it takes: measured on an RTX 5090 with a 14-song dataset, LoKR dim 512 with MLP projections at a window of about 28 seconds ran at about 2.7 seconds per epoch. Longer windows cost more per epoch.

## Picking a checkpoint

A run stops at its target loss or its epoch cap and exports one adapter. The done panel shows the final loss, the number of epochs and whether the target was hit. A lower loss is not always better, especially for the planner (see known limits below).

Ways to get more than one candidate:

- Set "Milestone loss step" above 0 to save a snapshot each time the loss falls by that amount. "Milestones to keep" caps how many are kept (6 by default). Each milestone is a full copy of the adapter, so they take disk space.
- Tick "Calibrate after training". For the planner it generates plans with the base LM, the previous adapter and the new one on your dataset's own captions. For the sound adapter it renders your songs from their real codes, so only the sound differs. Either way it measures which lands closest to your real audio, at more than one adapter scale, and ships the winner with its best scale baked in. "Update album preset to the winner" repoints your Lyric Studio album preset.
- For the planner, use Codes audition. It decodes the planner's codes straight through the detokenizer and VAE, with no DiT and no sound adapter, and can A/B the adapter against the base LM at a fixed seed. Judge structure and style there, not fidelity.

The final judge is a real generation with both adapters loaded.

## Using the result

New adapters are available straight away, with no restart. Planner adapters appear in the Planner Adapter dropdown and sound adapters in the Adapters dropdown. The trigger word is stored inside the adapter file and the server adds it to your caption at generation time; if your caption already contains it, it is not added a second time. Strength, stacking and per-section control are covered in [Adapters](../adapters.md).

## Known limits

- Adapters capture timbre and production far more readily than song structure. A sound adapter makes a render sound like the record; it does not make it arranged like the record.
- An overtrained planner shows it in its plans: looping sections, droning intros and songs that end early. Lower training loss makes this more likely, not less.
- Flash attention training runs on CUDA builds (and CPU) only. On a Vulkan or Metal build, untick Flash attention in both forms or the trainer refuses to start.
- The engine is stopped during preprocessing and training, so generation is unavailable until the job ends.
- Quantized bases, ConvRot ones included, are not training targets. Preprocess against a BF16 base.

## Related

- [Training Studio](../studios/training-studio.md)
- [Adapters](../adapters.md)
- [Backends](../backends.md)
- [Getting higher quality output](../quality.md)
- [Training internals](../../dev/training-internals.md)
