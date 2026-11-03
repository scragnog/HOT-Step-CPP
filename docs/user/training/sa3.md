# Stable Audio 3 training

Train a Stable Audio 3 Medium DoRA-rows adapter from instrumental audio in
[Training Studio](../studios/training-studio.md). The base DiT, SAME audio encoder
and T5Gemma text encoder stay frozen. Only the adapter parameters train.
Small, SFX and other adapter families are outside this workflow.

## Where to find it

Choose Stable Audio 3 in the engine picker at the top of the sidebar, then open
Training Studio. Training follows that global backend choice. While the open
dataset has a training job queued or running, the studio keeps that job's target.

SA3 has two tabs, **Dataset** and **Train**. It has no Preprocess tab: the Train
page prepares the audio and the trainer encodes it. Each dataset card shows four
chips that report what is on disk, not what a finished run implies:

| Chip | On when |
|---|---|
| Audio | Vocal-stripped, resampled instrumentals exist for the dataset (with the clip count) |
| Encoded | The trainer's encoding cache exists and still matches the prepared audio |
| Snapshots | At least one checkpoint snapshot was captured (with the count) |
| Adapter | A validated adapter was published (hover for its name and date) |

Training requires CUDA or CPU and the `ace-train` binary. The card lists missing
models. Install the SA3 Medium training base, inference DiT, SAME encoder,
T5Gemma encoder and tokenizer, and both SuperSep models through Model Manager.
See [Models](../models.md). Metal and Vulkan training are not supported by this
runner. No measured VRAM or step-time guarantee is available yet.

## Prepare and train

1. Create or open a dataset from your audio folder. Review its captions and
   musical metadata, and exclude tracks you do not want to train on.
2. Select the SA3 training target. You do not need ACE preprocessing or Build:
   the runner scans the current source files and prepares its own snapshot.
   The Train page follows the YuE2 layout: run and recipe, a collapsible
   Checkpoint previews block, progress and log, the checkpoint list, then the
   audition card.
3. Choose **New run**, enter an adapter name and review the recipe below.
4. Click **Prepare and train**. Preparation strips vocals with SuperSep level 5,
   then resamples the derived instrumentals to 44.1 kHz stereo and cuts them to
   their first 384 seconds. Originals and their sidecars stay intact. Derived
   instrumentals are cached using source, model and separation-setting
   fingerprints.
5. Follow the preparation count, encoding count and training steps in the card.
   The loss chart updates during training. Expand **Training log** for details.
   Generation pauses while the trainer runs, and the engine restarts afterward.
6. A completed run validates and publishes its final inference export into
   `models/sa3-adapters/`. Click Refresh if the picker has not updated yet.

Vocal stripping is HOT-Step's preparation policy for this instrumental workflow.
It is not an upstream training requirement.

## Captions and sidecars

Keep HOT-Step's existing key/value sidecars and `dataset.json`. No new source
format is needed. Preparation maps the parsed caption, genre, BPM, key and time
signature into a prompt such as `warm analogue pad, synthwave, 96 BPM, A minor,
4/4`. Lyrics and bookkeeping fields are omitted. An included track with none of
those five fields stops preparation. Duration conditioning comes from the
measured audio length, not a sidecar estimate.

Upstream-style plain caption `.txt` files exist only in the derived export.
Do not feed HOT-Step key/value sidecars directly to the upstream trainer: it
reads the entire file as prompt text.

## Recipe

Defaults come from the server. The initial recipe uses batch size 1.

| Control | Default | What it changes |
|---|---|---|
| Total steps | 1,000 | Total optimizer step target, including steps already completed on resume |
| Rank / Alpha | 16 / 16 | Adapter capacity and update scaling |
| Learning rate | 0.0001 | Optimizer update size |
| Crop seconds | 380 | Maximum crop, from 1 to 384 seconds; shorter crops use less memory |
| Training seed | 42 | Training randomness, preserved on resume |
| Checkpoint interval | 100 | Steps between saves of resumable optimizer state |

The form has no export interval of its own. With checkpoint previews on, the run
saves an inference snapshot at every preview step (see below); with them off, it
saves none before the final export.

Start with a short crop when checking whether your hardware can train the model.
The recipe is a starting point; listening comparisons decide which adapter to keep.

## Checkpoint previews

Previews work like YuE2's: while a run trains, the app renders short clips from
saved checkpoints so you can hear whether it is heading the right way. Open the
**Checkpoint previews** block on the Train card. It is on by default.

| Setting | Default | What it changes |
|---|---|---|
| Render previews at saved checkpoints | On | Off captures no snapshots and renders nothing |
| Preview every (steps) | 100, the default checkpoint interval | A preview round after every this many steps |
| Preview seconds | 30 | Length of each clip, 1 to 384 seconds; longer clips take longer at every round |
| Preview seeds | 424242 | Up to four whole-number seeds, comma separated |
| Preview prompts | None | One prompt per line, up to eight. Add at least one, or turn previews off |

Every round renders each prompt with each seed, so checkpoints compare like for
like. Use instrumental prompts that were not in your training captions, and keep
them the same for the whole run. The settings are fixed when the run starts; a
resumed run keeps the ones it began with.

The trainer and the render cannot share the GPU, so each round pauses training:
the trainer runs to the next preview step and exits, the app saves that
checkpoint, restarts the engine, renders the clips, stops the engine again and
resumes the trainer exactly where it stopped. Training in segments gives the same
result as one uninterrupted run within the trainer's numerical tolerance
(`tools/sa3-resume-proof.mjs` checks this on a short fixture). A failed preview
round is logged and training continues. A failed engine restore stops the run.

Checkpoints appear under **Checkpoint previews** below the card as they land,
one card per step with a player for each prompt and seed. The list opens on the
newest run; the menu above it switches to an earlier one. A cut-off round (cancel,
crash, server restart) shows its missing clips as failed, and a resumed run
renders what it still owes. **Retry** (or **Render more**) renders the step's clips
that have no finished render yet, using the settings the run started with. It needs
the engine up and no job running on the dataset. The last step is marked **final**, and its previews render after
the adapter is published.

Only the last step is published to the generation picker. Intermediate
checkpoints are previews only: they live in the run's `snapshots` folder, are
never loadable from the picker, and are not interchangeable with published
adapters.

## Cancel and resume

**Cancel training** stops the job. During training the runner requests a checkpoint
stop; cancellation during loading can terminate before a checkpoint exists. A run
appears in the resume picker only when saved state exists. Preparation failures
and runs without checkpoints must start again as a new run.

Select a saved run to resume. Its adapter name, rank, alpha, learning rate,
crop, seed, save intervals and checkpoint-preview settings stay fixed. You can increase **Total steps**.
Resume uses the saved prepared snapshot, so later edits to the source dataset
do not silently change the resumed run. A new run gets a separate dated folder.

Raw adapter weights and optimizer state remain under the adapters training root.
They are separate from the validated inference GGUF and cannot be loaded in the
generation picker. Publication happens after successful completion; a cancelled
run may be resumable without having a published adapter.

## Audition

Choose a published adapter in **Adapter audition**. Enter a held-out instrumental
caption, keep a fixed seed and duration, and click **Render audition**. Switch
**Use adapter** off and render again for the strength-0 base comparison. Strength 1
applies the full adapter. Each result has an audio player and is saved in the
library. Auditions use SA3 text-to-audio at 8 steps with MD Pingpong Blend and no
post-processing. Each audition explicitly selects SA3 and checks that routing,
independently of the active generation backend. The top-bar selection stays unchanged.

Auditions and the picker use this PC's catalogue. When training on a worker,
copy its published GGUF into this PC's `models/sa3-adapters/`, select **This PC**
and Refresh before auditioning. Automatic worker mirroring is not provided here.

For generation, the same catalogue is available in StableStep's adapter controls.
The inference export stores row scales against the inference base; community raw
training checkpoints are not interchangeable with this export.

## Related

- [Adapters](../adapters.md)
- [Backends](../backends.md)
- [Training internals](../../dev/training-internals.md)
