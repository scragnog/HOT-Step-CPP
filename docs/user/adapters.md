# Adapters

HeartMuLa's integration backend scans GGUF LM LoRAs in
`<adapters>/heartmula-lm-adapters/`, including up to two levels of subfolders.
It accepts one adapter per generation, with a strength from 0 to 2. The default
is 1; zero requests base logits. Safetensors exports need conversion before use.
The engine validates the adapter against the selected LM variant and component
fingerprints. See [HeartMuLa](backends.md#heartmula) for the integration status.

Adapters are small weight files, LoRA or LoKr, that pull a base model toward a style, a
voice or a production habit without replacing the model. You train them in the
[Training Studio](studios/training-studio.md) or get them elsewhere, then load one or
several per generation from the Adapters section of the generation bar.

What an adapter changes depends on the backend. On ACE-Step 1.5 the main adapter acts on
the DiT, the part that renders audio, and an optional planner adapter acts on the LM that
plans the song. On MiniMax-Music3 and YuE2 adapters act on the language model. Stable
Audio 3 adapters act on its DiT and have no picker in the generation bar: the Training
Studio trains them and renders auditions with them (see [File formats](#file-formats)).

## Where to find it

The **Adapters** section of the generation bar. Its contents follow the active backend
(see [Backends](backends.md)):

| Backend | What the Adapters section shows |
|---|---|
| ACE-Step 1.5 | A DiT adapter picker or stack, loading mode and VRAM options, basin re-base, group scales, and a planner (LM) adapter |
| MiniMax-Music3 | One LM adapter picker with strength dials |
| YuE2 | Two LM adapter slots, NAR (renderer) and AR (composer), each with its own dials |

A backend without adapter support shows a "not yet" panel instead.

All three read from one adapters directory. By default that is the `adapters` folder in
the data folder (see [Release data paths](#release-data-paths)). Change it under Settings >
Environment > **Adapters directory**; the new folder takes effect after a restart.

## The adapters folder

After deleting adapters, use **Settings > General > Adapters > Update adapter
references** to clear their paths from Lyric Studio album presets. The result lists
each changed preset and removed adapter. Live adapters and reference tracks stay in
place. This cleanup runs only when clicked and never substitutes another adapter.

The Training Studio writes into a fixed layout. Hand-installed files can sit at the top.

```
adapters/
  my-style.safetensors              hand-installed ACE-Step DiT adapter
  dit-<base>/<name>/<run>/          ACE-Step DiT adapters, one folder per base model
  lm-06b/ lm-17b/ lm-4b/<name>/<run>/   ACE-Step planner adapters, one folder per planner size
  mm3-lm-adapters/<run>/ckpt-<N>/   MiniMax-Music3 LM adapters, one folder per saved checkpoint
  yue2-nar-adapters/                YuE2 NAR (renderer) runs
  yue2-ar-adapters/                 YuE2 AR (composer) runs
  yue2-joint-adapters/              YuE2 joint runs, which save a NAR and an AR file per checkpoint
  sa3-medium/<dataset>/<name>/<run>/  Stable Audio 3 training runs (raw checkpoints, not inference files)
```

`dit-<base>` is a short name for the DiT the adapter was trained on, such as
`dit-xl-thirds` or `dit-turbo`. `<run>` is a date and time stamp like
`2026-08-06_11-05-06`. Training the same dataset again adds a new run folder and never
overwrites an old one.

## File formats

Stable Audio 3 Medium training publishes validated DoRA-rows GGUF files into
`models/sa3-adapters/`, shared with StableStep. Its raw adapter and optimizer
checkpoints remain in the training run folder and are not inference files.
Training Studio's SA3 card has a catalogue picker and fixed-seed auditions with
strength from 0 to 2. Strength 0 renders the base with an empty adapter list;
strength 1 applies the full adapter. See [Stable Audio 3 training](training/sa3.md) for vocal
stripping, the recipe and resume. This catalogue lives under the models root,
separately from the ACE-Step, MiniMax-Music3 and YuE2 adapter directories above. On
those three backends the same files are StableStep's adapters in the Post-Processing
section. With Stable Audio 3 active, the generation bar has no picker for them.

ACE-Step DiT adapters load from three shapes:

- A single `.safetensors` file. ComfyUI-style LoRA files, which store alpha per tensor,
  and flat LyCORIS LoKr files both load this way.
- A PEFT folder: `adapter_model.safetensors` with `adapter_config.json` beside it. Keep
  the config. Without it the engine falls back to alpha equal to rank and the strength
  comes out wrong, badly so for adapters trained with per-module ranks.
- A LoKr folder from the Training Studio, holding `lokr_weights.safetensors`. No config
  file is needed.

DoRA adapters support Merge mode and opt-in runtime magnitude rescaling. In the
ACE-Step Adapters section, turn on **Runtime DoRA magnitudes** to apply the learned
magnitudes in Runtime or Low-Rank mode, including when section controls or active
phases force runtime loading. This may materialize full-size deltas in Low-Rank mode.
The switch is saved in the browser and defaults to off, preserving the older runtime
result that skips magnitude rescaling. Merge mode and planner LM adapters keep their
existing behavior. The same saved switch is sent to MiniMax-Music3 requests; its DiT
training and adapter controls remain CLI-only.

ACE-Step checks DoRA files before applying their weights, including with the default
loading settings. A PEFT config with `use_dora: true` must have matching magnitude
tensors. If those tensors are missing, loading fails instead of treating the file as
plain LoRA. Keep the original config and weights together; replace an incomplete
adapter with a complete export.

MiniMax-Music3 adapters are `.safetensors` files, PEFT LoRA or LyCORIS LoKr, trained on
the MiniMax-Music3 LM.
Each can have a sidecar named `<file>.safetensors.json` that records the trigger word,
rank, dataset, step count, notes and recommended dial settings. The Training Studio writes
one for every checkpoint.

YuE2 adapters are `.safetensors` files whose header records which half of the model they
belong to. The app reads that field to sort them into the NAR and AR slots, and skips any
file in a scanned folder that does not have it.

## ACE-Step 1.5

The section has an **Adapter mode** switch, **Simple** or **Advanced**, under the
**Runtime DoRA magnitudes** toggle. Simple loads one adapter. Advanced loads a stack and
unlocks per-section control.

### Loading one adapter

1. Pick **Simple**.
2. Type a path in **Adapter Path**, or click the folder button and choose a
   `.safetensors` file.
3. Set **Adapter Scale** (0 to 4, default 1.0).

For a PEFT adapter you can choose `adapter_model.safetensors` itself: the engine reads the
`adapter_config.json` in the same folder.

### Stacking adapters

1. Pick **Advanced**.
2. Set **Adapter Folder** and click the search button to scan it. The scan lists
   `.safetensors` files in that folder, adapter folders directly inside it, and every run
   under any `dit-*` folder. Point it at the adapters root to see everything the Training
   Studio has made.
3. Click entries to add them to the stack. Click again, or the X on the stack row, to
   remove one.

Each stack row has its own scale slider (0 to 4) and an **Active phase** range. With two
or more adapters a **Stack Mode** switch appears:

| Stack Mode | How the sliders behave |
|---|---|
| Blend (default) | Sliders are relative weights. They are scaled so the effective strengths add up to **Combined Strength (Σ)**, default 0.75, so the total stays the same as you add adapters. Each row shows the effective scale it sends. |
| Sum | Sliders are absolute scales, added together. The total can go past 1 if you want to over-drive the stack. |

**Active phase** limits an adapter to part of the denoising run, as a percentage where 0%
is the first step. Early steps shape structure and rhythm, late steps shape timbre and
detail, and neighbouring windows crossfade. Setting any window forces Runtime mode with
each adapter's deltas held separately in VRAM, so set **Adapter VRAM** to Q8 or Q4 when
you use it.

### Loading mode

| Mode | What it does |
|---|---|
| Merge | Bakes the adapter into the model weights when the model loads. Sampling runs at full speed. **Merge VRAM** picks how the merged weights are stored: **HQ** keeps them at F32, about four times the VRAM on a Q8 base; **Low ¼** re-encodes them to the base's own quantization. FP4 bases always take the low path. Not available on ConvRot models. |
| Runtime (default) | Leaves the base weights alone and applies the adapter at every step. **Adapter VRAM** stores the deltas as **Full** (BF16), **Q8 ½** or **Q4 ¼**. Q4 fits the most stacked adapters for a small quality cost. |
| Low-Rank | Applies the raw adapter factors per step without building full-size deltas. The lowest VRAM of the three for LoRA and LoKr. With **Runtime DoRA magnitudes** on, DoRA uses full-size deltas to include magnitude rescaling. |

Changing the adapter, a scale, the mode or the VRAM setting reloads the DiT on the next
generation. Generations with unchanged settings reuse the loaded model. The first Runtime
load spends some seconds precomputing deltas, and you can cancel during that phase.

### Group scales

The **Group Scales** fold-out sets strength per weight group: **Self-Attn**, **Cross-Attn**,
**MLP** and **Conditioning** default to 1.0; **Timestep** and **Proj-In** default to 0,
which skips them. Hover each label for what the group covers. **Reset** puts them back. A
group the adapter has no weights in ignores its slider.

### Planner adapter

**Planner Adapter (LM)** loads an adapter on the LM that plans song structure. It is meant
to pair with a DiT adapter trained on the same material, which carries the timbre. The list
comes from `lm-06b`, `lm-17b` and `lm-4b`, or from the folder typed in the box above it.
Each entry shows its run date, the planner size it was trained for, and, when it has been
evaluated, an artist-match score where lower is closer (green for a "toward" verdict, red
for "away"). The refresh button rescans.

**Planner Strength** (0 to 2, default 1.0) also applies to planner adapters that come from
album presets. Above about 1.4 the planning tends to go repetitive; train longer instead of
pushing the slider.

Use a planner adapter trained for the planner size you have loaded. One with more layers
than the loaded LM is refused.

### No-adapter reference

The **No-adapter reference** switch adds a third output to each generation: a raw 20-step
render with the DiT adapter bypassed, the planner adapter kept, and no post-processing.
Switch to it on the play bar to hear what the adapter is contributing.

### Using an adapter on a different base

An adapter trained on one DiT and loaded on another of the same size can fall apart at full
strength, even when the two bases are nearly identical. **Basin Re-base** corrects for that:

1. In **Adapter trained on (home base)**, pick the model the adapter was trained on. Only
   safetensors DiT models are listed, so the home base has to be installed in that format.
2. Set **Re-base Strength (β)**, 0 to 1, default 0.75. At 1 you get home-base behaviour;
   lower keeps more of the loaded base's character.

Re-base works in all three loading modes; in Low-Rank the correction is stored as
full-size deltas. It has been checked by ear in Merge mode only. It is skipped when
per-section directives or Active phase windows are in use. It cannot move an adapter
between model sizes (XL and 2B): nothing in the app does that.

HeartMuLa does the same for its LM adapters without a home-base picker, because each
adapter records the variant it was trained on. See [HeartMuLa](backends.md#heartmula).
YuE2 does it for its AR and NAR adapters the same way, see [YuE2](backends.md#yue2).

### Per-section adapters

With two or more adapters stacked in Advanced mode, you can give each lyric section its own
mix by adding a directive right after the section header, on the same line:

```
[Verse 1]{#1=1; #2=0}
Streetlights hum along the empty road
...
[Chorus]{#1=0.3; #2=1}
...
[Bridge]
...
```

Each `key=value` pair, separated by `;` or `,`, sets one adapter's weight for that
section. A key is either the adapter's position in the stack (`#1`, `#2`, or a bare `1`,
`2`), or its name: the file name without `.safetensors`, or the folder name for a folder
adapter. Names are not case sensitive. Adapters from the Training Studio live in
date-stamped run folders, so their name is the date stamp; position keys are easier there.
The panel shows a two-line example built from your current stack.

How directives are read:

- An adapter the directive does not mention gets 0 for that section.
- A section with no directive, like the bridge above, uses the stack's normal scales. So
  does a section whose directive names no adapter in the stack, which usually means a
  typo; the server log lists the unknown keys.
- Stack Mode applies per section: in Blend the weights are rescaled to Combined Strength,
  in Sum they are used as written. A directive of all zeros gives that section the base
  model alone.
- Negative weights are treated as 0.
- Braces with no `key=value` pair, such as `{softly}`, are left in the lyrics as text.
- Directives are removed before the lyrics reach the model. With fewer than two adapters,
  or in Simple mode, they are removed and ignored.

Per-section mode forces Runtime loading and holds each adapter separately, so VRAM grows
with every adapter in the stack. Set **Adapter VRAM** to Q4 for large stacks.

**Alignment Timing** (0.2 to 0.85, default 0.55) sets when during denoising the section
boundaries are moved to where the model actually placed each section. Before that point the
song is split in proportion to the length of each section's lyrics. Earlier values lock each
section's adapter in sooner; too early makes boundaries fuzzier.

The separation is not perfect. The part of an adapter that shapes how the caption is read
is blended across the whole song, and the first adapter in the stack tends to dominate. If
it does, try Alignment Timing around 0.3 to 0.4, or raise the weight of the adapter you
want in a section to 1.2 to 1.4 in that section's directive.

### Trigger words

A trigger word is the word an adapter saw in its training captions. The adapter responds
most strongly when the caption carries it in the same position. For each loaded adapter,
the planner adapter included, the app takes the first trigger it finds from:

1. The trigger stored in the adapter file itself. The Training Studio writes one, and files
   carrying the common `modelspec.trigger_phrase` field are read too.
2. The file name, when Settings > General > **Use filename as trigger word** is on (off by
   default). **Trigger word placement** then sets Prepend, Append or Replace. Replace
   swaps your whole style caption for the trigger.

The trigger is added at the position the adapter was trained with. A caption that already
contains it is left alone, and it is added again after the planner LM rewrites the caption.
The **LoRA** box under the style caption in Custom-Gen prepends whatever you type there,
for adapters that record no trigger.

## MiniMax-Music3

1. Put adapters in `adapters/mm3-lm-adapters/`, up to two folder levels deep. The Training
   Studio already writes there. The list is read when the panel opens, so reopen it after
   adding files.
2. Pick one in **LM Adapter**. Picking fills the dials from the adapter's sidecar, or the
   defaults when it has none.
3. Choose **Application**: **Runtime** applies the adapter as live deltas, so dial changes
   cost nothing, but planning runs about 25% slower at rank 256. **Merge** folds it into
   the weights, so planning runs at full speed and any dial change costs a re-merge of a
   few seconds. **GPU merging** (Merge only) can be turned off to merge with CPU help.
4. Set **Strength**, **Attention** and **MLP** (0 to 2 each). The fold-out **Depth thirds
   (advanced)** has **Early third**, **Middle third** and **Late third** dials; leave them
   at 1.0 for renders you keep, since the late third controls where songs end.
   **Recommended** resets all seven, Plan included, to the adapter's own suggestion.
5. **Plan** (0 to 1, default 1) splits the adapter's two jobs. The planner picks one
   code per frame for the song's structure and melody, and the voice comes from the
   codes drawn after it. At 1 the adapter does both, as before. At 0 the base model
   plans the song and the adapter sings it; in between, the two models' choices are
   mixed. Below 1 the base model runs beside the adapter on every planning step, so
   the adapter renders in Runtime mode even when Merge is picked, and planning is slower.
   The planner also plans at most 2 takes at a time below 1 (4 otherwise), so Variations
   Per Render and Require Natural Ending get 2 candidates per round instead of 3 or 4.
   That keeps its memory at what 4 takes need without the dial. Adapters that carry an
   artist token or a trained prefix ignore Plan, with a line in the engine log.

When the adapter has a trigger, **Add trigger to caption** puts it in the caption at its
trained position. Leave it on unless you are placing the trigger yourself. If a run
recorded a trigger it never trained, the panel says so and does not add it.

Adapters often stop songs from ending on their own: the render runs to the 5-minute cap.
**Adapter Tail Fade** in the Generation dropdown (off by default) keeps the adapter at full
strength until the lyrics should be done, then steps it down to the base model over 60 s
so the base can close the song. The start comes from the word count: about 1.85 words a
second plus 30 s, never later than 4:00. The fade keeps what has already been generated,
so the timbre holds. In a test on one adapter, 9 of 12 plans ended on their own with the
fade against 2 of 12 without. It needs lyrics, and it renders the adapter in Runtime mode
even when Merge is picked, so planning is slower.

An adapter that fails to load fails the job. It never renders the base model silently.

## YuE2

YuE2's language model has two halves that share no weights, so it has two slots:
**NAR adapter (renderer)** for timbre and production, and **AR adapter (composer)** for
structure and phrasing. An adapter trained on one half is refused by the other.

1. Adapters trained in the Training Studio appear on their own, newest run first, with the
   final export at the top of each run and the earlier checkpoints under it. For adapters
   copied from another machine, set **Adapter folder**; every NAR and AR file in it,
   subfolders included, is added.
2. Pick an adapter for either slot or both. The picked adapter's card shows its trigger,
   dataset, rank, steps and training loss.
3. Set **Strength**, **Attention** and **MLP** (0 to 2), and optionally the depth thirds.
   Each dial saves when you let go of the slider, or when you leave or press Enter in its
   number box.

A picked adapter shows the trigger word it was trained with. One that carries none (no
trigger in the file, its run or its dataset) shows a **LoRA trigger word** box instead: type the
word if you know it and it is put at the start of the style prompt for every render with that
adapter file. It saves when you leave the box or press Enter; clear it to stop.

The AR slot splits Strength in two. **Planner Strength** applies while the AR writes the
score (melody, chords and sections); **Composer Strength** applies while it turns that
score into audio. Planner Strength follows Composer Strength until you move it, and
**Reset**, which puts every dial in the slot back to 1.0, makes it follow again. It needs
Chain of Thought on. When the two differ, the score is planned in its own job, so each
generation reloads the model twice: once to plan, once to render.

The picker sets the default for songs you queue from then on. Each song keeps the adapters
it was queued with, including an album's own adapters when it comes from Lyric Studio, so
changing the picker while songs wait does not change what they render with. Queued songs
with the same adapters still render together in one batch.

In [Cover Studio](studios/cover-studio.md), the **Render mode** sets the adapters for that
cover: **Base YuE2** clears both slots, **Existing Artist** loads a Lyric Studio album's
paired YuE2 adapters, and **Custom** takes the **AR composer** and **NAR renderer** you pick.
The cover captures the pair and its strength settings when you click Generate;
changing the global picker while it waits does not change the queued job. The
AR strength is Cover Studio's own setting (0 to 10, default 0.6) and replaces the
picker's AR scale for covers; values above 0.6 hurt adherence to the score. A
build by section can change it until its first keep. The
approved ABC score supplies the composition plan in every mode. An adapter
changes how its half of the model uses that plan, but the app does not promise
to preserve the source performance or match a particular voice.

YuE2 merges adapters into the resident model. When a song's adapters differ from the ones
loaded, the engine waits for the renders ahead of it to finish, then unloads and reloads
with that song's adapters, which adds a longer warm-up. The panel shows whether each slot
is "Merged into the resident model now" or waiting for the next load.

The trigger is written into the style prompt for you. With both slots filled only the NAR
adapter's trigger goes in; if the AR adapter was trained under a different trigger, add
that one to the caption yourself.

### Importing ComfyUI adapters

YuE2 adapters made in ComfyUI or ai-toolkit can come as a `.safetensors` file that holds
one or both halves, in naming the engine does not load. **Import adapter…** converts one:

1. Click **Import adapter…** and pick the file.
2. The app writes the available NAR and AR adapters under
   `yue2-joint-adapters/<name>_<date>_<time>/`, lists each as **Imported · <name>** in its
   slot and selects each available half. The original file is not changed.

The trigger is read from the file's training tags, and the step count from its training
info. A file with no alpha imports at the strength ComfyUI applies (alpha equal to the
rank, scale 1.0); one whose modules carry different alphas is refused. If the adapter
sounds too strong or too weak, use the **Strength** dials. LoRA and LoKr adapters are
both supported. Import accepts fused and separate projection names, plus
`lora_A`/`lora_B` and `lora_down`/`lora_up` names. Adapters trained in this app never
need importing; a file that is already a native YuE2 adapter is refused, so point
**Adapter folder** at it instead.

YuE2 weights are licensed CC BY-NC 4.0, with the authors' exception for individual
creators, and an adapter trained on them carries the same terms. The panel shows the full
notice.

## Adapters from the Training Studio

Each backend's training output lands in the layout above and shows up in the matching
picker without a restart.

- ACE-Step DiT and planner training writes one adapter per run, with the trigger stored in
  the file. Choosing between versions means choosing between runs.
- MiniMax-Music3 training saves a checkpoint folder at the save interval set for the run
  (`ckpt-100`, `ckpt-200`, ...), each with its own sidecar.
- YuE2 training lists the final export and the step snapshots for every run.
- Stable Audio 3 training publishes a validated GGUF to `models/sa3-adapters/` when a
  run completes, and the Training Studio's SA3 card lists it.

Later checkpoints are not always better. Which one to keep, and how to compare them by
ear, is covered per backend in [ACE-Step training](training/ace-step.md),
[MiniMax-Music3 training](training/minimax-music3.md) and [YuE2 training](training/yue2.md).

## When an adapter seems to do nothing or sounds wrong

| Symptom | Likely cause and fix |
|---|---|
| The output sounds like the base model | Turn on **No-adapter reference** and compare. If they match, check the scale, check that the trigger word is reaching the caption, and check the engine log for adapter warnings. In a stack, a Blend budget of 0.75 spread over several adapters leaves each one weak. |
| The job fails with a model mismatch error, or the engine log says an adapter "merged no tensors" | The adapter was trained for a different model size, XL against 2B for the DiT, or a larger planner than the one loaded. Load the matching base. In a Merge or Runtime stack, a mismatched adapter only logs the warning ("staged no deltas" in Runtime) and is left out while the others load. With per-section directives or Active phase windows every adapter must load, so one mismatched adapter fails the job. |
| Coherent on its home base, falls apart on another | Use **Basin Re-base** with the home base selected. |
| Weak or distorted PEFT adapter | `adapter_config.json` is missing from beside `adapter_model.safetensors`. |
| DoRA adapter sounds off, or is still audible at strength 0 | Runtime and Low-Rank modes skip the magnitude rescale unless **Runtime DoRA magnitudes** is on; use Merge mode or turn the switch on. Where the rescale is applied, strength scales only the rescale: the learned weight change goes in at full size, so strength 0 still changes the sound. |
| Merge fails on a ConvRot model | ConvRot models support Runtime and Low-Rank only. |
| Out of memory with several adapters | Per-section directives and Active phase windows hold every adapter separately. Set **Adapter VRAM** to Q4, drop an adapter, or use Merge with **Low ¼** when you do not need per-section control. |
| Out of memory in Merge mode | **HQ** stores merged weights at F32. Switch Merge VRAM to **Low ¼** or use Runtime. |
| Per-section adapters blur together | See the tips at the end of [Per-section adapters](#per-section-adapters). |
| A YuE2 adapter from ComfyUI is not listed, or is refused | It is a single joint file. Use **Import adapter…** in the YuE2 panel. |
| A YuE2 pick has no effect yet | It merges on the next model load. Generate once, or check the slot's status line. |

For reading the logs, see [Troubleshooting](troubleshooting.md#where-the-logs-are).

## Related

- [Generation](generation.md)
- [Backends](backends.md)
- [Getting higher quality output](quality.md)
- [Training Studio](studios/training-studio.md)
- [ACE-Step training](training/ace-step.md)
- [MiniMax-Music3 training](training/minimax-music3.md)
- [YuE2 training](training/yue2.md)
- [Stable Audio 3 training](training/sa3.md)
- [Troubleshooting](troubleshooting.md)

## Release data paths

Release adapters default to the shared data folder's `adapters/`. A managed-folder move updates saved adapter paths inside transferred folders. An adapter directory override outside that folder remains external. See [Data folder and upgrades](data-folder.md).
