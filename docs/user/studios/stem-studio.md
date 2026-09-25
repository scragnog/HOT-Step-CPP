# Stem Separator

Splits an existing track into isolated stems, vocals, drums, bass and more, so you can remix, sample, or feed a stripped-down source into another studio. Two engines live behind one panel: SuperSep, a neural separator that is the general-purpose default, and Extract, which asks the DiT model to regenerate each instrument from scratch while listening to the full mix. Reach for SuperSep for quick, accurate splits of a real recording; reach for Extract when you want an AI reinterpretation of a part rather than the original signal.

![Stem Separator](../../images/studio-stem-studio.webp)
<!-- screenshot: needed -->

## Where to find it

The "Stem Separator" entry in the sidebar (`/stem-studio`). The rest of the docs, FEATURES.md and README call the same panel "Stem Studio"; the sidebar label and this page's file name both say Stem Separator, they are the same feature.

You need source audio before anything else: upload a file or pick a track already in the library. SuperSep needs its model files downloaded first (see Tips and limits); Extract needs a base or SFT DiT model, not a turbo or merged one, and needs Model Manager to have at least one such model installed.

## Workflow

1. Add source audio: drag a file onto the upload zone or click it (MP3, WAV, FLAC, AIFF, OGG), or click **Browse Library** and pick a track, filtering by Create, Lyric Studio or Cover Studio and choosing the Raw or Mastered version if both exist.
2. Pick a mode: **SuperSep (GGML)**, the default, or **Extract (DiT)**.
3. For SuperSep, choose a **Separation Level** from the dropdown.
4. For Extract, toggle which of the 12 instrument tracks to generate and pick a DiT model from the **Extract Model** dropdown. Optionally expand **Optional: Style & Lyrics** to add a style hint and paste lyrics, lyrics only steer the vocals track.
5. Click **Separate Audio** (SuperSep) or **Extract N Tracks** (Extract). A progress bar tracks the SuperSep passes or each DiT render in turn.
6. When the job finishes, the stems load into the interactive Stem Mixer.
7. Preview the mix, adjust levels, then download individual stems or the whole set as a ZIP.
8. Past jobs stay listed under **Recent Extractions**; click one to reload its stems into the mixer at any time.

## Controls

| Control | What it does |
|---|---|
| Source Audio | Upload zone or library picker. Clearing the source resets the panel; the last source is remembered across visits. |
| Mode: Extract (DiT) / SuperSep (GGML) | Switches between generative re-synthesis and neural separation. Switching modes keeps the source audio. |
| Separation Level (SuperSep) | Basic (6 stems: vocals, bass, drums, guitar, piano, other), Vocal Split (+ lead/backing vocals, 8 stems), Full (+ 6 drum sub-stems, 12 stems), 2-Stem (BS-RoFormer) (vocals plus an instrumental that is the exact mix-minus complement), 2-Stem (Leap Xe) (vocals and instrumental each rendered by their own dedicated model, so neither is a residual of the other). |
| Select Tracks (Extract) | Toggles for the 12 track types: vocals, backing vocals, drums, bass, guitar, keyboard, percussion, strings, synth, fx, brass, woodwinds. Select All / Clear affect the whole grid. |
| Extract Model | DiT model used for generation. Only plain base checkpoints appear (`acestep-v15-base-*`, `acestep-v15-xl-base-*`); SFT, merged and turbo builds are filtered out. |
| Style Hint / Lyrics (Extract) | Optional free-text caption and lyrics to steer the DiT render. Lyrics only apply when the vocals track is selected; feeding them to other tracks pulls the lead vocal into that track instead. |
| Separate Audio / Extract N Tracks | Starts the job. Disabled until a source is set (and, for Extract, at least one track and a model are chosen). |
| Stem Mixer: Preview | Loads every stem into the browser and plays them together in sync via Web Audio, so you can audition the split before downloading anything. |
| Stem Mixer: Mute (M) / Solo (S) | Per-stem toggles. Soloing one or more stems silences the rest during preview; muting is independent per stem. |
| Stem Mixer: Volume | Per-stem slider, 0-200%, live during preview. |
| Stem Mixer: Download / Download All | Downloads one stem as WAV, or every stem in the job as a ZIP. |
| Recent Extractions | Past jobs (both modes, mixed), each tagged Extract or SuperSep with its stem count and age. Click to reload, or delete to remove its files from disk. |

## Tips and limits

Extract is generative, not literal separation: the DiT model listens to the full mix and paints a new instrument track that fits it, so the result won't be a clean isolation of the original recording. It needs a plain base or SFT checkpoint; turbo models remain selectable but the job runs with a visible warning that extraction quality will be poor. Each selected track is a full, separate DiT render, done one after another, so extracting several tracks takes proportionally longer, and the panel always ignores the global bar's step count, adapter and LoRA settings for this mode, generating with engine defaults, a random seed, and adapters forced off.

SuperSep needs its model files installed through Model Manager first. The "Stem Separation" pack (BS-RoFormer, Mel-Band RoFormer, MDX23C) covers Basic, Vocal Split, Full and the BS-RoFormer 2-Stem level; the separate "StableStep Separation (Leap Xe)" pack is needed for the Leap Xe 2-Stem level. Level 3 ("Maximum") was retired and no longer appears in the dropdown, though older saved jobs that used it still work.

<!-- TODO(verify): the model registry also lists a "Stem Separation Runtime" pack (ONNX Runtime + cuDNN, ~1.3 GB) described as required for GPU separation, while every individual SuperSep model file's own description says "native GGML, no ONNX Runtime". Confirm with a maintainer whether that runtime pack is still needed for any current separation level before telling users to download it. -->

Some SuperSep levels produce hidden intermediate stems, for example the raw combined vocal buss before it's split into lead and backing. These save to disk for debugging but never appear in the Stem Mixer or the stem count.

Both modes write into the same job history and disk layout, so Recent Extractions, downloads and deletion work identically regardless of which one produced a job. Settings has a stem storage counter and a "Clear All Stems" button that deletes every job's files at once.

There's no automatic hand-off into Stem Builder: download a stem (or the ZIP) from here, then upload it as the source track on that panel. Stem Studio's own output isn't offered as a library source inside either panel's picker.

A SuperSep job that never finishes separating times out after 2 hours; an Extract track render times out after 60 minutes. Either is reported as a failed job rather than left spinning.

## Related

- [Stem Builder](stem-builder.md)
- [Library](library.md)
- [Model Manager](model-manager.md)
- [Getting started](../getting-started.md)
