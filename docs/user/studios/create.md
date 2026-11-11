# Custom-Gen

Full manual song generation: type or paste a style description and lyrics, set BPM, key and the
other music parameters by hand, then queue the render. Reach for it when you want direct control
over every field instead of the genre-picker shortcuts in Auto-Gen, or when refining a song that
arrived here from Lyric Studio or Auto-Gen's preview screen.

For a single YuE2 song, Library's Edit action reproduces it exactly with the same model
and adapters. For a song generated in a batch, Edit restores its score and seeds but
renders a new performance: a song composed beside others does not decode the same as
that song composed alone.

Edit restores the track's rendered seed and NAR noise seed, and sets the song and
variation counts to one. If the engine received a score, including one selected by
auto-replan, Edit sends that saved score byte for byte. If the
engine planned and composed in one call, Edit leaves the score input empty and restores
the auto-replan setting so planning consumes the same random draws before composing.
The saved score remains available for display in both cases.

Older songs use their saved score-provenance flags when available. Without those flags,
Edit leaves the score input empty and plans again. Missing scores or settings cannot be
recovered, so exact reproduction of those older songs is not guaranteed.

For YuE2, **Continue from recording (experimental)** uploads audio and selects its start
and end in seconds.
Choose up to 30 seconds, then Generate. YuE2 tokenizes the recording, uses the selected
section as the beginning of a new song, and continues composing with score planning off.
The prefix is synthesized again, so it will differ from the original audio. Listening
checks found it works on simple material and HOT-Step renders, but is unreliable on real
vocal recordings, with smeared starts and looping passages. The feature renders one
song per job; the variation count still applies. It cannot be combined with a cover or a
build session. The selected recording section belongs only to Custom-Gen; it does not
change Cover Studio's requests. Library's Edit action restores the saved section for a
continuation, and clears it for an ordinary YuE2 song. Style and lyrics may both be empty for YuE2.

![Custom-Gen](../../images/studio-create.webp)
<!-- screenshot: needed -->

## Where to find it

The "Custom-Gen" entry in the sidebar, in the Create section above Library. It is the app's
default view, at `/`. The engine must be ready before you can submit; the app shows a "not
ready" error otherwise, and a "paused for training preprocessing" error if a training job
currently owns the GPU. While renders run, the sidebar entry carries a badge with the number of
running jobs and the active one's progress.

Which model, adapter and sampler settings a generation uses come from the global parameter bar
above this panel, not from anything on this page, see [Generation](../generation.md). Some
sections described below only appear for a particular active backend (ACE-Step 1.5,
MiniMax-Music3, YuE2 or Stable Audio 3), chosen with the engine picker at the top of the
sidebar, see [Backends](../backends.md).

## Workflow

1. Write a **Style Description**, or pick one from **Presets** for a ready-made ensemble
   description and edit it from there. Optionally add a **LoRA** trigger word and turn on
   **Beat I/O** for a percussive intro/outro. Both `{a|b|c}` wildcard syntax and the LoRA trigger
   prefix apply here. YuE2 shows neither: its adapters carry their own trigger word.
2. Optionally expand **Song Info** to set an Artist, Title and Subject.
3. Turn on **Instrumental**, or leave it off and write **Lyrics** (with `[Verse]`, `[Chorus]` and
   the other section tags).
4. Optionally add a **Negative Prompt** of what to avoid in the output (folded away on YuE2,
   which does not use it).
5. On MiniMax-Music3, if this song arrived from Lyric Studio carrying its training album's own
   captions, pick a **Caption source**; otherwise click **Compose Caption** to turn the style
   description into a MiniMax-Music3 Structured Caption.
6. On YuE2, open **Dataset Caption Options** under the Style Description to pick a **Dataset**
   and, once it has captioned tracks, a **Caption source**, the same three-way choice as
   MiniMax-Music3's, and the tags it offers.
7. Under **Music Parameters**, set **BPM**, **Duration**, **Key**, **Time Sig**, **Vocal
   Gender** and **Language** as needed. Which of these appear depends on the active backend, see the Controls table.
8. On ACE-Step 1.5, optionally import a **.latent** file to continue from a saved latent instead
   of a fresh prompt. Expand **Cover Art** to override the auto-generated art subject.
9. Or skip steps 1 to 7 entirely: click **Generate with AI** in the header, give it a
   genre/style, a subject or both, and let a configured LLM write the caption, lyrics, title and
   metadata.
10. Click **Generate** (**Queue Generation**, with a running count, once another job is already
    rendering). On YuE2, if "Preview the score first" is on in the global parameter bar and no
    lead sheet is pasted into its Lead sheet (ABC) box, this opens a lead sheet preview instead
    of queuing a render.
11. The song appears in the Generations list beside the panel and in [Library](library.md) once
    it finishes, tagged as having come from Custom-Gen.

The form's content, metadata and caption-source choices (MiniMax-Music3 track pick, YuE2
dataset and track pick) also save to a server draft, together with the active backend. Saving
and **Load saved draft** follow the same rules as Cover and Repaint: the draft belongs to this
browser, a conflicting edit elsewhere becomes a new draft, and a failed save keeps your edits
here. A draft loads only on the backend it was saved with; on any other backend the picker
says which backend to switch to. The draft controls sit in the **Drafts** fold under the
Generate button, collapsed by default. On YuE2 they also have a name box and **Save named
draft**, as in Cover Studio, and a named draft is listed under its name. Every fold in this
column (Song Info, Negative Prompt, Dataset Caption Options, Drafts) remembers whether it was
open.

Training Studio's **Send to Custom-Gen** opens this page with a bar at the top offering its
audition setup. **Apply** replaces the form's content and generation settings with it (LoRA
trigger cleared, adapter stack emptied, LM code caching turned off) and does not generate;
**Dismiss** leaves the form as it was.

## Controls

| Control | What it does |
|---|---|
| Style Description | The caption sent to the model. Supports `{a|b|c}` wildcards, the `{·} expand` button resolves them once in place, `auto` resolves them at Generate time instead, using the DiT seed (or a fresh draw when the seed is randomised); `auto` is not offered on YuE2. Drag its bottom edge to resize it, like Lyrics. Read-only, with a dataset caption showing through it, while a MiniMax-Music3 or YuE2 caption source below is locked to something other than Custom. |
| Presets | A button at the right of the Style Description title that opens a list of twelve ready-made ensemble descriptions, one sentence of instrumentation and mood each, from a smoky jazz quartet to a heavy metal band. Picking one writes it into the Style Description, replacing what was there; typing in the box afterwards clears the selection. Hidden while a caption source is locked. Handy with a lead sheet, where the melody and chords are already fixed and only the arrangement is left to say. |
| LoRA | A trigger word prepended to the style description before it is sent, for adapters that need one typed rather than embedded in the file. If the description already starts with that complete trigger, for example on a reused song, it is not added a second time. Not shown on YuE2: a YuE2 adapter without an embedded trigger takes one in the Adapters dropdown instead (see [Adapters](../adapters.md)). |
| Beat I/O | Appends a request for a clean percussive intro and outro (1, 2, 4 or 8 bars) to the style description, for DJ mixing. Not shown on YuE2. |
| Song Info (Artist / Title / Subject) | Optional metadata, collapsed by default. Fills in and expands when a song arrives with any of these set: Lyric Studio sends all three, Auto-Gen sends a title when its LLM wrote one. |
| Instrumental (no vocals) | Sends `[Instrumental]` as the lyrics and hides the Lyrics box. |
| Lyrics | Full lyric text with section tags. Supports the same wildcard syntax and expand/auto controls as Style Description. Hidden while Instrumental is on. |
| Negative Prompt | Free-text description of what to avoid, for example "jazz, acoustic, slow, ambient, piano". Only ACE-Step 1.5 has a negative prompt input; MiniMax-Music3, YuE2 and Stable Audio 3 report none. On YuE2 the box sits in a fold, collapsed by default. |
| Caption source (MiniMax-Music3) | Only shown for a song sent from Lyric Studio with its album's captioned tracks attached. Automatic from dataset picks the track whose BPM is closest to yours; a named Track uses that track's own Structured Caption verbatim; Custom uses this song's own caption, editable. Dismiss removes the picker for this panel. |
| Compose Caption (MiniMax-Music3) | Turns the plain-English Style Description into a MiniMax-Music3 Structured Caption, assembled from MiniMax's own 1,000 reference captions, no external AI model or API key involved. Reports the genre and family it routed to, how many reference captions it drew from, and any warnings (unrecognised genre, a control that conflicts with the prompt, or a control MiniMax-Music3 cannot express). The dice button composes again with a new seed. Disabled while the box is empty or already holds a Structured Caption; hidden while a Caption source above is locked in. |
| Dataset Caption Options (YuE2) | A fold under the Style Description, open by default. The tag chips sit in a box four rows tall that scrolls, and a leading "and" is dropped from each tag. Dataset picks which training dataset's captions this panel offers, defaulting to the dataset the active YuE2 adapter was trained on. Once that dataset has captioned tracks, Caption source offers the same Custom / Automatic / named-Track choice as MiniMax-Music3. |
| BPM | Target tempo, 0 to 240. 0 means Auto, letting the model pick. |
| Duration | Target length in seconds, from Auto (0 or below) up to the backend's ceiling, 600 s on ACE-Step 1.5 and Stable Audio 3. Hidden on MiniMax-Music3 and YuE2, which end the song themselves, so a length there could only cut the ending off early; their renders always run on Auto. |
| Key | Musical key, or blank for Auto. |
| Time Sig | 4/4, 3/4, 6/8 or 2/4, or blank for Auto. No wire field on MiniMax-Music3 or YuE2, hidden on both. |
| Vocal Gender | Any / Female / Male / Duet. Hidden on YuE2. No backend has a wire field for it; it only reaches the model when Compose Caption writes it into the caption's Vocal Details line for MiniMax-Music3. On ACE-Step 1.5 and Stable Audio 3, or on MiniMax-Music3 without Compose Caption, this control has no effect. |
| Vocal Language / Lyrics Language | Hidden on YuE2, which infers language from the lyrics' own characters. Relabelled Lyrics Language on MiniMax-Music3, since that backend has no language input either and follows the lyrics text the same way. |
| Import Latent (.latent) | ACE-Step 1.5 only. Loads a previously exported `.latent` (or `.hslat`) file to continue from that saved latent instead of a fresh prompt, skipping the VAE encode step the backend would otherwise do on source audio. Populates BPM, key, lyrics and caption from the file's embedded metadata where present. |
| Cover Art | Override the auto-generated cover art image subject while keeping the automatic art direction (genre visuals, quality modifiers). Only shown when cover art generation is enabled in post-processing settings. |
| Generate with AI | Opens a modal to have a configured LLM provider write the caption, lyrics, title, subject, BPM, key, time signature, duration and vocal language in one shot, from a genre/style, a subject you give it (Random asks the LLM for one), or both. Its Generate Song button needs an available provider and stays disabled when the server lists none. Turns Instrumental off. |
| Generate / Queue Generation | Submits the form. Disabled until there is a caption, lyrics, or Instrumental is on. Reads Queue Generation, with a count badge, once another job is already rendering. On YuE2, if "Preview the score first" is on (a toggle in the global parameter bar's YuE2 controls), this opens the lead sheet preview instead of queuing a render directly, unless a lead sheet is already pasted into the bar's Lead sheet (ABC) box. |
| Lead sheet preview (YuE2) | Shows the planner's lead sheet as staff notation before any audio renders, with a health verdict (healthy, long or runaway), a reason, an estimated duration, the seed and the section order. Play synthesises it in the browser, the first click is the user gesture the browser requires before it can make sound, and needs network access to fetch a soundfont. Continue renders exactly this score with its seed pinned; Retry plans a new one with a fresh seed; Cancel renders nothing. Download .abc saves the score to a text file named after its title line, for editing in MuseScore or EasyABC and dropping back into the global bar's Lead sheet (ABC) box. |

A reset arrow appears beside Instrumental, BPM, Key, Time Sig, Vocal Gender and
Vocal/Lyrics Language whenever the value has moved from the default; click it to snap
that one field back.

## Tips and limits

Adapters and Mastering are not on this page. Both moved to the global parameter bar, see
[Generation](../generation.md).

ACE-Step songs become playable when generation finishes. MiniMax-Music3 has a separate
**Play While Rendering** option in its backend controls for listening during a render.

The server setting `GENERATION_INTENT_PATH` selects how Create submits. By default the server's
generation intent resolver builds the request; set to `old`, the original browser-side request
builder does. The controls and saved form fields work the same way on either path, see
[Configuration](../../dev/config.md).

Vocal Gender is a dead control everywhere except through Compose Caption on MiniMax-Music3, see
the Controls table.

Generate only checks that there is a caption, lyrics or Instrumental. MiniMax-Music3, YuE2 and
Stable Audio 3 also need a Style Description, and a render without one fails with a message
saying so; on YuE2 an adapter's trigger word is enough. There is no minimum length.

On Stable Audio 3 only the Style Description and Duration reach the model. It renders exactly
the length asked for, 30 seconds when Duration is on Auto, and ignores the lyrics, BPM, key,
time signature, vocal fields and negative prompt that this page still shows. Import latent is
hidden there.

## Related

- [Auto-Gen](insta-gen.md)
- [Lyric Studio](lyric-studio.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
- [Adapters](../adapters.md)
