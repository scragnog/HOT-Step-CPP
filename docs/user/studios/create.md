# Custom-Gen

Full manual song generation: type or paste a style description and lyrics, set BPM, key and the
other music parameters by hand, then queue the render. Reach for it when you want direct control
over every field instead of the genre-picker shortcuts in Auto-Gen, or when refining a song that
arrived here from Lyric Studio or Auto-Gen's preview screen.

![Custom-Gen](../../images/studio-create.webp)
<!-- screenshot: needed -->

## Where to find it

The "Custom-Gen" entry in the sidebar, in the Create section above Library (`/create`). The
engine must be ready before you can submit; the app shows a "not ready" error otherwise, and a
"paused for training preprocessing" error if a training job currently owns the GPU.

Which model, adapter and sampler settings a generation uses come from the global parameter bar
above this panel, not from anything on this page, see [Generation](../generation.md). Some
sections described below only appear for a particular active backend (ACE-Step 1.5,
MiniMax-Music3 or YuE2), chosen in that same bar, see [Backends](../backends.md).

## Workflow

1. Write a **Style Description**, optionally add a **LoRA** trigger word and turn on **Beat I/O**
   for a percussive intro/outro. Both `{a|b|c}` wildcard syntax and the LoRA trigger prefix apply
   here.
2. Optionally expand **Song Info** to set an Artist, Title and Subject.
3. Turn on **Instrumental**, or leave it off and write **Lyrics** (with `[Verse]`, `[Chorus]` and
   the other section tags).
4. Optionally add a **Negative Prompt** of what to avoid in the output.
5. On MiniMax-Music3, if this song arrived from Lyric Studio carrying its training album's own
   captions, pick a **Caption source**; otherwise click **Compose Caption** to turn the style
   description into a MiniMax-Music3 Structured Caption.
6. On YuE2, pick a **Dataset** and, once it has captioned tracks, a **Caption source**, the same
   three-way choice as MiniMax-Music3's.
7. Set **BPM**, **Duration**, **Key**, **Time Signature**, **Vocal Gender** and **Language** as
   needed. Which of these appear depends on the active backend, see the Controls table.
8. Optionally import a **.latent** file to continue from a saved latent instead of a fresh
   prompt, or expand **Cover Art** to override the auto-generated art subject.
9. Or skip steps 1 to 7 entirely: click **Generate with AI** in the header, fill in a genre/style
   and a subject, and let a configured LLM write the caption, lyrics, title and metadata.
10. Click **Generate** (**Queue Generation**, with a running count, once another job is already
    rendering). On YuE2, if "Preview the score first" is on in the global parameter bar, this
    opens a lead sheet preview instead of queuing a render.
11. The song appears in [Library](library.md) once it finishes, tagged as having come from
    Custom-Gen.

## Controls

| Control | What it does |
|---|---|
| Style Description | The caption sent to the model. Supports `{a|b|c}` wildcards, the `{·} expand` button resolves them once in place, `auto` resolves them at Generate time instead, using the DiT seed (or a fresh draw when the seed is randomised). Read-only, with a dataset caption showing through it, while a MiniMax-Music3 or YuE2 caption source below is locked to something other than Custom. |
| LoRA | A trigger word prepended to the style description before it is sent, for adapters that need one typed rather than embedded in the file. |
| Beat I/O | Appends a request for a clean percussive intro and outro (1, 2, 4 or 8 bars) to the style description, for DJ mixing. |
| Song Info (Artist / Title / Subject) | Optional metadata, collapsed by default. Auto-populates and expands when a song is sent here from Lyric Studio or Auto-Gen. |
| Instrumental (no vocals) | Sends `[Instrumental]` as the lyrics and hides the Lyrics box. |
| Lyrics | Full lyric text with section tags. Supports the same wildcard syntax and expand/auto controls as Style Description. Hidden while Instrumental is on. |
| Negative Prompt | Free-text description of what to avoid, for example "jazz, acoustic, slow, ambient, piano". Sent to all three backends. |
| Caption source (MiniMax-Music3) | Only shown for a song sent from Lyric Studio with its album's captioned tracks attached. Automatic from dataset picks the track whose BPM is closest to yours; a named Track uses that track's own Structured Caption verbatim; Custom uses this song's own caption, editable. Dismiss removes the picker for this panel. |
| Compose Caption (MiniMax-Music3) | Turns the plain-English Style Description into a MiniMax-Music3 Structured Caption, assembled from MiniMax's own 1,000 reference captions, no external AI model or API key involved. Reports the genre and family it routed to, how many reference captions it drew from, and any warnings (unrecognised genre, a control that conflicts with the prompt, or a control MiniMax-Music3 cannot express). The dice button composes again with a new seed. Hidden while a Caption source above is locked in. |
| Dataset / Caption source (YuE2) | Dataset picks which training dataset's captions this panel offers, defaulting to the dataset the active YuE2 adapter was trained on. Once that dataset has captioned tracks, Caption source offers the same Custom / Automatic / named-Track choice as MiniMax-Music3. |
| BPM | Target tempo. 0 means Auto, letting the model pick. |
| Duration | Target length in seconds. Hidden entirely on MiniMax-Music3, whose planner decides when the song ends, so a length there could only cut the ending off early. Where the active backend supports an explicit end-of-song token, an Auto chip lets the model end the song itself instead of aiming for a fixed length. |
| Key | Musical key, or blank for Auto. |
| Time Signature | No wire field on MiniMax-Music3 or YuE2, hidden on both. |
| Vocal Gender | Any / Female / Male / Duet. Hidden on YuE2. Has no wire field on either remaining backend, it only reaches the model when Compose Caption writes it into the caption's Vocal Details line for MiniMax-Music3; on ACE-Step 1.5, or on MiniMax-Music3 without Compose Caption, this control has no effect. |
| Vocal Language / Lyrics Language | Hidden on YuE2, which infers language from the lyrics' own characters. Relabelled Lyrics Language on MiniMax-Music3, since that backend has no language input either and follows the lyrics text the same way. |
| Import latent | Loads a previously exported `.latent` file to continue from that saved latent instead of a fresh prompt, skipping the VAE encode step the backend would otherwise do on source audio. Populates BPM, key, lyrics and caption from the file's embedded metadata where present. |
| Cover Art | Override the auto-generated cover art image subject while keeping the automatic art direction (genre visuals, quality modifiers). Only shown when cover art generation is enabled in post-processing settings. |
| Generate with AI | Opens a modal to have a configured LLM provider write the caption, lyrics, title, subject, BPM, key, time signature, duration and vocal language in one shot, from a genre/style and a subject you give it (or a random one). Requires at least one LLM provider configured and reachable; disabled otherwise. Turns Instrumental off. |
| Generate / Queue Generation | Submits the form. Disabled until there is a caption, lyrics, or Instrumental is on. Reads Queue Generation, with a count badge, once another job is already rendering. On YuE2, if "Preview the score first" is on (a toggle in the global parameter bar's YuE2 controls), this opens the lead sheet preview instead of queuing a render directly. |
| Lead sheet preview (YuE2) | Shows the planner's lead sheet as staff notation before any audio renders, with a health verdict (healthy, long or runaway), a reason, an estimated duration, the seed and the section order. Play synthesises it in the browser, the first click is the user gesture the browser requires before it can make sound, and needs network access to fetch a soundfont. Continue renders exactly this score with its seed pinned; Retry plans a new one with a fresh seed; Cancel renders nothing. |

## Tips and limits

Adapters and Mastering are not on this page. Both moved to the global parameter bar, see
[Generation](../generation.md).

Vocal Gender is a dead control everywhere except through Compose Caption on MiniMax-Music3, see
the Controls table.

<!-- TODO(verify): confirm whether Custom-Gen enforces any server-side minimum on caption/lyrics length beyond the client-side "caption, lyrics, or Instrumental" check, and whether that check differs per backend. -->

## Related

- [Auto-Gen](insta-gen.md)
- [Lyric Studio](lyric-studio.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
- [Adapters](../adapters.md)
