# Auto-Gen

Genre-first song creation: pick one or more genres, choose how lyrics get written, and Auto-Gen fills in everything else, style caption, lyrics, BPM, key, time signature and title, then queues the song. Reach for it when you want a finished song fast and don't need to hand-tune every generation parameter; for that, use Custom-Gen instead.

![Auto-Gen](../../images/studio-insta-gen.webp)
<!-- screenshot: needed -->

## Where to find it

The "Auto-Gen" entry in the sidebar (`/insta-gen`). The engine must be ready before you can submit; the app shows a "not ready" error otherwise.

The "Lyrics + AI" vocal mode calls an external LLM, not the built-in one, so it also needs at least one provider configured with working credentials under Settings > AI Services > API Keys (Gemini, OpenAI, Anthropic, Ollama, LM Studio, llama.cpp, or a custom OpenAI-compatible endpoint). Only providers that respond successfully appear in the Provider dropdown.

Next to the panel, a Generations list shows songs made in Auto-Gen only, filtered out from the rest of the library.

## Workflow

1. Select one or more genres in **Select Genres**, or click **Random** to pick 2 to 4 at random.
2. Choose a **Vocal Mode**: Instrumental, Lyrics (the built-in LM writes lyrics on its own), or Lyrics + AI (an external LLM writes lyrics from a subject you give it).
3. For Lyrics + AI, enter a **Song Subject** or click its **Random** button to have the LLM invent one, then pick an **LLM Provider** and **Model**.
4. Optionally add **Additional style tags** and, unless the mode is Instrumental, a **Vocal Language**.
5. Leave **Preview lyrics first** on to review and edit before committing, or turn it off to generate straight through.
6. Click the action button. With preview on, this runs Inspire (or the AI lyric step) and stops at a preview of the lyrics, caption and metadata; click **Generate Song** there to queue the render, or **Refine in Custom-Gen** to send the same lyrics, caption and metadata into Custom-Gen instead. With preview off, the button queues generation directly.
7. Watch progress in the Generations list beside the panel. Queue more songs while one is still running; Auto-Gen runs its own jobs one at a time and marks later ones "Queued…".

## Controls

| Control | What it does |
|---|---|
| Select Genres | Search box over a curated, categorised genre taxonomy (Pop, Rock, Electronic, Hip-Hop and more). Selected genres appear as removable chips and feed the style caption. Random picks 2 to 4 genres. |
| Vocal Mode | Instrumental (no lyrics), Lyrics (built-in LM writes lyrics on a random topic), or Lyrics + AI (an external LLM writes lyrics from your subject). |
| System Prompt (Lyrics + AI only) | Collapsible editor for the system prompt sent to the external LLM. Save stores a custom prompt; Reset discards it and restores the built-in default. |
| Song Subject (Lyrics + AI only) | What the song is about, in your own words. Required unless Random is on, which asks the LLM to invent a subject instead. |
| LLM Provider / Model (Lyrics + AI only) | Which configured external LLM writes the lyrics, and which of its models. |
| Additional style tags | Free-text tags appended after the selected genres to build the style caption, for example "female vocals, melancholic, reverb-heavy guitar". |
| Vocal Language | Language the lyrics are written and sung in. Hidden when Vocal Mode is Instrumental. |
| Style Caption | Read-only preview of the caption that will be sent: selected genres plus any additional style tags. |
| Cover Art | Optional override for the auto-generated cover art subject, keeping the automatic art direction. Only shown when cover art generation is enabled in post-processing settings. |
| Caption Rewrite (CoT) | When on, the built-in LM rewrites the caption into a richer style description before generation. When off, your typed caption is sent as-is. |
| Preview lyrics first | When on, the action button stops at a preview of the generated lyrics and metadata before you commit. When off, it queues generation immediately. |
| Action button | Labelled Inspire, Generate Lyrics, or Auto-Gen depending on Vocal Mode and the preview toggle. Runs the lyric/metadata step, or queues generation straight away. |
| Edit Caption / Generated Lyrics (preview screen) | Editable text of the caption and lyrics Auto-Gen produced. Changes here are what actually gets generated. |
| Generate Song (preview screen) | Queues generation with the edited caption and lyrics, plus the BPM, key, time signature and duration shown in the metadata badges. |
| Refine in Custom-Gen (preview screen) | Copies the current caption, lyrics and metadata into Custom-Gen's fields and switches to that studio, so you can adjust generation parameters by hand before rendering. |

## Tips and limits

The song title comes from the external LLM when it supplies one, otherwise Auto-Gen derives it from the lyrics: the first line of the chorus, then verse 1, then the first verse, then whatever lyric line comes first.

When the active backend is MiniMax-Music3, the estimated duration from the lyric/metadata step is dropped rather than sent, since MM3 has no duration input of its own; length comes from the planner instead.

<!-- TODO(verify): confirm whether Auto-Gen enforces any minimum genre/caption length beyond "at least one genre or a typed style tag", and whether any backend other than MM3 changes its behaviour. -->

## Related

- [Create (Custom-Gen)](create.md)
- [Lyric Studio](lyric-studio.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
- [Getting started](../getting-started.md)
