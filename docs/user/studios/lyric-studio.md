# Lyric Studio

An AI-assisted lyric-writing workspace, backed by its own database of artists,
albums and generations. Fetch or type in a set of source lyrics, have an LLM
learn the writer's style as a profile, generate new lyrics from that profile,
refine them, then send the result to Custom-Gen to render as audio. It also
holds the per-album adapter presets (ACE-Step, MiniMax-Music3 or YuE2) that a
song renders under.

![Lyric Studio](../../images/hot-step-lyric-studio.webp)

## Where to find it

Sidebar entry **Lyric Studio**. Lyric generation, refinement, profile building
and captioning need at least one LLM provider configured. See
[Controls > LLM providers](#llm-providers) and
[Getting started](../getting-started.md) for API keys. Fetching lyrics from
Genius needs a Genius access token in the same place. None of this is required
to type lyrics in by hand and send them to Custom-Gen.

An external agent (Claude Code, Codex) can also drive Lyric Studio through the
`lyricstudio` MCP server. See
[tools/mcp-lyricstudio/README.md](../../../tools/mcp-lyricstudio/README.md).

## Workflow

1. Add an artist: **Fetch from Genius** (artist name or Genius URL, with an
   optional album and a max-songs cap) or **Add Manually** from the artist
   grid's add menu.
2. Select the artist, then select an album to open its detail view. Songs
   fetched with no album name land under **Top Songs**.
3. On the **Source Lyrics** tab, review the fetched songs, edit or delete any
   of them, or add one by hand.
4. On the **Profiles** tab, click **Build New Profile** to have the profiling
   LLM analyse the source lyrics and produce a style profile (themes, tone,
   rhyme schemes, structural patterns, vocabulary statistics).
5. On the **Generated Lyrics** tab, click **Generate Lyrics** (or the faster
   **Generate (No Thinking)**) to write a new song from the profile. Set a
   count to generate several at once, and an optional subject to steer the
   topic.
6. Expand a generated song to edit its title, subject, BPM, key, duration,
   lyrics, and its caption fields, or click **Refine** to have the refinement
   LLM revise the lyrics in place (title and lyrics change; the metadata
   generated with them does not).
7. Click **Generate Audio** to queue a render from this song's lyrics and
   caption, or **Send to Custom-Gen** to open [Custom-Gen](create.md) with
   the lyrics, caption, title, artist, key/BPM and the album's adapter preset
   already filled in.
8. The **Generated Songs** tab lists every audio render made from this
   album's lyrics, playable inline.

For working through many albums or songs at once, open **Bulk Operations**
from the album sidebar. See [Bulk operations](#bulk-operations) below.

## Controls

| Control | What it does |
|---|---|
| Fetch Lyrics (artist grid, or per-artist "Add Album") | Pulls songs from Genius by artist name/URL and optional album name/URL, up to a max-songs cap (1-100). Needs `GENIUS_ACCESS_TOKEN`. |
| Add Manually (artist / album / song) | Creates an artist, album, or a single song's lyrics without Genius. |
| Build New Profile | Runs the profiling LLM over every source song in the album and saves a style profile: themes, common subjects, tone and mood, vocabulary notes, structural and narrative patterns, rhyme schemes, meter and repetition statistics. Streams its output live. |
| Recalculate Stats (bulk operations) | Re-runs only the local statistical analysis (rhyme, meter, vocabulary) on existing profiles, with no LLM call. |
| Generate Lyrics / Generate (No Thinking) | Writes a new song from a profile with the generation LLM, using past generations for that artist to avoid repeating subjects, keys, titles, BPMs and durations. "No Thinking" asks the model to skip its reasoning phase, when it has one. The count selector queues several in a row. The subject field steers the topic without dictating content. |
| Refine | Rewrites a generated song's lyrics and title with the refinement LLM, keeping its captions and other metadata; the result is saved as a new generation linked back to the original. |
| Title / Subject / BPM / Key / Duration | Editable inline on an expanded song; blur to save. |
| Caption (ACE-Step / YuE2) | The caption box that ACE-Step and YuE2 both render from. On YuE2, when the album's dataset has per-track captions, a source picker offers the track nearest in tempo, a named track, or this song's own caption. |
| MM3 Caption | MiniMax-Music3's three-heading Structured Caption, a different text from the ACE-Step/YuE2 caption above, generated alongside the lyrics. Carries the same source picker (nearest tempo / named track / custom) when the album has captioned source tracks. |
| YuE2 Caption | YuE2's one-sentence planner caption (language, genre, vocal, instruments, mood, production, BPM in that order). Plain text; no source picker of its own, because a song with one no longer borrows a dataset track's caption. |
| Generate Audio | Queues an audio render of this song through the active backend, using the current global generation parameters plus the album's adapter preset. |
| Send to Custom-Gen | Opens [Custom-Gen](create.md) with this song's lyrics, caption, title, artist, subject, key/BPM/duration and the album's adapter preset (or MM3/YuE2 adapter) already applied. |
| Album Preset | Per-album settings for the active backend: on ACE-Step, a DiT adapter folder with per-group scale sliders (self-attn, cross-attn, MLP, cond-embed), a planner-LM adapter folder, and a reference audio track for timbre conditioning; on MiniMax-Music3, one LM adapter, picked from the runs trained on this album's own dataset or any other installed adapter; on YuE2, a folder scanned for its AR (song planner) and NAR (audio renderer) adapter halves. See [Adapters](../adapters.md) and [Backends](../backends.md). |
| Edit System Prompts | Views and overrides the four LLM system prompts (Generation, Metadata Planning, Artist Profiler, Refinement) per-provider default, with a reset to the built-in prompt. |
| LLM providers (Profile / Generate / Refine rows) | Independent provider+model choice for each of the three LLM roles. See [LLM providers](#llm-providers) below. |
| Filename Prepend | Text prepended to downloaded audio filenames for songs from this artist. |
| LLM Duration | When on, uses the LLM's own estimated song duration; when off, calculates duration from the lyrics and BPM instead, which tends to undershoot less than an LLM's guess. |
| Randomize Timbre | Picks a random track from the reference folder as the timbre conditioner on each render, instead of the album preset's one fixed reference track. |
| Use LM Adapter | Off by default. When on, Send-to-Custom-Gen also applies the album preset's planner-LM adapter alongside the DiT adapter. |
| Bulk Operations | Opens the queue panel described below. |

### Bulk operations

The **Bulk Operations** panel works across every artist and album at once, in
five modes: **Fetch Lyrics** (batch Genius fetches for a list of
artists/albums), **Build Profiles** (queue profile builds for every unprofiled
album), **Generate Lyrics** (queue lyric writing for profiled albums, with a
**Fill to target** mode that tops each profile up to a chosen total instead of
a fixed count), **Generate Audio** (queue audio rendering for already-written
songs, per album, with newest/oldest ordering and a filter to skip songs
already rendered or already kept), and **Assign Presets** (bulk-apply an
adapter and reference-track preset across multiple albums at once).

### LLM providers

Lyric Studio calls out to one of several LLM providers, configured with
environment variables (see [dev/config.md](../../dev/config.md) and
[Getting started](../getting-started.md) for where to set them):

| Provider | Env vars |
|---|---|
| Google Gemini | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| OpenAI | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| LM Studio | `LMSTUDIO_BASE_URL`, `LMSTUDIO_API_KEY`, `LMSTUDIO_MODEL` |
| llama.cpp | `LLAMACPP_BASE_URL`, `LLAMACPP_MODEL` |
| OpenAI-compatible (any other endpoint) | `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_MODEL`, `OPENAI_COMPAT_NAME`, `OPENAI_COMPAT_REASONING_EFFORT` |

Only providers with credentials configured show up in the provider dropdowns.
Three roles, Profiling, Generation and Refinement, each get an independent
provider and model choice, all persisted locally in the browser. `LLM_TIMEOUT_MS`
sets how long a single call may run (default 300 s). Raise it if a local
model shares the GPU with the engine and needs longer to respond.

## Tips and limits

Building a profile needs at least one source song. Generating lyrics needs at
least one profile.

Refining a song changes its lyrics and title only; its captions, BPM, key and
duration stay as they were. Edit those by hand afterward if the rewrite
changed the song enough to need it.

"Curated Profile" (building a profile from hand-picked songs across several
albums) is present in the UI but its backend endpoint is not implemented yet.
Expect it to fail.

MM3 and YuE2 caption source pickers (nearest tempo / named track / custom)
only appear when the album's own dataset carries per-track captions, which
normally means the album was exported from Training Studio rather than typed
in or fetched from Genius.

## Related

- [Custom-Gen](create.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
- [Adapters](../adapters.md)
- [Getting started](../getting-started.md)
- [Configuration reference](../../dev/config.md)
