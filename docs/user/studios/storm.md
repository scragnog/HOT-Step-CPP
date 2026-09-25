# STORM

A live streaming performance mode. Instead of queuing one song and waiting for it, STORM opens a continuous audio stream and renders it in back-to-back "slots" that crossfade into each other, while you retune the style, lyrics, seed and sampler settings for the next slot without stopping playback. It also has a one-off Sequential mode that queues a single song the normal way, and a DJ mode that runs two independent streams on a crossfader.

![STORM](../../images/studio-storm.webp)
<!-- screenshot: needed -->

## Where to find it

The "STORM" entry in the sidebar (`/storm`). The engine must be ready first.

STORM only works on the ACE-Step 1.5 backend. On any other active backend the page shows "Not supported by the active backend — switch back to ACE-Step 1.5 in the global bar" instead of the studio.

Streaming and the normal generation queue can't run at the same time: starting a stream while any job is active in the queue is refused ("N generation job(s) active — wait for the queue to drain before streaming"), and while a stream is running you can't start a normal generation elsewhere.

## Workflow

1. Pick a mode tab: **Sequential**, **Continuous**, **DJ**, or **Drift** (Drift is a placeholder, not built yet).
2. In the Compose column, write a **Style** caption and, unless **Instrumental** is on, **Lyrics**. Optionally add a LoRA trigger word or a percussive beat intro/outro for DJ mixing.
3. In the Sampler column, set guidance/steps/duration/BPM defaults, tune the extra params for whichever solver you're using, and optionally pick a specific Solver, Scheduler or Guider for the stream (leave them on "inherit global" to use the ones set in the global bar).
4. Click **Generate** (Sequential) or **Start** (Continuous). Sequential queues one song through the normal job queue, the same as any other studio, and STORM's live controls don't apply to it. Continuous opens the audio stream and starts playing slot 1 as soon as it renders.
5. While a Continuous stream plays, use the Live Controls column on the right to change the style, lyrics, seed, key, BPM or per-slot sliders for the *next* slot, then click **Send** (or drag a slider, which sends after you stop moving it). Changes land on the slot after the one currently rendering.
6. Click **Record** to start capturing the master output to a downloadable `.webm` file; click it again to stop and download. Recording only happens in the browser, nothing is saved to the library automatically.
7. Click **Stop** (or navigate away from the page) to end the stream.

For DJ mode: start Deck A and Deck B independently from their own panels, each with its own caption, lyrics and Start/Stop. The center column shows each deck's detected key on the Camelot wheel with a compatibility read-out, a crossfader, cut buttons, per-deck nudge buttons, and Quantize A→B to line up Deck B's next slot to Deck A's next beat.

## Controls

### Mode tabs

| Control | What it does |
|---|---|
| Sequential | Builds one song from the Compose/Sampler panels and sends it to the normal generation queue. No live stream, no crossfading. |
| Continuous | Opens a persistent audio stream: slots render and crossfade back-to-back until you stop it. Live Controls become active. |
| DJ | Two independent Continuous streams (Deck A, Deck B) mixed through a crossfader, with key-compatibility and beat-sync helpers. |
| Drift | Placeholder screen only ("Coming soon"). Not implemented. |

### Compose column

| Control | What it does |
|---|---|
| Style | The style caption for the stream. Supports `{A|B|C}` wildcard syntax, expandable in place with the wildcard button. |
| LoRA | A trigger word prepended to the style caption. |
| Beat I/O | Appends a note asking for a clean 1/2/4/8-bar percussive intro and outro, for cleaner DJ transitions. |
| Instrumental | Skips lyrics and forces an instrumental generation. |
| Lyrics | Lyrics for the stream. The loop/cycle/shuffle buttons control how sections advance automatically as slots play: loop repeats the same lyrics, cycle steps through sections in order, shuffle picks a random unseen section each slot. |
| Negative Prompt | Free-text negative prompt, sent with every slot. |

### Sampler column

| Control | What it does |
|---|---|
| Solver extra params | Sliders specific to the active solver (for example Detail Sens, Coherence, Early Focus and Cache Depth for the default `storm` solver). Which sliders appear depends on which Solver is selected below. |
| Precision | RK solver order (auto, 2 to 5). Applies from the next Start, not live. |
| Cache Ratio / CFG Cutoff | Skip redundant DiT steps / skip the unconditional guidance pass after a ratio of steps, to speed up rendering. Applies from the next Start. |
| LSS | Latent self-similarity strength, live-adjustable while streaming. |
| Solver / Scheduler / Guider | Overrides the stream's solver, noise scheduler or guidance mode. "— inherit global —" uses whatever is set in the global bar. Changing these while a stream plays sends the change live. |
| Scheduler params / Guider params | Extra sliders and dropdowns specific to the chosen Scheduler or Guider, shown only when one is picked that has extra params. |
| XFade | Crossfade length between slots, in beats. |
| MaxBuf | How far ahead of playback the client is allowed to buffer, in minutes or in slot multiples (toggle with the clock/note icon). The engine pauses rendering once the buffer fills and resumes once playback catches up. |

### Live Controls (Continuous, right column)

| Control | What it does |
|---|---|
| Guidance / Steps / Duration | Sliders for the next slot's CFG scale, inference steps and length. Click the value to type a number directly. |
| Seed | The base seed. Each slot advances the seed by one unless Seed Lock is on, in which case every slot reuses the same seed. Includes randomize, a save/load drawer, and the lock toggle. |
| Key | Auto shows the key detected client-side from the audio. Manual lets you type a key, but this does not currently change generation, it's a display-only field the server ignores. |
| BPM | Auto shows the BPM detected client-side from the audio. Manual sends a fixed BPM for the next slot. |
| AI ↻ | Enables automatic AI lyric continuation every 1, 2 or 4 slots, using the external LLM provider configured in AI Generate settings. The direction field is a persistent hint; the gear icon opens a preset manager and the prompt template used for continuation. |
| Style / Lyrics tabs | Fields for the next slot's style or lyrics. The pin icon next to each makes it "stick" (stays applied to every following slot) instead of clearing after Send. |
| Send | Queues the typed style/lyrics for the slot two ahead of the one currently playing. |
| Record | Captures the mixed output to a `.webm` file, downloaded when you stop recording. |

Timeline boxes above the panel show each slot's state (queued, rendering, rendered, playing, past). Clicking a box shows that slot's stamped metadata (seed, detected key/BPM, guidance, steps, style, lyrics) once it has played; clicking a future box does not seek playback.

## Tips and limits

Nothing STORM renders is written to SQLite or the library. Continuous and DJ streams exist only as audio in the browser tab; the only way to keep output is the Record button. Sequential mode is the exception, since it goes through the normal generation queue and lands in the library like any other song.

Leaving the STORM page stops whatever is streaming.

AI Continuation needs an LLM provider already configured under the same AI Generate settings used elsewhere in the app; with none configured it shows a warning instead of generating, and disables itself after three consecutive failures.

The STORM solver plugin and this streaming UI are a contribution from MDMAchine (Alexander Allan), shipped under GPLv3 alongside the rest of the plugin stack.

## Related

- [Create (Custom-Gen)](create.md)
- [Library](library.md)
- [Generation](../generation.md)
- [Backends](../backends.md)
