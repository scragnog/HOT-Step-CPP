# Captions and lyrics

> Reference for the `mm3-lm-adapter-training` skill. Read only when the task needs it.

## Captions: per-track .mm3.txt ONLY (the shared caption was removed 2026-09-09)

**The shared caption was the endings bug.** From 2026-08-24 the route
auto-picked an existing `_shared-caption.txt` and the trainer replaced every
per-track caption with it, while renders used the per-track captions. On
album B the identical recipe went 0/6 natural endings (shared) to 4/6
(per-track) with likeness Rob called perfect. Rob: "if shared captions break
endings, we should not offer it as a feature at all." The box, the route
fallback and the status field are gone (fdc4a970); five datasets carried the
file (renamed `*.retired-2026-09-09`) and adapters trained from them under it
need retraining. The prior-preservation default that had papered over it is
off unless a corpus is named (b0eacf91).

**Endings are decided by the caption x lyrics pair at render time.** Same
adapter, same seeds: one lyric ended 1/6 under its nearest-tempo training
caption and 4/6 under another training caption; that caption ended 4/6 with
different lyrics too. Plans copy nothing from the training track (no shared
run of four codes). So a training caption used word for word with new lyrics
is a strong style prompt, which is what the caption source picker in Lyric
Studio / Create does (Automatic = nearest tempo). Single-sentence edits flip
a failing pair in both directions; the official templates mention fades in
377/1000 and outros in 723/1000, so there is no banned word. The candidates
loop (3 plans, capped ones dropped) turns a two-thirds per-plan rate into a
96% first-round success.

**Training-side ending levers all failed by ear (2026-09-09):** end-only
score-last 0/6; score-last on all crops 2/6 + early stops, looping, pitch
drift; lyrics dropout 0/6; FAITHSL (bghira-shape recipe + score-last) 6/6 but
zero likeness. `--score-last` / `--score-last-end-only` exist as diagnostics
only. Do not spend GPU re-deriving this.

**Rob's direction, 2026-09-03, still the rule: the correct input is a per-track
`<stem>.mm3.txt` Structured Caption generated in the Training Studio's Enhance
panel with MOSS (local, hears the audio) or Gemini (hears the audio).** Never
suggest renaming ACE sidecar `.txt` files to `.mm3.txt`: the trainer's skip
exists because an ACE caption trains the wrong genre, and the rename also
puts the lyrics in the prompt twice. The server now refuses a run with no
captions at all and says so (2026-09-03).

Historical note: the 2026-08-23/24 sweep found per-song MOSS captions "hardly
worked" and shared captions bound the style better. That verdict predates the
crop fix and the acoustic loss, and Rob's direction above supersedes it.

At generation time the caption should look like a training caption: the
picker's Automatic mode copies the nearest-tempo dataset track's caption
verbatim. You do not have to type the trigger: the app adds it for you.

### The trigger is added for you, and typing it anyway is harmless

`ace-train mm3-lm-train` writes the trigger into the adapter's
`<file>.safetensors.json` sidecar. `readMm3AdapterTrigger` in
`server/src/services/backends/minimax/trigger.ts` reads it, and
`mapMinimaxParams` prepends it to the caption on every MM3 render, in the exact
shape the trainer used: `<trigger>, ` at the front of the caption's FIRST line,
which on a Structured Caption is the `Global Metadata` line.

```
first caption line, as trained  : albumA2, Global Metadata
first caption line, as you type :                Global Metadata
first caption line, as LM sees  : albumA2, Global Metadata
```

It is IDEMPOTENT, case-insensitively (`applyMm3Trigger`), so a caption that
already opens with the trigger is left alone rather than growing a second copy.
Type it or don't; the render is the same either way. The Adapters panel has a
checkbox to switch the injection off if you want to place the trigger yourself.

Two things gate it:

- **The sidecar must say the trigger was trained.** `--trigger` alone writes the
  sidecar without teaching the model anything; `--trigger-prepend` is what
  injects it into the captions. Runs now record `"triggerPrepend": true|false`
  in the sidecar and the server refuses to auto-add an untrained trigger.
  Sidecars written before that field existed are read as trained.
- **ACE is a different code path with different rules.** ACE reads the trigger
  from the adapter's safetensors `__metadata__` (`hot_step_trigger`) and
  `translateParams.ts` applies it WITHOUT `skipPresent`, so on ACE a caption
  that already opens with the artist name really does come out as
  `album b, album b, album b title, ...`. There, start your caption at the
  second item:

```
training caption : album b, album b title, pop punk, bright major-key ...
what you type    :            warning album, pop punk, bright major-key ...
what the LM sees : album b, album b title, pop punk, bright major-key ...
```

Passing `{skipPresent: true}` at the translateParams call site would make this
robust; it is not done today. See
[mm3-captioning](../mm3-captioning/SKILL.md) for the MM3 Structured Caption
format; training and rendering both use the three-section per-track format.

## Lyrics shape matters as much as the adapter

MM3 plans the whole track from the whole lyric, and `duration` only truncates.
Write a FULL song's worth of lyrics even for a 40 s render. Tidy four-line
stanzas produce mainstream rock and suppress the style; irregular line lengths
matching the artist's own writing work far better.
