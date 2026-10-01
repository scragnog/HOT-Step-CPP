# Driving HOT-Step with an agent (MCP)

Two local MCP servers let an agent such as Claude Code or Codex act on the
running app on your behalf, instead of (or alongside) you using the UI.

- **`lyricstudio`** drives Lyric Studio: artists, albums, profiles, lyric
  generation and refinement. It reads the database directly. See
  [tools/mcp-lyricstudio/README.md](../../tools/mcp-lyricstudio/README.md).
- **`hotstep`** drives generation and training across all three backends:
  submitting and polling songs, switching backends, and the dataset/training
  pipeline (datasets, labeling, prepare, start, runs). It talks to the running
  server over its HTTP API, the same one the browser UI uses. See
  [tools/mcp-hotstep/README.md](../../tools/mcp-hotstep/README.md) for the
  full tool and field reference; this page is a shorter tour.

## Setup

The app's server has to be running first; an MCP server with nothing to talk
to can't do anything. The MCP package also needs its own dependencies
installed once, separately from the app's own `install.bat`/`install.sh`,
which don't touch it:

```
cd tools/mcp-hotstep
npm install
```

**Claude Code** (or another client reading `mcpServers` JSON): add this to
the project's `.mcp.json`, adjusting the path to where your checkout lives:

```json
{
  "mcpServers": {
    "hotstep": {
      "command": "node",
      "args": [
        "/path/to/hot-step-cpp/tools/mcp-hotstep/node_modules/tsx/dist/cli.mjs",
        "/path/to/hot-step-cpp/tools/mcp-hotstep/src/index.ts"
      ],
      "env": {
        "HOTSTEP_URL": "http://127.0.0.1:3001"
      }
    }
  }
}
```

**Codex**: add the equivalent to its `config.toml`:

```toml
[mcp_servers.hotstep]
command = "node"
args = [
  "/path/to/hot-step-cpp/tools/mcp-hotstep/node_modules/tsx/dist/cli.mjs",
  "/path/to/hot-step-cpp/tools/mcp-hotstep/src/index.ts",
]
env = { HOTSTEP_URL = "http://127.0.0.1:3001" }
```

`HOTSTEP_URL` defaults to `http://127.0.0.1:3001` if omitted; set it only if
your server runs somewhere else. After editing the MCP server's own source
(not this config), reconnect it in your client: a session already connected
keeps its old tool list until you reconnect it, the same rule as the
`lyricstudio` server.

## Security posture

The app's HTTP API has no real login. Asking it for a token (which both MCP
servers do automatically) succeeds for anyone who can reach it, no password
involved. That is the app's existing single-user, local-machine design, not
something either MCP server weakens further. Anyone who can reach the
server's URL can submit generations and start jobs that occupy the GPU, some
for a long time. Keep it pointed at your own machine (`127.0.0.1`) unless
you have deliberately put something else, such as a firewall or an
authenticated tunnel, in front of it.

## Generation tour

Five tools cover generation, in the order you'd typically call them:

- `gen_backends` lists the three backends, which one is active, and what it
  supports.
- `gen_configure` switches the active backend, or picks a model on a backend
  whose model choice is server-side state (MiniMax-Music3, YuE2). Switching
  releases the outgoing backend's VRAM.
- `gen_submit` starts a generation: a backend, a caption, optionally lyrics
  and the rest of the usual knobs. It names the backend it expects to be
  active; if someone switched backends since you last checked, the server
  answers 409 rather than running on the wrong one.
- `gen_wait` polls a submitted job until it finishes or a time budget runs
  out, whichever comes first.
- `gen_song` looks up a finished song's metadata and audio location once you
  have its id. It never returns audio bytes, only a URL (and, when the MCP
  server runs on the same machine as the app, a file path).

A worked example, generating a short instrumental on whichever backend is
currently active:

1. `gen_backends` to read the active backend's id from the response.
2. `gen_submit` with that id, a caption such as "a calm piano piece", and
   `instrumental: true`. Returns a job id.
3. `gen_wait` with that job id. Once it reports a finished outcome, the
   response includes the finished job's result, including the new song's id.
4. `gen_song` with that song id to get the audio file's URL (and, running
   locally, its path on disk).

## Training tour

Training has more steps because it is a pipeline: a dataset has to be built
and labeled before it can be preprocessed, and preprocessed before it can
train. Starting a prepare or training job spends real GPU time, in some
cases for a long time; a cancel exists (via `train_job`) but a run already
under way does not refund the minutes already spent.

- `train_datasets` and `train_dataset_create` list and create datasets.
- `train_dataset_label` runs the three labeling stages (`label`, `caption`,
  `build`) that turn a folder of audio into a dataset ready to train from.
- `train_prepare` runs whatever per-backend data stage training needs
  first (tensors for ACE, RVQ codes for MiniMax-Music3, latents and a joint
  manifest for YuE2).
- `train_start` starts a training run, and `train_wait` polls it the same
  way `gen_wait` polls a generation.
- `train_runs` lists existing runs and checkpoints for a dataset and backend,
  and reports what is still missing before training can start.
- `train_fields` looks up the exact fields a given backend and stage accepts,
  each with its type, whether it's required and its default: the same
  information the README's field tables hold, but generated from the same
  validation the server runs, so it can't drift from it. Reach for this
  before guessing a field name.

One example per backend family, as which tool to call in which order (see
the README's field tables for what each one actually takes). Every step
below answers with a job id, not a finished result: call `train_wait` on
that id before moving to the next step, call it again if it reports
`outcome: "budget"`, and proceed only once it reports `outcome: "done"`.

- **ACE-Step 1.5**: `train_dataset_label` `stage: "label"` → wait → `stage:
  "build"` (ACE's prepare step refuses an unbuilt dataset) → wait →
  `train_prepare` `backend: "ace"` (encode tensors) → wait → `train_start`
  with `backend: "ace-lm"` or `backend: "ace-dit"` → wait.
- **MiniMax-Music3**: `train_dataset_label` `stage: "label"` → wait →
  `stage: "caption"` for its own caption format → wait → `stage: "build"`
  (its code export also refuses an unbuilt dataset) → wait → `train_prepare`
  `backend: "mm3"` (encode RVQ codes) → wait → `train_start` with
  `backend: "mm3-lm"` → wait.
- **YuE2**: `train_dataset_label` `stage: "label"` for captions → wait →
  `train_prepare` `backend: "yue2", stage: "preprocess"` (no build needed,
  unlike the other two) → wait → `train_prepare` `backend: "yue2", stage:
  "joint-prepare"` → wait → `train_start` with `backend: "yue2-joint"` →
  wait.

## Limits

- Pipelines and batches, checkpoint previews and audition, ladder rung
  scoring, and MiniMax-Music3 streaming and STORM are not exposed yet. Use
  the UI for those.
- A wait tool (`gen_wait`, `train_wait`) always returns once its time budget
  is up, whether or not the job has finished; the job itself keeps running
  on the server either way. Poll again, or raise the budget (and your MCP
  client's own request timeout to match it).
- No audio or other binary data crosses MCP. Generation and training tools
  return URLs and file paths, never bytes.

## Related

- [tools/mcp-hotstep/README.md](../../tools/mcp-hotstep/README.md)
- [tools/mcp-lyricstudio/README.md](../../tools/mcp-lyricstudio/README.md)
- [Generation](generation.md)
- [Backends](backends.md)
- [Training Studio](studios/training-studio.md)
- [ACE-Step 1.5 training](training/ace-step.md)
- [MiniMax-Music3 training](training/minimax-music3.md)
- [YuE2 training](training/yue2.md)
