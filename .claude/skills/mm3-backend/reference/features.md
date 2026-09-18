# Feature subsystems

> Reference for the `mm3-backend` skill. Read only when the task needs it.

## Sampler plugins: shared with ACE (built 2026-08-16 — NOT YET COMPILED OR HEARD)

The same Lua solver / scheduler / guidance plugins that drive the ACE DiT now
drive MM3's flow DiT. No plugin was modified and no plugin API was widened —
the plugin layer never had an ACE dependency (every `lua_call_*` entry point
takes raw `float *` + counts + a param map); what was ACE-specific was the
sampler, not the plugins. Bridge: **`engine/src/minimax/mm3-plugins.h`**.

Two conventions differ, and both mappings are exact:

1. **Time runs the other way.** ACE `t` descends 1→0 with `xt -= vt*dt`; MM3
   `sigma` ascends 0→1 with `x += dsigma*v`. Substituting `sigma = 1-t` and
   `v_ace = -v_mm3` makes them the same expression — *including the terminal
   step*, where MM3's last increment `(1 - sigma[steps-1])*v` is character-for-
   character ACE's engine-owned `x0 = xt - t_curr*vt`. MM3's steps+1 sigma array
   IS ACE's "N timesteps + engine-owned final step".
2. **The latents are transposed.** ACE is time-major `[T][Oc]`, MM3 is
   channel-major `[C=128][L]`. This is NOT cosmetic: `apg_forward` normalises
   per channel over time and indexes `[t*Oc+c]` to do it, so a channel-major
   buffer would be grouped along neither axis. The bridge transposes into the
   ACE view before any plugin sees a buffer. 4 transposes/step of 88k floats
   against two 2.4B forwards — free.

**Opt-in, and that is load-bearing.** Empty selection == the native arithmetic,
expression for expression, so the parity fixtures still cover the default path.
This holds even for `infer_method=euler` (Lua computes in double and rounds on
store; the native loop is float throughout — they can differ in the last ulp).
Server-side the picks only travel when `params.samplerPluginsEnabled` is on,
because ACE defaults solver/guidance to euler + apg — forwarding blindly would
move every MM3 render off the native loop silently, and `guidance_mode: "apg"`
is a genuinely different algorithm from MM3's plain CFG.

**The picks are per backend as of 07b2a5c** (they were one shared global before,
so changing the solver in MM3 mode rewrote ACE's). `BACKEND_SCOPED_FIELDS` in
`ui/src/stores/globalParamsStore.ts` lists them: `inferMethod`, `scheduler`,
`guidanceMode`, `pluginParams`, `backendParams`. Storage key is the bare `hs-*`
for `ace` and `hs-*@<backendId>` for anything else; a backend's first visit
seeds from the bare key, and `useBackendStore.subscribe` re-hydrates the store
on every switch. Anything writing one of those keys **outside the setters**
(`utils/paramProfiles.ts`, `sendAuditionToCustomGen.ts`) must route through the
exported `scopedKey()` — a bare-key write lands where the live store will never
read it back, and the next backend switch silently undoes it.

Not supported: **owns_loop solvers** (11 of them, mostly MDMAchine's — they'd
bypass the per-step overlap blend and break every window seam; engine warns and
falls back, and `SamplerPluginControls.tsx` filters them out of the dropdown),
**postprocess plugins** (they replace ACE's tiled VAE decode), **`composite:`
schedulers** (built by `sampler-schedule.h` from ACE's own globals), and the
**legacy sideband params** (`stork_substeps` etc., read from `g_hotstep_params`).

State resets **per window**, not per song: one 200-frame window == one ACE
"generation", so `MM3PluginRun::begin_window()` resets APG momentum and solver
history. Momentum leaking across a seam would smear one window's guidance
history onto the next one's first step, right where the overlap machinery is
trying to hide the join. Stochastic solvers are the open risk — they inject
noise that fights the overlap blend; test the seams by ear.

Wire fields (mm3-request.h, all optional): `infer_method`, `scheduler`,
`guidance_mode`, `flow_shift` (default 1.0 = MM3's own hardcoded shift; only
consulted with a scheduler plugin), `apg_norm_threshold`, `plugin_params`.
`POST /mm3/synth` echoes `sampler_plugins` back **only when something was
selected** — that is how you tell "I picked a solver" from "a solver ran".
Capability: `features.samplerPlugins` (distinct from `features.plugins`, which
selects WHICH Generation dropdown renders — MM3 needs the generic one for its
steps/cfg knobs, so it is `plugins: false, samplerPlugins: true`).

**Status: written, TypeScript type-checks clean on both tiers, C++ NOT compiled
and no audio generated.** Validate in this order: (1) `dev-rebuild.bat`;
(2) a generation with no plugins selected, confirming it is bit-identical to a
pre-change render on the same seed — that is the parity guarantee, and it is
the only claim here that can be checked without ears; (3) `linear` scheduler
alone, which should be near-identical to native (MM3's schedule IS shift=1
linear, differing only in float32 linspace rounding); (4) a solver, listening
specifically at window seams on a >8 s clip.

## Caption composer: plain English -> Structured Caption, no LLM (SHIPPED 2026-08-22, 3f80d84f)

`POST /api/lireek/mm3/compose` turns a plain-English brief into an MM3
Structured Caption by **selecting prose from MiniMax's own 1,000 reference
captions** — no model, no provider, no network. Same brief + controls + seed is
byte-identical; a new seed gives a different take in the same genre.

Why not an LLM: MM3 reads the caption's PROSE, not its genre label. A caption
whose Basic Attributes said "Hardcore Punk." rendered as southern rock every
seed because an LLM had invented southern-rock vocabulary for the body.
Selecting real prose makes that unreachable — the composer can only emit words
the target genre's templates contain.

| Piece | Where |
|---|---|
| corpus build (1019 upstream files -> one 4.3 MB JSON) | `server/scripts/build-mm3-corpus.mjs` |
| committed corpus (`.claude/` is absent in a release) | `server/src/data/mm3-corpus.json` |
| route / parseBrief / resolveSlots / compose | `server/src/services/lireek/mm3Compose.ts` |
| endpoints | `server/src/routes/lireek/mm3Routes.ts` |
| 51 self-checks | `cd server && npx tsx scripts/check-mm3-compose.ts` |
| UI (caption box IS the brief box) | `ui/src/components/create/Mm3ComposeButton.tsx` |

Slot precedence is **explicit control > brief prose > corpus default**; on
conflict the control wins AND the conflict is surfaced. Re-run
`build-mm3-corpus.mjs` after any upstream refresh.

### Three Create-page controls have NO path to MM3 — do not re-derive this

| Control | Why it cannot reach the model |
|---|---|
| **time signature** | Zero hits for `time_?sig\|signature` anywhere in `engine/src/minimax/`. The caption cannot carry it either: 26/1000 reference captions state a meter, all inside Groove prose, never in Basic Attributes — and they are 4/4 x24, 3/4 x1, 6/8 x1. **Hidden in MM3 mode.** |
| **language** | MM3 has **no language input at all**. The tokenizer is a byte-level Qwen3/GPT-2 BPE (`mm3-tokenizer.h:17-20`), so any UTF-8 encodes and the language simply follows the characters of the lyrics. No reference caption states a language. **Relabelled "Lyrics Language".** |
| **duration** | A real wire param (`generate.ts:277`) but a **CEILING, not a target**: `max_frames = min(round(duration*25), 9000)`, the AR loop breaks there, and EOS can fire earlier — `mm3-ar-loop.h:98-99` records a 7500-frame request stopping naturally at 1200. A short render means EOS fired; a song that seems to "want to be longer" is being **truncated at your number**. |

**bpm and keyScale are also ignored on the MM3 wire** (`generate.ts:192`) — they
were dead knobs until the composer started writing them into Basic Attributes.
`vocalGender` is deliberately NOT in the generation request for the same reason:
no backend has a wire field for it; it travels inside the caption.

### Gender pools are thin in guitar genres

metal-heavy-rock has **2** female captions of 78; hip-hop-rap **2** of 74;
country-americana 6 of 50. A thin family borrows its **vocal columns only** from
the family its own cards name under `Secondary routes`. Do NOT "fix" this by
gender-flipping male templates (tenor->alto) — that invents prose the corpus
never had, which is the exact failure the composer exists to prevent.

### Not yet heard

Genre fidelity is measured (8/8 routing, 0 out-of-distribution terms in composed
punk captions). **Nothing has been rendered.** The ear test is still the bar.

## Lyric timestamps: use Whisper, not attention (measured 2026-08-14)

ACE derives LRC from its **DiT's lyric cross-attention**. MM3 has no analogue:
its flow DiT has no cross-attention and never sees lyrics — conditioning is
channel concatenation from the condition encoder. The only place lyric tokens
and audio frames coexist is the **LM decode loop**, so that was probed
(`MM3_ALIGN_DUMP=1`, `MM3_ALIGN_FILE=<path>`; forces the manual F32 attention
path because flash fuses the softmax away).

Result: **viable at LINE level, which is the granularity that matters.**
`lrc_align()` emits lines ("consensus → DTW → sentence grouping → LRC text"),
not words, so line onset is the bar. Three heads track the lyric across every
test clip — **L12/H27, L19/H7, L24/H29** — and a naive DTW over their consensus
gives median line-onset error **0.83 s (indie)** and **0.71 s (synth)**, all
lines within 2 s. Folk was inconclusive (only 2 lines matched, and Whisper
itself renders that clip as a single 20 s segment).

Errors skew consistently NEGATIVE — attention leads the audio by ~0.6–0.8 s,
which is expected (the LM attends to a token as it begins generating that
content) and is a constant offset worth calibrating out, not noise.

Do NOT judge this at word level. An earlier pass did, called it unusable, and
was wrong twice over: it compared "fraction through BPE tokens" against
"fraction through words" — curves that differ even for a PERFECT alignment,
because tokens-per-word varies — and it used a naive DTW over 3 heads rather
than `lrc_align()`'s consensus denoising with ACE's 7-head-scale config.

Traps, each of which produced a wrong answer first:
1. **55 % of all 1152 heads sit permanently on one structural token.** Any head
   ranking must reward MOVEMENT — scoring "monotonic" as `delta >= 0` counts a
   pinned head as perfectly monotonic (it scored 0.98 and ranked first).
2. **Single-clip results do not generalise.** The best head on one clip (L16/H13)
   did not make the top 14 across three. Rank by the WORST clip, never the mean.
3. **Whisper `base` is not adequate ground truth for sung vocals** — it returned
   11 word timings for a folk clip `large-v3-turbo` transcribes as one 10-word
   line, which made attention look 7 s wrong. Validate against `large-v3-turbo`.
4. **Capture requires the manual attention path on EVERY layer, not just the
   three that are read.** This is the expensive, counter-intuitive one. Leaving
   the other 33 layers on flash is cheaper (9.8 vs 11.2 ms/step) and produces
   identical audio — but WRONG alignment: same seed, selective capture stamped
   the lines at 0.9/3.6/7.5/10.3 s where the vocal sits at 0.1/10.2/15.1/19.4,
   while all-manual gave 2.2/9.9/16.5/19.5. The captured layers' attention
   depends on whether the layers feeding them ran flash or manual by far more
   than that difference is supposed to matter. Not understood; do not "optimise"
   it back without re-validating against Whisper.
   Net cost WAS ~+49 % on the LM step — **superseded 2026-08-21 by the
   post-hoc replay pass** (`mm3_lm_lrc_replay`, mm3-lm-graph.h): decode runs
   pure flash (audio bit-identical to a no-LRC render, verified) and the
   alignment attention is recomputed afterwards from the sampled codes —
   teacher-forced 256-query chunks, all-manual, blocks 0..24 only, single CFG
   row via ne3=1 views onto KV row 0 — at ~0.1-0.5 s per song. Causality makes
   prefill attention identical to decode attention over the same tokens, and
   all-manual keeps it out of the mixed-graph trap above; LRC verified
   character-identical to the live path on the same forced codes.
   `MM3_LRC_LIVE=1` restores live capture (validation/fallback).
5. The dump is **MM3ALIGN2**: an ASCII header line, then `tokens` × int32 lyric
   token ids, then f32 in `[frame][layer][head][token]` order. v1 omitted the
   ids, which is what forced the bogus token-progress-vs-word-progress
   comparison. Resolve ids to text via `tokenizer.ggml.tokens` in the LM GGUF.

**SHIPPED** (`features.lyricTimestamps: true`, request field `get_lrc`). The
engine emits LRC on `Job::result_lrc` → the existing `x-lrc-text` header → the
server writes `<uuid>.lrc`, same contract ACE uses. Measured against Whisper:
median line error **0.39 s** over the matched lines of two clips (synth
+0.20/+0.52/+0.62/−0.02, indie +2.12/−0.26/+1.36/+0.14).

`lrc_align()` is NOT used, and that was measured rather than assumed: fed the
same captured attention it stamps every line in the first half of the song, and
sweeping violence_level 2.0 → 0.0 changes nothing. A plain monotonic DTW over
the head-averaged matrix, grouped on newline tokens, is what works — see
`mm3-align.h`. `lrc_align()` stays untouched for ACE.

## Analysis tool: where the vocals sit (2026-09-10)

`tools/vocal-end/vocal_end.py <audio...>`: SuperSep (engine, level 4) vocal stem -> energy-based vocal activity
(first/last vocal, tail, share, gaps; the reliable signal) + whisper-cli per-word on the stem (hallucinates on stems;
lyric-repeat hint only). Needs the engine up. The MM3 LRC is a forced alignment and cannot answer "did the singing
stop"; this can. README in the folder.
