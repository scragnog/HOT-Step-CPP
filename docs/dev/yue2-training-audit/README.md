# YuE2 training audit

Why YuE2 adapters can train cleanly and still miss the album, and which changes would make
the two shipped training paths comparable. Written 2026-11-09 against commit `be818f66`, from
the code only. No training run was executed and no source file was changed.

This page is deliberately long. Each finding has four parts: what the code does (quoted), what
that means mechanically, why it produces the symptom a listener hears, and what would confirm
it. Section 10 lists the command that re-derives every quantitative claim, and section 8 defines
how a change becomes a measured improvement instead of a plausible one.

## How to read the evidence

| Label | Meaning |
|---|---|
| Verified | Read in the named file at the named line |
| Derived | Arithmetic from verified shapes and defaults, no run |
| Likely | Follows from the code, but needs a listening test or run log |
| Open | Depends on behaviour not yet traced |

Line numbers are from `be818f66` and will drift. Quoted code is verbatim, trimmed with `...`
where noted. File:line references are the evidence; do not take a claim on this page without
following one.

## The problem in one page

The user-facing promise is narrow and reasonable: point the trainer at an album, get an
adapter, and hear that album in the output. Three things have to line up for that to happen:

1. the adapter must be able to change the part of the model that carries the thing you want
   (timbre, arrangement, diction),
2. it must be optimised toward the same thing you will later judge it on, and
3. it must see the same prompt at training time that it will see at render time.

The joint trainer is the only method the Training Studio now offers. It trains two adapters at
once, one for the planner (AR) and one for the decoder (NAR), and it is a careful
implementation: the losses are the report's, the sampler reproduces the reference noise path,
and the export writes tensors the existing loader can merge.

The problem is not that it is wrong. It is that it is *narrower than the thing it is compared
against*, and the comparison is invisible in the UI:

- It can move 4 fused matrices per block on each half and nothing else. The retained NAR-only
  endpoint can also move the four flow heads, which sit at the input and output of the decoder.
  Those are exactly the places a timbre adapter would want.
- The two presets on the same card optimise different objectives, not just different step
  counts. Base-matched drops the KL anchor, drops caption dropout, sets `arTargets` to `base`
  and weights the planner at 0.25. The fixed recipe keeps all four the other way.
- The default ranks and adapter types across the four entry points are 32/32, 64/256, 128/128
  and 256/256. The same alpha field means scale 1, 2 or 4 depending on what sits beside it.
- The render path composes one style sentence for two adapters, driven by the NAR trigger. A
  joint AR adapter whose trigger differs is merged but never addressed. Joint exports also do
  not record their CoT modes, so the renderer cannot warn about a CoT mismatch.

So an A/B between "Balanced" and "HOT-Approved", or between a joint adapter and an older
NAR-only adapter, changes five to eight variables at once. The listening result is real, but it
is not evidence about any one of them. That is the finding this page documents.

The same gap applies to judging the result. A run logs its planner KL, its reconstruction error
and its decoder drift, and those say whether training converged, not whether the output sounds
more like the album. Because the two shipped paths change five to eight objectives at once, even
a careful listen cannot price one of them. That is what section 8 is for: without a fixed panel,
a control arm and a decision rule, every adjustment stays an opinion with a loss curve attached.

## 1. What one training step actually does

### 1.1 Two forwards, three loss terms

The joint step runs the AR half through the AR adapter and the NAR half through the NAR adapter
in the same update:

```c
// engine/src/train/yue2-aitk-joint-step.h:140, 267
yue2_aitk_stack::forward(backend,model,&state.ar_adapters(),false,embeds.values, ... ,&ar, ...);
yue2_aitk_stack::forward(backend,model,&state.nar_adapters(),true,frontend.values,frames+2, ... ,&nar, ...);
```

The planner is scored with cross-entropy against the frozen base's distribution, plus a KL
anchor. The decoder is scored with flow-matching MSE. The weights are not applied to the loss
value and left there; they are folded into the gradient:

```c
// yue2-aitk-joint-step.h:146-149
// The head kernel's gradient is d(CE + kl_weight * KL). --ar-loss-weight w
// wants d(w * CE + kl_weight * KL) = w * d(CE + (kl_weight / w) * KL): the
// kernel runs at kl_weight / w and the hidden gradient is scaled by w.
```

So the objective is

```
L = ar_loss_weight * CE_ar + kl_weight * KL_ar + MSE_flow_nar + cursor_weight * CE_cursor
```

and the reported composite agrees (`engine/src/train/yue2-aitk-runtime.cpp:1527`):

```c
const double composite = metrics.ar_ce + (double) config.kl_weight * metrics.ar_kl
                       + metrics.nar_mse + metrics.cursor_ce * (double) cursor_weight;
```

The important detail is that `ar_loss_weight` does not scale the total loss in a way Adam
would cancel. Because Adam normalises per-parameter step size, a global scale does nothing;
what this weight changes is the ratio of planner to decoder gradient *before* the shared clip,
and therefore which half consumes the 1.0 clip budget. The config comment says this outright
(`yue2-aitk-runtime.h`, the `ar_loss_weight` field). A 0.25 to 0.5 change is a real change in
who gets the step, not a cosmetic one.

### 1.2 What a "site" is, and why it is fused

The adapter dims are fixed at construction:

```c
// engine/src/train/yue2-aitk-train-state.h:52
const Yue2AitkDims dims{2048, 2048, 1024, 6144};   // hidden, q, kv, feed-forward
```

Every layer gets four sites, and each site carries one delta over a fused base matrix
(`yue2-aitk-lora.h`, `yue2_aitk_make_expert_adapters`):

```c
// yue2-aitk-lora.h, the four make_site calls
make_site(... dims.hidden, dims.q_width + 2 * dims.kv_width, ... prefix + "attn_qkv");
make_site(... dims.q_width, dims.hidden,                     ... prefix + "attn_output");
make_site(... dims.hidden, 2 * dims.feed_forward,           ... prefix + "ffn_gate_up");
make_site(... dims.feed_forward, dims.hidden,               ... prefix + "ffn_down");
```

| Site | Base shape (in to out) | Contains | LoRA params at r=64 |
|---|---|---|---|
| `attn_qkv` | 2048 to 4096 | q 2048, k 1024, v 1024 | 393,216 |
| `attn_output` | 2048 to 2048 | o | 262,144 |
| `ffn_gate_up` | 2048 to 12288 | gate 6144, up 6144 | 917,504 |
| `ffn_down` | 6144 to 2048 | down | 524,288 |
| **Per layer** | | | **2,097,152** |

There are 28 layers (`kLayers`), so one half holds 58,720,256 parameters and the AR+NAR pair
holds 117,440,512, about 470 MB in F32 before gradients or optimizer state. That is derived
arithmetic (`r * (in + out)` per site), not a measurement.

The fusion matters for capacity and it is why the export needs a bridge. One rank-64 A factor
is shared by q, k and v, so the adapter cannot give q a different low-rank basis from k. At
export the fused B is cut by row into ordinary per-projection tensors:

```c
// yue2-aitk-native-adapter-io.h:264-272
if (site == "self_attn.qkv_proj") {
    if (!add_pair(out, native + "attn_q", "", *a, *b, rank, 0, 2048, m) ||
        !add_pair(out, native + "attn_k", "", *a, *b, rank, 2048, 1024, m) || ...
} else if (site == "mlp.gate_up_proj") {
    if (!add_pair(out, native + "ffn_gate", ... , 0, 6144) || ...
```

and `add_pair` duplicates the same A pointer into each destination while slicing B:

```c
// yue2-aitk-native-adapter-io.h:63-72
out.push_back({"yue2." + module + ".lora_A.weight", ashape, a.data, -1});
out.push_back({"yue2." + module + ".lora_B.weight", bshape, b.data + first_row * rank, -1});
```

Net: the adapter trains one delta per fused matrix, the export presents it as the separate
native sites the loader expects, and the shared A constraint is real but invisible in the
output file. A LoRA file cannot express "q and v need different bases"; it can only express
different output rows. That is a capacity limit the UI does not state.

### 1.3 What is deliberately outside the adapter

Two omissions are by design, and the difference between them is the subject of section 2.

The AR output head is out because the graph builds its gradient by hand:

```c
// yue2-ar-train-graph.h:136-141
// - No `proj` tier. `vae2llm` / `llm2vae` / `time_embd.{0,1}` belong to the flow
//   head and have no part in the AR forward. An AR adapter that names them is
//   malformed.
```

`token_embd` and `output` are also excluded: the chunked cross-entropy computes `dL/dh` itself,
so a trainable head would need a second gradient path through that seam. That is the same
choice the standalone AR trainer makes, and it is defensible.

The NAR flow heads are different. The inference loader treats them as ordinary adapter sites:

```c
// engine/src/yue2/yue2-adapter.h:73-76
//   NAR  196 block sites (28 x `nar_attn_{q,k,v,output}`, `nar_ffn_{gate,up,down}`)
//        + 4 flat flow heads (`vae2llm`, `llm2vae`, `time_embd.{0,1}`) = 200
//   AR   196 block sites (28 x `attn_{q,k,v,output}`, `ffn_{gate,up,down}`)
//        and nothing flat — the flow heads have no part in the AR forward
```

So the loader is ready for 200 NAR sites. The joint trainer produces 196. That gap is section 2.

## 2. Finding 1: the joint adapter cannot reach the flow heads

### Observation

`Yue2AitkTrainState::initialize` builds the AR and NAR adapters from the same four-site maker,
for LoRA, LoKr and DoRA alike (`yue2-aitk-train-state.h:56-66`). Nothing in the joint trainer
names `vae2llm`, `llm2vae`, `time_embd.0` or `time_embd.1`.

The retained NAR-only endpoint does name them. Its `nar_attn_mlp_proj` preset is the server
default for a NAR run (`server/src/services/training/yue2Train.ts:199`), and the engine adds
exactly four head sites:

```c
// engine/src/train/yue2-nar-train-graph.h:803-806
mk(&ad->vae2llm,  LD, H, "vae2llm");    // 64 x 2048
mk(&ad->llm2vae,  H, LD, "llm2vae");    // 2048 x 64
mk(&ad->time0,   256, H, "time_embd.0"); // 256 x 2048
mk(&ad->time1,     H, H, "time_embd.1"); // 2048 x 2048
```

The same file records why that preset is the one used for measurements
(`yue2-nar-train-graph.h:702-710`), and `yue2Train.ts` records that the campaign's ear-tested
renders were `nar_attn_mlp_proj`, rank 256, 10,000 steps.

### Mechanism

Those four tensors are the decoder's interface to the latent space, not decoration:

- `vae2llm` maps a 64-dim latent frame into the 2048-wide model at the decoder input.
- `time_embd.0` and `time_embd.1` turn the flow timestep into a 2048-wide conditioning vector
  added to that input. Every denoising step is conditioned through them.
- `llm2vae` maps the final 2048 hidden state back to the 64-dim velocity the sampler integrates.

With only block sites adapted, the joint adapter can change what happens *inside* the stack but
not how audio enters, how the timestep is represented, or how velocity leaves. A NAR-only
adapter at `nar_attn_mlp_proj` can change all three. That is why "same rank, same dataset, same
steps" does not make the two adapters comparable: they do not have the same degrees of freedom.

### Why it produces the symptom

Timbre complaints are usually about the spectral envelope and the transient shape, which are
output-side properties. `llm2vae` is the last linear map before those properties exist. It is
plausible that a large part of what users call "likeness" lives in the output head plus the
last blocks, and the joint trainer can only reach the blocks. That is a hypothesis, not a
measurement, and section 8 gives the test that settles it.

### The companion interaction

Every YuE2 pack ships a tokenizer companion adapter, merged under every generation. The joint
trainer loads it frozen (`yue2-aitk-runtime.cpp:743` prints `(frozen)`), and the model code is
specific about what it does (`engine/src/train/yue2-aitk-model.h:240-247`):

```c
// its block LoRA becomes a frozen branch on every decoder linear ...,
// llm2vae takes its full F32 replacement ..., and its vae2llm bias are
// overwritten. Nothing here is a parameter.
```

So at render time the flow heads are already the companion's, not the base's. The NAR-only
trainer still trains those heads with `--companion` on (`yue2-nar-train-run.h:137-140`), which
means the renderer already composes an adapter delta with a replaced `llm2vae` somehow.

The merge order is settled in code. The companion goes on first, then user adapters:

```c
// engine/src/yue2/yue2-model.h:1638-1640
    // the decoder, and user NAR adapters are trained on top of it.
```

The adapter loop follows at `:1643-1668`. So a joint adapter delta on `llm2vae` would merge
onto the companion's replacement weight rather than be discarded by it. That removes the
plausible defence for excluding the heads: the exclusion is a trainer-side allocation choice,
not a format limitation. The NAR-only path's measurements are consistent with this order,
because it trains those heads with the companion already applied.

### What to do

Either add the four heads to the joint trainer behind a flag, or declare the restriction and
surface it. The loader already accepts the sites, so this is a trainer-side allocation and a
graph change, not a format change. Adding them to DoRA and LoKr needs the same per-parameter
handling the other sites get.

Whatever is chosen, the adapter card should say which sites the file contains. Today a user
cannot tell a 196-site joint NAR adapter from a 200-site NAR-only one by looking at it.

## 3. Finding 2: the two joint recipes train different objectives

Both paths are reachable from the same card. The values below are the shipped constants, not
defaults inferred from the UI.

| Setting | Base-matched (Legacy, Fast, Balanced, Thorough) | Fixed recipe (HOT-Approved) |
|---|---|---|
| AR targets | `base` | `tuned` |
| AR loss weight | 0.25 | 0.5 |
| KL weight | forced 0 | 0.2 |
| Caption dropout | forced 0 | 0.5 |
| Text / lyric / both dropout | 0.1 / 0.1 / 0.1 | 0 / 0 / 0 |
| ABC dropout | 0.5 | 0.5 |
| Adapter | card choice, dim-128 LoKr except Legacy at 256 | rank-64 LoRA, alpha 64 |
| Updates | Fast 100, Balanced 200, Thorough 300, Legacy 500 | 1500 |
| Songs per update | 4, except Legacy at 1 | 1 |
| LR, weight decay | 1e-4, 0.1 | 1e-4, 0.01 |
| Beta2 | 0.95 | 0.999 |
| Schedule | cosine-floor, floor 0.1, warmup 3% | constant, no warmup |
| Planner LR scale | 1 | 0.5 |
| Drift meters | `--nar-drift` on | off |

Sources: `BASE_MATCHED_FORCED` and `BASE_MATCHED_DEFAULTS` in
`server/src/services/training/yue2JointTrainRunner.ts:341-370`, `applyFixedRecipe` at
`:382-400`, `YUE2_JOINT_FIXED_RECIPE_DROPOUTS` in
`server/src/contracts/trainingRecipes.ts:352-354`.

### What each difference does to the gradient

- **`arTargets`.** `base` removes the tokens the model is never asked to predict:
  MUSIC_START always, and ABC_END when the lead sheet is dropped
  (`engine/src/train/yue2-aitk-batch.h:183-205`). `tuned` keeps them. The fixed recipe therefore
  spends part of the planner's budget on structural markers that base-matched does not.
- **`ar_loss_weight`.** As in section 1.1, this changes the planner's share of the shared clip.
  0.25 against 0.5 is a twofold difference in how much of each update the planner takes.
- **KL.** Base-matched sets it to 0, so there is no anchor to the base planner and no
  teacher forward. The fixed recipe keeps 0.2. With KL on, the planner is pulled toward the
  base distribution while it learns the album; with it off, it is free.
- **Caption dropout.** The sampler picks the `*_nocap` prefix when the draw fails
  (`yue2-aitk-batch.h:96-97`), which requires a dataset prepared with trigger-only prefixes
  (`:219`). The fixed recipe trains half its rows with the style reduced to the trigger, so a
  render with a full caption is only in distribution for the other half. Base-matched never
  does this, so a render with the bare trigger is out of distribution for it.
- **Condition dropouts.** Base-matched's 0.1 each means about 30% of steps train with text,
  lyrics or both missing. The fixed recipe never degrades the prompt this way. Two adapters can
  therefore disagree about how much the lyrics are supposed to matter.

### Worked example on a 15-track album

Derived from the constants above, not measured:

| | Balanced | HOT-Approved |
|---|---|---|
| Updates | 200 | 1500 |
| Songs per update | 4 | 1 |
| Song draws | 800 | 1500 |
| Draws per track | 53 | 100 |
| Decoder window | 60 s crop | 60 s crop |
| Planner window | whole song | whole song |
| Planner share of clip | 0.25 CE, no KL | 0.5 CE + 0.2 KL |
| Prompt mix | 70% full, 10/10/10 degraded | 50% full, 50% trigger-only |

That is why a listening comparison between the two presets says little about either one. They
differ in the number of examples, the objective balance, the anchoring, the prompt
distribution and the optimizer schedule.

### Dead stop values

`YUE2_JOINT_LORA_STOP` sets `targetKl 1.4`, `plannerLrScale 0.3` and
`YUE2_JOINT_LOKR_STOP` sets `targetKl 1.0`, `plannerLrScale 0.6` (`trainingRecipes.ts:360-362`).
`BASE_MATCHED_FORCED` then sets `targetKl` undefined, `stopMode` steps, and both LR scales to 1
(`yue2JointTrainRunner.ts:347-355`). Under the presets none of those numbers can take effect.
They survive only for the retired tuned path. Either delete them or stop overriding them, so
the values a preset displays are the values the run uses.

## 4. Finding 3: rank, alpha and type are not comparable across entry points

| Entry point | Adapter | Default | Effective scale |
|---|---|---|---|
| Joint engine | LoRA | rank 32, alpha 32 | 1.0 |
| Joint card | LoKr | rank 64, alpha 256, dim 128, factor 4 | 2.0 on factorised sites, 1.0 on monolithic |
| Joint fixed recipe | LoRA | rank 64, alpha 64 | 1.0 |
| NAR-only endpoint | LoRA | rank 256, alpha 256 | 1.0 |
| AR-only endpoint | LoRA | rank 128, alpha 128 | 1.0 |

For LoRA the delta is `(alpha / rank) * B A`, and the engine builds the site with exactly that
scale (`yue2_aitk_detail::make_site`, `yue2-aitk-lora.h:144-160`). For LoKr it is `alpha / dim`,
except that monolithic sites are forced to 1, the LyCORIS K6 rule:

```c
// yue2-aitk-lora.h, make_lokr_site
const float a_eff = k.mono ? (float) dim : alpha;
k.scale = a_eff / (float) dim;
```

Two consequences:

- The joint card's shared alpha field defaults to 256. Switch the type from LoKr to LoRA without
  touching alpha and a rank-64 LoRA would train at scale 4. Whether the card rewrites alpha on
  a type switch is not established here; the effective scale should be shown next to the field
  either way, because the number the user reads is not the number the optimizer applies.
- The engine rewrites alpha to `lokr_dim` when alpha was not passed explicitly
  (`yue2-aitk-runtime.cpp:463-464`). So "alpha unset", "alpha 128" and "alpha 256" are three
  different runs at dim 128, and only the first two coincide.

Scale also has to travel with the file, because merging uses it. An adapter whose header is
missing or wrong merges at the wrong strength; the DiT trainer has a hard post-check for
exactly this failure (`server/src/services/training/trainDitRunner.ts:421-424`), which is the
precedent for treating a missing scale as a failed export rather than a warning.

The resume binding is a separate trap. `ar_targets` is refused on a mismatch
(`yue2-aitk-runtime.cpp:211`) and `ar_loss_weight` is accepted with a note (`:212`), while rank,
alpha and adapter type are part of the same fingerprint (`yue2AitkRuns.ts:304-306`). A ladder
that changes parameterisation cannot resume. That is correct behaviour, but it should be said
before the run starts.

## 5. Finding 4: the prompt the adapter saw is not always the prompt it gets

This is the most likely cause of "it trained but it does not sound like the album", ahead of
site coverage. The two sides are built in different places and only some mismatches are
guarded.

Training side:

- The prefix is chosen from prepared variants, and caption dropout swaps to the trigger-only
  variant (`yue2-aitk-batch.h:93-97`).
- The style template is either `upstream` (a sentence) or `bare` (`<trigger>, <caption>`), and
  the choice is written into the adapter header (`yue2ArTrain.ts:91-95`).
- The genre/BPM/key tail is part of the trained sentence, built from the dataset sidecar.
- ABC dropout 0.5 means half the rows train without the lead sheet, so the adapter is expected
  to serve `cot=full` and `cot=off`.

Render side:

```c
// server/src/services/backends/yue2/generate.ts:134-145
// Both YuE2 trainers only ever showed the model a trigger INSIDE the style
// sentence ... sending the caption box verbatim is off-distribution — the reason
// adapters capped and sang weakly until the template x adapter x seed grid of
// 2026-09-14 pinned it.
```

The renderer now composes the sentence from the adapter's own header. Four gaps remain:

1. **One sentence, two adapters.** The template carries one trigger, and the NAR slot drives it
   (`generate.ts:172-180`). If the AR adapter was trained on a different dataset, its trigger is
   absent from the prompt and the adapter is merged but not addressed. The renderer logs a note
   (`:192-195`); it cannot fix it.
2. **CoT is not recorded by joint exports.** `generate.ts:186-188`: "A file that does not record
   its CoT modes (joint exports do not yet) is unknown, never guessed as off." So the renderer
   cannot warn that a joint adapter trained with `abc_dropout 0` is being used with `cot=off`.
   Fixed-recipe adapters always have both arms, so this is mostly a legacy-run risk.
3. **Absent template means `bare`.** `generate.ts:216-219` treats a missing field as the old
   behaviour rather than guessing `upstream`. Correct, but it means an old file silently gets
   the bare grammar.
4. **Empty caption.** `generate.ts:259-267` warns whether the adapter trained for a bare trigger
   or not. The opposite case, a long caption for a caption-dropout adapter, has no warning.

Mismatch matrix:

| Adapter trained on | Render sends | What the model sees | Expected symptom |
|---|---|---|---|
| Trigger-only 50% | Bare trigger | In distribution | Adapter responds strongly |
| Trigger-only 50% | Long style sentence | Half-known | Weak likeness, caption bleed |
| Full caption only | Long style sentence | In distribution | Normal |
| Full caption only | Bare trigger | Unknown short prompt | Weak or generic output |
| Upstream template | Bare composition | Wrong sentence shape | Capping, weak singing |
| `cot=full` only | `cot=off` | Missing lead-sheet regime | Structure drift, no warning |

**What would confirm it:** the run log's `[YuE2] Style sent:` line plus the adapter header's
`style_template`, `caption_dropout` and `cot`. If the three disagree, the listening result is
about the prompt, not the adapter capacity.

## 6. Finding 5: the latent cache is readable but not inspectable

The cache is well specified, and that makes the absence of tooling more frustrating than the
format warrants.

```c
// engine/src/train/yue2-preprocess-run.h:667-669
// A cache file is frames * latent_dim raw f32, frame-major. Its byte size is
// therefore the whole of its metadata: a short or misaligned file is a torn
// write from an interrupted run and is re-encoded rather than trusted.
```

Details that matter when reading a cache by hand:

- Path is `latents/<safe name>_<key>.f32`, where the key hashes file size, mtime, VAE variant
  and loudness tag (`:1186-1193`). A re-cut album gets new filenames; nothing is overwritten.
- The VAE returns channel-major, and preprocessing reorders to frame-major before writing
  (`:1291-1298`), so frame `t` occupies bytes `t * 64 * 4` to `(t + 1) * 64 * 4`.
- `.gain` beside it is plain text, written as `%.6f\n` (`:1307`) and read with `%lf` (`:1199`).
  It is always present, 0 included, and holds either the LUFS gain or the peak-scale gain.
- The loudness tag is `|peak` when normalization is off and `|lufs%.2f` when it is on
  (`:1186-1189`), so two caches of the same album can sit side by side and differ in level.
- The server cuts joint caches at `YUE2_CACHE_LUFS = 0` (`yue2Train.ts:566`). Tracks that peak
  past full scale are scaled to -0.3 dBFS, not clipped (`yue2-preprocess-run.h:1254-1257`).
- Frame count is derived from byte size and refuses a non-multiple (`:670-681`), which is why a
  torn write is silently re-encoded rather than trained on.

What exists to inspect it: `yue2-probe --vae-parity <fixture-dir>` decodes fixture latents, and
`--vae-roundtrip <in.f32> <out.f32>` takes *audio* in and writes audio out
(`engine/tools/yue2-probe.cpp:17, 123, 5622`). There is no command that takes a cached
`latents/*.f32` plus its `.gain` and writes a WAV, and nothing in the UI surfaces per-song clip
offsets, frames or gain.

**Why this matters:** a user cannot hear what the adapter is being shown. If a track was
peak-scaled by 6 dB, cut at a different offset, or encoded from a different VAE variant, the
only evidence is a log line they would have to know to look for. A decode command would turn
most cache questions into a listening test.

## 7. Finding 6: mix correction mostly exists, the gap is narrower than it looks

| Need | Existing tool | Gap |
|---|---|---|
| Match tone and level to a reference | Matchering mastering (`routes/mastering.ts`) | None |
| Broad tonal correction | VST3 chain, presets, live monitor | No native parametric EQ |
| Denoise, extend highs, tame transients | Spectral Lifter | None |
| Clean fizz, second autoencoder pass | PP-VAE | None |
| Vocal artefacts | Vocal Naturalizer | None |
| Loudness target | LUFS normalizer with peak limiter | None |
| Diagnose metallic, word cuts, noise | Quality evaluator | No cause classification |
| Rebalance instruments | Stem Separator plus mixer | No stem remix stage inside post-processing |

So "fix the mix" is already possible with a reference track and the VST chain. What is missing
is a native EQ and a way to say whether an artefact is in the recording or in the model. The
order matters: measure in Compare, correct with VST or matchering, and build a native stage only
if the same correction is being repeated. If the artefact survives those, it is a decoder or
latent problem, and section 9 is the lever.

## 8. Making improvement measurable

### Why the current loop cannot show improvement

A run already logs three numbers per checkpoint: the planner's KL to the base, the decoder's
reconstruction error, and the decoder's drift from base. The user guide says so directly
(`docs/user/training/yue2.md`):

> Every joint run, whatever the preset, logs the planner's KL to the base model at every update
> and the decoder's reconstruction and drift meters at every checkpoint, so runs can be compared.

Those meters answer "did the adapter move, and did training converge". They do not answer "is
the output closer to the album", for three reasons:

- **Fit is not likeness.** Reconstruction and flow MSE are measured on the training
  distribution. An adapter can lower both while becoming more confidently wrong about the artist:
  the loss sees the album's own frames, the listener hears a new song.
- **Drift is a magnitude, not a direction.** It is defined as "how far the decoder's flow
  prediction has moved from the base decoder's on a fixed probe set, as a relative squared
  error" (`engine/src/train/yue2-aitk-runtime.h:119-122`). A larger drift can be the change you
  wanted or the damage you did not.
- **The recipes are not single variables.** The two shipped paths change six to eight objectives
  at once (section 3), so a meter that moves cannot be attributed to one of them.

The one instrument that measures the target is the ear test, and it has a measured noise floor:
the same condition scored twice differed by about ±1 per criterion, differences under about 0.5
on a series average are noise, and renders are not reproducible from the seed
(`.claude/skills/ear-test-scoresheet/SKILL.md`). A single listen, unblinded, without a base
reference, is therefore not a measurement. It is an impression with high variance.

This is how the current presets were chosen. The user guide records it: "The current values, 200
updates and dim 128 for Balanced, come from an overnight batch of albums at those settings that
sounded right by ear." That was reasonable when the goal was a first working recipe. It is not a
method for deciding whether four flow heads, a different `arTargets`, or twice the rank is worth
1,500 updates.

### What a solution delivers

The deliverable is not a better loss curve. It is a gate that turns a run into one of three
decisions, with the evidence attached.

| Problem today | What the gate changes | Value |
|---|---|---|
| "Better" is unfalsifiable; two people can hear opposite results from one adapter | One pre-registered primary metric and a confidence interval | Ship, iterate or abandon in one session |
| Every knob is unpriced; heads, rank, recipe and dropout all move together | One delta per arm, in likeness points | GPU time goes to the knob that moves the score |
| A likeness gain can hide a quality loss | Quality guardrail and hard gates read before the primary | No silent trade of diction or artefacts for likeness |
| Entry points cannot be compared (196 vs 200 sites, 64/256 vs 256/256) | Same panel, same control, matched parameter count | Heads, rank and recipe priced in the same unit |
| Preset provenance is an overnight listening session | Recorded panel result per preset and per change | A preset that survives staff changes and re-runs |
| The adapter card cannot state a claim | Panel size and measured delta on the card | User trust and fewer "it does nothing" reports |

**Definition of done for a training change.** All hard gates pass, the primary likeness delta is
at least +0.5 with a 95% confidence interval above zero on the fixed panel, the quality series
is not worse by more than 0.3, and no render is flagged as a near-duplicate of training audio.
A result that misses any of these is "not demonstrated", which is a useful and publishable
outcome, not a failure to hide.

### The measurement stack

Four tiers, read in order. A lower tier can veto a higher one; a higher tier cannot rescue a
failed gate.

| Tier | Question | Instrument that exists today | Pass rule |
|---|---|---|---|
| T0 validity | Did the render complete and is it usable? | `server/scripts/yue2-ladder.mjs` (job status, stall and engine-restart detection), WAV duration and peak checks | Every planned render exists, is finite, does not clip, and has the expected duration |
| T1 manipulation | Did the adapter do what the config asked? | `meters.json` (`ar_kl_mean20`, `nar_drift`), `--meter-only`, `--eval-adapted` CE and flow MSE | Direction matches the change; never used as a quality claim |
| T2 primary | Does it sound more like the album? | Blind ear score on the likeness series (voice, writing, sound), 1 to 5; aligner diction as a secondary | Likeness delta at least +0.5 with a CI above zero |
| T3 guardrail | Did anything break? | Ear score on the quality series (diction, audio, coherence) plus the T0 gates | No criterion worse by more than 0.3; zero gate failures |
| T4 generalization | Is it the artist, or the training set? | Fresh panel songs, memorization check | Delta holds on unseen tracks; no near-duplicate |

Why the cheap metrics cannot be promoted:

- **CE and flow MSE** measure fit. They are the right T1 check, because a change that does not
  move them did not train. They are the wrong T2, because a model can reduce them by memorising
  the album's frames rather than learning its style.
- **`nar_drift`** is a magnitude. It is deliberately conditioned by the base planner so it
  measures the decoder alone (`engine/src/train/yue2-aitk-joint-step.h:329-331`). A rising number
  means the decoder changed, not that it improved.
- **Aligner diction** is a gate, not the target. The ear-test workflow measured its correlation
  with the user's diction scores at 0.37, so it catches garbled words and misses the judgement.

### The panel

A fixed panel is what makes results comparable across weeks and entry points.

- 10 to 12 tracks, none present in any training run for the adapter under test. Three families:
  new songs from the target artist, one adjacent artist, and one out-of-domain control. The last
  two catch an adapter that only learned "more of the same".
- Every track carries the exact caption and lyrics the backend will receive, stored with the
  panel, not retyped per run.
- The base model is an arm on every panel, with no adapter. It anchors the scale and makes "did
  the adapter help at all" answerable.
- Two takes per track per arm, with seeds recorded. This is not optional: renders are not
  reproducible from the seed, so a single take confounds the arm with the take.
- A manifest with dataset ids, prompt hashes, seeds, adapter file hashes, engine commit, sampler
  settings and decode settings. The join step refuses a run whose manifest differs from the
  panel's on any pinned field.
- A leakage assertion: panel ids must not intersect the training manifest. A held-out album
  stays out of every run, not just the one being judged.

Panel size comes from the noise floor, not from convenience. With a paired per-track standard
deviation near 0.7 points and a target effect of 0.5, 20 pairs give a standard error near 0.16
and a 95% interval half-width near 0.31, which separates 0.5 from zero. Ten tracks with one take
gives about 0.43, which does not. So 10 tracks x 2 takes is the minimum and 12 x 3 is comfortable.
This is Derived from the ear-test noise figure; the first panel should replace the assumed 0.7
with its own measured value.

### Controls and blinding

- One variable per arm. Prompt, seed set, take count, sampler, CFG, decode path, loudness
  normalization and engine commit are pinned in the manifest.
- Randomize row order and hide arm names until scores are saved. The score sheet keys scores by
  track id, so A and B labels can be decoded afterwards.
- Include the base reference in the same session. A score without an anchor drifts between
  sessions.
- Duplicate one condition under a second id, disguised. That gives an in-situ noise estimate and
  catches a scorer who is matching labels.
- Score in one sitting where possible, and never score an arm you just watched train. The ±1
  noise acquires a direction when the scorer knows what to expect.

### The experiment ladder

Each rung answers one question and is cheap enough to stop at. Do not skip E0 or E1; they cost
less than a wrong E2.

| Rung | Question | Arms | Panel | Decision |
|---|---|---|---|---|
| E0 | Does the harness produce complete, comparable rows? | Base twice | 2 tracks, 2 takes | Renders complete, meters parse, scores save. Stop if not |
| E1 | What is the noise floor here? | Base vs base, disguised | 10 tracks, 2 takes | Absolute delta under 0.3 and CI contains zero. Otherwise fix the panel or the scoring |
| E2 | Do the four flow heads matter? | Base, joint-196, joint-200, NAR-only-281 | 10 tracks, 2 takes | The table below |
| E3 | Which recipe term matters? | Fixed recipe, then one change at a time: `arTargets`, AR weight, KL, caption dropout | 10 tracks, 2 takes | One factor with a CI above zero; keep it, revert the rest |
| E4 | Is capacity or adapter type the constraint? | Rank 64 vs 256; LoRA vs LoKr at matched effective scale | 10 tracks, 2 takes | Delta per million parameters; pick the cheaper arm if tied |
| E5 | Does the winner generalize? | Winner vs base on fresh songs and a held-out album | New 10 tracks, 2 takes | Same direction and gate result. Only E5 can change a shipped preset |

E2 to E4 each need one training run per arm. That is the real cost, and the reason to run them
as decisions rather than as a default sweep.

### The first experiment: flow heads

This is the audit's largest finding (section 2) and the cheapest one to price, because the arms
already exist as code paths.

- Base: no adapter.
- Joint-196: the shipped joint trainer at rank 64, alpha 64, fixed recipe, 1,500 steps.
- Joint-200: the same, with the four flow-head sites enabled (the change in section 2).
- NAR-only-281: the retained `nar_attn_mlp_proj` path at rank 256, as the measured ceiling.

All four use the same dataset, steps, seed and panel. 4 arms x 10 tracks x 2 takes = 80 renders,
about 53 minutes of scoring at roughly 40 seconds per render.

| Result | Reading | Next |
|---|---|---|
| 200 above 196, 196 about base | The heads are the binding constraint; the joint trainer caps likeness by construction | Add the heads behind a flag and make them default in the fixed recipe |
| 200 about 196, both above base | The heads add nothing at this data and step count; the block adapter already reaches what the data supports | Spend on data, steps or rank (E3, E4); do not add the sites |
| 196 and 200 both about base | The problem is optimisation or the recipe, not representation | Go to E3 before touching the graph |
| 281 above 200 | The NAR-only endpoint has more than the heads (site count, rank, steps); compare at matched parameters before concluding | Run a matched-parameter arm |
| Likeness up and quality down | The adapter is overfitting or overwriting the companion | Check memorization, drift and caption dropout before shipping |

If the interval covers zero, the answer is "not demonstrated", not "no effect". Record it and
move to the next rung. That distinction is the whole point of the gate.

### The decision rule

1. Pre-register the panel manifest hash, the primary metric, the threshold, the number of tracks
   and takes, and the stopping rule before the first render. Store it beside the run.
2. Read the gates first. Any T0 failure, missing render, clipped file, or near-duplicate fails the
   arm regardless of its score.
3. Compute the primary as a paired per-track difference, then its mean and 95% interval
   (bootstrap over tracks with 10,000 resamples, or a paired t interval). Ship only when the
   lower bound is above zero and the mean is at least 0.5.
4. Check the guardrail. A quality series drop larger than 0.3, or any single quality criterion
   below its base by more than 0.5, rejects the arm even if likeness rose.
5. Flag memorization. A generation that shares a long contiguous code run, or is near-identical
   in the audio fingerprint to a training track, is a failure, not a win.
6. Declare one primary. Everything else is exploratory and labelled as such. Do not choose the
   winning criterion after seeing the scores.
7. Publish the negative result in the run registry. A flat result prices the knob at zero, which
   is information the next decision needs.

### Instrumentation to build

Most of the pieces exist. The missing part is the join and the decision layer.

| Piece | Status | Where |
|---|---|---|
| Checkpoint renders with a base arm, takes, diction and meters | Exists | `server/scripts/yue2-ladder.mjs` writes `results.jsonl` with `ar_kl`, `nar_drift` and aligner scores |
| Meter-only scoring of a finished run | Exists | `--meter-only` in `yue2-aitk-runtime.h` |
| Held-out CE and flow MSE, base vs adapted | Exists | `--eval-base-loss`, `--eval-adapted`, `--holdout-items`, planner gate |
| Blind 1 to 5 score sheet with likeness and quality series | Exists | `.claude/skills/ear-test-scoresheet/` writes `scores.json` |
| Arm orchestration for a matrix, not a ladder | Missing | Extend the ladder or add `server/scripts/yue2-eval.mjs`; the ACE eval scripts show the idempotent pattern |
| Join of scores, meters and gates into one table | Missing | New report module; `server/scripts/lm-adapter-eval-report.ts` is the pattern for the statistics and the report |
| Memorization check for YuE2 | Missing | The LCS idea is in `lm-adapter-eval-report.ts`, but it needs the planner's code sequence captured per render |
| Run registry with commit, config hash, panel hash and verdict | Missing | One JSONL file per experiment under the run directory |

### Cost and cadence

- A full rung (4 arms, 10 tracks, 2 takes) is 80 renders and about an hour of scoring, plus one
  training run per arm. That is a weekly decision, not a daily loop.
- The daily loop stays cheap: meters plus a 3-track smoke after each code change. The full panel
  is for changes that would alter a preset or a shipped claim.
- Rotate the confirm panel (E5) each quarter or after two shipped changes, so the gate does not
  become the thing being optimised.

### Failure modes and their mitigations

- **Optimising the proxy.** Aligner diction, loudness and duration can all improve while the
  song gets worse. Mitigation: they are gates, never the primary.
- **Panel leakage.** A "held-out" song that appeared in an earlier run makes every later result
  optimistic. Mitigation: assert the intersection, and keep one album out of all runs.
- **Scorer drift and expectation bias.** Mitigation: a base reference in every session, a
  disguised duplicate, randomized order, and arm names hidden until scores are saved.
- **Small n and multiple comparisons.** Mitigation: pre-register one primary, use paired
  differences, and report intervals rather than stars.
- **Non-reproducible seeds.** Mitigation: two takes minimum, paired by track, seeds recorded.
- **Unmatched arms.** A rank-256 NAR adapter and a rank-64 joint adapter differ in capacity,
  not just sites. Mitigation: report parameter count and effective scale beside every delta.
- **Overfitting the panel after E5.** Mitigation: rotate the confirm set and never train on any
  panel track.

### How this changes the recommendations

The audit's findings are hypotheses until a rung prices them. The mapping:

| Finding | Rung | Primary read |
|---|---|---|
| Flow heads (section 2) | E2 | Likeness delta of 200 vs 196 |
| Recipe divergence (section 3) | E3 | Delta per factor, one at a time |
| Rank, alpha, type (section 4) | E4 | Delta per million parameters, at matched effective scale |
| Prompt fidelity (section 5) | E3 caption arm, plus a matched-prompt A/B | Delta when training and render prompts are identical |
| Latent cache (section 6) | E0 | Base vs decode round-trip identity and artefact rate |
| Mix correction (section 7) | Outside training | Artefact rate before and after correction, same panel |

The recommended order in section 9 changes in one way: nothing is done until the harness in this
section exists, and nothing is called done until it has a rung result. That adds one item at the
top and one condition to every acceptance check, not a new program of work.

## 9. Recommended order of work

| # | Change | Files | Acceptance check |
|---|---|---|---|
| 0 | Stand up the measurement harness in section 8 | `server/scripts/yue2-ladder.mjs`, new join and report | Base vs base on the fixed panel has a CI containing zero |
| 1 | Decide the flow-head contract: add a joint flag for the four heads, or document block-only | `yue2-aitk-lora.h`, `yue2-aitk-train-state.h`, adapter card | A joint export either carries 200 NAR sites or the card says 196, and E2 prices the difference |
| 2 | Prove a joint `llm2vae` delta survives the companion merge | adapter export, `yue2-model.h` | An adapter that changes only `llm2vae` shows its effect at render |
| 3 | Put both joint recipes on one shared table | `yue2JointTrainRunner.ts`, `trainingRecipes.ts` | One test asserts every shared field |
| 4 | Show effective scale next to rank and alpha | Training Studio card | Scale comes from the same constants the engine uses |
| 5 | Delete or wire the dead stop values | `trainingRecipes.ts` | No constant is overridden before it can act |
| 6 | Record CoT in joint exports and warn on mismatch | `yue2-aitk-adapter-io.h`, `generate.ts` | Joint adapter header carries `cot` |
| 7 | Add a cached-latent decode command | `engine/tools/yue2-probe.cpp` | A cached latent plus gain produces a WAV |
| 8 | Port `--audio-loss` to the joint trainer | `yue2-aitk-joint-step.h` | Flag accepted, alignment gate passes on one clip |
| 9 | Fix the stale cache loudness line | `docs/dev/training-internals.md` | Page matches `YUE2_CACHE_LUFS` |

Item 3 is the cheapest and the one that most changes what a user can conclude. Item 1 is the
most expensive and the only one that changes what a joint adapter can represent.

Item 0 is now the precondition: without the panel, the control arm and the decision rule, the
rest of the table produces runs, not evidence. A row is done when its rung in section 8 clears
the rule in that section, not when the code merges.

## 10. Reproducing every claim

Run from the repository root. Each line re-derives one section.

```
git rev-parse --short HEAD                                  # audit applies to this commit
rg -n "dims\{2048" engine/src/train/yue2-aitk-train-state.h # 1.2 site dims
rg -n "make_site\(" engine/src/train/yue2-aitk-lora.h      # 1.2 four sites
rg -n "vae2llm|llm2vae|time_embd" engine/src/train/yue2-nar-train-graph.h  # 2 head sites
rg -n "vae2llm|llm2vae" engine/src/train/yue2-aitk-lora.h  # 2 absence in joint
rg -n "BASE_MATCHED_FORCED|applyFixedRecipe" server/src/services/training/yue2JointTrainRunner.ts
rg -n "YUE2_JOINT_(LORA|LOKR)_STOP" server/src/contracts/trainingRecipes.ts
rg -n "ar_targets|ar_loss_weight" engine/src/train/yue2-aitk-runtime.cpp
rg -n "keep_base_targets" engine/src/train/yue2-aitk-batch.h
rg -n "selective|retain_caption" engine/src/train/yue2-aitk-batch.h
rg -n "YUE2_CACHE_LUFS" server/src/services/training/yue2Train.ts
rg -n "frame-major|yp_write_cache" engine/src/train/yue2-preprocess-run.h
rg -n "vae-roundtrip|vae-parity" engine/tools/yue2-probe.cpp
```

Line numbers in this page are the output of these searches at `be818f66`.

## 11. Limits and open questions

- No training and no render was run. Everything about listening is a hypothesis tied to a code
  path, not a result.
- The protocol in section 8 is a design, not a result. The panel has not been built, no rung has
  been run, and its thresholds are derived from the ear-test noise figure rather than measured on
  YuE2.
- Parameter counts are arithmetic from the site shapes; they were not read off a checkpoint.
- Merge order is resolved in code (`yue2-model.h:1638-1671`), but not demonstrated with a file
  that targets `llm2vae`. That is item 2 in section 9.
- The NAR-only and AR-only endpoints are still served but are no longer offered by the Training
  Studio card. Every comparison with "the campaign's adapter" is a comparison with an endpoint a
  user cannot reach from the current UI.
- Whether a quantized base changes the picture is out of scope. The recipe comments say the AR
  path is only proven on bf16, and the NAR VRAM model was fitted on `nar_attn_mlp`.

## Related

- [YuE2 training](../../user/training/yue2.md)
- [Training Studio](../../user/studios/training-studio.md)
- [Adapters](../../user/adapters.md)
- [Training internals](../training-internals.md)
- [Architecture](../architecture.md)
- [Model files](../../user/models.md)
