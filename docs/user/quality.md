# Getting higher quality output

This answers a recurring question ([#152](https://github.com/scragnog/HOT-Step-CPP/issues/152)):
which settings trade generation speed for better output. There is no single
"quality" knob. In rough order of how much each one actually moves the
result:

1. Model precision.
2. The prompt and lyrics.
3. Step count and solver.
4. Post-processing.

The defaults in each Create panel are a reasonable starting point, not a
floor. Everything below is optional and mostly off by default, because most
of it costs render time.

## Model precision comes first

Quant choice affects the result more than any sampler setting. Higher
precision needs more VRAM and runs slower per step.

### ACE-Step 1.5

| Component | Fastest / smallest | Recommended | Highest quality |
|---|---|---|---|
| DiT | Standard Turbo Q4_K_M | XL Turbo Q8_0 | XL Turbo BF16 |
| Planner LM | 1.7B Q8_0 | 4B Q8_0 | 4B BF16 |
| VAE decoder | Standard VAE BF16 | ScragVAE BF16 | Standard or ScragVAE at F32 |

The XL Turbo DiT needs about 12 GB of VRAM. ScragVAE is a fine-tuned decoder
with more high-frequency energy and dynamic range than the standard decoder;
select it from the Models dropdown, no retraining involved. The base (non-turbo)
DiT variants exist too, but they need 60 to 100 steps to reach a comparable
result, so they are a poor trade for anyone chasing quality per minute.

### MiniMax-Music3

| Pack | LM / DiT precision | Download | Notes |
|---|---|---|---|
| Q4_K_M | Q4_K_M | ~8 GB | Smallest, for limited VRAM |
| Q6_K | Q6_K, depth Q8_0 | ~10 GB | |
| Balanced | LM Q8_0, DiT Q4_K_M | ~12 GB | Quality lives in the LM, VRAM lives in the DiT |
| Q8_0 | Q8_0 throughout | ~13 GB | Recommended: near-lossless, about half the F16 size |
| F16 | Full precision | ~24 GB | Reference quality bar |

The Q8_0 and below quants use an importance matrix measured from the
full-precision model, so error is spent on the weights that carry signal
rather than spread evenly.

### YuE2

| Pack | LM precision | Download | Measured drift (NAR-stage rel-L2 vs bf16 output) |
|---|---|---|---|
| Tiny (imatrix) | Q2_K-imat | ~2.2 GB | 0.389, audibly degraded |
| Compact (imatrix) | Q4_K_M-imat | ~3.0 GB | 0.116 |
| Recommended | Q8_0 | ~4.4 GB | 0.019 |
| Full precision | BF16 | ~7.8 GB | 0.013 (the baseline itself) |

The drift figures were measured on the NAR stage against the bf16 output.
Below Q5, YuE2 quants also use an importance matrix. Generate from the Q8_0 GGUF unless you
have measured that the ConvRot checkpoint is faster on your card. ConvRot
generation does not yet apply the joint companion decoder that GGUF
generation does, so treat it as a training-time checkpoint first.

## The prompt and lyrics

A vague caption produces vague music no matter what the sampler does. Three
things move the result more than any dial:

- Section tags. Only `[Intro]`, `[Verse]`, `[Verse 1]`, `[Verse 2]`,
  `[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Interlude]`,
  `[Outro]`, `[Hook]` and `[Refrain]` are recognised. Each must be alone on
  its own line.
- Concrete genre and instrumentation. "Nylon guitar, brushed percussion,
  velvet crooning vocals, golden-age Latin romance" gives the model more to
  work with than "romantic ballad". Name the instruments, the playing style
  and the production era, not just an adjective.
- An explicit BPM and language. Left unspecified, generation tends to drift
  toward a mid-tempo default regardless of genre, so a slow ballad can come
  back sounding like mid-tempo pop. State the tempo. State the language
  rather than leaving it to be inferred from the lyrics.

## Steps and solver

More steps help up to a point, then the extra render time buys nothing. Past
roughly 40 steps on ACE-Step, further steps mostly add noise rather than
detail.

### ACE-Step 1.5

Default is 12 steps. The in-app assistant has a one-click "Maximum quality"
preset: the JKASS Quality solver, 35 steps, the `sgm_uniform` scheduler, APG
guidance at scale 6.0, and a VAE re-encode pass at 0.15 blend. JKASS Quality
is a 2-NFE solver, so each step costs two network evaluations; 35 steps there
is roughly what 70 would cost on a 1-NFE solver like Euler or DPM++2M.

Other solver/scheduler pairings worth knowing: DPM++2M with `sgm_uniform` is
the general-purpose combination, and JKASS with `beta57` favours a warmer,
more musical result at the same step count.

### MiniMax-Music3

The Flow Steps slider (`mm3Steps`) defaults to the checkpoint's own 30 and
ranges from 2 to 60. Fewer steps are proportionally faster, since the flow
stage dominates render time on longer tracks. Below 30 steps, Low-Step
Compensation reshapes the noise schedule automatically to keep the low end
and stereo image intact; without it, low step counts go thin and phasey
rather than merely soft. Raising the slider above 30 trades speed for more
refinement in the same way extra ACE-Step steps do.

### YuE2

The ODE Steps slider (`yue2OdeSteps`) defaults to 32 and ranges from 8 to 64.
Each step is a midpoint solver step, meaning two network evaluations. Lower
is faster and less refined; there is no published ceiling analogous to
ACE-Step's diminishing-returns point, so this one is worth an A/B rather than
assuming higher is always better.

## Post-processing

Post-processing runs after the main render and, in the current build, every
stage below defaults to off. None of it happens unless you turn it on.

- StableStep re-renders the instrumental through a Stable Audio 3 refiner
  while keeping the vocal stem untouched, replacing autoencoder fizz with
  real spectral detail. Refine Strength (`stableStepStrength`) defaults to
  0.30 if you enable it; lower values are closer to cleanup, higher values
  re-interpret the instrumentation.
- Mastering (`masteringEnabled`) runs matchering-based mastering against a
  reference track.
- PP-VAE re-encodes the finished mix through the post-processing VAE to
  smooth spectral artifacts. If you turn it on, F32 gives the best result of
  the available precisions.
- ScragVAE, described above, is a decoder swap rather than a post-process
  stage, and applies during the main render.

Turning several of these on at once compounds render time on top of whatever
you already spent on steps, so treat post-processing as a separate quality
budget from the solver settings above.

## Comparing two settings on the same seed

The Audio Quality Evaluator (`qualityEvalEnabled`, off by default) scores
each track from 0 to 100 percent using three weighted spectral metrics:
metallic sound at 85th-percentile spectral rolloff (40 percent), word-cut
discontinuities from spectral flux z-scores (40 percent), and noise or hiss
from zero-crossing rate (20 percent). You can evaluate the raw render, the
mastered render, or both, and the score is stored per track and shown as a
colour-coded badge in the Library.

To isolate one variable, set `randomSeed` to false and pin a specific seed,
then change only the setting you are testing between two generations. The
Library's A/B compare view lists every parameter that differs between two
pinned tracks side by side, so you can confirm the seed and everything else
actually matched before comparing the quality scores or listening yourself.
A quality score difference of a few points is within the noise of the
metric; treat it as a tiebreaker alongside your own ears, not a verdict on
its own.

## Backend cheat sheet

What to change first, per backend, when you want slower but better:

- ACE-Step 1.5: XL Turbo DiT at Q8_0 or BF16, 4B LM at Q8_0, apply the
  Maximum quality assistant preset, then consider StableStep and PP-VAE.
- MiniMax-Music3: the Q8_0 pack, Flow Steps above 30 up to 60, Flow Guidance
  (`mm3CfgFlow`) raised from its 1.7 default if the render is drifting from
  the caption.
- YuE2: the Q8_0 or full-precision LM pack, ODE Steps raised from 32 toward
  64, generate from the GGUF rather than the ConvRot checkpoint.
