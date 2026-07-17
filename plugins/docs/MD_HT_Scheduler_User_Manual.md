# MD HT Scheduler — User Manual
## MDMAchine | A&E Concepts | GPL v3
### Plugin Version: V3 | Internal Version: v5.0

---

## What Does HT Do?

Most schedulers space your denoising steps evenly (uniform) or with a simple curve (Karras, exponential). HT uses physics to place steps where they actually matter.

It combines two density functions:

**HAP (Hamiltonian Action-Principle)** simulates a particle falling through a gravity well with drag. The particle accelerates as it falls (stretching steps in the mid-sigma structure zone) and slows as drag increases (compressing steps at the end for detail refinement).

**TPT (Thermodynamic Phase Transition)** creates a gravity well at a specific sigma level (the "critical temperature") where the latent undergoes its most important structural change. Steps cluster around this point so the model has finer control during the crystallization moment.

The result: more steps where they matter, fewer where they don't. Works from 12 to 150+ steps.

---

## Quick Start

If you just want it working, use these and go:

| Setting | Value | Why |
|---|---|---|
| Kinetic Energy | 0.3 | Standard front-loading |
| Damping Friction | 2.2 | Moderate tail compression |
| Critical Temp | 0.6 | Cluster at the structure/detail boundary |
| Phase Intensity | 1.0 | Moderate clustering |
| Well Width | 0.25 | Balanced spread |
| Density Floor | 0.1 | Gentle minimum everywhere |
| Everything else | OFF / defaults | Turn on one at a time |

**For 12-step turbo:** add `Poly Slope = 0.8` (more structure steps) and `Uniformity Blend = 0.2` (gentle uniformity).

**For 50+ step runs:** try `SNR Space = ON` for perceptually-weighted step placement.

---

## The Controls

### HAP Controls (Base Schedule Shape)

#### Kinetic Energy
How aggressively the schedule front-loads steps into the structure-formation zone.

- 0.0 = uniform (no front-loading)
- 0.3 = standard (default, gentle front emphasis)
- 1.0-1.5 = noticeable structure emphasis
- 2.0+ = aggressive (lots of steps early, sparse late)

Higher values mean more steps during the "big decisions" phase of generation and fewer during detail refinement. Good for complex prompts that need strong early structure. Too high and the detail phase gets starved.

#### Damping Friction
How fast the schedule compresses steps toward the end.

- 0.0 = no compression (uniform tail)
- 2.2 = standard (default)
- 4.0+ = heavy end compression (many detail steps)

Think of it as atmospheric drag on the particle. Higher drag = the particle slows down more at the end = more steps compressed into the final detail phase.

### TPT Controls (Phase Transition Clustering)

#### Critical Temp
Where on the sigma axis the gravity well sits (as a fraction of 0-1).

- 0.6 = default (structure/detail boundary)
- 0.3-0.4 = clusters steps later (detail-focused)
- 0.7-0.8 = clusters steps earlier (structure-focused)

This is the moment in the generation where the latent "crystallizes" from noise into structure. Placing the well here gives the model more steps at the most information-dense moment.

#### Phase Intensity
How strong the clustering effect is.

- 0.0 = off (pure HAP, no clustering)
- 1.0 = moderate (default)
- 2.0+ = strong (heavy step concentration at critical temp)

At 0 you get a pure HAP schedule. As you increase, more steps pile up around the critical temp and fewer are available elsewhere.

#### Well Width
How wide the clustering zone spreads around the critical temp.

- 0.1 = tight (steps concentrated in a narrow band — can sound "overdriven")
- 0.25 = balanced (default)
- 0.4+ = broad (gentle clustering over a wide zone)

Pairs with Phase Intensity: intensity controls depth (how many steps cluster), width controls spread (how wide the cluster zone is). Both together shape the gravity well.

### Density Floor

Guarantees a minimum step density everywhere in the schedule. Without a floor, some zones can end up with very few steps (sparse gaps), which forces the solver to make oversized jumps that cause artifacts.

- 0.0 = no floor (old behavior, maximum clustering contrast)
- 0.1 = gentle floor (default, prevents worst-case sparse gaps)
- 0.3+ = significant floor (schedule becomes more uniform overall)

**This is the key setting for pairing with stabilization solvers** like Trajectory Anchor. If the output sounds "crispy" or harsh, increase the floor.

### SNR Space

When ON, the integration grid is built uniform in SNR (signal-to-noise ratio) space instead of sigma space. Steps automatically track perceptual importance since SNR maps to how much "useful information" vs "noise" the model is working with at each point.

- OFF = sigma-uniform grid (default, standard behavior)
- ON = SNR-uniform grid (steps cluster where SNR changes fastest)

At 12 steps you'll barely notice the difference. At 50-150 steps it meaningfully improves how the schedule distributes effort across the perceptual range, especially for audio where mid-frequency detail matters more than extreme high or low sigma regions.

### Post-Processing Controls

These apply after the HT schedule is computed, in order. All are default off. They compose cleanly with each other and with any step count.

#### LINA Warp
Time-axis resampling ported from the MD Causal scheduler. This is different from the native Shift Warp.

- **Shift Warp** transforms the sigma *values* (changes what sigma each step lands on)
- **LINA Warp** transforms *where on the curve* each step samples from (resamples the schedule itself)

Settings:
- 1.0 = off (default)
- < 1.0 = front-load (more high-sigma / structure steps)
- > 1.0 = back-load (more low-sigma / detail steps)

Subtle at small deviations from 1.0. Start with 0.9 or 1.1 and adjust.

#### Poly Slope
Power curve applied to the sigma values after everything else.

- 1.0 = off (default, no change)
- > 1.0 = compress toward zero (more detail steps, good for long runs 50+)
- < 1.0 = compress toward one (more structure steps, good for 12-step turbo)

This is the simplest global shape control. If you're running a turbo model at 12 steps and need more structural emphasis, drop poly to 0.7-0.8. If you're running 150 steps and want finer detail distribution, push it to 1.1-1.3.

#### Uniformity Blend
Blends the HT schedule with a pure linear uniform schedule.

- 0.0 = pure HT (default, maximum clustering character)
- 0.2-0.3 = gentle uniformity (tames clustering, good for Trajectory Anchor)
- 0.5 = half and half
- 1.0 = pure uniform (no HT character, just linear)

**This is the most direct fix for "HT doesn't pair well with my solver."** If the output sounds over-processed, harsh, or unstable with your solver, increase the blend. You're trading HT's intelligent step placement for the safety of uniform spacing.

#### Schedule Smoothing
Moving average on the final sigma sequence. Eliminates sharp transitions between dense and sparse zones.

- 0 = off (default)
- 3 = mild smoothing
- 5+ = heavy smoothing

Helps stabilization solvers (Trajectory Anchor's inertia engine, memory buffer) by giving them gradual step-size transitions instead of sudden jumps. Slight cost: smoothing blurs the clustering precision.

### Engine Controls

#### CDF Resolution
Resolution of the integration grid used internally. Higher = smoother CDF sampling, slightly slower to compute.

- 1000 = default (fine for most uses)
- 200 = fast but rough
- 5000 = very smooth (overkill for < 50 steps)

#### Shift Warp
Native HOT-Step sigma warp. Applied first in the post-processing chain, before LINA, poly, uniformity, and smoothing.

- 1.0 = off
- > 1.0 = shifts sigma distribution

#### Verbose
Prints per-step sigma values, step sizes (gaps), and the min/max gap ratio to the console. Turn this on when tuning — the gap ratio tells you at a glance how uniform your schedule is.

- Ratio 2:1 = gentle variation (very even)
- Ratio 5:1 = moderate (noticeable clustering)
- Ratio 10:1+ = aggressive (sparse gaps likely, may cause artifacts)

---

## Pairing Guide

### With Trajectory Anchor (recommended settings)
Trajectory Anchor's stabilization stack needs reasonably uniform step sizes to work properly. Oversized gaps in the schedule fight the inertia engine and concept lock.

| Setting | Value | Why |
|---|---|---|
| Density Floor | 0.1-0.2 | Prevents sparse gaps |
| Uniformity Blend | 0.1-0.3 | Tames clustering |
| Smooth Window | 3 | Gradual transitions |
| Poly Slope | 0.8 (at 12 steps) | More structure steps for turbo |
| Phase Intensity | 0.5-1.0 | Don't over-cluster |

### With STORM
STORM handles stiffness internally and adapts per-step, so it's more tolerant of non-uniform schedules.

| Setting | Value | Why |
|---|---|---|
| Density Floor | 0.0-0.1 | STORM handles gaps |
| Uniformity Blend | 0.0 | Let HT do its thing |
| Phase Intensity | 1.0-2.0 | STORM benefits from clustering |

### With Omni Relational
Omni Relational is step-based like Trajectory Anchor but lighter (no memory buffer, no concept lock). More tolerant of non-uniform schedules.

| Setting | Value | Why |
|---|---|---|
| Density Floor | 0.1 | Gentle minimum |
| Uniformity Blend | 0.0-0.1 | Light touch |

### Step count guidelines

| Steps | Recommended adjustments |
|---|---|
| 8-12 (turbo) | Poly Slope 0.7-0.8, Uniformity Blend 0.2, Floor 0.1 |
| 25-35 (standard) | Defaults work well |
| 50-100 (quality) | SNR Space ON, Poly Slope 1.1 |
| 100-150 (maximum) | SNR Space ON, Poly Slope 1.2, CDF Resolution 2000+ |

---

## Troubleshooting

**Output sounds "crispy" or harsh**
Increase Density Floor (0.2-0.3). Add Uniformity Blend (0.2). The schedule has sparse gaps causing oversized solver steps.

**Output sounds over-smoothed or lacks punch**
Reduce Uniformity Blend and Smooth Window. Increase Phase Intensity. You're flattening the schedule too much — let it cluster.

**Trajectory Anchor fighting the schedule**
See the Trajectory Anchor pairing guide above. Floor + blend + smoothing, all gentle.

**No audible difference from uniform schedule**
Phase Intensity is probably at 0 or very low. Increase to 1.0+. Also check that Uniformity Blend isn't at 1.0 (which = pure uniform).

**Not sure what the schedule looks like**
Turn on Verbose. Read the console output. The gap ratio tells you everything — 2:1 is gentle, 10:1+ is aggressive.

**12-step turbo sounds thin / lacks structure**
Poly Slope 0.7-0.8 shifts more steps into the structure phase. Also consider dropping Kinetic Energy to 0.0-0.1 so HAP doesn't front-load *too* aggressively at very low step counts.

**150-step run sounds no better than 50**
Turn on SNR Space. At high step counts, sigma-uniform spacing wastes steps in perceptually unimportant regions. SNR-uniform puts them where they matter.

---

*© 2026 Alexander Allan (MDMAchine) — A&E Concepts — GPL v3*
