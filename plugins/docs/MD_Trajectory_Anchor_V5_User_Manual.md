# MD Trajectory Anchor — User Manual
## MDMAchine | A&E Concepts | GPL v3
### Plugin Version: V5 | Internal Version: v5.0

---

## What Does Trajectory Anchor Do?

Every other solver in the MD suite takes over the entire sampling loop and controls how the model steps from noise to audio. Trajectory Anchor is different. It's a **step() solver**, which means it works alongside whatever guidance module you have active (STORM Guidance, Clarity, APG, etc.) instead of replacing it. Your guider stays active.

What it actually does: at each denoising step, it applies a stack of stabilization systems to the latent — momentum, structure locking, tonal correction, energy management — that keep the trajectory from drifting, oscillating, or losing coherence during the generation.

Think of it like guardrails on a mountain road. The model drives, the guidance steers, and Trajectory Anchor keeps you from going off the cliff.

---

## What's New in V5

### Step-Budget Auto-Scaling (invisible, zero new params)

All system strengths now automatically adapt to your step count. The solver knows whether you're running 12 steps or 150, and adjusts how hard each system pushes per step.

- **12 steps (turbo):** each step matters a lot, systems push ~1.7x harder per step
- **35 steps (standard):** baseline, no change
- **150 steps (quality):** each step matters less, systems push ~0.5x per step

This is why V4 users noticed "more steps = better" — with V5, the solver explicitly accounts for step budget so you get cleaner results at every step count without retuning params.

### Anti-Ringing on Identity Anchor (invisible, zero new params)

When the identity anchor pulls the latent toward its captured snapshot, V4 could overshoot — the latent moves past the anchor, then gets pulled back, then overshoots again. This creates a subtle oscillation ("ringing") that sounds rough.

V5 detects when the latent is already moving toward the anchor and automatically reduces the pull. If the latent is moving away, full pull is maintained. Result: smooth convergence toward the anchor without oscillation.

---

## Quick Start — Recommended Defaults

If you just want it to work well out of the box:

| Setting | Value | Why |
|---|---|---|
| Warmup Steps | 2 | Skip first 2 steps (pure noise, nothing to stabilize) |
| Inertia Engine | ON | Smooths trajectory direction changes |
| Inertia Alpha | 0.15 | Gentle momentum (auto-scales with step count in V5) |
| Concept Lock | ON | Protects settled structure from noise |
| Tonal Anchor | ON | Keeps tonal balance from drifting |
| Everything else | OFF / defaults | Turn on one at a time as you learn them |

**For 12-step turbo:** use these exact defaults. V5's auto-scaling handles the rest.

**For 150-step quality:** same defaults. V5 backs off automatically. Optionally enable Identity Anchor (blend 0.08) and Look-Back (lambda 0.15) for even more stability.

Pair with: **MD HT Scheduler V3** (recommended over HAP for Trajectory Anchor) and **STORM Guidance V2** or **Clarity** or **APG**.

---

## The Systems (What Each One Does)

Trajectory Anchor has 8 independent systems. Each can be toggled on or off. They run in order, top to bottom, every step. All system strengths are automatically scaled by the step-budget system in V5.

### 1. Warmup Steps

**What:** Disables all stateful systems (inertia, concept lock, anchors, memory) for the first N steps.

**Why:** At the very start of generation, the latent is pure noise. Anchoring into chaos doesn't help — it just locks you into random structure. Warmup lets the model find its footing before the stabilization kicks in.

**Setting:** 2 is the sweet spot. 0 means everything is active from step 1 (not recommended). 3-4 if you're running very few total steps (like 8-12).

### 2. Inertia Engine

**What:** Carries a fraction of the previous step's velocity into the current step. Like momentum in physics — the trajectory resists sudden direction changes.

**Why:** Without inertia, each step is independent and the trajectory can jitter between competing solutions. With inertia, the path smooths out and commits to a direction.

**Inertia Alpha** controls how much carry-over (before V5 auto-scaling):
- 0.10 = subtle, barely noticeable
- 0.15 = recommended starting point
- 0.20-0.25 = noticeable smoothing, may soften transients
- 0.30+ = strong, can make things sluggish

The solver automatically scales alpha down when the latent has low entropy (already structured), so it pushes harder during chaotic early steps and backs off during refinement. V5 additionally scales by step budget.

### 3. Memory Buffer

**What:** Keeps the last 3 step outputs in a ring buffer and blends their average into the current output.

**Why:** Suppresses step-to-step jitter without redirecting the trajectory. Different from inertia — inertia smooths the velocity (direction), memory smooths the position (output).

**Memory Blend** controls the blend fraction:
- 0.12 = subtle (default)
- 0.25+ = heavy, may soften transients and fast attacks in audio

**Default: OFF.** Turn this on if you're hearing jittery artifacts or instability in the output. Leave it off if things sound clean — it adds a slight smoothing cost.

### 4. Concept Lock

**What:** Detects which parts of the latent are "settled" (small step-to-step change) and gently pulls them back toward their previous state. Dynamic regions are left alone.

**Why:** Once a structural element crystallizes mid-generation, noise in subsequent steps can erode it. Concept lock protects what's already formed while letting unfinished parts keep evolving.

**Concept Lock Sigma Power** controls how fast the lock fades as sigma drops:
- 1.0 = linear fade (default, recommended)
- 2.0 = quadratic — lock concentrated on early structure steps only, off during detail phase
- 0.5 = slow fade — lock persists deep into detail steps (more conservative, may over-constrain)

**Default: ON.** This is one of the most impactful systems. Leave it on unless you specifically want maximum creative freedom in the late steps.

### 5. Identity Anchor

**What:** At a specific sigma level (anchor_sigma), captures a full snapshot of the latent. Then on every subsequent step, gently pulls the latent back toward that snapshot.

**Why:** Prevents late-stage structural drift — the model sometimes "changes its mind" about the overall shape of the output in the last few steps. The anchor keeps it committed to the structure it chose at the anchor point.

**V5 improvement:** Anti-ringing automatically detects when the latent is already moving toward the anchor and reduces the pull. No more overshoot oscillation. This makes higher anchor_blend values safer to use.

**Anchor Sigma** = when the snapshot is taken:
- 0.5 = mid-generation (default) — captures after initial structure but before fine detail
- Lower (0.3) = captures more detail, locks in later
- Higher (0.7) = captures coarser structure only

**Anchor Blend** = how hard it pulls (before V5 auto-scaling):
- 0.08 = gentle (default, recommended)
- 0.15 = noticeable pull (safer in V5 thanks to anti-ringing)
- 0.20+ = strong (was risky in V4, more usable in V5)

**Default: OFF.** Turn this on if you're hearing late-stage structural drift (the output sounds like it "forgot" what it was doing toward the end). Start at 0.08 blend.

### 6. Tonal Anchor

**What:** Captures spectral centroid and band energy ratios at anchor_sigma (same timing as identity anchor). Applies a small tonal correction each step to prevent tonal balance from drifting.

**Why:** The model can gradually shift tonal balance during generation — bass gets louder, highs get softer, or vice versa. This corrects for that drift without changing the content.

**Tonal Strength:**
- 0.10-0.20 = recommended for audio
- Correction is hard-capped at 0.1% per step regardless of this value, so even high settings are gentle

**Default: ON.** Low-cost, high-value. Keeps tonal balance stable without audible artifacts.

### 7. Look-Back Smoother

**What:** Blends the current step output toward the previous step output, weighted by sigma. Heavy smoothing at high sigma (early steps, structure phase), fading to zero at low sigma (detail phase).

**Why:** Suppresses ODE manifold shearing — the technical root cause of the "metallic twinge" artifact in flow-matching audio. Same mechanism used in STORM internally.

**Look-Back Lambda** (before V5 auto-scaling):
- 0.15 = gentle (default)
- 0.35 = moderate (good for 35-step simple schedule)
- 0.55 = strong (good for 25-step DDIM schedule)

**Look-Back SNR Power:**
- 1.3 = standard (25-step DDIM)
- 1.5 = faster fade (35-step simple)
- Higher = smoothing concentrated on very early steps only

**Default: OFF.** Turn this on if you're hearing metallic or harsh artifacts. If you're already running STORM Guidance (which has its own CFG adaptation), you may not need this.

### 8. RMS Servo

**What:** Downward-only energy ceiling. If the latent RMS exceeds the target range, scales it down. Never scales up — only prevents energy runaway.

**Why:** Some configurations (high CFG, aggressive guidance, long generations) can cause the latent energy to ramp up over the course of generation, leading to clipping or distortion.

**RMS Target Min / Max:**
- Min = ceiling at low sigma (detail phase). Start at 1.2-1.8.
- Max = ceiling at high sigma (structure phase). Start at 2.5.
- ACE-Step latents typically run ~2.0 RMS at x0.

**RMS Servo Gain:**
- 0.6 = gradual correction (default, recommended)
- 1.0 = hard snap each step (aggressive)

**Default: OFF.** Only turn this on if you're experiencing energy runaway (clipping, distortion, "blown out" sound). Most setups don't need it.

---

## Advanced Settings

### Latent Pressure

Like RMS Servo but smarter — monitors the entropy * RMS product (a measure of "how chaotic and how energetic") and nudges toward a target value. Correction capped at 0.05% per step, so it accumulates gently over many steps.

**Default: OFF.** Experimental. If you enable it, run with verbose output first to see what your latent's actual entropy distribution looks like before setting targets.

### Relational Weight (Barbour Best Matching)

Per-block velocity equalization from the Omni Relational solver, available here as an optional addon. Equalizes component magnitudes across 4 blocks of the velocity vector, fading with sigma.

- 0.0 = off (default)
- 0.3-0.5 = balanced

**Important:** Do NOT use this if you're also running Confluence as your solver — the two velocity-reshaping systems fight each other.

### Eta (SDE Noise)

Stochastic noise injection per step. 0 = pure deterministic ODE (default). Small values (0.05-0.15) add subtle variation without overwhelming the stabilization. Scaled by sigma so it fades during detail phase.

### Safety Clamp

Hard absolute value clamp on the latent after all corrections. 2.5 is standard. Raise to 4.0+ if you hear clamping artifacts (sounds like hard limiting / pumping). NaN/Inf triggers a full rollback to raw Euler before clamping.

---

## V5 Step-Budget Scaling — How It Works

You don't need to touch anything for this to work. But if you're curious about the math:

The solver reads the total step count from the engine and computes a scaling factor:

```
budget_scale = sqrt(35 / your_step_count)
```

| Your Steps | Scale Factor | Effect |
|---|---|---|
| 8 | 2.09x | Systems push much harder per step |
| 12 | 1.71x | Strong push (turbo sweet spot) |
| 25 | 1.18x | Slight push |
| 35 | 1.00x | Reference — no change |
| 50 | 0.84x | Slight pullback |
| 100 | 0.59x | Moderate pullback |
| 150 | 0.48x | Systems very gentle per step |

This multiplier is applied to: Inertia Alpha, Memory Blend, Anchor Blend, Tonal Strength, and Look-Back Lambda. The values you set in the params are the *base* values at 35 steps. At other step counts, V5 adjusts them automatically.

**Why sqrt?** Linear scaling would be too aggressive — halving the steps would double the push, which overshoots. Square root gives diminishing returns that match how the trajectory actually behaves.

---

## Pairing Guide

### Best pairings (tested and validated)

**Trajectory Anchor V5 + HT Scheduler V3 + STORM Guidance V2:**
The gold pairing as of V5. HT V3's density floor and uniformity blend prevent the sparse step gaps that fought V4's stabilization stack. STORM Guidance's CFG rolloff prevents late-step ringing. For 150 steps, enable HT's SNR Space mode.

**Trajectory Anchor V5 + HT Scheduler V3 + Clarity:**
Lighter guidance. Good for exploring. HT V3 handles the schedule side, Clarity handles post-CFG cleanup.

**Trajectory Anchor V5 + any schedule + APG:**
Safest baseline. APG is always clean. Use when other pairings aren't working and you need to isolate whether the issue is guidance or schedule.

**Trajectory Anchor V5 + double composite schedule:**
Illynir's original preferred pairing. Still works well, though HT V3 has been validated as superior by the same tester.

### What NOT to pair with

- **Trajectory Anchor + Confluence** with relational_weight > 0 — two velocity-reshaping systems fighting. Keep relational_weight at 0 if using Confluence.
- **Trajectory Anchor + another step() solver** — only one solver active at a time.
- **Trajectory Anchor + MD HAP (standalone)** — HAP's clustering creates sparse gaps that fight the stabilization stack. Use HT V3 instead, which has density floor and uniformity blend to prevent this. If you must use HAP, keep kinetic_energy at 1.0 or below and damping_friction at 0.5 or below.

### Schedule recommendations by step count

| Steps | Recommended Schedule | Key Settings |
|---|---|---|
| 8-12 (turbo) | HT V3 | poly_slope=0.8, uniform_blend=0.2, floor=0.1 |
| 25-35 (standard) | HT V3 or double composite | HT defaults work well |
| 50-100 (quality) | HT V3 | snr_space=ON, poly_slope=1.1 |
| 100-150 (maximum) | HT V3 | snr_space=ON, poly_slope=1.2, resolution=2000+ |

---

## Troubleshooting

**"Several songs playing simultaneously"**
You're running an old version of the plugins or your HOT-Step build is behind. Grab the latest release and redeploy the whole plugin batch together. Don't mix old and new files.

**Output sounds "mushy" or over-smoothed**
Turn off Memory Buffer. Reduce Inertia Alpha. If Look-Back is on, reduce lambda. You're over-stabilizing. At high step counts (100+), V5's auto-scaling should prevent this — if it's still mushy, your base values are too high.

**Output sounds harsh or metallic**
Turn on Look-Back Smoother (lambda 0.15-0.35). If already on, increase lambda. Also check your guidance module — Clarity or STORM Guidance V2 help here.

**Output sounds "rough" or "drafted" but musical**
This is the Trajectory Anchor tradeoff — best musicality but can leave rough edges. Try enabling Identity Anchor (blend 0.08-0.15, safer in V5 thanks to anti-ringing). Also try increasing Concept Lock Sigma Power to 1.5-2.0 to lock structure more aggressively.

**Late-stage structural drift ("forgot what it was doing")**
Turn on Identity Anchor (blend 0.08, sigma 0.5). V5's anti-ringing makes this safer than in V4.

**Tonal balance shifting during generation**
Tonal Anchor should already be ON by default. If it is and you're still hearing drift, increase tonal_strength to 0.20-0.30.

**Energy runaway / clipping / distortion**
Turn on RMS Servo (min 1.2, max 2.5, gain 0.6). If severe, also turn on Latent Pressure.

**"More steps sounds worse" (shouldn't happen in V5)**
If you're experiencing this, the step-budget auto-scaling may not be reading the step count correctly. Check that you're on the latest HOT-Step build. As a manual workaround, reduce Inertia Alpha and other strengths proportionally when increasing steps.

**HAP schedule sounds bad with Trajectory Anchor**
Switch to HT Scheduler V3. This is a known pairing issue — HAP's clustering creates step gaps that fight the stabilization stack. HT V3 has density floor and uniformity blend specifically designed to solve this.

---

## Community Findings

**Illynir (2026-07-16):** "By far the best in terms of musicality, beyond any doubt." Spent 7+ hours across two sessions testing V3/V4. Found that more steps consistently improved output quality (confirmed by V5's step-budget scaling design). HT V3 scheduler validated as superior to HAP and double composite for Trajectory Anchor pairing: "Much MUCH better than HAP" and "beats my two-stage composite."

---

*© 2026 Alexander Allan (MDMAchine) — A&E Concepts — GPL v3*
