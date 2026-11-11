# Solvers, schedulers and guidance modes

A render starts from noise and cleans it up over a set number of steps. Three choices shape
how that happens, and each one is a Lua plugin you pick from a dropdown:

| Plugin type | What it decides |
|---|---|
| Solver | How each step moves from noisier to cleaner audio. Some solvers run the model once per step; others run it two or more times. That count is the NFE (network function evaluations) per step, and a 2-NFE solver takes about twice as long per step as a 1-NFE one. |
| Scheduler | How the steps are spaced between full noise and clean audio: evenly, or bunched toward the start, where structure forms, or the end, where detail forms. |
| Guidance mode | How strongly, and in what way, the caption and lyrics pull on the result at each step, working with the Guidance Scale slider. |

Because they are plugins, the lists can grow without an app update. Writing your own is
covered in [plugin authoring](../dev/plugins-authoring.md).

## Where they appear

On ACE-Step 1.5 the pickers are in the **Generation** section of the generation bar:

- **Solver**. One-evaluation solvers come first, then multi-evaluation solvers with their
  NFE added to the name, then adaptive solvers that choose their own count. The default is
  Euler.
- **Schedule**. Besides the scheduler plugins, the list has three entries built into the
  app that are not plugins: **Beta (Custom)**, **Power** and **Composite (2-Stage)**, each
  with its own sliders. The default is Linear.
- **Guidance**. The default is APG.

Hover a picker's label to read the selected plugin's own description.

When the selected plugin declares settings of its own, a **\<plugin name\> Controls**
fold-out appears under its dropdown. It starts collapsed. Inside are the plugin's sliders,
drop-downs, toggles or text boxes, each with a "?" for its explanation, and a **Reset**
button that returns them to the plugin's defaults. Some settings only appear when another
one is set a certain way. The app remembers values per plugin, and keeps a separate set of
picks and values for each backend.

On MiniMax-Music3 the plugins are off by default. Turn on **Sampler Plugins** in the
Generation section to get Solver, Schedule and Guidance pickers. Each has a **Native**
option that keeps MiniMax-Music3's own tested sampling for that stage. This mode is
experimental: the plugins were written and tuned for the ACE-Step DiT, and stochastic
solvers can disturb the seams between the windows MiniMax-Music3 renders. Solvers that
take over the whole sampling loop, and the Beta, Power and Composite schedules, are not
offered there.

On Stable Audio 3 the Generation section has its own **Sampler** choice. Its default, MD
Pingpong Blend, is itself a solver plugin and needs nothing else switched on. The shared
pickers work as on MiniMax-Music3: turn on **Sampler Plugins**, and each picker has a
**Native** option. Full-loop solvers and the Beta, Power and Composite schedules are not
offered there either. Guidance does little, because Stable Audio 3 has no unconditional
prediction to steer against.

YuE2 does not offer these pickers. Its Generation section has its own **NAR Solver** and
**NAR Scheduler** choices instead. The defaults, Midpoint and Uniform, are YuE2's own
sampling; the experimental choices run the YuE2 plugins in the tables below, plus MD HT
Scheduler V3. Most of them show their settings as YuE2 controls under the choice.
Ancestral Euler, Plan-Aware and Plan-Loose have none and run on their plugin defaults
unless a profile sets them, as the [Plan Play presets](presets/yue2-plan-play/README.md)
do for Ancestral Euler and Plan-Loose.

The StableStep refine in the **Post-processing** section has its own Solver, Schedule and
Guidance pickers, separate from the generation ones. Guidance does little there, for the
same reason. Solvers that take over the whole sampling loop are listed there too, but the
refine runs its native ping-pong step in their place and says so in the engine log.

A fourth plugin type, post-process, replaces the ACE-Step VAE decode. Those plugins appear
in the Post-processing section as the Tiled Decoder and are listed at the end of this
page.

## Adding a plugin

Plugins are `.lua` files, and the engine looks in up to three places, each with one
subfolder per type (`solvers`, `schedulers`, `guidance`, `postprocess`):

| Folder | What goes there |
|---|---|
| `engine/plugins/<type>/` | The built-in set that ships with the app |
| `plugins/<type>/` at the top of the install folder | Community plugins. The app ships its third-party plugins here (the `md_*` files and STORM), with user manuals for two of them in `plugins/docs/` |
| `plugins/<type>/` in the data folder | Your own plugins in a release build. The data folder stays put when the app is upgraded, and a plugin here replaces a shipped one of the same name |

To add one, copy the `.lua` file into the right subfolder and restart the app. The engine
reads the folders once at startup, in the order above, and the new plugin appears in its
dropdown with its settings. Nothing needs rebuilding. Some plugins load a companion file
from their own folder with `require`: the shipped `md_*` solvers share
`md_solver_commons.lua`, for example. Copy any companion file into the same subfolder. Keep
your own plugins in the data folder's `plugins/` in a release build, or in `plugins/` in a
source checkout, so they stay apart from the built-in set.

If a plugin does not show up, check the newest `logs/<session>/ace_engine.log`. The
engine logs a `[Plugins] Loaded ...` count at startup, an error with the Lua message for a
file that fails to load, and a warning for each file it skips: a file that declares the
wrong type for its folder, or a name that is already taken. When two plugins share a
name the first one loaded wins, so a community plugin in the app's `plugins/` folder
cannot replace a built-in one of the same name. The exception is the data folder's
`plugins/` in release builds: a plugin there replaces the built-in of the same name, and
the log says which file won.

Plugins run in a sandbox with no access to files or the operating system. They can still
produce bad audio if they are wrong, so treat a new one like any other setting and compare
it against a known result.

## Solvers

NFE 0 means the solver decides how many evaluations each step needs. Stochastic solvers
add fresh noise during sampling. Params
counts the settings the plugin adds to the UI. Source says which folder the file is in.

<!-- generated:start solvers -->
<!-- generated by tools/docs/build-docs.mjs, do not edit by hand -->
| Solver | Id | NFE | Order | Stochastic | Params | Description | Source |
|---|---|---|---|---|---|---|---|
| A-FloPS (1 NFE) | `aflops` | 1 | 2 |  |  | Exponential integrator with residual tracking | built in, `engine/plugins/solvers/aflops.lua` |
| A-FloPS Midpoint (2 NFE) | `aflops2` | 2 | 2 |  |  | Midpoint-corrected exponential integrator | built in, `engine/plugins/solvers/aflops2.lua` |
| DOP853 (13 NFE) | `dop853` | 13 | 8 |  |  | 8th order Dormand-Prince (maximum accuracy) | built in, `engine/plugins/solvers/dop853.lua` |
| DOPRI5 (7+ NFE) | `dopri5` | 0 | 5 |  |  | Adaptive Dormand-Prince 5th order with error control | built in, `engine/plugins/solvers/dopri5.lua` |
| DPM++ 2M | `dpm2m` | 1 | 2 |  |  | 2nd order Adams-Bashforth multistep (1 NFE) | built in, `engine/plugins/solvers/dpm2m.lua` |
| DPM++ 2M Adaptive | `dpm2m_ada` | 1 | 2 |  |  | Step-ratio-corrected AB2 for non-uniform schedules | built in, `engine/plugins/solvers/dpm2m_ada.lua` |
| DPM++ 3M | `dpm3m` | 1 | 3 |  |  | 3rd order Adams-Bashforth multistep (1 NFE) | built in, `engine/plugins/solvers/dpm3m.lua` |
| Euler (ODE) | `euler` | 1 | 1 |  |  | First-order Euler step (default) | built in, `engine/plugins/solvers/euler.lua` |
| Gauss-Legendre 2s (6 NFE) | `gl2s` | 6 | 4 |  |  | Implicit 4th-order A-stable symplectic integrator | built in, `engine/plugins/solvers/gl2s.lua` |
| Heun (2 NFE) | `heun` | 2 | 2 |  |  | Second-order predictor-corrector | built in, `engine/plugins/solvers/heun.lua` |
| JKASS Fast | `jkass_fast` | 1 | 1 |  | 3 | Euler with beat stability, frequency damping, and temporal smoothing | built in, `engine/plugins/solvers/jkass_fast.lua` |
| JKASS Quality (2 NFE) | `jkass_quality` | 2 | 2 |  |  | Heun predictor-corrector with derivative averaging | built in, `engine/plugins/solvers/jkass_quality.lua` |
| MD Ancestral Euler (YuE2) | `md_ancestral_yue2` | 1 | 1 |  | 1 | One-evaluation Euler step with per-step noise re-injection (churn). churn 0 = plain Euler. | community, `plugins/solvers/md_ancestral_yue2.lua` |
| MD Confluence V3 | `md_confluence_v3` | 1 | 1 | yes | 40 | STORM / Trajectory Anchor hybrid with batch-aware routing. Disagreement- and inertia-modulated latent blend. Per-batch tonal anchor, spectral guard, and RMS servo. | built in, `engine/plugins/solvers/md_confluence_v3.lua` |
| MD Confluence V4 | `md_confluence_v4` | 1 | 1 | yes | 42 | STORM / Trajectory Anchor hybrid with batch-aware routing. Disagreement- and inertia-modulated latent blend. Per-batch tonal anchor, spectral guard, and RMS servo. | community, `plugins/solvers/md_confluence_v4.lua` |
| MD Confluence V5 | `md_confluence_v5` | 1 | 1 | yes | 42 | STORM / Trajectory Anchor hybrid with batch-aware routing. Disagreement- and inertia-modulated latent blend. Per-batch tonal anchor, spectral guard, and RMS servo. | community, `plugins/solvers/md_confluence_v5.lua` |
| MD Dual-Time V2 | `md_dual_time_v2` | 1 | 1 | yes | 7 | Inner convergence loop sampler. Zero-NFE velocity history interpolation. Sigma-adaptive inner blend. Batch-aware, shared anchor stack. | community, `plugins/solvers/md_dual_time_v2.lua` |
| MD Dual-Time V3 | `md_dual_time_v3` | 1 | 1 | yes | 7 | Inner convergence loop sampler. Zero-NFE velocity history interpolation. Sigma-adaptive inner blend. Batch-aware, shared anchor stack. | community, `plugins/solvers/md_dual_time_v3.lua` |
| MD Eigenflow V1 | `md_eigenflow_v1` | 1 | 1 | yes | 6 | PCA trajectory filtering sampler. Power iteration on velocity history, batch-aware, shared anchor stack. | community, `plugins/solvers/md_eigenflow_v1.lua` |
| MD Eigenflow V2 | `md_eigenflow_v2` | 1 | 1 | yes | 6 | PCA trajectory filtering sampler. Power iteration on velocity history, batch-aware, shared anchor stack. | community, `plugins/solvers/md_eigenflow_v2.lua` |
| MD Fractional Solver (FABM v1.1) | `md_fractional_solver` | 1 | 2 |  | 5 | Fractional Adams-Bashforth-Moulton predictor-corrector. Alpha < 1 = subdiffusive (memory-heavy, sticky), alpha > 1 = superdiffusive (aggressive jumps). GL kernel provides built-in structural memory. Validated baseline: pair with smooth non-uniform schedulers (e.g. MD Psychoacoustic), avoid scattered/randomized schedulers. | community, `plugins/solvers/md_fractional_solver_v1.1.lua` |
| MD Granular Collider (Kinetic Intermodulation) | `md_granular_collider` | 1 | 1 |  | 7 | Shatters velocity into grains. High energy deltas between grains spawn intermodulation sidebands. True granular physics inside the ODE. | community, `plugins/solvers/md_granular_collider_v2.0.lua` |
| MD Hamiltonian V2 | `md_hamiltonian_v2` | 1 | 1 | yes | 14 | Energy-conserving momentum-augmented sampler. Euler + momentum correction, spectral weighting, Hamiltonian tracking. Shared anchor stack. | community, `plugins/solvers/md_hamiltonian_v2.lua` |
| MD Hamiltonian V3 | `md_hamiltonian_v3` | 1 | 1 | yes | 14 | Energy-conserving momentum-augmented sampler. Euler + momentum correction, spectral weighting, Hamiltonian tracking. Shared anchor stack. | community, `plugins/solvers/md_hamiltonian_v3.lua` |
| MD Latent Channel EQ (YuE2) | `md_latent_eq_yue2` | 1 | 1 |  | 7 | YuE2 NAR: Euler with per-channel-group velocity gain (8 groups of 8 latent channels). | community, `plugins/solvers/md_latent_eq_yue2.lua` |
| MD Lévy Flight (Alpha-Stable SDE v1.1) | `md_levy_flight` | 1 | 1 | yes | 5 | SDE solver with Lévy alpha-stable noise. Heavy tails enable latent space exploration beyond Gaussian reach. CMS algorithm, truncated for stability. Validated baseline: alpha=1.65, eta=0.8. | community, `plugins/solvers/md_levy_flight_v1.1.lua` |
| MD OmniRelational V3 (Sigma Adaptive) | `md_omni_relational_V3` | 1 | 1 | yes | 8 | Barbour Best Matching with sigma-adaptive relational weight and look-back smoother. rw fades toward pure Euler at low sigma. Proper state reset between generations. | community, `plugins/solvers/md_omni_relational_V3.lua` |
| MD OmniRelational V4 (Per-Block Equalization) | `md_omni_relational_V4` | 1 | 1 | yes | 8 | Barbour Best Matching with sigma-adaptive relational weight and look-back smoother. rw fades toward pure Euler at low sigma. Proper state reset between generations. | community, `plugins/solvers/md_omni_relational_V4.lua` |
| MD p-Adic Solver (Ultrametric v1.1) | `md_padic_solver` | 1 | 1 |  | 5 | Ultrametric velocity weighting via p-adic distance. Hierarchically similar velocities dominate regardless of recency. Validated clean at stock defaults. | community, `plugins/solvers/md_padic_solver_v1.1.lua` |
| MD Pingpong Blend | `md_pingpong_blend` | 1 | 1 | yes |  | Pingpong sampler (full fresh renoise each step). Recommended default for SA3 at 8 steps. | community, `plugins/solvers/md_pingpong_blend.lua` |
| MD PingPong Simple (Ancestral) | `md_pingpong_simple` | 1 | 1 | yes | 11 | Ancestral Euler with momentum, noise coherence, look-back SNR smoother, and domain-tunable RMS servo. Single-branch stochastic sampler. Port of MD_PingPong_Samplers v3.5. | community, `plugins/solvers/md_pingpong_simple.lua` |
| MD STORM V4 | `md_storm_V4` | 0 | 5 |  | 15 | Adaptive STORK/DPM++3M hybrid with relational velocity decomposition, pseudo-LTE error estimation, look-back SNR smoother, stiffness dispatch. | community, `plugins/solvers/md_storm_core_V4.lua` |
| MD Timestep Lie (YuE2) | `md_tlie_yue2` | 1 | 1 |  | 6 | YuE2 NAR: Euler on the true schedule, model queried at a falsified t. + lie = more invented detail, - lie = conservative. | community, `plugins/solvers/md_tlie_yue2.lua` |
| MD Trajectory Anchor | `md_trajectory_anchor` | 1 | 1 | yes | 25 | Latent path stabilizer. step() solver — guiders stay active. Inertia engine, concept lock, identity anchor, tonal anchor, memory buffer, look-back smoother, RMS servo. All stateful. State resets cleanly between generations. Eta=0 for pure ODE, eta>0 for SDE noise. | community, `plugins/solvers/md_trajectory_anchor.lua` |
| MD Trajectory Anchor V3 | `md_trajectory_anchor_V3` | 1 | 1 | yes | 27 | Latent path stabilizer. step() solver — guiders stay active. Inertia engine, concept lock, identity anchor, tonal anchor, memory buffer, look-back smoother, RMS servo. All stateful. State resets cleanly between generations. | community, `plugins/solvers/md_trajectory_anchor_V3.lua` |
| MD Trajectory Anchor V5 | `md_trajectory_anchor_V5` | 1 | 1 | yes | 27 | Latent path stabilizer. step() solver — guiders stay active. Inertia engine, concept lock, identity anchor, tonal anchor, memory buffer, look-back smoother, RMS servo. All stateful. State resets cleanly between generations. | community, `plugins/solvers/md_trajectory_anchor_V5.lua` |
| MD Vortex V1 | `md_vortex_v1` | 1 | 1 | yes | 6 | Vorticity damping sampler. Multi-scale curl proxy, enstrophy tracking, targeted shedding. Batch-aware, shared anchor stack. | community, `plugins/solvers/md_vortex_v1.lua` |
| MD Vortex V2 | `md_vortex_v2` | 1 | 1 | yes | 6 | Vorticity damping sampler. Multi-scale curl proxy, enstrophy tracking, targeted shedding. Batch-aware, shared anchor stack. | community, `plugins/solvers/md_vortex_v2.lua` |
| MD Wasserstein Flow (Proximal OT v1.0) | `md_wasserstein_flow` | 1 | 1 |  | 6 | Proximal optimal transport solver. Wasserstein-regularized steps with spectral flatness and RMS penalties. Smoother transport than Euler with anti-metallic properties. | community, `plugins/solvers/md_wasserstein_flow_v1.0.lua` |
| MD Wasserstein Flow (Proximal OT v1.2.1) | `md_wasserstein_flow_v121` | 1 | 1 |  | 7 | Proximal optimal transport solver. Split projection: spectral flatness transverse to velocity, RMS radial. Trust region bounded, zero-alloc scratch buffers. | community, `plugins/solvers/md_wasserstein_flow_v1_2_1.lua` |
| MD Wasserstein Flow (YuE2) | `md_wasserstein_yue2` | 1 | 1 |  | 8 | One-evaluation Euler step with the shared proximal latent regularizer. Tuned for YuE2 NAR. | community, `plugins/solvers/md_wasserstein_yue2.lua` |
| RF-Solver (2 NFE) | `rfsolver` | 2 | 2 |  |  | Rectified-flow-aware midpoint solver | built in, `engine/plugins/solvers/rfsolver.lua` |
| RK4 (4 NFE) | `rk4` | 4 | 4 |  |  | Classic 4th-order Runge-Kutta | built in, `engine/plugins/solvers/rk4.lua` |
| RK5 (6 NFE) | `rk5` | 6 | 5 |  |  | 5th-order Cash-Karp Runge-Kutta | built in, `engine/plugins/solvers/rk5.lua` |
| SDE (Stochastic) | `sde` | 1 | 1 | yes |  | SDE Euler with Philox re-noising for diversity | built in, `engine/plugins/solvers/sde.lua` |
| STORK 2 | `stork2` | 1 | 2 |  | 1 | 2nd-order stabilized Taylor-Chebyshev (1 NFE) | built in, `engine/plugins/solvers/stork2.lua` |
| STORK 4 | `stork4` | 1 | 4 |  | 1 | 4th-order stabilized Taylor-ROCK4 (1 NFE) | built in, `engine/plugins/solvers/stork4.lua` |
| STORM | `storm` | 0 | 5 |  | 6 | Adaptive STORK/DPM++3M hybrid with stiffness detection | community, `plugins/solvers/storm_sampler_core.lua` |
| UniPC (2 NFE) | `unipc` | 2 | 2 |  |  | Unified predictor-corrector in log-SNR space | built in, `engine/plugins/solvers/unipc.lua` |
| UniPC Predictor (1 NFE) | `unipc_p` | 1 | 2 |  |  | UniPC predictor-only (no corrector, 1 NFE) | built in, `engine/plugins/solvers/unipc_p.lua` |
<!-- generated:end solvers -->

## Schedulers

MD HT Scheduler v5.1 ships twice, built in and as a community copy with the same name.
The engine keeps the first one it loads, the built-in copy, and logs the other as a
duplicate.

<!-- generated:start schedulers -->
<!-- generated by tools/docs/build-docs.mjs, do not edit by hand -->
| Scheduler | Id | Params | Description | Source |
|---|---|---|---|---|
| Beta 57 | `beta57` |  | Beta(0.5,0.7) — smooth S-curve from RES4LYF | built in, `engine/plugins/schedulers/beta57.lua` |
| Cosine | `cosine` |  | Cosine annealing — balanced S-curve | built in, `engine/plugins/schedulers/cosine.lua` |
| DDIM Uniform | `ddim_uniform` |  | Log-SNR uniform (S-shaped) | built in, `engine/plugins/schedulers/ddim_uniform.lua` |
| Linear | `linear` |  | Uniform spacing (default) | built in, `engine/plugins/schedulers/linear.lua` |
| Linear-Quadratic | `linear_quadratic` |  | Linear start, quadratic finish | built in, `engine/plugins/schedulers/linear_quadratic.lua` |
| MD Causal (LINA + 14 Modes) | `md_causal` | 10 | 14 base schedule modes with LINA time-axis warp. Karras, smoothstep, beta, AYS, bong, DDIM, SGM, blended, variance-preserving, KL-optimal and more. Port of md_causal_scheduler_core v2.2. | community, `plugins/schedulers/md_causal.lua` |
| MD HAP (Hamiltonian Potential Well) | `md_hap` | 2 | Particle-in-potential-well sigma schedule. Kinetic energy stretches mid steps, damping friction compresses end steps. Port of hap_scheduler_core v1.0. | community, `plugins/schedulers/md_hap.lua` |
| MD HT Scheduler (HAP+TPT) V3 | `md_ht_scheduler V3` | 14 | Thermodynamic timestep schedule: HAP + TPT additive density, density floor, LINA warp, SNR-space, poly slope, uniformity blend, smoothing. 12 to 150+ steps. | community, `plugins/schedulers/md_ht_scheduler_V3.lua` |
| MD HT Scheduler (HAP+TPT) v5.1 | `md_ht_scheduler v5.1` | 15 | Thermodynamic timestep schedule: HAP + TPT additive density, density floor, LINA warp, SNR-space, poly slope, uniformity blend, smoothing, auto-shift compensation for low step counts. 12 to 150+ steps. | built in, `engine/plugins/schedulers/md_ht_scheduler_v5_1.lua` |
| MD HT Scheduler (HAP+TPT) v5.1 | `md_ht_scheduler v5.1` | 15 | Thermodynamic timestep schedule: HAP + TPT additive density, density floor, LINA warp, SNR-space, poly slope, uniformity blend, smoothing, auto-shift compensation for low step counts. 12 to 150+ steps. | community, `plugins/schedulers/md_ht_scheduler_V5.1.lua` |
| MD Noise Decay (Multi-Algorithm) V2 | `md_noise_decay V2` | 8 | 6-algorithm noise decay scheduler: polynomial, sigmoidal, piecewise, fourier, exponential, gaussian. Memory-optimized. | community, `plugins/schedulers/md_noise_decay_V2.lua` |
| MD Plan Overdrive (YuE2) | `md_plan_overdrive_yue2` | 5 | YuE2 only: stock uniform timesteps plus a per-step plan strength curve (w up to 3; engine clamp 0..4). | community, `plugins/schedulers/md_plan_overdrive_yue2.lua` |
| MD Plan-Aware (YuE2) | `md_plan_aware_yue2` | 2 | YuE2 only: denser high-noise steps in NAR chunks that overlap a chosen plan section; uniform elsewhere. | community, `plugins/schedulers/md_plan_aware_yue2.lua` |
| MD Plan-Loose (YuE2) | `md_plan_loose_yue2` | 2 | YuE2 only: uniform steps, plan conditioning loosened early and locked in later. | community, `plugins/schedulers/md_plan_loose_yue2.lua` |
| MD Psychoacoustic (Complexity-Adaptive) | `md_psychoacoustic` | 6 | Clusters step density around phase transition zones using psychoacoustic masking curves. Optional Fibonacci golden-ratio spacing. Focuses compute where perceptual impact is highest. | community, `plugins/schedulers/md_psychoacoustic_v1_2.lua` |
| MD QRD Scatter (Acoustic Diffusion) | `md_qrd_scatter` | 4 | Uses prime number quadratic residues to scatter sigma step intervals. Eliminates harmonic step-ringing by destroying uniform grids. | community, `plugins/schedulers/md_qrd_scatter_v2.0.lua` |
| Power (p=2) | `power` | 1 | Power-law t^p, front-loaded | built in, `engine/plugins/schedulers/power.lua` |
| SGM-Uniform (Karras) | `sgm_uniform` |  | Karras σ-ramp (ρ=7), front-loads structural steps | built in, `engine/plugins/schedulers/sgm_uniform.lua` |
| Tangent | `bong_tangent` |  | Front-loaded (structural focus) | built in, `engine/plugins/schedulers/bong_tangent.lua` |
<!-- generated:end schedulers -->

## Guidance modes

<!-- generated:start guidance -->
<!-- generated by tools/docs/build-docs.mjs, do not edit by hand -->
| Guidance mode | Id | Params | Description | Source |
|---|---|---|---|---|
| ADG (ACE-Step) | `adg` |  | Angle-based Dynamic Guidance for ACE-Step base models. Rotates clean-latent estimates with a clipped angle. Other backends use ordinary CFG. | built in, `engine/plugins/guidance/adg.lua` |
| APG | `apg` | 2 | Adaptive perpendicular guidance (default) | built in, `engine/plugins/guidance/apg.lua` |
| CFG-MP | `cfg_mp` | 1 | Manifold projection guidance (Su et al. 2025) | built in, `engine/plugins/guidance/cfg_mp.lua` |
| CFG-Zero⋆ | `cfg_zero_star` | 1 | Zero-init + APG guidance (Fan et al. 2025) | built in, `engine/plugins/guidance/cfg_zero_star.lua` |
| CFG++ | `cfg_pp` |  | Step-scaled guidance for few-step models | built in, `engine/plugins/guidance/cfg_pp.lua` |
| Dynamic CFG | `dynamic_cfg` |  | Cosine-decaying guidance schedule with APG projection | built in, `engine/plugins/guidance/dynamic_cfg.lua` |
| MD ADG (Angle-based Dynamic Guidance) | `md_adg` | 4 | Rotates cond toward the cond-uncond direction. Angle scales with sigma — aggressive at high sigma, settles to identity. Port of pingpong_trilogy v1.4.2. | community, `plugins/guidance/md_adg.lua` |
| MD APG (Adaptive Projected Guidance) V4 | `md_apg V4` | 4 | APG: decomposes cond-uncond diff. Momentum buffer + norm threshold. Memory-optimized. | community, `plugins/guidance/md_apg_V4.lua` |
| MD APT (Temporal Phase Locker) | `md_apt` | 5 | Two-phase guidance: transport (enhanced flow) above exceptional point, trapping (damped) below. Sigmoid blend between phases. Port of apt_symmetry_core v1.0. | community, `plugins/guidance/md_apt.lua` |
| MD Bucket Brigade (Analog Feedback) | `md_bucket_brigade` | 7 | Captures the latent, degrades it, and feeds it back into the generation steps later like a physical analog delay pedal. Creates intense transient glue. | community, `plugins/guidance/md_bucket_brigade_v2.0.lua` |
| MD Clarity V1 | `md_clarity_v1` | 5 | Lightweight post-CFG cleanup. HF smoothing, spike clamping, orthogonal projection. Pairs with MD solvers. | community, `plugins/guidance/md_clarity_v1.lua` |
| MD Custom Noise (Structured Injection) V3 | `md_custom_noise V3` | 4 | 10-algorithm structured noise injection via post_step. Zero-allocation optimized. | community, `plugins/guidance/md_custom_noise_V3.lua` |
| MD Firefly Ringdown (QNM Decay) V2 | `md_firefly_ringdown V2` | 5 | Quasi-normal mode exponential decay damping on guidance velocity. Marginalize-phase preserves direction. Port of firefly_ringdown_core v1.0. | community, `plugins/guidance/md_firefly_ringdown_V2.lua` |
| MD Liquid PRO (Non-Newtonian Fluid) | `md_liquid_pro` | 6 | Belt/Gear fluid guidance with oobleck shear-thickening. tanh flow state splits advective vs frictional regimes. Port of liquid_guider_pro_core v1.0. | community, `plugins/guidance/md_liquid_pro.lua` |
| MD Phase Inversion (Anti-Phase Cancellation) | `md_phase_inversion` | 10 | Noise cancellation for latent space: decomposes uncond into smooth (generic) + specific components, generates anti-phase vector to cancel AI smoothness. Sigma-adaptive. Optional detail recovery via post_step. | community, `plugins/guidance/md_phase_inversion_v1_2.lua` |
| MD STORM Guidance V2 | `md_storm_guidance` | 10 | Companion guidance plugin for STORM. (1) Sigma-aware CFG adaptation — flat at full scale until knee, then tail rolloff to prevent late-step over-sharpening and harmonic ringing. (2) NAG post-step latent suppression — soft-clamps spiking elements to suppress resonance buildup in LoRA inference. Generation-boundary safe. | community, `plugins/guidance/md_storm_guidance.lua` |
| Rescaled CFG | `rescaled_cfg` |  | Std-matched to prevent saturation | built in, `engine/plugins/guidance/rescaled_cfg.lua` |
| SMC-CFG | `smc_cfg` | 2 | Sliding mode control guidance (Han et al. 2025) | built in, `engine/plugins/guidance/smc_cfg.lua` |
<!-- generated:end guidance -->

## Post-process plugins

Replace the VAE decode: the plugin receives the DiT's latents, decodes them itself (in
tiles, for the ones below) and can process the audio before handing it back. Only
community plugins exist for this type so far.

<!-- generated:start postprocess -->
<!-- generated by tools/docs/build-docs.mjs, do not edit by hand -->
| Post-process plugin | Id | Params | Description | Source |
|---|---|---|---|---|
| MD Audio Tiled Decoder | `md_audio_tiled` | 6 | Tiled VAE decode with OLA crossfade, dual-pass merge, LSS, and DSP chain | community, `plugins/postprocess/md_audio_tiled.lua` |
| MD Audio Tiled Decoder V2 | `md_audio_tiled V2` | 17 | Tiled VAE decode with OLA crossfade, dual-pass merge, LSS, and DSP chain | community, `plugins/postprocess/md_audio_tiled_V2.lua` |
| MD Audio Tiled Decoder V2.1 | `md_audio_tiled V2.1` | 19 | Tiled VAE decode with OLA crossfade, dual-pass merge, LSS, and DSP chain | community, `plugins/postprocess/md_audio_tiled_V2.1.lua` |
<!-- generated:end postprocess -->

## Choosing

Start from the defaults. Which combinations are worth trying, how many steps each solver
needs, and what a multi-NFE solver costs are covered in
[Steps and solver](quality.md#steps-and-solver). Change one picker at a time on a fixed
seed, as described in
[Comparing two settings on the same seed](quality.md#comparing-two-settings-on-the-same-seed).

## Related

- [Generation](generation.md)
- [Backends](backends.md)
- [Getting higher quality output](quality.md)
- [Plugin authoring](../dev/plugins-authoring.md)

## Release data paths

In release builds, put your plugins in the shared data folder's `plugins/`, under the appropriate category. Shipped plugins stay with the app; user plugins with the same name take precedence. See [Data folder and upgrades](data-folder.md).
