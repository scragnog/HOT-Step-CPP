# DCW Correction Fixes — Upstream Parity + Future Improvements

## Problem

When DCW (Differential Correction in Wavelet domain) is enabled, generated tracks
exhibit abrupt starts and premature cut-offs, especially in `skipLm` (text2music)
mode. The issue disappears when DCW is disabled.

## Root Cause — Two Bugs in Our C++ Port

Comparing our `hot-step-sampler.h` and `dcw.h` against the upstream Python
implementation (`ace-step-vanilla/acestep/models/common/dcw_primitives.py`,
`dcw_correction.py`, and the sampler loop in `modeling_acestep_v15_xl_base.py`)
reveals **two bugs and a default-value mismatch**.

---

### Bug 1: Denoised Computed from Post-Step Latent (CRITICAL)

The predicted clean sample `denoised = x - v*t` must be computed from the
**pre-step** latent (`xt_before_step`) and the **raw** velocity from the model
evaluation — NOT from the post-step `xt` that the solver has already moved.

**Upstream Python** (correct):
```python
# BEFORE solver step:
xt_before_step = xt          # cache pre-step state
vt_for_denoise = vt          # cache raw velocity

# solver step updates xt...
xt = xt - vt * dt_tensor

# DCW uses pre-step state:
denoised = xt_before_step - vt_for_denoise * t_unsq
xt = dcw_corrector.apply(xt, denoised, t_curr_f)
```

**Our C++** (wrong — `hot-step-sampler.h:694-699`):
```cpp
// solver step modifies xt[] in-place
solver_info->step_fn(xt.data(), vt.data(), ...);

// DCW uses POST-step xt (BUG):
for (int i = 0; i < n_total; i++) {
    denoised[i] = xt[i] - vt[i] * t_curr;  // xt is already stepped!
}
```

**Impact:** The post-step `xt` has already been moved toward `x0` by the solver.
Computing `denoised = xt_post - v*t` yields a bogus quantity that is neither the
predicted clean sample nor the current noisy state. The wavelet correction then
pushes the latent in an incorrect direction, systematically corrupting temporal
structure. This is the most likely cause of the abrupt starts/ends.

**Fix:** Cache `xt` before the solver step, use the cached copy to compute denoised.

---

### Bug 2: Spurious Step-Count Normalization

Our C++ applies a `step_norm = dt * 20` multiplier to the effective scaler:

```cpp
float dt = t_curr - t_next;
float step_norm = dt * 20.0f;
float eff_scaler = t_curr * g_hotstep_params.dcw_scaler * step_norm;
```

The upstream Python does NOT do this:
```python
low_s = t * self.scaler         # just t * scaler, no dt or step normalization
high_s = (1.0 - t) * self.scaler
```

**Impact:** At 50 steps, `dt ~ 0.02`, `step_norm ~ 0.4` — roughly comparable to
upstream. At 8 steps (turbo), `dt ~ 0.125`, `step_norm ~ 2.5` — applying **5x**
the intended correction, severely distorting the latent.

**Fix:** Remove `step_norm`. Use `t_curr * scaler` directly, matching upstream.

---

### Mismatch 3: Default Scaler Values Are 2x Upstream

| Parameter        | Our default | Upstream default |
|------------------|-------------|------------------|
| `dcw_scaler`     | **0.10**    | 0.05             |
| `dcw_high_scaler`| **0.05**    | 0.02             |

**Fix:** Update defaults in `hot-step-params.h` to match upstream (0.05 / 0.02).

---

## Phase 0 — Upstream Parity Fix (Do This First)

These three changes restore correctness before any experimental work.

### [MODIFY] `engine/src/hot-step-sampler.h`

1. **Cache pre-step latent:** Before the solver dispatch (line ~686), copy `xt`
   into a `xt_pre` vector. After solver step, compute
   `denoised[i] = xt_pre[i] - vt[i] * t_curr`.

2. **Remove step normalization:** Change effective scaler from
   `t_curr * scaler * step_norm` to just `t_curr * scaler` (low) and
   `(1 - t_curr) * high_scaler` (high). Remove the `dt` / `step_norm` variables.

### [MODIFY] `engine/src/hot-step-params.h`

3. **Update defaults:** `dcw_scaler = 0.05f`, `dcw_high_scaler = 0.02f`.

### Verification

- Rebuild via `dev-rebuild.bat`
- Generate with DCW enabled (double mode, default scalers)
- Confirm: no abrupt starts, no premature cut-offs
- Compare perceptual quality against DCW-disabled baseline

---

## Phase 1 — Optional Improvements (After Parity Is Confirmed)

Once Phase 0 restores correctness, these are optional enhancements to further
reduce any remaining boundary sensitivity.

### Option A — Channel-Axis DWT (Alternative Decomposition)

**Rationale:** Even with the bugs fixed, the DWT still operates along the T
(temporal) axis, meaning corrections modify the song's temporal envelope. For
audio latents where T = sequential time, the "low-frequency" band is the song's
pacing, not a quality metric. Applying DWT along the Oc (channel = 64) axis
instead would decompose each frame into spectral envelope + spectral detail,
correcting audio quality without touching temporal structure.

**Implementation:**
- Add new DCW mode `"channel"` or `"spectral"` to `dcw.h`
- For each (batch, frame), extract 64 channel values and run Haar DWT on those
- Correct low/high spectral bands, IDWT, write back
- Add UI toggle to select T-axis vs channel-axis DCW

**Pros:** Fundamentally avoids temporal artifacts; aligns with audio-domain
intuition (spectral correction ~ image spatial correction).

**Cons:** 64 channels is small for DWT (only 32+32 coefficients per frame);
effectiveness is unproven. Upstream chose T-axis deliberately. Also, the Haar
wavelet on 64 channels may not capture meaningful spectral structure since the
64 latent channels are learned representations, not ordered frequency bins.

**Risk:** Medium. Requires A/B testing to confirm it actually improves quality.

---

### Option B — Temporal Margin Window

**Rationale:** Even with correct implementation, the first and last DWT
coefficients directly affect frames 0-1 and T-2 to T-1. If there's any
discontinuity at the song boundaries (silence -> content), the DWT captures it
and the correction amplifies it.

**Implementation:**
- After DCW correction, apply a Tukey-style window that blends the corrected
  and uncorrected latents at the boundaries:
  ```
  margin = 250 frames (10 seconds at 25 Hz)
  for t in [0, margin):
      w = t / margin  (ramp from 0 to 1)
      xt[t] = w * xt_corrected[t] + (1-w) * xt_uncorrected[t]
  ```
  Same at the end boundary.
- Expose `dcw_margin_frames` as a configurable parameter (default: 250)

**Pros:** Simple, low-risk, preserves interior correction behaviour.

**Cons:** Doesn't address the fundamental T-axis question; the margin size is
arbitrary and may need tuning per song length.

**Risk:** Low. Minimal code change, easy to A/B test.

---

## Recommended Execution Order

1. **Phase 0** — Fix the two bugs + defaults. This alone may fully resolve the
   issue since the upstream Python doesn't exhibit these artifacts.
2. **Test Phase 0** — If starts/ends are now clean, we're done.
3. **If artifacts persist** -> Implement Option B (temporal margin) as a safety
   net. It's the lowest-risk additional measure.
4. **Option A** (channel-axis) is a research experiment — only pursue if there's
   a specific quality motivation beyond the temporal boundary issue.
