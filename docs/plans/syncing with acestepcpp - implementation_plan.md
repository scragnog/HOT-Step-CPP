# Upstream Sync + Extension Layer — acestep.cpp → HOT-Step-CPP Engine

> Sync upstream changes, then refactor so future syncs are trivial.

## Background

Our engine (`D:\Ace-Step-Latest\hot-step-cpp\engine\`) is a heavily modified fork of [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp). We diverged at commit `9c234756` ~1 week ago. Since then, upstream has 12 src/ commits (+2,700 lines changed), and we've modified all 25 shared files.

**This plan has two goals:**
1. **Sync** — Pull in upstream changes (model-store, safety fixes, simplifications)
2. **Refactor** — Restructure so future syncs are a 30-minute direct-copy instead of a multi-day merge

### Key Assets

| Asset | Path |
|-------|------|
| Our engine | `D:\Ace-Step-Latest\hot-step-cpp\engine\` |
| Vanilla upstream clone | `D:\Ace-Step-Latest\acestepcpp\acestep.cpp\` |
| Upstream remote | `https://github.com/ServeurpersoCom/acestep.cpp` (master) |
| Divergence commit | `9c234756` |
| Current upstream HEAD | `3222db7` |

### Confirmed Decisions

- ✅ **Model-store**: ADOPT — centralise GPU module ownership
- ✅ **DiT sampler simplification**: ADOPT — rebuild plugins on simplified base
- ✅ **ace-server.cpp**: Replace with our own `hot-step-server.cpp`
- ✅ **Extension Layer**: Refactor all our additions into separate files

---

## The Extension Layer Architecture

### The Problem

We modify 25+ upstream files. Every upstream change is a conflict. Syncing is painful.

### The Solution

Treat upstream files as a **library we consume, not source we edit**. All our additions live in our own files.

### Target Directory Layout

```
engine/src/
├── [UPSTREAM ZONE — vanilla files, direct-copy on sync]
│   ├── adapter-merge.h            ← upstream, unmodified
│   ├── audio-io.h                 ← upstream, unmodified
│   ├── audio-resample.h           ← upstream, unmodified
│   ├── backend.h                  ← upstream, unmodified
│   ├── bpe.h                      ← upstream, unmodified
│   ├── cond-enc.h                 ← upstream, unmodified
│   ├── debug.h                    ← upstream, unmodified
│   ├── dit-graph.h                ← upstream, unmodified
│   ├── dit-sampler.h              ← upstream, unmodified
│   ├── dit.h                      ← upstream, unmodified
│   ├── fsq-detok.h                ← upstream, unmodified
│   ├── fsq-tok.h                  ← upstream, unmodified
│   ├── gguf-weights.h             ← upstream, unmodified
│   ├── metadata-fsm.h             ← upstream, unmodified
│   ├── model-store.cpp            ← upstream, unmodified (NEW)
│   ├── model-store.h              ← upstream, unmodified (NEW)
│   ├── philox.h                   ← upstream, unmodified
│   ├── pipeline-lm.cpp            ← upstream, unmodified
│   ├── pipeline-lm.h              ← upstream, unmodified
│   ├── pipeline-synth-impl.h      ← upstream, unmodified
│   ├── pipeline-synth-ops.cpp     ← upstream, unmodified
│   ├── pipeline-synth-ops.h       ← upstream, unmodified
│   ├── pipeline-synth.cpp         ← upstream, unmodified
│   ├── pipeline-synth.h           ← upstream, unmodified
│   ├── pipeline-understand.cpp    ← upstream, unmodified
│   ├── pipeline-understand.h      ← upstream, unmodified
│   ├── prompt.h                   ← upstream, unmodified
│   ├── qwen3-enc.h                ← upstream, unmodified
│   ├── qwen3-lm.h                 ← upstream, unmodified
│   ├── request.cpp                ← upstream, unmodified
│   ├── request.h                  ← upstream, unmodified
│   ├── safetensors.h              ← upstream, unmodified
│   ├── sampling.h                 ← upstream, unmodified
│   ├── task-types.h               ← upstream, unmodified
│   ├── timer.h                    ← upstream, unmodified
│   ├── vae-enc.h                  ← upstream, unmodified
│   ├── vae.h                      ← upstream, unmodified
│   ├── wav.h                      ← upstream, unmodified
│   └── weight-ctx.h               ← upstream, unmodified
│
├── [HOT-STEP ZONE — our files, never in upstream, never conflict]
│   ├── hot-step-server.cpp        ← OUR server binary (replaces ace-server.cpp)
│   ├── hot-step-sampler.h         ← OUR sampling loop with solver/scheduler/guidance
│   ├── hot-step-synth.h           ← Wrapper: upstream synth + mastering + VAE selection
│   ├── hot-step-request.h         ← Extended request fields for our API
│   ├── hot-step-ops.cpp           ← Our custom pipeline ops
│   ├── mastering.h                ← Already separate ✓
│   ├── adapter-runtime.h          ← Already separate ✓
│   ├── model-registry.h           ← Already separate ✓
│   ├── guidance/                  ← Already separate ✓
│   ├── schedulers/                ← Already separate ✓
│   ├── solvers/                   ← Already separate ✓
│   └── vendor/dr_flac/            ← Already separate ✓
│
engine/tools/
│   ├── ace-server.cpp             ← upstream, NOT COMPILED by us
│   ├── hot-step-server.cpp        ← OUR server binary
│   └── ... (upstream tools, unmodified)
```

### What Moves Where

#### ace-server.cpp → hot-step-server.cpp
Our 1,448-line ace-server.cpp becomes `hot-step-server.cpp` in `engine/tools/`. CMakeLists.txt compiles `hot-step-server` instead of `ace-server`. The upstream ace-server.cpp stays as a reference but is not compiled.

This eliminates the single biggest merge target.

#### request.h extensions → hot-step-request.h

Upstream's `request.h` has evolved differently from ours. Key differences:

| Field | Upstream | Ours |
|-------|----------|------|
| `repaint_strength` | REMOVED (97be220) | Still present |
| `lm_mode` | ADDED | We use URL params instead |
| `output_format` | ADDED | We handle format in Node.js |
| `synth_model`, `lm_model`, `adapter`, `adapter_scale` | Now in AceRequest | We have these in ServerFields |
| `scheduler` | Not present | ADDED by us |
| `guidance_mode` | Not present | ADDED by us |
| `infer_method` | `"ode"` default | `""` default (our solver dispatch) |
| `stork_substeps`, `beat_stability`, etc. | Not present | ADDED by us |
| `apg_momentum`, `apg_norm_threshold` | Not present | ADDED by us |

**Solution**: Keep upstream's `request.h` vanilla. Create `hot-step-request.h`:

```cpp
// hot-step-request.h — HOT-Step extensions to AceRequest
#pragma once
#include "request.h"

struct HotStepRequest {
    AceRequest base;  // upstream request (composition, not inheritance)

    // solver/scheduler/guidance dispatch
    std::string scheduler;        // "" = linear
    std::string guidance_mode;    // "" = apg
    std::string infer_method;     // "" = auto (our solver dispatch)

    // solver sub-parameters
    int   stork_substeps     = 0;   // 0 = default (10)
    float beat_stability     = -1;  // -1 = default
    float frequency_damping  = -1;
    float temporal_smoothing = -1;

    // guidance sub-parameters
    float apg_momentum       = 0;  // 0 = default (0.75)
    float apg_norm_threshold = 0;  // 0 = default (2.5)
};

// Parse JSON into HotStepRequest (upstream fields → base, ours → extensions)
bool hot_step_request_parse_json(HotStepRequest * r, const char * json);
std::string hot_step_request_to_json(const HotStepRequest * r, bool sparse = true);
```

#### dit-sampler.h → hot-step-sampler.h

Upstream's simplified `dit-sampler.h` stays vanilla. Our solver/scheduler/guidance dispatch lives in `hot-step-sampler.h`:

```cpp
// hot-step-sampler.h — multi-solver sampling loop
#pragma once
#include "dit-sampler.h"  // upstream base (for shared types/helpers)
#include "solvers/solver-interface.h"
#include "schedulers/scheduler-interface.h"
#include "guidance/guidance-interface.h"

// Our sampling loop with plugin dispatch
int hot_step_dit_sample(/* ... params including solver, scheduler, guidance ... */);
```

#### pipeline-synth.cpp → hot-step-synth.h (wrapper)

Mastering runs AFTER full synthesis (after VAE decoding). So we don't need to modify `pipeline-synth.cpp`:

```cpp
// hot-step-synth.h — wraps upstream synth with mastering
#pragma once
#include "pipeline-synth.h"
#include "mastering.h"

struct HotStepSynthExt {
    const char * mastering_ref_path;  // NULL = no mastering
    const char * vae_override;        // NULL = default VAE
    const char * adapter_mode;        // "merge" or "runtime"
};

// Calls ace_synth_generate() then applies mastering post-hoc
int hot_step_synth_generate(AceSynth * ctx, AceRequest * req,
                            const HotStepSynthExt * ext, AceAudio * out);
```

#### pipeline-synth.h — AdapterGroupScales

Our `AdapterGroupScales` struct is currently in `pipeline-synth.h`. Upstream doesn't have it. Two options:
1. Move it to `hot-step-request.h` (cleanest separation)
2. Accept this one small addition to pipeline-synth.h (minimal diff)

**Recommendation**: Move to `hot-step-request.h` for full separation.

---

## Merge Surface Comparison

| | Before Refactor | After Refactor |
|---|---|---|
| Files needing manual merge on sync | **25+** | **1** (hot-step-sampler.h) |
| Lines of conflict per sync | **~2,700** | **~50–100** |
| Sync process | Multi-day merge | 30-min copy + rebuild |
| Risk of breaking our features | High | Low |

The only file that needs manual attention on future syncs is `hot-step-sampler.h`, because our solver dispatch IS the sampling loop and can't be trivially wrapped. But it's ONE file, and upstream changes to the sampler are infrequent.

---

## Execution Plan

### Phase A: Setup (Task 1)

#### Task 1: Create Workflow Document + Marker File
**Files:**
- Create: `engine/UPSTREAM_SYNC` — marker with commit `9c234756`
- Create: `.agents/workflows/upstream-sync.md` — the repeatable 6-phase process

**Commit:** `chore: add upstream sync workflow and marker file`

---

### Phase B: Upstream Sync (Tasks 2–5)

Bring our engine up to date with upstream HEAD (`3222db7`).

#### Task 2: Direct-Copy Upstream Files We Haven't Substantively Modified

Copy the upstream HEAD version of files where our modifications were minimal (formatting, minor additions that the upstream version now supersedes).

**Strategy**: For each file, diff our version against the upstream divergence point. If our changes are purely covered by the upstream changes (e.g., we added OOM checks that upstream also added), take the upstream version.

**Files to evaluate:**
| File | Likely Action |
|------|---------------|
| `audio-resample.h` | Take upstream (our changes were minor) |
| `cond-enc.h` | Take upstream |
| `fsq-tok.h` | Take upstream |
| `qwen3-enc.h` | Take upstream |
| `weight-ctx.h` | Take upstream |
| `wav.h` | Take upstream |
| `vae-enc.h` | Take upstream |
| `vae.h` | Take upstream (ScragVAE selection moves to hot-step-synth.h) |
| `gguf-weights.h` | Take upstream (tensor bounds verification) |
| `task-types.h` | Take upstream |
| `adapter-merge.h` | Take upstream (runtime adapter logic stays in adapter-runtime.h) |
| `backend.h`, `bpe.h`, `debug.h`, `philox.h`, etc. | Take upstream (verify identical) |

**For each**: compare baseline→ours diff vs baseline→theirs diff. If our additions are already in upstream OR can move to our extension files, take upstream wholesale.

**Commit:** `sync: take upstream vanilla for non-modified files`

---

#### Task 3: Adopt Model-Store + Pipeline Refactor

**Sub-steps:**

**3a. Copy new upstream files:**
- `model-store.cpp` → `engine/src/model-store.cpp`
- `model-store.h` → `engine/src/model-store.h`
- Update `CMakeLists.txt` to compile `model-store.cpp`
- Build to verify clean compile

**3b. Take upstream pipeline files wholesale:**
- `pipeline-synth.cpp` → take upstream version (our mastering/VAE extensions move to hot-step-synth.h)
- `pipeline-synth.h` → take upstream version (AdapterGroupScales moves to hot-step-request.h)
- `pipeline-synth-impl.h` → take upstream version
- `pipeline-synth-ops.cpp` → take upstream version (our custom ops move to hot-step-ops.cpp)
- `pipeline-synth-ops.h` → take upstream version
- `pipeline-lm.cpp` → take upstream version
- `pipeline-lm.h` → take upstream version
- `pipeline-understand.cpp` → take upstream version
- `pipeline-understand.h` → take upstream version
- `dit.h` → take upstream version
- `dit-graph.h` → take upstream version (verify if changed)

**3c. Take upstream request files:**
- `request.h` → take upstream version (our extensions move to hot-step-request.h)
- `request.cpp` → take upstream version

**Build won't compile yet** — our ace-server.cpp references the old API. That's expected, it gets replaced in Task 4.

**Commit:** `sync: adopt model-store + take upstream pipeline/request files`

---

#### Task 4: Create Extension Layer Files

This is where we rebuild our additions on top of the upstream base.

**4a. Create `hot-step-request.h` + `hot-step-request.cpp`:**
- Move `AdapterGroupScales` from pipeline-synth.h
- Move solver/scheduler/guidance fields from request.h
- Move solver sub-params (`stork_substeps`, etc.) from request.h
- Move guidance sub-params (`apg_momentum`, etc.) from request.h
- Implement JSON parse/serialize for our extension fields

**4b. Create `hot-step-sampler.h`:**
- Take our current dit-sampler.h as the starting point
- Refactor to use upstream's simplified types/helpers where possible
- Keep our solver/scheduler/guidance plugin dispatch
- Port upstream DiT batching fix (`0986cca`)
- Verify all 14 solvers, 8 schedulers, 4 guidance modes compile cleanly

**4c. Create `hot-step-synth.h`:**
- Wrapper around upstream's `ace_synth_generate()` / `ace_synth_job_run_dit()` / `ace_synth_job_run_vae()`
- Adds mastering post-processing (calls into mastering.h after VAE decode)
- Handles our VAE model selection (resolving ScragVAE)
- Handles adapter-mode dispatch (merge vs runtime)

**4d. Create `hot-step-ops.cpp`:**
- Move our custom pipeline ops from pipeline-synth-ops.cpp
- Mastering op implementation
- Adapter-runtime op implementation

**4e. Move `hot-step-server.cpp` (from ace-server.cpp):**
- Copy our current ace-server.cpp → `engine/tools/hot-step-server.cpp`
- Update it to use `HotStepRequest` for our extension fields
- Update it to call `hot_step_synth_generate()` wrapper instead of direct pipeline calls
- Update it to use model-store API (from upstream) for model lifecycle
- Keep our ServerFields, adapter resolution, VAE selection, per-group scales
- Update `CMakeLists.txt`: compile `hot-step-server` instead of `ace-server`

**Build + test**: Full compile, basic generation test.

**Commit:** `refactor: extension layer — hot-step-request, hot-step-sampler, hot-step-synth, hot-step-server`

---

#### Task 5: Take Upstream dit-sampler.h + Remaining Files

Now that our sampling logic is in `hot-step-sampler.h`, we can take upstream's vanilla `dit-sampler.h`:

- `dit-sampler.h` → take upstream version (kept as reference, our binary uses hot-step-sampler.h)
- `audio-io.h` → take upstream version (FLAC support: verify dr_flac include still works from our vendor/)

**Commit:** `sync: take upstream dit-sampler.h + audio-io.h`

---

### Phase C: Verification + Completion (Tasks 6–7)

#### Task 6: Build & Full Test

1. Full build: `dev-rebuild.bat`
2. Functional tests:
   - Text2music generation (Euler solver, linear scheduler, CFG guidance)
   - Multi-solver: RK4 + cosine scheduler + APG guidance
   - Adapter: LoRA merge mode
   - Adapter: LoKr runtime mode
   - Mastering: with matchering reference
   - Model switching mid-session (model-store cache)
   - ScragVAE selection from dropdown
   - `/props` endpoint returns correct model lists
3. Regression check:
   - All 14 solvers selectable + functional
   - All 8 schedulers selectable + functional
   - All 4 guidance modes selectable + functional
   - Composite scheduler blending
   - Auto-shift
   - Dynamic CFG
   - WAV lossless pipeline
   - Cover mode (source audio)

---

#### Task 7: Finalize Sync

1. Update `engine/UPSTREAM_SYNC`:
   ```
   # Last synced upstream commit (acestep.cpp master branch)
   # Updated: 2026-04-21
   3222db7c255971c71e8441108584b85fdf448445
   ```
2. Commit: `sync: upstream sync complete at 3222db7 + extension layer refactor`
3. Tag: `v1.5-upstream-sync-3222db7`

---

## Future Sync Process (Post-Refactor)

After the Extension Layer is in place, future syncs follow this streamlined process:

```
1. git pull vanilla repo
2. Read engine/UPSTREAM_SYNC → LAST_SYNCED
3. git log LAST_SYNCED..HEAD -- src/ → see what changed
4. For each changed file in src/:
     → Direct-copy from upstream to engine/src/  (no merge needed!)
5. Check if upstream changed dit-sampler.h:
     → If yes: review changes, port relevant fixes to hot-step-sampler.h
     → If no: done
6. Rebuild (dev-rebuild.bat)
7. Test
8. Update marker, commit, tag
```

**Time estimate**: 30 minutes for most syncs. Only longer if upstream changes the sampling loop fundamentals (rare).

---

## Open Questions

> [!IMPORTANT]
> **FLAC decoding**: We added `vendor/dr_flac/dr_flac.h` and our `audio-io.h` includes it for FLAC support in the mastering pipeline. Upstream doesn't have FLAC. After taking upstream's `audio-io.h`, we'll need to add FLAC support back — either:
> 1. Add it in `hot-step-synth.h` (our wrapper handles FLAC decoding before passing to upstream synth)
> 2. Accept ONE small patch to `audio-io.h` (a conditional `#include "vendor/dr_flac/dr_flac.h"` + FLAC decode function)
> 
> Option 1 is cleaner for sync. Option 2 is simpler to implement. Preference?

> [!NOTE]
> **Node.js server updates**: Our Node.js server (`server/src/services/aceClient.ts`) sends JSON to the C++ server with fields like `scheduler`, `guidance_mode`, `adapter_mode`. After the refactor, `hot-step-server.cpp` will parse these from JSON using `hot-step-request.h`. No changes needed to the Node.js side — the JSON contract stays the same.

---

## Verification Plan

### Build Verification
```powershell
cmd /c "d:\Ace-Step-Latest\hot-step-cpp\dev-rebuild.bat"
```

### Smoke Test
```
Generate text2music → verify audio output
Generate with adapter → verify merge/runtime modes
Generate with mastering → verify reference matching
Switch DiT model → verify model-store caching
```
