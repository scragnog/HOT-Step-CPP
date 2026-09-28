# verify-hooks.ps1 — Post-sync verification of HOT-Step integration hooks
#
# Run after any upstream sync to verify all HOT-Step hooks are intact.
# Exit code 0 = all good, 1 = broken hooks detected.
#
# Usage: powershell -File engine\verify-hooks.ps1

$src   = "$PSScriptRoot\src"
$tools = "$PSScriptRoot\tools"
$ggml  = "$PSScriptRoot\ggml"
$errors = 0

Write-Host "`n=== HOT-Step Hook Verification ===" -ForegroundColor Cyan
Write-Host ""

# ── Hook 1: pipeline-synth-ops.cpp must include hot-step-sampler.h ────
$content = Get-Content "$src\pipeline-synth-ops.cpp" -Raw
if ($content -match '#include\s+"hot-step-sampler\.h"') {
    Write-Host "  [OK] pipeline-synth-ops.cpp -> hot-step-sampler.h" -ForegroundColor Green
} elseif ($content -match '#include\s+"dit-sampler\.h"') {
    Write-Host "  [FAIL] pipeline-synth-ops.cpp includes dit-sampler.h (should be hot-step-sampler.h)" -ForegroundColor Red
    Write-Host "         Fix: change #include `"dit-sampler.h`" to #include `"hot-step-sampler.h`"" -ForegroundColor Yellow
    $errors++
} else {
    Write-Host "  [WARN] pipeline-synth-ops.cpp: no sampler include found" -ForegroundColor Yellow
    $errors++
}

# ── Hook 2: model-store.h must include hot-step-params.h ──────────────
$content = Get-Content "$src\model-store.h" -Raw
if ($content -match '#include\s+"hot-step-params\.h"') {
    Write-Host "  [OK] model-store.h -> hot-step-params.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] model-store.h missing hot-step-params.h include" -ForegroundColor Red
    $errors++
}

# ── Hook 3: dit.h must include adapter-merge.h and adapter-runtime.h ──
$content = Get-Content "$src\dit.h" -Raw
if ($content -match '#include\s+"adapter-merge\.h"') {
    Write-Host "  [OK] dit.h -> adapter-merge.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] dit.h missing adapter-merge.h include" -ForegroundColor Red
    $errors++
}
if ($content -match '#include\s+"adapter-runtime\.h"') {
    Write-Host "  [OK] dit.h -> adapter-runtime.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] dit.h missing adapter-runtime.h include" -ForegroundColor Red
    $errors++
}

# ── Hook 4: hot-step-server.cpp must include hot-step-params.h ────────
$content = Get-Content "$tools\hot-step-server.cpp" -Raw
if ($content -match '#include\s+"hot-step-params\.h"') {
    Write-Host "  [OK] hot-step-server.cpp -> hot-step-params.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] hot-step-server.cpp missing hot-step-params.h include" -ForegroundColor Red
    $errors++
}

# ── Hook 4-family: per-family include/call pairs on hot-step-server.cpp ──
#    Each row is one family's wiring into the shared server (its own
#    include + its own registration call), checked generically instead of
#    one hand-written block per family. A new family (YuE2, ...) adds a row
#    here when it gains its own hook include/call pair; nothing else in this
#    script has to change for that.
$content = Get-Content "$tools\hot-step-server.cpp" -Raw
$familyHooks = @(
    @{
        Family  = "MiniMax-Music3"
        Include = '#include\s+"minimax/mm3-server\.h"'
        Call    = 'mm3_register_routes\s*\('
        LostMsg = "Single hook for the whole MiniMax-Music3 backend subsystem (engine/src/minimax/). Without the include, /mm3/props, /mm3/warm and /mm3/unload routes vanish. The include alone registers nothing - the call site is the other half."
    },
    @{
        Family  = "MiniMax-Music3"
        Include = '#include\s+"minimax/mm3-job\.h"'
        Call    = 'mm3_register_job_routes\s*\('
        LostMsg = "MID-FILE include (after the job system: Job/job_create/work_push), not next to mm3-server.h at the top - re-add it AFTER job_status_str(), which is where Job, job_create, job_set_phase, work_push and g_store are defined. Lose the include and POST /mm3/synth vanishes, leaving only the deprecated /mm3/synth-e2e bring-up path that does GPU work on an httplib thread. Without the call, POST /mm3/synth and GET /mm3/job are not routed."
    }
    @{
        Family  = "YuE2"
        Include = '#include\s+"yue2/yue2-server\.h"'
        Call    = 'yue2_register_routes\s*\('
        LostMsg = "Single hook for the whole YuE2 backend subsystem (engine/src/yue2/). Without the include, every /yue2/* route vanishes, including the production POST /yue2/synth (which rides the SHARED /job routes -- there is no separate /yue2/job route to lose). The include alone registers nothing - the call site is the other half."
    }
)
foreach ($hook in $familyHooks) {
    if ($content -match $hook.Include) {
        Write-Host "  [OK] hot-step-server.cpp -> $($hook.Family) include ($($hook.Include))" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] hot-step-server.cpp missing $($hook.Family) include ($($hook.Include))" -ForegroundColor Red
        Write-Host "         $($hook.LostMsg)" -ForegroundColor Yellow
        $errors++
    }
    if ($content -match $hook.Call) {
        Write-Host "  [OK] hot-step-server.cpp calls $($hook.Family) registration ($($hook.Call))" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] hot-step-server.cpp never calls $($hook.Family) registration ($($hook.Call))" -ForegroundColor Red
        Write-Host "         $($hook.LostMsg)" -ForegroundColor Yellow
        $errors++
    }
}

# -- Hook 4-gate: hot-step-server.cpp must ask the FAMILY TABLE, not one name -
#    The two boot gates (registry_scan failure; ACE pipeline unusable) must
#    call hot_step_any_family_weights_present(), not one family's own probe,
#    and must include hot-step-families.h to get it. This hook exists so an
#    upstream sync cannot silently restore the pre-generalisation gate.
if ($content -match '#include\s+"hot-step-families\.h"') {
    Write-Host "  [OK] hot-step-server.cpp -> hot-step-families.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] hot-step-server.cpp missing hot-step-families.h include" -ForegroundColor Red
    Write-Host "         Without it neither boot gate can see any family's weights-present probe." -ForegroundColor Yellow
    $errors++
}
$gateMatches = [regex]::Matches($content, 'hot_step_any_family_weights_present\s*\(')
if ($gateMatches.Count -ge 2) {
    Write-Host "  [OK] hot-step-server.cpp calls hot_step_any_family_weights_present() at both boot gates" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] hot-step-server.cpp calls hot_step_any_family_weights_present() only $($gateMatches.Count) time(s), expected 2+" -ForegroundColor Red
    Write-Host "         An upstream sync restored the old single-name gate (registry_scan failing with" -ForegroundColor Yellow
    Write-Host "         no family present falls straight to 'return 1;') - MM3-only installs (issue #118)" -ForegroundColor Yellow
    Write-Host "         exit at boot with no error; re-add the family gate at both call sites." -ForegroundColor Yellow
    $errors++
}

# ── Hook 5: fsq-detok.h must include fsq-quant.h, and neither fsq-detok.h
#            nor fsq-tok.h may carry upstream's own FSQ quantizer copies ──
$content = Get-Content "$src\fsq-detok.h" -Raw
if ($content -match '#include\s+"fsq-quant\.h"') {
    Write-Host "  [OK] fsq-detok.h -> fsq-quant.h" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] fsq-detok.h missing fsq-quant.h include" -ForegroundColor Red
    Write-Host "         Upstream's FSQ encode/decode are NOT reference-conformant" -ForegroundColor Yellow
    Write-Host "         (plain tanh, no ResidualFSQ soft clamp = 40% index match)." -ForegroundColor Yellow
    $errors++
}
if ($content -match 'static\s+void\s+fsq_decode_index' -or $content -match 'static\s+const\s+int\s+FSQ_LEVELS') {
    Write-Host "  [FAIL] fsq-detok.h re-declares FSQ_LEVELS/fsq_decode_index (upstream copy is back)" -ForegroundColor Red
    $errors++
}
$content = Get-Content "$src\fsq-tok.h" -Raw
if ($content -match 'static\s+int\s+fsq_encode_index') {
    Write-Host "  [FAIL] fsq-tok.h re-declares fsq_encode_index (upstream copy is back)" -ForegroundColor Red
    Write-Host "         Delete it; the conformant one lives in src/fsq-quant.h" -ForegroundColor Yellow
    $errors++
} else {
    Write-Host "  [OK] fsq-tok.h uses shared fsq_encode_index" -ForegroundColor Green
}

# ── Hook 6: linker sentinel present in hot-step-sampler.h ─────────────
$content = Get-Content "$src\hot-step-sampler.h" -Raw
if ($content -match 'hotstep_sampler_linked_') {
    Write-Host "  [OK] hot-step-sampler.h has linker sentinel" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] hot-step-sampler.h missing linker sentinel (hotstep_sampler_linked_)" -ForegroundColor Red
    $errors++
}

# ── Hook 7: ggml-cuda's out_prod must carry the BF16 patch ────────────
#            engine/ggml is a SUBMODULE, so a submodule update reverts it.
$outProd = "$ggml\src\ggml-cuda\out-prod.cu"
if (Test-Path $outProd) {
    $content = Get-Content $outProd -Raw
    if ($content -match 'HOT-Step patch: BF16 out_prod') {
        Write-Host "  [OK] ggml-cuda/out-prod.cu has the BF16 patch" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] ggml-cuda/out-prod.cu is missing the BF16 out_prod patch" -ForegroundColor Red
        Write-Host "         Without it, train-dit --mirror bf16 aborts on the first backward pass." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\bf16-out-prod.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $outProd not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# -- Hook 8: ggml.c's MUL_MAT backward must carry the mm-backward patch ------
#            Also a SUBMODULE file, so a submodule update reverts it.
$ggmlC = "$ggml\src\ggml.c"
if (Test-Path $ggmlC) {
    $content = Get-Content $ggmlC -Raw
    if ($content -match 'HOT-Step patch: mm-backward') {
        Write-Host "  [OK] ggml/src/ggml.c has the mm-backward patch" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] ggml/src/ggml.c is missing the mm-backward patch" -ForegroundColor Red
        Write-Host "         Without it, ace-train --bwd mm silently runs the slow out_prod path" -ForegroundColor Yellow
        Write-Host "         and train-dit --mirror bf16 loses its tensor-core backward." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\mm-backward.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $ggmlC not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# -- Hook 9: ggml-cuda's quant->F32 copies must carry the occupancy patch -----
#            Also a SUBMODULE file. This one fails SILENTLY: without it every
#            quantized-base training run still produces correct numbers, just
#            ~3x slower, so nothing crashes to tell you it is gone.
$cpyCu = "$ggml\src\ggml-cuda\cpy.cu"
if (Test-Path $cpyCu) {
    $content = Get-Content $cpyCu -Raw
    if ($content -match 'HOT-Step patch: cpy-q-occupancy') {
        Write-Host "  [OK] ggml-cuda/cpy.cu has the quant-copy occupancy patch" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] ggml-cuda/cpy.cu is missing the cpy-q-occupancy patch" -ForegroundColor Red
        Write-Host "         SILENT: quantized-base LM training stays correct but runs ~3x slower" -ForegroundColor Yellow
        Write-Host "         (every quant->F32 dequant launches 1 thread per CUDA block)." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\cpy-q-occupancy.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $cpyCu not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# -- Hook 10: ggml-cuda's CPY must reach the generic quant->F32 converter -----
#             Also a SUBMODULE file. Without it, K-quant / MXFP4 / IQ bases are
#             rejected by supports_op and LM training on anything below q8_0 is
#             impossible - which puts the VRAM floor back above 22 GB.
$cpyH = "$ggml\src\ggml-cuda\cpy.cuh"
if (Test-Path $cpyH) {
    $content = Get-Content $cpyH -Raw
    if ($content -match 'HOT-Step patch: quant-cpy-generic') {
        Write-Host "  [OK] ggml-cuda/cpy.cuh has the generic quant->F32 copy patch" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] ggml-cuda/cpy.cuh is missing the quant-cpy-generic patch" -ForegroundColor Red
        Write-Host "         Without it, only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 bases can be trained -" -ForegroundColor Yellow
        Write-Host "         every K-quant, MXFP4 and IQ base fails at graph build." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\quant-cpy-kquant.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $cpyH not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# -- Hook 11: ggml-cuda's F16 GEMM must accumulate in F32 --------------------
#             Also a SUBMODULE file, and the WORST of the silent failures:
#             without it, f16 weights run under CUBLAS_COMPUTE_16F, which
#             accumulates and writes dst in half precision. Any partial sum
#             past 65504 becomes +inf and everything after it NaN. Nothing
#             errors - the GEMM succeeds and the render comes out garbled.
#             The MM3 LM trips this whenever an LM adapter is loaded.
$cudaCu = "$ggml\src\ggml-cuda\ggml-cuda.cu"
if (Test-Path $cudaCu) {
    $content = Get-Content $cudaCu -Raw
    if ($content -match 'HOT-Step patch: f16-f32-accumulate') {
        Write-Host "  [OK] ggml-cuda/ggml-cuda.cu has the F16 F32-accumulate patch" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] ggml-cuda/ggml-cuda.cu is missing the f16-f32-accumulate patch" -ForegroundColor Red
        Write-Host "         SILENT: f16 + an LM adapter renders noise instead of music," -ForegroundColor Yellow
        Write-Host "         because the f16 GEMM overflows its own half-precision accumulator." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\f16-f32-accumulate.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $cudaCu not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# -- Hook 12: ggml.c must carry the flash-attn-train autodiff case -----------
#             SUBMODULE file. The two fused attention ops the DiT trainer's
#             --attn flash mode runs on are registered in three places: the
#             enum and constructors here, the CPU reference in ops.cpp, and
#             the CUDA kernels in the two NEW files ggml-cuda/fattn-train.{cu,cuh}.
#             Those two are new files, so a submodule re-checkout deletes them
#             outright rather than reverting them - but the autodiff case in
#             ggml.c is the piece that cannot be reconstructed from them.
#             Failure is loud (ace-train stops compiling, or the trainer's
#             supports_op probe aborts with attn-unsupported), so this hook is
#             here to name the cause before someone spends a build cycle on it.
$ggmlC2 = "$ggml\src\ggml.c"
$fattnCu = "$ggml\src\ggml-cuda\fattn-train.cu"
if (Test-Path $ggmlC2) {
    $content = Get-Content $ggmlC2 -Raw
    if ($content -match 'HOT-Step patch: flash-attn-train') {
        if (Test-Path $fattnCu) {
            Write-Host "  [OK] ggml.c has the flash-attn-train ops, and fattn-train.cu is present" -ForegroundColor Green
        } else {
            Write-Host "  [FAIL] ggml.c has the flash-attn-train ops but ggml-cuda/fattn-train.cu is GONE" -ForegroundColor Red
            Write-Host "         The CUDA kernels are new files, so a submodule re-checkout deletes them." -ForegroundColor Yellow
            Write-Host "         Fix (from the repo root): git apply engine\patches\flash-attn-train.patch" -ForegroundColor Yellow
            $errors++
        }
    } else {
        Write-Host "  [FAIL] ggml.c is missing the flash-attn-train patch" -ForegroundColor Red
        Write-Host "         ace-train will not compile, and train-dit --attn flash is gone." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\flash-attn-train.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $ggmlC2 not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# -- Hook 13: ggml-alloc.c must carry the enlarged free-block table ----------
#             SUBMODULE file. MAX_FREE_BLOCKS 256 -> 1024. Without it the DiT
#             trainer's LoKR at dim 256 (a ~19k-node graph) trips
#             "out of free blocks" in ggml_dyn_tallocr_insert_block a few
#             epochs in. Failure is loud (GGML_ASSERT), but it only shows on
#             that configuration, so a lost patch looks like a LoKR bug.
$ggmlAlloc = "$ggml\src\ggml-alloc.c"
if (Test-Path $ggmlAlloc) {
    $content = Get-Content $ggmlAlloc -Raw
    if ($content -match 'HOT-Step patch: alloc-free-blocks') {
        Write-Host "  [OK] ggml-alloc.c has the alloc-free-blocks patch" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] ggml-alloc.c is missing the alloc-free-blocks patch" -ForegroundColor Red
        Write-Host "         LoKR dim 256 training will abort with 'out of free blocks'." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\alloc-free-blocks.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $ggmlAlloc not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# -- Hook 14: ggml-vulkan must carry the training ops (BF16_ROUND, fused attention)
#             SUBMODULE file. Without them the YuE2 joint trainer on a Vulkan
#             build refuses to start (BF16_ROUND unsupported) or, worse, falls
#             back to exact attention, whose memory rules out whole songs.
$ggmlVk = "$ggml\src\ggml-vulkan\ggml-vulkan.cpp"
if (Test-Path $ggmlVk) {
    $content = Get-Content $ggmlVk -Raw
    if ($content -match 'HOT-Step patch: flash-attn-train \(Vulkan\)' -and $content -match 'HOT-Step patch: BF16_ROUND' -and
        (Test-Path "$ggml\src\ggml-vulkan\vulkan-shaders\fa_train_fwd.comp")) {
        Write-Host "  [OK] ggml-vulkan has the training ops and their shaders" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] ggml-vulkan is missing the zzzz-vulkan-train-ops patch (or its shader files)" -ForegroundColor Red
        Write-Host "         YuE2 joint training on Vulkan will refuse to start or lose fused attention." -ForegroundColor Yellow
        Write-Host "         Fix (from the repo root): git apply engine\patches\zzzz-vulkan-train-ops.patch" -ForegroundColor Yellow
        $errors++
    }
} else {
    Write-Host "  [WARN] $ggmlVk not found - ggml submodule not checked out?" -ForegroundColor Yellow
}

# ── Summary ───────────────────────────────────────────────────────────
Write-Host ""
if ($errors -gt 0) {
    Write-Host "  $errors hook(s) broken! Fix before building." -ForegroundColor Red
    Write-Host "" 
    exit 1
} else {
    Write-Host "  All hooks intact." -ForegroundColor Green
    Write-Host ""
    exit 0
}
