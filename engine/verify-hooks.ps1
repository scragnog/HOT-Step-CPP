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

# ── Hooks 7-16: engine/ggml must be HOT-ggml with its HOT-Step changes ──────
#    engine/ggml is pinned to HOT-ggml's hot-step branch, which carries these
#    changes as commits (docs/dev/ggml-fork.md). Each hook greps for the
#    capability itself, so a stock ggml-org checkout (usually a submodule URL
#    cached from before the fork) fails here in seconds instead of twenty
#    minutes into a CUDA compile. The fix for every one of them is the same:
#        git submodule sync -- engine/ggml
#        git submodule update --init engine/ggml
#    A hook passing proves the source is present, not that a backend runs it.
$ggmlFix = "Fix (from the repo root): git submodule sync -- engine/ggml; git submodule update --init engine/ggml"

function Test-GgmlHook([string]$Label, [string[]]$Files, [string[]]$Patterns, [string]$LostMsg) {
    foreach ($f in $Files) {
        if (-not (Test-Path $f)) {
            Write-Host "  [FAIL] $Label - $f is missing" -ForegroundColor Red
            Write-Host "         $LostMsg" -ForegroundColor Yellow
            Write-Host "         $ggmlFix" -ForegroundColor Yellow
            $script:errors++
            return
        }
    }
    $content = ($Files | ForEach-Object { Get-Content $_ -Raw }) -join "`n"
    foreach ($p in $Patterns) {
        if ($content -notmatch $p) {
            Write-Host "  [FAIL] $Label - pattern not found: $p" -ForegroundColor Red
            Write-Host "         $LostMsg" -ForegroundColor Yellow
            Write-Host "         $ggmlFix" -ForegroundColor Yellow
            $script:errors++
            return
        }
    }
    Write-Host "  [OK] $Label" -ForegroundColor Green
}

$cuda = "$ggml\src\ggml-cuda"
$vk   = "$ggml\src\ggml-vulkan"

# Hook 7: BF16 src0 in ggml-cuda's OUT_PROD.
Test-GgmlHook "ggml-cuda out_prod accepts BF16 (Hook 7)" @("$cuda\out-prod.cu") @('HOT-Step patch: BF16 out_prod') `
    "Without it, train-dit --mirror bf16 aborts on the first backward pass."

# Hook 8: env-gated mul_mat formulation of the MUL_MAT backward.
Test-GgmlHook "ggml.c mm-backward (Hook 8)" @("$ggml\src\ggml.c") @('HOT-Step patch: mm-backward') `
    "Without it, ace-train --bwd mm silently runs the slow out_prod path and --mirror bf16 loses its tensor-core backward."

# Hook 9: quant->F32 copies launch CUDA_CPY_BLOCK_SIZE threads per block.
#   This is upstream since ggml-org b64fb805, so there is no HOT-Step marker:
#   the hook checks the launch shape itself. SILENT if lost: quantized-base
#   LM training stays correct but runs ~3x slower.
Test-GgmlHook "ggml-cuda quant->F32 copy occupancy (Hook 9)" @("$cuda\cpy.cu") `
    @('cpy_q_f32<cpy_blck_q8_0_f32,\s*QK8_0><<<num_blocks,\s*CUDA_CPY_BLOCK_SIZE') `
    "SILENT: quantized-base LM training stays correct but runs ~3x slower (one thread per CUDA block)."

# Hook 10: CPY reaches the generic quant->F32 converter (K-quant/IQ/MXFP4 bases).
Test-GgmlHook "ggml-cuda generic quant->F32 copy (Hook 10)" @("$cuda\cpy.cuh") @('HOT-Step patch: quant-cpy-generic') `
    "Without it, only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 bases can be trained; every K-quant, MXFP4 and IQ base fails at graph build."

# Hook 11: F16 cuBLAS GEMMs accumulate and write F32. The worst silent failure:
#   f16 + an LM adapter renders noise because the GEMM overflows its half accumulator.
Test-GgmlHook "ggml-cuda F16 GEMM accumulates in F32 (Hook 11)" @("$cuda\ggml-cuda.cu") @('HOT-Step patch: f16-f32-accumulate') `
    "SILENT: f16 + an LM adapter renders noise instead of music."

# Hook 12: fused training attention - op, autodiff case and CUDA kernels.
Test-GgmlHook "flash-attn-train ops, autodiff and CUDA kernels (Hook 12)" @("$ggml\src\ggml.c", "$cuda\fattn-train.cu") `
    @('HOT-Step patch: flash-attn-train') `
    "ace-train will not compile, and train-dit --attn flash is gone."

# Hook 13: ggml-alloc's per-chunk free-block table is 1024, not 256.
Test-GgmlHook "ggml-alloc free-block table (Hook 13)" @("$ggml\src\ggml-alloc.c") @('HOT-Step patch: alloc-free-blocks') `
    "LoKR dim 256 training will abort with 'out of free blocks'."

# Hook 14: Vulkan training ops. Upstream split ggml-vulkan.cpp, so the pipeline
#   fields now live in ggml-vulkan-types.h; check both files and a shader.
Test-GgmlHook "ggml-vulkan training ops (Hook 14)" `
    @("$vk\ggml-vulkan.cpp", "$vk\ggml-vulkan-types.h", "$vk\vulkan-shaders\fa_train_fwd.comp") `
    @('HOT-Step patch: flash-attn-train \(Vulkan\)', 'HOT-Step patch: BF16_ROUND', 'pipeline_fa_train_fwd', 'pipeline_bf16_round') `
    "YuE2 joint training on Vulkan will refuse to start or lose fused attention."

# Hook 15: ConvRot8, the CUDA int8 training op YuE2 adapters on int8 bases use.
Test-GgmlHook "ConvRot8 op and CUDA kernels (Hook 15)" @("$ggml\include\ggml.h", "$cuda\ggml-cuda.cu", "$cuda\convrot8.cu") `
    @('GGML_OP_CONVROT8,', 'ggml_cuda_op_convrot8\s*\(') `
    "YuE2 training on an int8 base will not compile or will refuse the op."

# Hook 16: BF16_ROUND on CUDA (Hook 14 covers the Vulkan half).
Test-GgmlHook "BF16_ROUND unary op on CUDA (Hook 16)" @("$ggml\include\ggml.h", "$cuda\unary.cu", "$cuda\ggml-cuda.cu") `
    @('GGML_UNARY_OP_BF16_ROUND', 'op_bf16_round') `
    "The YuE2 joint trainer refuses to start on CUDA."

# Hook 17: engine/ggml is clean and at the commit this repo pins. The fork is
#   the only place ggml changes live now; local edits in engine/ggml are lost
#   on the next submodule update and are not what CI or a user builds. Untracked
#   files count too: ggml-cuda's CMakeLists globs *.cu, so a stray kernel file
#   is compiled in. Fails closed: in a git checkout, any git command that does
#   not succeed is a failure, never a pass.
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repoRoot ".git"))) {
    Write-Host "  [WARN] not a git checkout, engine/ggml pin not checked (Hook 17)" -ForegroundColor Yellow
} else {
    $hook17 = $null
    try {
        $pin = (& git -C $repoRoot rev-parse ":engine/ggml" 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "git rev-parse :engine/ggml failed (exit $LASTEXITCODE): $pin" }
        $head = (& git -C $ggml rev-parse HEAD 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "git -C engine/ggml rev-parse HEAD failed (exit $LASTEXITCODE): $head" }
        $dirty = (& git -C $ggml status --porcelain --untracked-files=all 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "git -C engine/ggml status failed (exit $LASTEXITCODE): $dirty" }
        $pin = "$pin".Trim(); $head = "$head".Trim()
        if ($pin -notmatch '^[0-9a-f]{40}$' -or $head -notmatch '^[0-9a-f]{40}$') { throw "unexpected pin/HEAD '$pin' / '$head'" }
        if ($pin -ne $head) {
            $hook17 = "engine/ggml is at $head but this repo pins $pin"
        } elseif ($dirty) {
            $hook17 = "engine/ggml has local changes or untracked files:`n" + (($dirty | ForEach-Object { "           $_" }) -join "`n")
        }
    } catch {
        $hook17 = "could not verify the engine/ggml pin: $_"
    }
    if ($hook17) {
        Write-Host "  [FAIL] $hook17 (Hook 17)" -ForegroundColor Red
        Write-Host "         ggml changes belong on HOT-ggml's hot-step branch (docs/dev/ggml-fork.md)." -ForegroundColor Yellow
        Write-Host "         $ggmlFix" -ForegroundColor Yellow
        $errors++
    } else {
        Write-Host "  [OK] engine/ggml is clean at the pinned commit $($pin.Substring(0, 8)) (Hook 17)" -ForegroundColor Green
    }
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
