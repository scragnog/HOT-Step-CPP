# ggml-pre-fork-overlay.ps1 - recognise and remove the pre-fork ggml patch overlay
#
# Used by update.bat (update.sh carries the same check inline). Before the
# HOT-ggml fork, engine/ggml was ggml at c044c6f0 with engine/patches applied
# as uncommitted edits plus eight untracked kernel files. That exact state, and
# only that state, may be restored so the submodule can move to the fork:
#
#   -Mode Check    exit 0: engine/ggml is clean
#                  exit 2: engine/ggml is exactly the pre-fork overlay
#                  exit 1: anything else (a local edit, an extra edit inside an
#                          overlay file, a partial overlay, a stray file, another
#                          base, or a git failure). Nothing is changed.
#   -Mode Restore  re-runs the check, then restores exactly the overlay's files
#                  (tracked ones from git, new ones deleted). exit 0 / 1.
#
# "Exactly" means HEAD is c044c6f0 and all 34 files hash to the overlay's
# content (sha256 after stripping CR, so CRLF checkouts match).
param(
    [Parameter(Mandatory)] [ValidateSet('Check', 'Restore')] [string]$Mode,
    [string]$GgmlDir = ''
)
if (-not $GgmlDir) { $GgmlDir = Join-Path $PSScriptRoot 'ggml' }   # PS 5.1: no $PSScriptRoot in param defaults

$Base = 'c044c6f03892f9d5e98213b05f8afea1f8b0d3c9'
$Overlay = [ordered]@{
    "include/ggml-rpc.h" = "54e097e2f54a0a6874ac6da34910f05738e5aa9c780a447f25f85641ff48d499"
    "include/ggml.h" = "28603404395a9293da16ee552ddbd25ffd65d7db06c95315847e8dd26c859285"
    "src/ggml-alloc.c" = "0a5175e1ba169c9a7b6ada3d637e9af75fac0ff02c503ace6f0ed77df1d260e9"
    "src/ggml-backend-meta.cpp" = "fea0b35d80f1592041a64d14bb16697aa589879a76aa54153288b02323f2f158"
    "src/ggml-backend.cpp" = "4958ef48f11be667546c33c54f5028e4b4271b06a0b4adef7a10294023d871e6"
    "src/ggml-cpu/ggml-cpu.c" = "4f87e04e947eadd4bb01ee5dff4cd79a5209cb7f368f19ef135ac765953086a8"
    "src/ggml-cpu/ggml-cpu.cpp" = "d3f614f0695bd123007d27b95f6130ac617a09fa89c6acde9220a0cc5cef7ee5"
    "src/ggml-cpu/ops.cpp" = "3e70ed8d52990b067c91dc7b769b5b3fb38b08fb02fca944db120ba29814a332"
    "src/ggml-cpu/ops.h" = "dedb188e9f187fd89216c27c90889140ae63655ee5ca58686754f59dab2d0523"
    "src/ggml-cuda/CMakeLists.txt" = "d6f42a9eec46c1e828b10256110e9d21394ca3168ae9da0ae1ec604e99e15da4"
    "src/ggml-cuda/convrot8.cu" = "697be95992b30190250e5edee00cd6612e2d6a1284a9c2da09044cce295cc3af"
    "src/ggml-cuda/convrot8.cuh" = "d6db00cd1d973c7de46c2b8ccc83ac273c9a4f30642ec8e62cf6508db8741005"
    "src/ggml-cuda/cpy.cu" = "89d89c749b3667aa541aeefb818436df3a4b282b209803947796915a0743bed8"
    "src/ggml-cuda/cpy.cuh" = "648bd317d9b980de8ade995e3f5b2323c720336de123733e5fdeedc1912ba9af"
    "src/ggml-cuda/fattn-train.cu" = "e22cfad8df782c6fa4e42e972a11ae935abfb91f745e98bf5335b19a8f304660"
    "src/ggml-cuda/fattn-train.cuh" = "92ce96329ab084ef4ae8882c75a24661bd2d89175dd3548c542e33cfac432dca"
    "src/ggml-cuda/ggml-cuda.cu" = "857a1e030c92e7aa985aac0205796e4b242876caabc3e70ebba07d11c3b2f3c1"
    "src/ggml-cuda/out-prod.cu" = "fba28be9a58f851ea3e7e4df7305d1ea90589537e57ef1363efc60e98aa457d0"
    "src/ggml-cuda/unary.cu" = "0b4b5ff52b6aad57e4774023cfb8068a5e2551b98266bdb86fc09cb6f9e0cf89"
    "src/ggml-cuda/unary.cuh" = "f77920497798b8d6d6e338d12901f84baedaecdc33412be04a25080130355ead"
    "src/ggml-hip/CMakeLists.txt" = "32f0bedd7775c8117efd87f139a003b18fa367627ff14e772371858e35d443b4"
    "src/ggml-metal/ggml-metal-device.cpp" = "1562b93c826e1017872f9dfcb51b4b527a3f73c2a77fbe8a64402656c197fdd1"
    "src/ggml-metal/ggml-metal-device.h" = "6a253e5b44756162b0b9d03c3bbeb7f27209088468b532c9c004fbfec8b5c483"
    "src/ggml-metal/ggml-metal-ops.cpp" = "795cdcdbbc3ada0b150345d0eaeeda7582c5d624884402dbe740d1f93c610035"
    "src/ggml-metal/ggml-metal.metal" = "51b72ed95ca4e79e46d67bd80da4f6e5645f9433f88bd01f5aa7203850629cb8"
    "src/ggml-musa/CMakeLists.txt" = "2adfa5007f429cdb71e78066d587d1e942f5e6d3daefde3814f859bbed2e82c3"
    "src/ggml-vulkan/ggml-vulkan.cpp" = "2c1ff93f645ee28a08acc643203273560cad58a13b13efd191a29e2ab5c3f97d"
    "src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dkv.comp" = "cc659c1c46ee2964cb20dea408951ee12144850f30f9e5a7880db2b767dbca2f"
    "src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dq.comp" = "576409fdc697cc58dfffc9f2ca44299f258b91f6636cb5ca6d1050516c40aa59"
    "src/ggml-vulkan/vulkan-shaders/fa_train_common.glsl" = "d6e66ba8d75bad1fa6a6c85c4e87252c7da4a1a66f46e72d8477248d6f492fc2"
    "src/ggml-vulkan/vulkan-shaders/fa_train_fwd.comp" = "ec25fce777e748c4da65a1e7f5e3139fdb337a928ee3a72aaaed20fb27559074"
    "src/ggml-vulkan/vulkan-shaders/unary.comp" = "8b018353eef8fc4b34ac559f77a32a1944b1450d2f608c8fb23760de5751f944"
    "src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp" = "4432d0084e3a053367e75a76051f4371af1143b589bc351e3229a7b140d8ce45"
    "src/ggml.c" = "24484864b88f541e66a62e379249a23f4c8acce9084bb9c3a32d18eee34697c0"
}
$New = @(
    'src/ggml-cuda/convrot8.cu', 'src/ggml-cuda/convrot8.cuh',
    'src/ggml-cuda/fattn-train.cu', 'src/ggml-cuda/fattn-train.cuh',
    'src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dkv.comp', 'src/ggml-vulkan/vulkan-shaders/fa_train_bwd_dq.comp',
    'src/ggml-vulkan/vulkan-shaders/fa_train_common.glsl', 'src/ggml-vulkan/vulkan-shaders/fa_train_fwd.comp'
)

function Get-NormalizedSha256([string]$Path) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $kept = New-Object System.Collections.Generic.List[byte] ($bytes.Length)
    foreach ($b in $bytes) { if ($b -ne 13) { $kept.Add($b) } }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    ($sha.ComputeHash($kept.ToArray()) | ForEach-Object { $_.ToString('x2') }) -join ''
}

function Test-Overlay {
    if (-not (Test-Path (Join-Path $GgmlDir '.git'))) { return 0 }
    $status = & git -C $GgmlDir status --porcelain --untracked-files=all 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR: cannot read engine/ggml status: $status"; return 1 }
    if (-not $status) { return 0 }

    $problems = New-Object System.Collections.Generic.List[string]
    foreach ($line in @($status)) {
        $code = $line.Substring(0, 2); $path = $line.Substring(3)
        $ok = ($code -eq ' M' -and $Overlay.Contains($path) -and $New -notcontains $path) -or
              ($code -eq '??' -and $New -contains $path)
        if (-not $ok) { $problems.Add("    $line") }
    }
    $head = & git -C $GgmlDir rev-parse HEAD 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR: cannot read engine/ggml HEAD: $head"; return 1 }
    if ("$head".Trim() -ne $Base) { $problems.Add("    base is $("$head".Trim()), not $Base") }
    foreach ($path in $Overlay.Keys) {
        $file = Join-Path $GgmlDir $path
        if (-not (Test-Path $file -PathType Leaf)) { $problems.Add("    missing overlay file $path"); continue }
        if ((Get-NormalizedSha256 $file) -ne $Overlay[$path]) { $problems.Add("    $path differs from the known overlay") }
    }
    if ($problems.Count -eq 0) { return 2 }
    Write-Host ""
    Write-Host "  ERROR: engine/ggml has local changes this updater will not touch:"
    $problems | ForEach-Object { Write-Host $_ }
    Write-Host ""
    Write-Host "  ggml changes belong on the HOT-ggml fork (docs/dev/ggml-fork.md)."
    Write-Host "  Move or commit them, then check with: git -C engine/ggml status"
    Write-Host "  Nothing has been changed."
    return 1
}

$state = Test-Overlay
if ($Mode -eq 'Check') { exit $state }

# Restore
if ($state -eq 0) { exit 0 }
if ($state -ne 2) { exit 1 }
$tracked = @($Overlay.Keys | Where-Object { $New -notcontains $_ })
& git -C $GgmlDir checkout -- @tracked
if ($LASTEXITCODE -ne 0) { Write-Host "  ERROR: could not restore the overlay's tracked files in engine/ggml"; exit 1 }
foreach ($path in $New) {
    try { Remove-Item -LiteralPath (Join-Path $GgmlDir $path) -ErrorAction Stop }
    catch { Write-Host "  ERROR: could not remove engine/ggml/$path : $_"; exit 1 }
}
Write-Host "  Restored the pre-fork overlay files in engine/ggml."
exit 0
