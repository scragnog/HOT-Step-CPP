# release/build-release.ps1 — Build a complete portable release of HOT-Step CPP
#
# Usage:
#   .\release\build-release.ps1 [-Version "1.5.0"] [-SkipEngine] [-SkipUI] [-Variant cuda -CudaAsset cuda13.1] [-BuildBackend all] [-UpdaterProtocol 1]
#
# -UpdaterProtocol 1 builds the versioned layout the Windows launcher updates
# (release/updater/package-layout.mjs), for updater test fixtures. Releases
# ship protocol 0, the flat package, until the updater has passed acceptance.
#
# Produces:
#   release/out/HOT-Step-CPP-v{version}-win-x64-{variant}.zip
#
# Requirements:
#   - Node.js 24 LTS (for building with correct native module ABI)
#   - Visual Studio 2022 Build Tools with C++ workload
#   - CUDA Toolkit 12.x (for CUDA variant)
#   - Vulkan SDK (for Vulkan variant)

param(
    [string]$Version = "0.0.0",
    [switch]$SkipEngine,
    [switch]$SkipUI,
    [ValidateSet("cuda", "vulkan", "cpu")]
    [string]$Variant = "cuda",       # Packaged backend
    [ValidateSet("all", "cuda", "vulkan", "cpu")]
    [string]$BuildBackend = "all",   # Backends to compile
    [ValidateSet("", "cuda13.1", "cuda12.8", "cuda12-volta")]
    [string]$CudaAsset = "",         # CI asset this CUDA build stands in for (build-info.json)
    [string]$NodeVersion = "24.18.0", # Node.js LTS version to bundle
    [ValidateSet(-1, 0, 1)]
    [int]$UpdaterProtocol = -1        # -1 = build-info.mjs default (UPDATER_PROTOCOL)
)

$ErrorActionPreference = "Stop"
# Reject mismatches before downloads, staging cleanup or builds, including SkipEngine.
if ($BuildBackend -ne "all" -and $BuildBackend -ne $Variant) {
    throw "BuildBackend '$BuildBackend' must match Variant '$Variant', or use BuildBackend 'all'."
}
# build-info.json names an exact asset variant, and the CUDA toolkit decides which one.
if ($Variant -eq "cuda" -and -not $CudaAsset) {
    throw "Variant cuda needs -CudaAsset cuda13.1, cuda12.8 or cuda12-volta to match the toolkit you build with."
}
$assetVariant = if ($Variant -eq "cuda") { $CudaAsset } else { $Variant }
$buildCuda = if ($BuildBackend -in @("all", "cuda")) { "ON" } else { "OFF" }
$buildVulkan = if ($BuildBackend -in @("all", "vulkan")) { "ON" } else { "OFF" }
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleaseDir = Join-Path $ProjectRoot "release"
$StagingDir = Join-Path $ReleaseDir "staging"
$OutputDir = Join-Path $ReleaseDir "out"

# Portable Node.js download cache
$NodeCacheDir = Join-Path $ReleaseDir ".node-cache"
$NodeZipName = "node-v${NodeVersion}-win-x64.zip"
$NodeUrl = "https://nodejs.org/dist/v${NodeVersion}/${NodeZipName}"
$NodeExe = Join-Path (Join-Path $NodeCacheDir "node-v${NodeVersion}-win-x64") "node.exe"

Write-Host "`n════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  HOT-Step CPP Release Builder" -ForegroundColor Cyan
Write-Host "  Version: $Version | Variant: $Variant" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════`n" -ForegroundColor Cyan

# ── Phase 0: Clean staging ────────────────────────────────────────────

Write-Host "[Phase 0] Cleaning staging directory..." -ForegroundColor Yellow
if (Test-Path $StagingDir) { Remove-Item -Recurse -Force $StagingDir }
New-Item -ItemType Directory -Force $StagingDir | Out-Null
New-Item -ItemType Directory -Force $OutputDir | Out-Null

# ── Phase 1: Download portable Node.js ────────────────────────────────

Write-Host "`n[Phase 1] Portable Node.js $NodeVersion..." -ForegroundColor Yellow

if (-not (Test-Path $NodeExe)) {
    Write-Host "  Downloading from nodejs.org..."
    New-Item -ItemType Directory -Force $NodeCacheDir | Out-Null
    $zipPath = Join-Path $NodeCacheDir $NodeZipName

    if (-not (Test-Path $zipPath)) {
        Invoke-WebRequest -Uri $NodeUrl -OutFile $zipPath -UseBasicParsing
    }

    Write-Host "  Extracting..."
    Expand-Archive -Path $zipPath -DestinationPath $NodeCacheDir -Force
    Remove-Item $zipPath -ErrorAction SilentlyContinue

    if (-not (Test-Path $NodeExe)) {
        throw "Node.js extraction failed - $NodeExe not found"
    }
}

$nodeVer = & $NodeExe --version
Write-Host "  Using: $NodeExe ($nodeVer)" -ForegroundColor Green

# ── Phase 2: Build C++ Engine ─────────────────────────────────────────

if (-not $SkipEngine) {
    Write-Host "`n[Phase 2] Building C++ engine ($Variant)..." -ForegroundColor Yellow

    $engineDir = Join-Path $ProjectRoot "engine"

    # Use buildall.cmd for multi-arch release builds
    $buildScript = Join-Path $engineDir "buildall.cmd"

    Write-Host "  Running buildall.cmd ($BuildBackend) with static runtime..."

    # Smart CMake cache handling: only clear if build flags have changed.
    # Deleting CMakeCache.txt forces a full reconfigure -> full rebuild of ALL
    # targets including 100+ CUDA vendor kernels (~1 hour). If the cache already
    # has the correct release flags, we skip deletion for a fast incremental build.
    $buildDir = Join-Path $engineDir "build"
    $cacheFile = Join-Path $buildDir "CMakeCache.txt"
    $needsCacheClear = $false

    if (Test-Path $cacheFile) {
        $cacheContent = Get-Content $cacheFile -Raw
        $requiredFlags = @{
            "HOT_STEP_STATIC_RUNTIME:BOOL=ON" = "Static runtime (/MT)"
            "GGML_CUDA:BOOL=$buildCuda"        = "CUDA backend"
            "GGML_VULKAN:BOOL=$buildVulkan"    = "Vulkan backend"
            "GGML_BACKEND_DL:BOOL=ON"         = "Dynamic backend loading"
            "GGML_CPU_ALL_VARIANTS:BOOL=ON"   = "CPU all variants"
        }
        foreach ($flag in $requiredFlags.Keys) {
            if ($cacheContent -notmatch ("(?m)^" + [regex]::Escape($flag) + "\r?$")) {
                Write-Host "  Cache mismatch: $($requiredFlags[$flag]) ($flag)" -ForegroundColor Yellow
                $needsCacheClear = $true
            }
        }
        if ($needsCacheClear) {
            Write-Host "  Clearing CMake cache (flags changed since last build)..." -ForegroundColor Yellow
            Remove-Item $cacheFile -Force
        } else {
            Write-Host "  CMake cache valid -- incremental build (CUDA vendor files will NOT rebuild)" -ForegroundColor Green
        }
    } else {
        Write-Host "  No CMake cache -- full build required" -ForegroundColor Yellow
    }

    # Set cmake flags for the build
    # buildall.cmd runs cmake with its own flags, so we override via env
    $previousCmakeExtra = $env:RELEASE_CMAKE_EXTRA
    $env:RELEASE_CMAKE_EXTRA = "-DHOT_STEP_STATIC_RUNTIME=ON"

    Push-Location $engineDir
    try {
        # Use the absolute path even when current-directory executable lookup is disabled.
        & $buildScript $BuildBackend
        if ($LASTEXITCODE -ne 0) { throw "Engine build failed (exit code $LASTEXITCODE)" }
    } finally {
        $env:RELEASE_CMAKE_EXTRA = $previousCmakeExtra
        Pop-Location
    }

    # Verify output
    $aceServerExe = Join-Path (Join-Path $buildDir "Release") "ace-server.exe"
    if (-not (Test-Path $aceServerExe)) {
        throw "Build succeeded but ace-server.exe not found at $aceServerExe"
    }
    Write-Host "  Engine build complete" -ForegroundColor Green
} else {
    Write-Host "`n[Phase 2] SKIPPED (engine build)" -ForegroundColor DarkGray
}

# ── Phase 3: Bundle Server ────────────────────────────────────────────

Write-Host "`n[Phase 3] Bundling server..." -ForegroundColor Yellow

# Install server deps with system npm (dev machine only)
Write-Host "  Installing server dependencies..."
Push-Location (Join-Path $ProjectRoot "server")
try {
    # Use cmd /c to prevent PS treating npm stderr warnings as errors
    cmd /c "npm install --ignore-scripts 2>&1" | Out-Null
    # Verify portable Node works
    & $NodeExe --version | Out-Null
    Write-Host "  Dependencies installed" -ForegroundColor Green
} finally {
    Pop-Location
}

# Rebuild better-sqlite3 native addon for the portable Node.js ABI
# Match the staged native addon to the portable Node 24 runtime (ABI 137).
# IMPORTANT: We backup and restore the original .node file so the dev environment
# is not contaminated by the release build.
Write-Host "  Rebuilding better-sqlite3 for Node $NodeVersion..."
$bsqlPkg = Join-Path (Join-Path $ProjectRoot "server") "node_modules\better-sqlite3"
$nativeAddon = Join-Path $bsqlPkg "build\Release\better_sqlite3.node"
$nativeBackup = Join-Path $bsqlPkg "build\Release\better_sqlite3.node.dev-backup"

# Backup the dev machine's native addon
if (Test-Path $nativeAddon) {
    Copy-Item $nativeAddon $nativeBackup -Force
    Write-Host "    Backed up dev addon"
}

Push-Location $bsqlPkg
try {
    cmd /c "npx prebuild-install --runtime node --target $NodeVersion --arch x64 --platform win32 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  prebuild-install failed, trying npm rebuild..."
        Pop-Location
        Push-Location (Join-Path $ProjectRoot "server")
        cmd /c "npm rebuild better-sqlite3 2>&1" | Out-Null
    }
    Write-Host "  Native addon rebuilt for Node $NodeVersion" -ForegroundColor Green
} finally {
    Pop-Location
}

# Copy the Node 24 addon to staging before restoring dev version
$stagingBsqlBuild = Join-Path $StagingDir "server\node_modules\better-sqlite3\build\Release"
New-Item -ItemType Directory -Force $stagingBsqlBuild | Out-Null
Copy-Item $nativeAddon (Join-Path $stagingBsqlBuild "better_sqlite3.node") -Force -ErrorAction SilentlyContinue

# Restore the dev machine's native addon
if (Test-Path $nativeBackup) {
    Copy-Item $nativeBackup $nativeAddon -Force
    Remove-Item $nativeBackup -Force
    Write-Host "    Restored dev addon"
}

# Install esbuild in release dir (dev dependency for bundling)
Write-Host "  Installing esbuild..."
Push-Location $ReleaseDir
try {
    if (-not (Test-Path (Join-Path $ReleaseDir "node_modules\esbuild"))) {
        npm install esbuild --save-dev 2>&1 | Out-Null
    }
} finally {
    Pop-Location
}

# Run esbuild (cmd /c prevents PS from treating esbuild stderr warnings as fatal)
Write-Host "  Running esbuild..."
cmd /c "node `"$(Join-Path $ReleaseDir 'esbuild.config.mjs')`" 2>&1"
if ($LASTEXITCODE -ne 0) { throw "esbuild bundle failed" }

Write-Host "  Server bundle complete" -ForegroundColor Green

# ── Phase 4: Build UI ─────────────────────────────────────────────────

if (-not $SkipUI) {
    Write-Host "`n[Phase 4] Building UI..." -ForegroundColor Yellow

    Push-Location (Join-Path $ProjectRoot "ui")
    try {
        cmd /c "npm install 2>&1" | Out-Null
        cmd /c "npm run build 2>&1"
        if ($LASTEXITCODE -ne 0) { throw "UI build failed" }
    } finally {
        Pop-Location
    }

    Write-Host "  UI build complete" -ForegroundColor Green
} else {
    Write-Host "`n[Phase 4] SKIPPED (UI build)" -ForegroundColor DarkGray
}

# ── Phase 5: Assemble Release ─────────────────────────────────────────

Write-Host "`n[Phase 5] Assembling release..." -ForegroundColor Yellow

$dist = $StagingDir

# Runtime: portable Node.js
Write-Host "  Copying runtime..."
New-Item -ItemType Directory -Force (Join-Path $dist "runtime") | Out-Null
Copy-Item $NodeExe (Join-Path $dist "runtime\node.exe")

# Engine: binaries + DLLs
Write-Host "  Copying engine binaries..."
$engineOut = Join-Path $dist "engine"
New-Item -ItemType Directory -Force $engineOut | Out-Null
$buildRelease = Join-Path (Join-Path $ProjectRoot "engine") "build\Release"

$engineFiles = @(
    "ace-server.exe", "ace-train.exe", "ace-midi.exe", "ace-caption.exe", "mastering.exe", "mp3-codec.exe",
    "neural-codec.exe", "vst-host.exe", "quantize.exe",
    "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"
)

# Add variant-specific DLLs
switch ($Variant) {
    "cuda"   { $engineFiles += "ggml-cuda.dll" }
    "vulkan" { $engineFiles += "ggml-vulkan.dll" }
    # cpu: no GPU backend DLL needed
}

foreach ($file in $engineFiles) {
    $src = Join-Path $buildRelease $file
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $engineOut $file)
    } else {
        throw "Missing required engine file: $file"
    }
}

# Also copy any ggml-cpu-* variant DLLs (from GGML_CPU_ALL_VARIANTS)
Get-ChildItem (Join-Path $buildRelease "ggml-cpu-*.dll") -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $engineOut $_.Name)
}

# Lua plugins — dual directory system:
#   engine/plugins/  = native/built-in plugins (solvers, schedulers, guidance)
#   plugins/         = community/user plugins (same structure, overrides native)
# ace-server scans both via engine_dir and project_dir resolution from binary path
Write-Host "  Copying Lua plugins..."
$enginePluginsSrc = Join-Path (Join-Path $ProjectRoot "engine") "plugins"
if (Test-Path $enginePluginsSrc) {
    $enginePluginsDst = Join-Path $engineOut "plugins"
    Copy-Item -Recurse $enginePluginsSrc $enginePluginsDst
    $pluginCount = (Get-ChildItem -Recurse $enginePluginsDst -Filter "*.lua" | Measure-Object).Count
    Write-Host "    $pluginCount native plugins (engine/plugins/)" -ForegroundColor Green
} else {
    Write-Warning "  engine/plugins/ not found -- engine will have no built-in plugins!"
}
$communityPluginsSrc = Join-Path $ProjectRoot "plugins"
if (Test-Path $communityPluginsSrc) {
    Copy-Item -Recurse $communityPluginsSrc (Join-Path $dist "plugins")
    Write-Host "    Community plugins dir copied (plugins/)" -ForegroundColor Green
}

# MSVC C++ Runtime - bundle so the app works on clean machines without VC++ Redistributable
$msvcRedist = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter "vcruntime140.dll" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'x64' -and $_.FullName -notmatch 'debug' -and $_.FullName -match 'Redist' } |
    Select-Object -First 1
if ($msvcRedist) {
    $msvcDir = $msvcRedist.DirectoryName
    foreach ($dll in @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")) {
        $src = Join-Path $msvcDir $dll
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $engineOut $dll)
        }
    }
    Write-Host "  Bundled MSVC runtime DLLs"
} else {
    Write-Warning "  MSVC Redist not found - vcruntime DLLs not bundled"
}

# Server: bundled JS + native deps
Write-Host "  Copying server..."
$serverOut = Join-Path $dist "server"
# server.mjs was already placed by esbuild into staging/server/
# Copy better-sqlite3 minimal package
$bsqlSrc = Join-Path (Join-Path $ProjectRoot "server") "node_modules\better-sqlite3"
$bsqlDst = Join-Path $serverOut "node_modules\better-sqlite3"
New-Item -ItemType Directory -Force $bsqlDst | Out-Null

# Copy the JS package + native binding
Copy-Item (Join-Path $bsqlSrc "package.json") (Join-Path $bsqlDst "package.json")
New-Item -ItemType Directory -Force (Join-Path $bsqlDst "lib") | Out-Null
Copy-Item -Recurse (Join-Path $bsqlSrc "lib\*") (Join-Path $bsqlDst "lib")
# NOTE: Do NOT copy build/Release/better_sqlite3.node from source tree here!
# The correct Node 24 addon was already staged at Phase 3 (line ~166).
# Copying from $bsqlSrc would overwrite the addon selected for the portable runtime.
# See: https://github.com/scragnog/HOT-Step-CPP/issues/18

# Copy better-sqlite3 runtime dependencies: bindings + file-uri-to-path
$nmSrc = Join-Path (Join-Path $ProjectRoot "server") "node_modules"
foreach ($dep in @("bindings", "file-uri-to-path")) {
    $depSrc = Join-Path $nmSrc $dep
    $depDst = Join-Path $serverOut "node_modules\$dep"
    if (Test-Path $depSrc) {
        Copy-Item -Recurse $depSrc $depDst
        Write-Host "    Copied $dep"
    } else {
        Write-Warning "  Missing dependency: $dep"
    }
}

# Copy ffmpeg.exe
$ffmpegSrc = Join-Path (Join-Path (Join-Path $ProjectRoot "server") "node_modules\ffmpeg-static") "ffmpeg.exe"
if (Test-Path $ffmpegSrc) {
    Copy-Item $ffmpegSrc (Join-Path $serverOut "ffmpeg.exe")
} else {
    Write-Warning "  ffmpeg.exe not found - audio conversion will be limited"
}

# Copy data files (model-registry.json, assistant-knowledge.md)
$dataOut = Join-Path $serverOut "data"
New-Item -ItemType Directory -Force $dataOut | Out-Null
Copy-Item (Join-Path $ProjectRoot "server/src/data/*") $dataOut -Recurse -Force

# Gate: the bundle must boot before anything is zipped; runs after the addon and its deps are staged (a bad bundle once shipped with exit 13).
Write-Host "  Boot-testing server.mjs with the portable runtime..."
& $NodeExe (Join-Path $ReleaseDir "smoke-boot.mjs") (Join-Path $StagingDir "server\server.mjs")
if ($LASTEXITCODE -ne 0) { throw "Server bundle failed its boot test - not releasing" }

# UI: pre-built dist
Write-Host "  Copying UI..."
$uiSrc = Join-Path (Join-Path $ProjectRoot "ui") "dist"
$uiDst = Join-Path (Join-Path $dist "ui") "dist"
if (Test-Path $uiSrc) {
    New-Item -ItemType Directory -Force (Join-Path $dist "ui") | Out-Null
    Copy-Item -Recurse $uiSrc $uiDst
} else {
    Write-Warning "  UI dist not found - skipping"
}

# Essentia
$essentiaSrc = Join-Path $ProjectRoot "Essentia"
if (Test-Path $essentiaSrc) {
    Write-Host "  Copying Essentia..."
    Copy-Item -Recurse $essentiaSrc (Join-Path $dist "Essentia")
}

# Noise samples
$noiseSrc = Join-Path $ProjectRoot "noise_samples"
if (Test-Path $noiseSrc) {
    Copy-Item -Recurse $noiseSrc (Join-Path $dist "noise_samples")
}

# Mutable data folders are initialized beside the app at first boot.

# Config and docs
Copy-Item (Join-Path $ReleaseDir ".env.example") (Join-Path $dist ".env.example")
Copy-Item (Join-Path $ReleaseDir "profile-path.mjs") (Join-Path $dist "profile-path.mjs")
Copy-Item (Join-Path $ReleaseDir "HOT-Step.bat") (Join-Path $dist "HOT-Step.bat")
Copy-Item (Join-Path $ReleaseDir "README.txt") (Join-Path $dist "README.txt")
Copy-Item (Join-Path $ProjectRoot "open-browser-if-needed.mjs") (Join-Path $dist "open-browser-if-needed.mjs")
Set-Content -Path (Join-Path $dist "VERSION") -Value $Version -NoNewline

# Exact package identity for the updater (docs/dev/releasing.md)
$protocolArgs = if ($UpdaterProtocol -ge 0) { @("--protocol", "$UpdaterProtocol") } else { @() }
& $NodeExe (Join-Path $ReleaseDir "build-info.mjs") $dist --os windows --arch x64 --variant $assetVariant --tag "v$Version" @protocolArgs
if ($LASTEXITCODE) { throw "build-info.mjs failed" }
& $NodeExe (Join-Path $ReleaseDir "check-layout.mjs") --package $dist --os windows --variant $assetVariant
if ($LASTEXITCODE) { throw "build-info.json check failed" }

Write-Host "  Assembly complete" -ForegroundColor Green

# ── Phase 6: Package ──────────────────────────────────────────────────

Write-Host "`n[Phase 6] Packaging..." -ForegroundColor Yellow

# Protocol 1+: move the package into app/<id>/ under the stable launcher root.
# The same script runs in release.yml, so this is what CI zips.
& $NodeExe (Join-Path $ReleaseDir "updater/package-layout.mjs") $dist
if ($LASTEXITCODE) { throw "package-layout.mjs failed" }
& $NodeExe (Join-Path $ReleaseDir "check-layout.mjs") --package $dist --os windows --variant $assetVariant
if ($LASTEXITCODE) { throw "package layout check failed" }

$zipName = "HOT-Step-CPP-v${Version}-win-x64-${Variant}.zip"
$zipPath = Join-Path $OutputDir $zipName

if (Test-Path $zipPath) { Remove-Item $zipPath }

# Compress — use .NET for better compression than Compress-Archive
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($dist, $zipPath, [System.IO.Compression.CompressionLevel]::Optimal, $false)

$zipSize = (Get-Item $zipPath).Length
$zipSizeMB = [math]::Round($zipSize / 1MB, 1)

# Generate SHA256
$hash = (Get-FileHash $zipPath -Algorithm SHA256).Hash
$hashFile = Join-Path $OutputDir "${zipName}.sha256"
"$hash  $zipName" | Set-Content $hashFile -NoNewline

Write-Host "  $zipName ($zipSizeMB MB)" -ForegroundColor Green
Write-Host "  SHA256: $hash" -ForegroundColor DarkGray

Write-Host "`n════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Release build complete!" -ForegroundColor Green
Write-Host "  Output: $zipPath" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════`n" -ForegroundColor Cyan
