# release/build-release.ps1 — Build a complete portable release of HOT-Step CPP
#
# Usage:
#   .\release\build-release.ps1 [-Version "1.5.0"] [-SkipEngine] [-SkipUI] [-Variant cuda]
#
# Produces:
#   release/out/HOT-Step-CPP-v{version}-win-x64-{variant}.zip
#
# Requirements:
#   - Node.js 22 LTS (for building with correct native module ABI)
#   - Visual Studio 2022 Build Tools with C++ workload
#   - CUDA Toolkit 12.x (for CUDA variant)
#   - Vulkan SDK (for Vulkan variant)

param(
    [string]$Version = "0.0.0",
    [switch]$SkipEngine,
    [switch]$SkipUI,
    [string]$Variant = "cuda",       # cuda, vulkan, cpu
    [string]$NodeVersion = "22.16.0" # Node.js LTS version to bundle
)

$ErrorActionPreference = "Stop"
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

    # For release builds, we want static MSVC runtime and multi-arch CUDA
    # buildall.cmd already does multi-arch + Vulkan.
    # We modify the cmake flags by setting them before calling the script.
    Write-Host "  Running buildall.cmd with static runtime..."

    # Clean the build to ensure /MT is applied everywhere
    $buildDir = Join-Path $engineDir "build"
    if (Test-Path (Join-Path $buildDir "CMakeCache.txt")) {
        Write-Host "  Clearing CMake cache for clean release build..."
        Remove-Item (Join-Path $buildDir "CMakeCache.txt") -Force
    }

    # Set cmake flags for the build
    # buildall.cmd runs cmake with its own flags, so we override via env
    $env:RELEASE_CMAKE_EXTRA = "-DHOT_STEP_STATIC_RUNTIME=ON"

    Push-Location $engineDir
    try {
        # buildall.cmd handles vcvars, ORT download, and multi-config build
        cmd /c "buildall.cmd"
        if ($LASTEXITCODE -ne 0) { throw "Engine build failed (exit code $LASTEXITCODE)" }
    } finally {
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
# Dev machine may run Node 24 (ABI 137) but portable bundle ships Node 22 (ABI 127)
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

# Copy the Node 22 addon to staging before restoring dev version
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
    "ace-server.exe", "mastering.exe", "mp3-codec.exe",
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
        Write-Warning "  Missing engine file: $file"
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
    Write-Warning "  engine/plugins/ not found — engine will have no built-in plugins!"
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
# The correct Node 22 addon was already staged at Phase 3 (line ~166).
# Copying from $bsqlSrc would overwrite it with the restored dev addon (Node 24 ABI).
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
Copy-Item (Join-Path (Join-Path (Join-Path $ProjectRoot "server") "src\data") "model-registry.json") (Join-Path $dataOut "model-registry.json")
Copy-Item (Join-Path (Join-Path (Join-Path $ProjectRoot "server") "src\data") "assistant-knowledge.md") (Join-Path $dataOut "assistant-knowledge.md")

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

# Empty directories for user content
New-Item -ItemType Directory -Force (Join-Path $dist "models") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $dist "adapters") | Out-Null

# Config and docs
Copy-Item (Join-Path $ProjectRoot ".env.example") (Join-Path $dist ".env.example")
Copy-Item (Join-Path $ReleaseDir "HOT-Step.bat") (Join-Path $dist "HOT-Step.bat")
Copy-Item (Join-Path $ReleaseDir "README.txt") (Join-Path $dist "README.txt")

Write-Host "  Assembly complete" -ForegroundColor Green

# ── Phase 6: Package ──────────────────────────────────────────────────

Write-Host "`n[Phase 6] Packaging..." -ForegroundColor Yellow

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
