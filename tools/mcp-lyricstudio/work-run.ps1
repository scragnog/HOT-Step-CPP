<#
Run a foreground command under the Work reservation guard.

The preferred runtime is the existing Node 22 collaboration environment at
%LOCALAPPDATA%\HOT-Step\collaboration-node22. That directory holds matching
node_modules and remains separate from the app's Node 24 runtime. This launcher
never runs npm or rebuilds native modules. Without the dedicated environment,
the workspace's Node executable must report major 24 and have tsx and
better-sqlite3 installed.

The TypeScript runner uses --preserve-symlinks, --preserve-symlinks-main and
--import tsx.  .cmd/.bat child commands are quoted by work-run.ts and reject
cmd metacharacters (including parentheses) in arguments; use an executable
directly for arbitrary arguments.
#>
# Keep this a plain forwarding script: Windows PowerShell 5.1 -File cannot
# bind GNU-style flags reliably through an advanced remaining-arguments param.
$RunnerArguments = @($args)

$ErrorActionPreference = 'Stop'
$packageRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $packageRoot '../..')).Path
$callerCwd = (Get-Location).Path
$dedicatedRoot = Join-Path $env:LOCALAPPDATA 'HOT-Step/collaboration-node22'
$pinnedNode = Join-Path $repoRoot 'release/.node-cache/node-v24.18.0-win-x64/node.exe'

function Get-NodeMajor([string]$NodeExecutable) {
    $value = & $NodeExecutable -p "process.versions.node.split('.')[0]" 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    return [string]$value
}

function Assert-NodeRuntime([string]$NodeExecutable, [string]$RuntimeRoot, [string]$ExpectedMajor) {
    if (!(Test-Path -LiteralPath $NodeExecutable -PathType Leaf)) {
        throw "Node executable was not found: $NodeExecutable"
    }
    $major = Get-NodeMajor $NodeExecutable
    if ($major -ne $ExpectedMajor) {
        throw "work-run requires Node $ExpectedMajor for '$RuntimeRoot'; '$NodeExecutable' reports Node $major. No npm rebuild is attempted."
    }
    foreach ($dependency in @('tsx', 'better-sqlite3')) {
        if (!(Test-Path -LiteralPath (Join-Path $RuntimeRoot "node_modules/$dependency"))) {
            throw "Node $ExpectedMajor runtime '$RuntimeRoot' is missing node_modules/$dependency. Install the matching runtime separately; work-run will not run npm."
        }
    }
    $probe = "try { const Database = require('better-sqlite3'); new Database(':memory:').close(); require.resolve('tsx'); } catch (error) { console.error(error.message); process.exit(1); }"
    Push-Location $RuntimeRoot
    try { & $NodeExecutable --preserve-symlinks --preserve-symlinks-main -e $probe } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) {
        throw "Node $ExpectedMajor dependencies in '$RuntimeRoot' do not load with '$NodeExecutable' (native ABI mismatch or incomplete install). No npm rebuild is attempted."
    }
}

$nodeExecutable = $null
$runtimeRoot = $null
if (Test-Path -LiteralPath $dedicatedRoot -PathType Container) {
    $runtimeRoot = $dedicatedRoot
    $portableCandidates = @(
        (Join-Path $dedicatedRoot 'node.exe'),
        (Join-Path $dedicatedRoot 'bin/node.exe')
    )
    $nodeCache = Join-Path $repoRoot 'release/.node-cache'
    if (Test-Path -LiteralPath $nodeCache -PathType Container) {
        $legacyCache = Get-ChildItem -LiteralPath $nodeCache -Directory -Filter 'node-v22*-win-x64' |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($legacyCache) { $portableCandidates += (Join-Path $legacyCache.FullName 'node.exe') }
    }
    foreach ($candidate in $portableCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $nodeExecutable = $candidate; break }
    }
    if (!$nodeExecutable) {
        $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
        if ($nodeCommand) { $nodeExecutable = $nodeCommand.Source }
    }
    if (!$nodeExecutable) { throw "Dedicated Node 22 runtime exists at '$dedicatedRoot' but no node.exe was found." }
} else {
    $runtimeRoot = $packageRoot
    $portableCandidates = @($pinnedNode, (Join-Path $packageRoot 'node.exe'))
    foreach ($candidate in $portableCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $nodeExecutable = $candidate; break }
    }
    if (!$nodeExecutable) {
        $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
        if ($nodeCommand) { $nodeExecutable = $nodeCommand.Source }
    }
    if (!$nodeExecutable) { throw 'No Node executable was found for work-run.' }
}

$expectedMajor = if ($runtimeRoot -eq $dedicatedRoot) { '22' } else { '24' }
Assert-NodeRuntime $nodeExecutable $runtimeRoot $expectedMajor
$runner = Join-Path $runtimeRoot 'src/work-run.ts'
if (!(Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Runner source is missing: $runner" }

# Some PowerShell invocation forms consume a literal `--` as a parameter
# terminator. Recover that boundary
# from the known runner flags so `work-run.ps1 --name X -- node ...` remains
# equivalent to invoking work-run.ts directly.
$separator = [Array]::IndexOf($RunnerArguments, '--')
if ($separator -lt 0 -and $RunnerArguments.Count -gt 0) {
    $valueFlags = @('--db', '--channel', '--name', '--resource', '--reason', '--cwd')
    $commandStart = -1
    for ($index = 0; $index -lt $RunnerArguments.Count; $index++) {
        if ($valueFlags -contains $RunnerArguments[$index]) { $index++; continue }
        if ($RunnerArguments[$index] -in @('--help', '-h')) { break }
        $commandStart = $index
        break
    }
    if ($commandStart -ge 0) {
        $before = if ($commandStart -gt 0) { @($RunnerArguments[0..($commandStart - 1)]) } else { @() }
        $after = @($RunnerArguments[$commandStart..($RunnerArguments.Count - 1)])
        $RunnerArguments = @($before + '--' + $after)
    }
}
$separator = [Array]::IndexOf($RunnerArguments, '--')
$commandIsBatch = $false
if ($separator -ge 0 -and $separator + 1 -lt $RunnerArguments.Count) {
    $commandIsBatch = [IO.Path]::GetExtension([string]$RunnerArguments[$separator + 1]).ToLowerInvariant() -in @('.bat', '.cmd')
}
$hasCwd = $false
for ($index = 0; $index -lt $RunnerArguments.Count; $index++) {
    if ($RunnerArguments[$index] -eq '--') { break }
    if ($RunnerArguments[$index] -eq '--cwd') {
        $hasCwd = $true
        if ($index + 1 -ge $RunnerArguments.Count) { throw '--cwd requires a path' }
        if (![IO.Path]::IsPathRooted([string]$RunnerArguments[$index + 1])) {
            $RunnerArguments[$index + 1] = [IO.Path]::GetFullPath((Join-Path $callerCwd $RunnerArguments[$index + 1]))
        }
    }
}
$childCwd = if ($commandIsBatch) { $repoRoot } else { $callerCwd }
$projectDb = Join-Path $repoRoot 'data/collaboration.db'

$nodeArgs = @(
    '--preserve-symlinks',
    '--preserve-symlinks-main',
    '--import', 'tsx',
    $runner,
    '--db', $projectDb
)
if (!$hasCwd) { $nodeArgs += @('--cwd', $childCwd) }
if ($RunnerArguments) { $nodeArgs += $RunnerArguments }

Push-Location $runtimeRoot
try {
    & $nodeExecutable @nodeArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
