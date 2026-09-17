param(
    [Parameter(Mandatory = $true)][string]$Executable,
    [string[]]$TestArguments = @()
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$dllDirectory = Join-Path $repo 'engine/build/Release'
$exe = (Resolve-Path -LiteralPath $Executable).Path
foreach ($dll in @('ggml-base.dll', 'ggml-cpu-x64.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $dllDirectory $dll))) {
        throw "Required test dependency missing: $dllDirectory/$dll"
    }
}
# Children inherit this mode: loader failures return an exit code, not a dialog.
Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class AitkTestErrorMode {
    [DllImport("kernel32.dll")]
    public static extern uint SetErrorMode(uint mode);
}
'@
$previousMode = [AitkTestErrorMode]::SetErrorMode(0x8003)
$previousPath = $env:PATH
try {
    $env:PATH = "$dllDirectory;$previousPath"
    # Windows PowerShell converts redirected native stderr into ErrorRecords.
    # CUDA initialization messages are diagnostics, not invocation failures.
    $ErrorActionPreference = 'Continue'
    & $exe @TestArguments
    $result = $LASTEXITCODE
} finally {
    $env:PATH = $previousPath
    [void][AitkTestErrorMode]::SetErrorMode($previousMode)
}
exit $result
