# open-browser-if-needed.ps1
# Opens the given URL in the default browser ONLY if no browser tab already has
# the app loaded. Waits for the server to come up, then asks /api/health how
# many tabs are connected (each open tab holds an SSE stream to
# /api/health/presence, so tabs left over from a previous run re-register within
# a few seconds of the server returning). Window-title matching was the old
# method; it only sees each Chrome window's foreground tab, so a background tab
# meant a duplicate.
#
# Usage: powershell -ExecutionPolicy Bypass -File open-browser-if-needed.ps1 <url>

param(
    [Parameter(Mandatory=$true)]
    [string]$Url
)

$health = $Url.TrimEnd('/') + '/api/health'

function Get-Clients {
    try { return [int](Invoke-RestMethod $health -TimeoutSec 5).clients } catch { return $null }
}

# Wait for the server (tsx compile + DB open take a few seconds; a first-run
# UI build under LAUNCH.bat can take longer).
$deadline = (Get-Date).AddMinutes(5)
$clients = $null
while ($null -eq $clients -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 1
    $clients = Get-Clients
}
if ($null -eq $clients) {
    Write-Host "[HOT-Step] Server did not come up at $health - not opening a browser."
    exit 1
}

# Give existing tabs a moment to reconnect (SSE retry is 2 s).
$grace = (Get-Date).AddSeconds(8)
while ($clients -eq 0 -and (Get-Date) -lt $grace) {
    Start-Sleep -Seconds 1
    $clients = Get-Clients
}

if ($clients -gt 0) {
    Write-Host "[HOT-Step] $clients browser tab(s) already connected - skipping."
} else {
    Write-Host "[HOT-Step] Opening browser: $Url"
    Start-Process $Url
}
