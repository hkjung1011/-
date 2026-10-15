param([int]$Port = 8893)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Python runtime missing. Run Setup.ps1 to install a local environment.'
}
Set-Location -LiteralPath $projectRoot
try {
    $runningPlatform = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/status" -TimeoutSec 2
    if ($runningPlatform.inference_device -eq 'cpu' -and $runningPlatform.models) {
        Write-Host "Already running: http://127.0.0.1:$Port"
        return
    }
} catch { }
Write-Host "Wafer inspection: http://127.0.0.1:$Port"
Write-Host 'Stop: Ctrl+C. This app does not send stage or conveyor motion commands.'
& $projectPython -m uvicorn app:app --host 127.0.0.1 --port $Port
if ($LASTEXITCODE -ne 0) { throw 'Inspection server exited with an error.' }
