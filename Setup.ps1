param([string]$Python = 'py')
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv'
if (-not (Test-Path -LiteralPath (Join-Path $venvRoot 'Scripts\python.exe'))) {
    & $Python -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$projectPython = Join-Path $venvRoot 'Scripts\python.exe'
& $projectPython -m pip install 'torch==2.12.1+cpu' 'torchvision==0.27.1+cpu' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed' }
& $projectPython -m pip install -r (Join-Path $projectRoot 'requirements-app.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
Write-Host 'Installed. Run Start-Platform.cmd.'
