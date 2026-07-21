$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found. Follow the installation section in README.md."
}

Push-Location $projectRoot
try {
    & $python -m pytest -q
    & $python main.py --profile quick --datasets breast_cancer,wine --verbose
}
finally {
    Pop-Location
}

