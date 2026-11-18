$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonCmd = $null
foreach ($candidate in @(
    { py -3.12 -V },
    { py -3.11 -V },
    { python -V }
)) {
    try {
        & $candidate
        $pythonCmd = "python"
        break
    }
    catch {
        continue
    }
}

if (-not $pythonCmd) {
    throw "Python 3.11 or 3.12 is required to run the demo."
}

if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv .venv
    } else {
        python -m venv .venv
    }
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe scripts/generate_datasets.py
& .\.venv\Scripts\python.exe scripts/seed_companies.py

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

if (-not $env:SECRET_KEY) {
    $env:SECRET_KEY = "demo-secret-key"
}
if (-not $env:ADMIN_REGISTRATION_CODE) {
    $env:ADMIN_REGISTRATION_CODE = "demo-admin-code"
}
& .\.venv\Scripts\python.exe -m flask --app app run --debug
