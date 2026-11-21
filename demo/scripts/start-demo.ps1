<#
.SYNOPSIS
    Start Oslian Demo: Flask, the frontend, the FHIR stream and the SQLite ledger,
    in one command.

.DESCRIPTION
    Everything the demo owns runs inside this one process. Flask serves frontend/
    directly; the stream playback clock, the SSE bus and the run scheduler are
    threads; the ledger is SQLite in WAL mode under var/. Forecasts come from the
    forecast API in api/, a separate process this script does not start.

    Detached mode is the safe default and needs no configuration. Api mode calls
    the forecast API at -ApiUrl.

.PARAMETER Mode
    detached (default) or api.

.PARAMETER ApiUrl
    The forecast API. Falls back to $env:OSLIAN_FORECAST_API_URL, then
    http://127.0.0.1:8000.

.PARAMETER Python
    Interpreter that runs Flask. Falls back to $env:OSLIAN_PYTHON, then python.

.PARAMETER Port
    HTTP port. Falls back to $env:OSLIAN_PORT, then 5000.

.PARAMETER Database
    SQLite ledger path. Falls back to $env:OSLIAN_STREAM_DB, then var\oslian_stream.sqlite3.
    A relative path is taken from the current directory. The parent directory is
    created when missing; an existing ledger is never deleted or overwritten.

.EXAMPLE
    .\scripts\start-demo.ps1

.EXAMPLE
    .\scripts\start-demo.ps1 -Mode api -ApiUrl http://127.0.0.1:8000

.EXAMPLE
    .\scripts\start-demo.ps1 -Database .\var\oslian_stream_ranges_v2.sqlite3
#>
[CmdletBinding()]
param(
    [ValidateSet('detached', 'api')] [string] $Mode = 'detached',
    [string] $ApiUrl,
    [string] $Python,
    [int]    $Port,
    [string] $Database
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Resolve-Setting([string] $value, [string] $envName, [string] $fallback) {
    if ($value) { return $value }
    $fromEnv = [Environment]::GetEnvironmentVariable($envName)
    if ($fromEnv) { return $fromEnv }
    return $fallback
}

$Python = Resolve-Setting $Python 'OSLIAN_PYTHON' 'python'
if (-not $Port) { $Port = [int](Resolve-Setting '' 'OSLIAN_PORT' '5000') }

# ---- preflight -------------------------------------------------------------
if (-not (Test-Path (Join-Path $root 'backend\oslian_demo\app.py'))) {
    throw "not an oslian-demo checkout: $root"
}
if (-not (Test-Path (Join-Path $root 'frontend\index.html'))) {
    throw "frontend is missing: $(Join-Path $root 'frontend\index.html')"
}
$pythonExe = (Get-Command $Python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) { throw "python not found: $Python (pass -Python or set OSLIAN_PYTHON)" }
& $pythonExe -c "import flask" | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "flask is not installed for $pythonExe. Run: & '$pythonExe' -m pip install -r requirements.txt"
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw "port $Port is already in use (pid $($listener.OwningProcess)). Stop it or pass -Port."
}

$varDir = Join-Path $root 'var'
if (-not (Test-Path $varDir)) { New-Item -ItemType Directory -Path $varDir | Out-Null }

# ---- ledger ----------------------------------------------------------------
# Explicit -Database first, then OSLIAN_STREAM_DB, then the default under var/.
# The path is resolved the way the server will read it (relative to the current
# directory), its parent is created when missing, and an existing file is left
# exactly as it is: a ledger is append-only and is never deleted or replaced here.
$Database = Resolve-Setting $Database 'OSLIAN_STREAM_DB' (Join-Path $varDir 'oslian_stream.sqlite3')
if (-not [IO.Path]::IsPathRooted($Database)) {
    $Database = Join-Path (Get-Location).Path $Database
}
$Database = [IO.Path]::GetFullPath($Database)
$ledgerDir = Split-Path -Parent $Database
if (-not (Test-Path $ledgerDir)) { New-Item -ItemType Directory -Path $ledgerDir -Force | Out-Null }
$env:OSLIAN_STREAM_DB = $Database

$env:PYTHONPATH = Join-Path $root 'backend'
$env:PYTHONUTF8 = '1'
$env:OSLIAN_PORT = "$Port"
$env:OSLIAN_FORECAST_RUNTIME = $Mode
if ($Mode -eq 'api') {
    $env:OSLIAN_FORECAST_API_URL = Resolve-Setting $ApiUrl 'OSLIAN_FORECAST_API_URL' 'http://127.0.0.1:8000'
    Write-Host "forecast API     : $env:OSLIAN_FORECAST_API_URL"
}

Write-Host "mode             : $Mode"
Write-Host "frontend         : $(Join-Path $root 'frontend')"
Write-Host "ledger           : $env:OSLIAN_STREAM_DB$(if (Test-Path $env:OSLIAN_STREAM_DB) { ' (existing)' } else { ' (new)' })"
Write-Host "open             : http://127.0.0.1:$Port"
& $pythonExe -m oslian_demo
