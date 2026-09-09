<#
.SYNOPSIS
    Start Oslian Demo: Flask, the frontend, the FHIR stream, the SQLite ledger and
    the WholeBody runtime, in one command.

.DESCRIPTION
    Everything runs inside this one process. Flask serves frontend/ directly; the
    stream playback clock, the SSE bus and the run scheduler are threads; the
    ledger is SQLite in WAL mode under var/; the WholeBody runtime is either the
    pinned fixture contract or a worker subprocess inside the research submodule.

    Fixture mode is the safe default and needs no configuration. Release mode
    defaults to the submodule at backend/research/wholebody-jepa; run
    setup-model-runtime.ps1 once first.

.PARAMETER Mode
    fixture (default) or release.

.PARAMETER ReleaseDir
    Override the release checkout. Only for a different working copy of the SAME
    commit model-lock.json approves. Falls back to $env:OSLIAN_WHOLEBODY_RELEASE_DIR,
    then backend/research/wholebody-jepa.

.PARAMETER ReleasePython
    Interpreter for the release worker; it needs scikit-learn 1.7.2.
    Falls back to $env:OSLIAN_WHOLEBODY_PYTHON, then -Python.

.PARAMETER ReleasePythonPath
    Dependency overlay prepended to that interpreter's path.
    Falls back to $env:OSLIAN_WHOLEBODY_PYTHONPATH.

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
    .\scripts\start-demo.ps1 -Mode release -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\sklearn-overlay

.EXAMPLE
    .\scripts\start-demo.ps1 -Database .\var\oslian_stream_ranges_v2.sqlite3

.EXAMPLE
    .\scripts\start-demo.ps1 -Mode release -Database .\var\oslian_stream_ranges_v2.sqlite3 -Python C:\path\to\python.exe -ReleasePython C:\path\to\python.exe -ReleasePythonPath C:\path\to\sklearn-overlay
#>
[CmdletBinding()]
param(
    [ValidateSet('fixture', 'release')] [string] $Mode = 'fixture',
    [string] $ReleaseDir,
    [string] $ReleasePython,
    [string] $ReleasePythonPath,
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
$env:OSLIAN_WHOLEBODY_RUNTIME = $Mode

if ($Mode -eq 'release') {
    $lock = Get-Content (Join-Path $root 'backend\model-lock.json') -Raw | ConvertFrom-Json
    $default = Join-Path $root $lock.submodule_path
    $ReleaseDir = Resolve-Setting $ReleaseDir 'OSLIAN_WHOLEBODY_RELEASE_DIR' $default
    if (-not (Test-Path $ReleaseDir)) { throw "release checkout not found: $ReleaseDir" }
    if (-not (Test-Path (Join-Path $ReleaseDir 'models'))) {
        throw "no models directory in $ReleaseDir. Run: .\scripts\setup-model-runtime.ps1"
    }
    $head = (& git -C $ReleaseDir rev-parse HEAD).Trim()
    if ($head -ne $lock.validated_commit) {
        throw "release checkout is at $head; model-lock.json approves $($lock.validated_commit). Run: .\scripts\setup-model-runtime.ps1"
    }

    $ReleasePython = Resolve-Setting $ReleasePython 'OSLIAN_WHOLEBODY_PYTHON' $pythonExe
    if (-not (Test-Path $ReleasePython)) {
        $resolved = (Get-Command $ReleasePython -ErrorAction SilentlyContinue).Source
        if (-not $resolved) { throw "release interpreter not found: $ReleasePython" }
        $ReleasePython = $resolved
    }
    $ReleasePythonPath = Resolve-Setting $ReleasePythonPath 'OSLIAN_WHOLEBODY_PYTHONPATH' ''
    if ($ReleasePythonPath -and -not (Test-Path $ReleasePythonPath)) {
        throw "release dependency overlay not found: $ReleasePythonPath"
    }

    $env:OSLIAN_WHOLEBODY_RELEASE_DIR = (Resolve-Path $ReleaseDir).Path
    $env:OSLIAN_WHOLEBODY_PYTHON = $ReleasePython
    if ($ReleasePythonPath) {
        $env:OSLIAN_WHOLEBODY_PYTHONPATH = (Resolve-Path $ReleasePythonPath).Path
    }
    Write-Host "release checkout : $env:OSLIAN_WHOLEBODY_RELEASE_DIR"
    Write-Host "locked commit    : $($lock.validated_commit)"
    Write-Host "release python   : $env:OSLIAN_WHOLEBODY_PYTHON"
    if ($ReleasePythonPath) { Write-Host "release overlay  : $env:OSLIAN_WHOLEBODY_PYTHONPATH" }
} else {
    Remove-Item Env:\OSLIAN_WHOLEBODY_RELEASE_DIR -ErrorAction SilentlyContinue
}

Write-Host "mode             : $Mode"
Write-Host "frontend         : $(Join-Path $root 'frontend')"
Write-Host "ledger           : $env:OSLIAN_STREAM_DB$(if (Test-Path $env:OSLIAN_STREAM_DB) { ' (existing)' } else { ' (new)' })"
Write-Host "open             : http://127.0.0.1:$Port"
& $pythonExe -m oslian_demo
