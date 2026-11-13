<#
.SYNOPSIS
    Run the Oslian Demo test suite.

.DESCRIPTION
    Runs tests/ with the repository's pytest configuration. The release-runtime
    test skips itself in fixture mode; pass -Mode release to run it against the
    commit model-lock.json approves.

.EXAMPLE
    .\scripts\test-demo.ps1

.EXAMPLE
    .\scripts\test-demo.ps1 -Mode release -ReleasePythonPath C:\path\to\sklearn-overlay

.EXAMPLE
    .\scripts\test-demo.ps1 tests\test_stream_api.py -v
#>
# PositionalBinding=$false so bare arguments go to pytest rather than being
# swallowed by the first named parameter that happens to accept a position.
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('fixture', 'release')] [string] $Mode = 'fixture',
    [string] $ReleaseDir,
    [string] $ReleasePython,
    [string] $ReleasePythonPath,
    [string] $Python,
    [Parameter(ValueFromRemainingArguments = $true)] [string[]] $PytestArgs
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
$pythonExe = (Get-Command $Python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) { throw "python not found: $Python (pass -Python or set OSLIAN_PYTHON)" }

$env:PYTHONUTF8 = '1'
$env:OSLIAN_WHOLEBODY_RUNTIME = $Mode

if ($Mode -eq 'release') {
    $lock = Get-Content (Join-Path $root 'backend\model-lock.json') -Raw | ConvertFrom-Json
    $default = Join-Path $root $lock.submodule_path
    $ReleaseDir = Resolve-Setting $ReleaseDir 'OSLIAN_WHOLEBODY_RELEASE_DIR' $default
    if (-not (Test-Path (Join-Path $ReleaseDir 'models'))) {
        throw "no models directory in $ReleaseDir. Run: .\scripts\setup-model-runtime.ps1"
    }
    $env:OSLIAN_WHOLEBODY_RELEASE_DIR = (Resolve-Path $ReleaseDir).Path
    $env:OSLIAN_WHOLEBODY_PYTHON = Resolve-Setting $ReleasePython 'OSLIAN_WHOLEBODY_PYTHON' $pythonExe
    $overlay = Resolve-Setting $ReleasePythonPath 'OSLIAN_WHOLEBODY_PYTHONPATH' ''
    if ($overlay) { $env:OSLIAN_WHOLEBODY_PYTHONPATH = (Resolve-Path $overlay).Path }
}

if (-not $PytestArgs) { $PytestArgs = @('tests', '-q') }
Push-Location $root
try { & $pythonExe -m pytest @PytestArgs }
finally { Pop-Location }
