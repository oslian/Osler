<#
.SYNOPSIS
    Run the Oslian Demo test suite.

.DESCRIPTION
    Runs tests/ with the repository's pytest configuration. The attached runtime
    is tested against a stub of the forecast API, so no model files are needed.

.EXAMPLE
    .\scripts\test-demo.ps1

.EXAMPLE
    .\scripts\test-demo.ps1 tests\test_stream_api.py -v
#>
# PositionalBinding=$false so bare arguments go to pytest rather than being
# swallowed by the first named parameter that happens to accept a position.
[CmdletBinding(PositionalBinding = $false)]
param(
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
$env:OSLIAN_FORECAST_RUNTIME = 'detached'

if (-not $PytestArgs) { $PytestArgs = @('tests', '-q') }
Push-Location $root
try { & $pythonExe -m pytest @PytestArgs }
finally { Pop-Location }
