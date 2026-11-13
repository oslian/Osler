<#
.SYNOPSIS
    Verify the release runtime end to end: git, manifest, hashes, worker, health.

.DESCRIPTION
    Runs every fail-closed check the demo relies on, in the same order the worker
    does, and then asks a live server for /api/model/health so the answer comes
    from the running system rather than from a script's own opinion.

    Read-only: it starts a temporary server on a spare port and stops it again.

.PARAMETER Python
    Interpreter that runs Flask. Falls back to $env:OSLIAN_PYTHON, then python.

.PARAMETER ReleasePython
    Interpreter for the release worker (needs scikit-learn 1.7.2).
    Falls back to $env:OSLIAN_WHOLEBODY_PYTHON, then -Python.

.PARAMETER ReleasePythonPath
    Dependency overlay for that interpreter.
    Falls back to $env:OSLIAN_WHOLEBODY_PYTHONPATH.

.PARAMETER Port
    Port for the temporary server. Default 5099.

.EXAMPLE
    .\scripts\check-model-runtime.ps1 -ReleasePythonPath C:\path\to\sklearn-overlay
#>
[CmdletBinding()]
param(
    [string] $Python,
    [string] $ReleasePython,
    [string] $ReleasePythonPath,
    [int]    $Port = 5099
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$lock = Get-Content (Join-Path $root 'backend\model-lock.json') -Raw | ConvertFrom-Json
$subPath = Join-Path $root $lock.submodule_path
$failures = @()

function Step([string] $name, [bool] $ok, [string] $detail) {
    if ($ok) { Write-Host ("  ok    {0,-22} {1}" -f $name, $detail) }
    else { Write-Host ("  FAIL  {0,-22} {1}" -f $name, $detail); $script:failures += $name }
}

Write-Host 'git'
$head = (& git -C $subPath rev-parse HEAD).Trim()
Step 'pinned_commit' ($head -eq $lock.validated_commit) "HEAD=$head"
$url = (& git -C $subPath remote get-url origin).Trim()
Step 'submodule_remote' ($url -eq $lock.repository) $url
$selfModules = & git -C $subPath ls-tree --name-only $lock.validated_commit '.gitmodules'
Step 'no_self_submodule' (-not $selfModules) 'the locked tree declares no submodule'

Write-Host 'artifacts'
$models = Join-Path $subPath 'models'
foreach ($name in $lock.files.PSObject.Properties.Name) {
    $spec = $lock.files.$name
    $file = Join-Path $models $name
    if (-not (Test-Path $file)) { Step $name $false 'missing'; continue }
    $hash = (Get-FileHash -Algorithm SHA256 -Path $file).Hash.ToLower()
    $size = (Get-Item $file).Length
    Step $name (($hash -eq $spec.sha256) -and ($size -eq $spec.bytes)) $hash.Substring(0, 16)
}
$manifestHash = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $models 'MODEL_MANIFEST.json')).Hash.ToLower()
Step 'manifest_sha256' ($manifestHash -eq $lock.manifest_sha256) $manifestHash.Substring(0, 16)
$registryHash = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $models 'runtime_registry.json')).Hash.ToLower()
Step 'registry_sha256' ($registryHash -eq $lock.registry_sha256) $registryHash.Substring(0, 16)

Write-Host 'interpreters'
$Python = if ($Python) { $Python } elseif ($env:OSLIAN_PYTHON) { $env:OSLIAN_PYTHON } else { 'python' }
$pythonExe = (Get-Command $Python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) { throw "python not found: $Python" }
Step 'demo_python' $true $pythonExe

$ReleasePython = if ($ReleasePython) { $ReleasePython }
                 elseif ($env:OSLIAN_WHOLEBODY_PYTHON) { $env:OSLIAN_WHOLEBODY_PYTHON }
                 else { $pythonExe }
$ReleasePythonPath = if ($ReleasePythonPath) { $ReleasePythonPath } else { $env:OSLIAN_WHOLEBODY_PYTHONPATH }

# Resolve sklearn exactly the way the worker will: the overlay first, then the
# interpreter's own site-packages. Probing without the overlay would report a
# version the worker never loads.
$savedPythonPath = $env:PYTHONPATH
if ($ReleasePythonPath) { $env:PYTHONPATH = $ReleasePythonPath }
$sk = & $ReleasePython -c "import sklearn; print(sklearn.__version__)" 2>$null
$env:PYTHONPATH = $savedPythonPath
Step 'scikit_learn' ($sk -eq $lock.scikit_learn) "installed=$sk expected=$($lock.scikit_learn) overlay=$ReleasePythonPath"

$mods = & $ReleasePython -c @"
import sys; sys.path.insert(0, r'$subPath')
import osler_jepa, whole_body_joint_jepa
print(osler_jepa.__file__)
"@ 2>$null
Step 'module_origin' ([bool]$mods -and $mods.StartsWith($subPath)) "$mods"

Write-Host 'live runtime'
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { throw "port $Port is in use (pid $($listener.OwningProcess)); pass -Port" }

$env:PYTHONPATH = Join-Path $root 'backend'
$env:PYTHONUTF8 = '1'
$env:OSLIAN_PORT = "$Port"
$env:OSLIAN_WHOLEBODY_RUNTIME = 'release'
$env:OSLIAN_WHOLEBODY_RELEASE_DIR = $subPath
$env:OSLIAN_WHOLEBODY_PYTHON = $ReleasePython
if ($ReleasePythonPath) { $env:OSLIAN_WHOLEBODY_PYTHONPATH = $ReleasePythonPath }

$proc = Start-Process -FilePath $pythonExe -ArgumentList '-m', 'oslian_demo' -WorkingDirectory $root `
    -PassThru -WindowStyle Hidden
try {
    $health = $null
    foreach ($attempt in 1..40) {
        Start-Sleep -Milliseconds 1500
        try { $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/model/health" -TimeoutSec 20; break }
        catch { continue }
    }
    if (-not $health) { throw 'the server never answered /api/model/health' }
    Step 'runtime_mode' ($health.runtime_mode -eq 'release') $health.runtime_mode
    Step 'runtime_attached' ($health.runtime_attached -eq $true) "$($health.runtime_attached)"
    Step 'status' ($health.status -eq 'healthy') $health.status
    Step 'ready' ($health.ready -eq $true) "$($health.ready)"
    Step 'worker_sklearn' ($health.scikit_learn -eq $lock.scikit_learn) "$($health.scikit_learn)"
    foreach ($check in $health.checks) {
        Step $check.check ($check.status -eq 'passed') $check.detail
    }
    $caps = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/model/capabilities" -TimeoutSec 20
    Step 'catalog_cells' ($caps.catalog_cells -eq $lock.runtime_contract.catalog_cells) "$($caps.catalog_cells)"
    Step 'model_cells' ($caps.model_cells -eq $lock.runtime_contract.model_capable_cells) "$($caps.model_cells)"
    Step 'persistence_cells' ($caps.persistence_cells -eq $lock.runtime_contract.persistence_fallback_cells) "$($caps.persistence_cells)"
    Step 'causal_claim_allowed' ($caps.causal_claim_allowed -eq $false) 'false'
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}

Write-Host ''
if ($failures) { throw "failed checks: $($failures -join ', ')" }
Write-Host 'release runtime verified.'
