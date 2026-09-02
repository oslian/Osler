<#
.SYNOPSIS
    Show what the WholeBody-JEPA research repository has that the locked commit does not.

.DESCRIPTION
    A dry run by default: it fetches, reports, and changes nothing. It never
    checks out, never rewrites model-lock.json, and never pushes.

    With -Update it will check out the candidate commit and run the runtime
    contract tests against it. Even then it does not touch model-lock.json: a
    candidate becomes the demo runtime only when you edit the lock deliberately,
    after the tests have passed.

.PARAMETER Update
    Try a candidate commit (default: the submodule's tracked branch tip) and run
    the runtime contract tests against it. Refused if the submodule is dirty.

.PARAMETER Commit
    The candidate commit to try with -Update. Defaults to the tracked branch tip.

.PARAMETER Python
    Interpreter used for the contract tests. Falls back to $env:OSLIAN_PYTHON, then python.

.EXAMPLE
    .\scripts\sync-model-research.ps1

.EXAMPLE
    .\scripts\sync-model-research.ps1 -Update
#>
[CmdletBinding()]
param(
    [switch] $Update,
    [string] $Commit,
    [string] $Python
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$lock = Get-Content (Join-Path $root 'backend\model-lock.json') -Raw | ConvertFrom-Json
$subPath = Join-Path $root $lock.submodule_path
if (-not (Test-Path (Join-Path $subPath '.git'))) {
    throw "submodule is not initialised. Run .\scripts\setup-model-runtime.ps1 first."
}

# ---- never overwrite work in progress --------------------------------------
$dirty = & git -C $subPath status --porcelain
if ($dirty) {
    Write-Host 'submodule has uncommitted changes:'
    $dirty | ForEach-Object { Write-Host "  $_" }
    throw 'stopping: commit or clean the submodule before syncing. Nothing was fetched or changed.'
}

Write-Host "fetching $($lock.repository) ..."
& git -C $subPath fetch --prune origin
if ($LASTEXITCODE -ne 0) { throw 'fetch failed' }

$branch = $lock.submodule_branch
$head = (& git -C $subPath rev-parse HEAD).Trim()
$remote = (& git -C $subPath rev-parse "origin/$branch").Trim()

Write-Host ''
Write-Host ("{0,-18} : {1}" -f 'locked commit', $lock.validated_commit)
Write-Host ("{0,-18} : {1}" -f 'submodule HEAD', $head)
Write-Host ("{0,-18} : {1}" -f "origin/$branch tip", $remote)
Write-Host ''

if ($head -ne $lock.validated_commit) {
    Write-Host 'WARNING: the submodule is not on the locked commit. Run setup-model-runtime.ps1.'
    Write-Host ''
}

if ($remote -eq $lock.validated_commit) {
    Write-Host "origin/$branch is the locked commit. Nothing to review."
    if (-not $Update) { return }
} else {
    Write-Host "commits on origin/$branch since the locked commit:"
    & git -C $subPath log --oneline "$($lock.validated_commit)..origin/$branch"
    Write-Host ''
    Write-Host 'files that differ under models/:'
    $changed = & git -C $subPath diff --name-only $lock.validated_commit "origin/$branch" -- models
    if ($changed) { $changed | ForEach-Object { Write-Host "  $_" } } else { Write-Host '  (none)' }
    Write-Host ''
}

if (-not $Update) {
    Write-Host 'Dry run only. Nothing was checked out, no lock was rewritten, nothing was pushed.'
    Write-Host 'To try the candidate:  .\scripts\sync-model-research.ps1 -Update'
    return
}

# ---- candidate trial -------------------------------------------------------
if (-not $Commit) { $Commit = $remote }
if ($Commit -eq $lock.validated_commit) { Write-Host 'candidate is the locked commit; nothing to try.'; return }

$selfModules = & git -C $subPath ls-tree --name-only $Commit '.gitmodules'
if ($selfModules) { throw "candidate $Commit carries its own .gitmodules; a recursive clone could loop. Refusing." }

Write-Host "trying candidate $Commit ..."
& git -C $subPath checkout --force --detach $Commit
& git -C $subPath checkout --force -- .

$Python = if ($Python) { $Python } elseif ($env:OSLIAN_PYTHON) { $env:OSLIAN_PYTHON } else { 'python' }
$pythonExe = (Get-Command $Python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) { throw "python not found: $Python" }

$env:PYTHONUTF8 = '1'
$env:OSLIAN_WHOLEBODY_RUNTIME = 'release'
$env:OSLIAN_WHOLEBODY_RELEASE_DIR = $subPath
Push-Location $root
try {
    & $pythonExe -m pytest tests/test_stream_runtime_contract.py tests/test_stream_api.py -q
    $passed = ($LASTEXITCODE -eq 0)
} finally { Pop-Location }

Write-Host ''
if ($passed) {
    Write-Host "candidate $Commit PASSED the runtime contract tests."
    Write-Host 'It is still NOT the demo runtime. To adopt it, deliberately:'
    Write-Host '  1. update backend/model-lock.json (commit, hashes, release date, counts)'
    Write-Host '  2. commit the submodule pointer and the lock together on the oslian-demo branch'
} else {
    Write-Host "candidate $Commit FAILED the runtime contract tests and must not become the demo runtime."
}
Write-Host 'restoring the locked commit ...'
& git -C $subPath checkout --force --detach $lock.validated_commit
& git -C $subPath checkout --force -- .
if (-not $passed) { throw 'candidate rejected' }
