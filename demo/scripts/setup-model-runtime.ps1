<#
.SYNOPSIS
    Initialise the oslian/Model release submodule at the commit model-lock.json approves.

.DESCRIPTION
    Idempotent and non-destructive. It initialises the one submodule, checks that
    it points at the expected repository, checks out the locked commit, verifies
    every model file against model-lock.json, and proves the locked tree declares
    no submodule of its own, so a recursive clone stops one level deep.

    It does NOT install packages, fetch newer commits, or push anything. Use
    sync-model-research.ps1 to look at what upstream has.

.PARAMETER Force
    Re-checkout the locked commit even when the submodule already sits on it.
    Refused if the submodule has uncommitted changes.

.EXAMPLE
    .\scripts\setup-model-runtime.ps1
#>
[CmdletBinding()]
param([switch] $Force)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $root 'backend\model-lock.json'
if (-not (Test-Path $lockPath)) { throw "model-lock.json not found: $lockPath" }
$lock = Get-Content $lockPath -Raw | ConvertFrom-Json
$subPath = Join-Path $root $lock.submodule_path
$commit = $lock.validated_commit

function Fail([string] $message) { throw $message }

Write-Host "repository  : $($lock.repository)"
Write-Host "submodule   : $($lock.submodule_path) (branch $($lock.submodule_branch))"
Write-Host "locked at   : $commit"
Write-Host ''

# ---- 1. the submodule declaration must be the only one ----
$modulesFile = Join-Path $root '.gitmodules'
if (-not (Test-Path $modulesFile)) { Fail '.gitmodules is missing' }
$declared = @(& git -C $root config --file .gitmodules --get-regexp 'submodule\..*\.path' |
    ForEach-Object { ($_ -split ' ', 2)[1] })
if ($declared.Count -ne 1) { Fail "expected exactly one submodule, found: $($declared -join ', ')" }
if ($declared[0] -ne $lock.submodule_path) { Fail "submodule path is $($declared[0]), expected $($lock.submodule_path)" }

$declaredBranch = & git -C $root config --file .gitmodules --get "submodule.$($lock.submodule_path).branch"
if ($declaredBranch -ne $lock.submodule_branch) {
    Fail "submodule branch is '$declaredBranch', model-lock.json says '$($lock.submodule_branch)'"
}

# ---- 2. initialise / update ------------------------------------------------
# deliberately NOT --recursive: this branch owns exactly one submodule and must
# never follow one it did not declare
& git -C $root submodule update --init -- $lock.submodule_path
if ($LASTEXITCODE -ne 0) { Fail 'git submodule update failed' }

$url = & git -C $subPath remote get-url origin
if ($url -ne $lock.repository) { Fail "submodule remote is $url, expected $($lock.repository)" }

# ---- 3. the artifacts are hashed byte for byte, so the checkout must be LF ----
# The model repository ships no .gitattributes, so a clone on a machine with
# core.autocrlf=true lands CRLF in the working tree and every text file then
# reads as modified. That is noise, not work, and the lock says to fix the
# checkout. Real edits are a different matter and are still refused.
& git -C $subPath config core.autocrlf false | Out-Null
& git -C $subPath config core.eol lf | Out-Null

$dirty = @(& git -C $subPath status --porcelain)
if ($dirty) {
    $realChanges = @(& git -C $subPath diff --ignore-cr-at-eol --name-only)
    if ($realChanges -and -not $Force) {
        Write-Host 'submodule has local changes:'
        $realChanges | ForEach-Object { Write-Host "  $_" }
        Fail 'refusing to touch a dirty submodule; commit or clean it first, or pass -Force to re-checkout'
    }
    if (-not $realChanges) {
        Write-Host "line endings : re-materialising $($dirty.Count) file(s) as LF"
        & git -C $subPath checkout --force -- .
    }
}

# models/ is what the lock hashes, and git will not rewrite a file it believes is
# already current -- which is exactly the case after a clone that wrote CRLF
# under a different config. Delete and restore them so the bytes on disk are the
# bytes in the commit, unconditionally.
foreach ($tracked in @(& git -C $subPath ls-files -- models)) {
    Remove-Item -LiteralPath (Join-Path $subPath $tracked) -Force -ErrorAction SilentlyContinue
}
& git -C $subPath checkout --force -- models

$head = (& git -C $subPath rev-parse HEAD).Trim()
if ($head -ne $commit -or $Force) {
    & git -C $subPath checkout --force --detach $commit
    if ($LASTEXITCODE -ne 0) { Fail "cannot check out $commit; fetch it first with sync-model-research.ps1" }
    # re-materialise anything that was checked out with the wrong line endings
    & git -C $subPath checkout --force -- .
}
$head = (& git -C $subPath rev-parse HEAD).Trim()
if ($head -ne $commit) { Fail "submodule HEAD is $head, expected $commit" }
Write-Host "HEAD        : $head  (matches the lock)"

# ---- 4. the locked tree must not contain a submodule of its own -------------
$selfModules = & git -C $subPath ls-tree --name-only $commit '.gitmodules'
if ($selfModules) {
    Fail "the locked commit carries its own .gitmodules; a recursive clone could loop. Refusing."
}
Write-Host 'recursion   : the locked tree declares no submodule of its own'

# ---- 5. every model file must match the lock -------------------------------
$models = Join-Path $subPath 'models'
$bad = @()
foreach ($name in $lock.files.PSObject.Properties.Name) {
    $spec = $lock.files.$name
    $file = Join-Path $models $name
    if (-not (Test-Path $file)) { $bad += "$name is missing"; continue }
    $size = (Get-Item $file).Length
    $hash = (Get-FileHash -Algorithm SHA256 -Path $file).Hash.ToLower()
    if ($size -ne $spec.bytes) { $bad += "$name is $size bytes, expected $($spec.bytes)"; continue }
    if ($hash -ne $spec.sha256) { $bad += "$name sha256 $hash, expected $($spec.sha256)"; continue }
    Write-Host ("  ok  {0,-42} {1}" -f $name, $hash.Substring(0, 16))
}
if ($bad) {
    $bad | ForEach-Object { Write-Host "  FAIL $_" }
    Fail 'model files do not match model-lock.json. A CRLF checkout is the usual cause; fix the checkout, never the check.'
}

Write-Host ''
Write-Host 'model runtime is ready. Start it with:  .\scripts\start-demo.ps1 -Mode release'
