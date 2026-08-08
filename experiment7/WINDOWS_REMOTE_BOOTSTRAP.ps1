[CmdletBinding()]
param(
    [string]$LocalRepo = (Join-Path $HOME 'source\pocketmon-experiment7'),
    [string]$SourceBranch = 'agent/experiment7-multideck-challengers-20260808',
    [string]$WorkBranch = 'codex/experiment7-multideck-challengers-20260808',
    [string]$Coordinator = 'doraemon02',
    [string]$ArchivePath = '',
    [string[]]$Servers = @('doraemon02','doraemon03','doraemon15','doraemon16','doraemon19','doraemon20')
)

$ErrorActionPreference = 'Stop'
$RepoUrl = 'https://github.com/LZhangGJ/pocketmon.git'
$ExpectedArchiveSha256 = '9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229'
$ExpectedArchiveBytes = 94038
$RemoteRepo = '/homes/lzhang/pocketmon'
$RemotePython = '/homes/lzhang/mypath/new/envs/trans/bin/python'
$RemoteArchive = '/homes/lzhang/pocketmon/data/imports/experiment7_code_for_gpt_2026-08-08.zip'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with code $LASTEXITCODE"
    }
}

foreach ($Program in @('git','ssh','scp')) {
    if (-not (Get-Command $Program -ErrorAction SilentlyContinue)) {
        throw "Required Windows command is missing: $Program"
    }
}

if (-not (Test-Path (Join-Path $LocalRepo '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $LocalRepo) | Out-Null
    Invoke-Checked git @('clone', $RepoUrl, $LocalRepo)
}

Set-Location $LocalRepo
Invoke-Checked git @('remote','set-url','origin',$RepoUrl)
Invoke-Checked git @('fetch','origin','--prune')
Invoke-Checked git @('switch','--force-create',$WorkBranch,"origin/$SourceBranch")

Write-Host "Repository: $RepoUrl"
Write-Host "Local repo: $LocalRepo"
Write-Host "Source branch: $SourceBranch"
Write-Host "Work branch: $WorkBranch"
Invoke-Checked git @('status','--short','--branch')

Write-Host "`n=== SSH and GPU inventory ==="
foreach ($Server in $Servers) {
    Write-Host "===== $Server ====="
    Invoke-Checked ssh @(
        $Server,
        "hostname; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; test -d $RemoteRepo && echo REPO_OK; test -x $RemotePython && echo PYTHON_OK"
    )
}

if ($ArchivePath) {
    $ResolvedArchive = (Resolve-Path $ArchivePath).Path
    $ArchiveInfo = Get-Item $ResolvedArchive
    $ArchiveSha = (Get-FileHash -Algorithm SHA256 $ResolvedArchive).Hash.ToLowerInvariant()
    if ($ArchiveInfo.Length -ne $ExpectedArchiveBytes) {
        throw "Archive size mismatch: expected=$ExpectedArchiveBytes actual=$($ArchiveInfo.Length)"
    }
    if ($ArchiveSha -ne $ExpectedArchiveSha256) {
        throw "Archive SHA-256 mismatch: expected=$ExpectedArchiveSha256 actual=$ArchiveSha"
    }
    Invoke-Checked ssh @($Coordinator, "mkdir -p $RemoteRepo/data/imports")
    Invoke-Checked scp @($ResolvedArchive, "${Coordinator}:$RemoteArchive")
    Invoke-Checked ssh @($Coordinator, "test `$(stat -c '%s' '$RemoteArchive') -eq $ExpectedArchiveBytes; test `$(sha256sum '$RemoteArchive' | awk '{print `$1}') = '$ExpectedArchiveSha256'")
    Write-Host "Verified archive copied to ${Coordinator}:$RemoteArchive"
}
else {
    Write-Warning 'ArchivePath was not supplied. Remote bootstrap will require an existing verified archive at the remote import path.'
}

$Dirty = (git status --porcelain)
if ($Dirty) {
    throw 'Working tree is dirty. Commit intended code changes before remote execution.'
}

$Commit = (git rev-parse HEAD).Trim()
Invoke-Checked git @('push','--set-upstream','origin',$WorkBranch)

$RemoteWorktree = "/homes/lzhang/worktrees/experiment7-coordinator-$Commit"
$RemoteCommand = @"
set -euo pipefail
cd '$RemoteRepo'
git fetch origin --prune
git cat-file -e '$Commit^{commit}'
if [ ! -d '$RemoteWorktree' ]; then
  git worktree add --detach '$RemoteWorktree' '$Commit'
fi
cd '$RemoteWorktree'
export PYTHON='$RemotePython'
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EXPERIMENT7_ARCHIVE='$RemoteArchive'
bash experiment7/unpack_source.sh
\$PYTHON -m compileall -q runs/experiment7/source experiment7 scripts rl tests
printf '{"host":"%s","commit":"%s","worktree":"%s","archive":"%s"}\n' "\$(hostname)" '$Commit' '$RemoteWorktree' '$RemoteArchive'
"@

Invoke-Checked ssh @($Coordinator, "bash -lc " + [Management.Automation.Language.CodeGeneration]::QuoteArgument($RemoteCommand))
Write-Host "Remote coordinator bootstrap completed at $Coordinator using commit $Commit"
Write-Host 'Next: follow experiment7/CODEX_TRAINING_PROMPT.md from the Windows Codex session.'
