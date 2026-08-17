$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONDONTWRITEBYTECODE = '1'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ControlRoot = $PSScriptRoot
$Python = 'python'
$Scheduler = Join-Path $RepoRoot 'experiment7\integration\multi_gpu_scheduler.py'
$Sources = '/homes/lzhang/pocketmon/runs/experiment7-multideck-20260809/multiday_7d/prepared/training_sources.json'
$RunRoot = '/homes/lzhang/pocketmon/runs/experiment7-multideck-20260809/multiday_7d/training'
$Worktree = '/homes/lzhang/worktrees/experiment7-7ec3c685c421'
$Commit = '7ec3c685c421d09fb1663f94868d061ae5c6d7a8'
$RemotePython = '/homes/lzhang/mypath/new/envs/trans/bin/python'
$Worker = "$Worktree/experiment7/integration/remote_worker.py"
$StatusLog = Join-Path $ControlRoot 'run_7d_training_status.log'

function Write-Status([string] $Message) {
    $line = "$(Get-Date -AsUTC -Format o) $Message"
    Add-Content -LiteralPath $StatusLog -Value $line -Encoding utf8
    Write-Host $line
}

function Invoke-Scheduler([string[]] $Arguments) {
    $output = @(& $Python $Scheduler @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    foreach ($line in $output) {
        Add-Content -LiteralPath $StatusLog -Value ([string] $line) -Encoding utf8
    }
    if ($exitCode -ne 0) {
        throw "scheduler failed with exit code $exitCode"
    }
    return ($output[-1] | ConvertFrom-Json)
}

function Wait-ForSources {
    while ($true) {
        & ssh doraemon03 "test -f '$Sources'"
        if ($LASTEXITCODE -eq 0) {
            Write-Status "sealed training sources are ready: $Sources"
            return
        }
        Write-Status 'sealed training sources are not ready; retrying in 60 seconds'
        Start-Sleep -Seconds 60
    }
}

function Get-Inventory([string] $Tag, [int] $Required) {
    $inventoryPath = Join-Path $ControlRoot "gpu_inventory_7d_$Tag.json"
    while ($true) {
        try {
            Invoke-Scheduler @(
                'inventory', '--hosts', 'doraemon15', '--output', $inventoryPath,
                '--minimum-free-mib', '40000', '--maximum-utilization', '5',
                '--ssh-timeout-seconds', '90'
            ) | Out-Null
            $inventory = Get-Content -Raw -LiteralPath $inventoryPath | ConvertFrom-Json
            $eligible = @($inventory.gpus | Where-Object { $_.eligible }).Count
            Write-Status "$Tag inventory found $eligible eligible GPU(s); need $Required"
            if ($eligible -ge $Required) {
                return $inventoryPath
            }
        }
        catch {
            Write-Status "$Tag inventory transient failure: $($_.Exception.Message)"
        }
        Start-Sleep -Seconds 60
    }
}

function Wait-ForPlan([string] $PlanPath) {
    while ($true) {
        try {
            $payload = Invoke-Scheduler @(
                'status', '--plan', $PlanPath, '--remote-python', $RemotePython, '--worker', $Worker
            )
            $statuses = @($payload.statuses)
            $stateText = (($statuses | ForEach-Object { "$($_.jobId)=$($_.status)" }) -join ', ')
            Write-Status "plan status: $stateText"
            $terminalFailures = @($statuses | Where-Object { $_.status -in @('failed', 'blocked_gpu_lock') })
            if ($terminalFailures.Count -gt 0) {
                throw "training plan failed: $stateText"
            }
            if ($statuses.Count -gt 0 -and @($statuses | Where-Object { $_.status -ne 'succeeded' }).Count -eq 0) {
                return
            }
        }
        catch {
            if ($_.Exception.Message -like 'training plan failed:*') {
                throw
            }
            Write-Status "status check transient failure: $($_.Exception.Message)"
        }
        Start-Sleep -Seconds 60
    }
}

function Start-Stage(
    [string] $Stage,
    [string] $InventoryPath,
    [string] $PlanName,
    [string] $PretrainCheckpoint = ''
) {
    $planPath = Join-Path $ControlRoot $PlanName
    $arguments = @(
        'make-training-plan', '--inventory', $InventoryPath, '--output', $planPath,
        '--worktree', $Worktree, '--commit', $Commit, '--python', $RemotePython,
        '--sources', $Sources, '--run-root', $RunRoot, '--stage', $Stage
    )
    if ($PretrainCheckpoint) {
        $arguments += @('--pretrain-checkpoint', $PretrainCheckpoint)
    }
    Invoke-Scheduler $arguments | Out-Null
    Invoke-Scheduler @(
        'launch', '--plan', $planPath, '--remote-python', $RemotePython, '--worker', $Worker
    ) | Out-Null
    Write-Status "$Stage launched from $planPath"
    Wait-ForPlan $planPath
    Write-Status "$Stage succeeded"
}

try {
    Write-Status "7-day Experiment 7 training controller started at commit $Commit"
    Wait-ForSources

    & ssh doraemon03 "test ! -e '$RunRoot'"
    if ($LASTEXITCODE -ne 0) {
        throw "refusing to overwrite existing training root: $RunRoot"
    }

    $smokeInventory = Get-Inventory 'smoke' 1
    Start-Stage 'smoke' $smokeInventory 'training_plan_7d_smoke.json'

    $pretrainInventory = Get-Inventory 'pretrain' 1
    Start-Stage 'pretrain' $pretrainInventory 'training_plan_7d_pretrain.json'

    $pretrainCheckpoint = "$RunRoot/pretrain/pretrain_model.pt"
    & ssh doraemon15 "test -s '$pretrainCheckpoint'"
    if ($LASTEXITCODE -ne 0) {
        throw "pretrain checkpoint is missing: $pretrainCheckpoint"
    }

    $finetuneInventory = Get-Inventory 'finetune' 3
    Start-Stage 'finetune' $finetuneInventory 'training_plan_7d_finetune.json' $pretrainCheckpoint

    $completion = [ordered]@{
        schemaVersion = 1
        status = 'succeeded'
        finishedAt = (Get-Date -AsUTC -Format o)
        commit = $Commit
        sources = $Sources
        runRoot = $RunRoot
    }
    $completion | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ControlRoot 'run_7d_training_completion.json') -Encoding utf8
    Write-Status 'all 7-day training stages succeeded'
}
catch {
    $failure = [ordered]@{
        schemaVersion = 1
        status = 'failed'
        finishedAt = (Get-Date -AsUTC -Format o)
        error = $_.Exception.Message
        commit = $Commit
        sources = $Sources
        runRoot = $RunRoot
    }
    $failure | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ControlRoot 'run_7d_training_failure.json') -Encoding utf8
    Write-Status "controller failed: $($_.Exception.Message)"
    exit 1
}
