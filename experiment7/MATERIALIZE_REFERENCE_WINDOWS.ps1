[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,

    [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$ExpectedBytes = 94038
$ExpectedSha256 = '9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229'

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$Archive = (Resolve-Path -LiteralPath $ArchivePath).Path
$ArchiveInfo = Get-Item -LiteralPath $Archive
$ArchiveSha = Get-Sha256 $Archive
if ($ArchiveInfo.Length -ne $ExpectedBytes) {
    throw "Experiment 7 ZIP size mismatch: expected=$ExpectedBytes actual=$($ArchiveInfo.Length) path=$Archive"
}
if ($ArchiveSha -ne $ExpectedSha256) {
    throw "Experiment 7 ZIP SHA-256 mismatch: expected=$ExpectedSha256 actual=$ArchiveSha path=$Archive"
}

$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("experiment7-reference-" + [guid]::NewGuid().ToString('N'))
$Extracted = Join-Path $TempRoot 'source'
$Destination = Join-Path $RepositoryRoot 'experiment7\reference'
New-Item -ItemType Directory -Force -Path $Extracted | Out-Null

try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Extracted)

    $ManifestPath = Join-Path $Extracted 'PACKAGE_MANIFEST.csv'
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "PACKAGE_MANIFEST.csv is missing from the verified ZIP"
    }

    $ManifestRows = Import-Csv -LiteralPath $ManifestPath
    $Errors = [System.Collections.Generic.List[string]]::new()
    foreach ($Row in $ManifestRows) {
        $Relative = [string]$Row.path
        $ExpectedFileBytes = [int64]$Row.bytes
        $ExpectedFileSha = ([string]$Row.sha256).ToLowerInvariant()
        $FilePath = Join-Path $Extracted $Relative
        if (-not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
            $Errors.Add("missing:$Relative")
            continue
        }
        $ActualFileBytes = (Get-Item -LiteralPath $FilePath).Length
        if ($ActualFileBytes -ne $ExpectedFileBytes) {
            $Errors.Add("bytes:$Relative expected=$ExpectedFileBytes actual=$ActualFileBytes")
        }
        $ActualFileSha = Get-Sha256 $FilePath
        if ($ActualFileSha -ne $ExpectedFileSha) {
            $Errors.Add("sha256:$Relative expected=$ExpectedFileSha actual=$ActualFileSha")
        }
    }
    if ($Errors.Count -gt 0) {
        throw ("Experiment 7 package manifest validation failed:`n" + ($Errors -join "`n"))
    }

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination) | Out-Null
    Copy-Item -LiteralPath $Extracted -Destination $Destination -Recurse -Force

    $Receipt = [ordered]@{
        schemaVersion = 1
        archivePath = $Archive
        archiveBytes = $ArchiveInfo.Length
        archiveSha256 = $ArchiveSha
        manifestFiles = $ManifestRows.Count
        manifestErrors = 0
        materializedPath = 'experiment7/reference'
        generatedAtUtc = [DateTime]::UtcNow.ToString('o')
    }
    $ReceiptPath = Join-Path $Destination 'IMPORT_RECEIPT.json'
    $Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8NoBOM

    $Required = @(
        'training\deck_identity_model.py',
        'training\train_multideck_identity.py',
        'data_pipeline\features.py',
        'data_pipeline\tokenizer.py',
        'data_pipeline\build_token_cache.py',
        'data_pipeline\build_sequence_cache.py',
        'data_pipeline\build_deck_identity_cache.py',
        'runtime_agent\main.py',
        'validation\arena_isolated.py',
        'docs\EXPERIMENT7_CLEANROOM_DESIGN.md'
    )
    foreach ($Relative in $Required) {
        $RequiredPath = Join-Path $Destination $Relative
        if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
            throw "Materialized reference source is incomplete: missing experiment7/reference/$($Relative -replace '\\','/')"
        }
    }

    Write-Host "Experiment 7 ordinary source materialized and verified."
    Write-Host "REFERENCE_PATH=$Destination"
    Write-Host "ARCHIVE_SHA256=$ArchiveSha"
    Write-Host "MANIFEST_FILES=$($ManifestRows.Count)"
}
finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    }
}
