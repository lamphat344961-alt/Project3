param(
    [string]$ReferenceReceipt = 'results/labse_stage03/reference_probe_local_001/execution_receipt.json',
    [string]$ApprovedBy = '',
    [string]$ApprovalReference = '',
    [string]$RunPrefix = '',
    [switch]$RunInference
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$labsePython = Join-Path $projectRoot '.venv-labse-local\Scripts\python.exe'
$snapshot = Join-Path $projectRoot 'results\labse_stage03\checkpoint\836121a0533e5664b21c7aacc5d22951f2b8b25b'
if (-not (Test-Path -LiteralPath $labsePython)) { throw 'Missing .venv-labse-local Python.' }
$referencePath = if ([IO.Path]::IsPathRooted($ReferenceReceipt)) { $ReferenceReceipt } else { Join-Path $projectRoot $ReferenceReceipt }
if (-not (Test-Path -LiteralPath $referencePath)) { throw "Missing checkpoint reference receipt: $referencePath" }
$reference = Get-Content -LiteralPath $referencePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($reference.status -ne 'CHECKPOINT_REFERENCE_PASS') { throw 'Checkpoint reference probe has not passed.' }
if ($reference.revision -ne '836121a0533e5664b21c7aacc5d22951f2b8b25b') { throw 'Reference revision mismatch.' }
if (-not $RunPrefix) { $RunPrefix = 'labse_local_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
if ($RunPrefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid run prefix.' }
if ($RunInference -and (-not $ApprovedBy.Trim() -or -not $ApprovalReference.Trim())) {
    throw 'Supply -ApprovedBy and -ApprovalReference for the actual inference run.'
}

Push-Location -LiteralPath $projectRoot
try {
    $head = (& git rev-parse HEAD).Trim()
    if ($reference.sources.commit -ne $head) { throw 'Source commit changed since reference probe; use a matching reference receipt.' }
    & $labsePython -B -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback'"
    if ($LASTEXITCODE -ne 0) { throw 'Local GPU runtime check failed.' }
    $cells = @('tn1:vi:char_budget', 'tn1:vi:native', 'tn2:en:native', 'tn2:vi:native')
    foreach ($cell in $cells) {
        $runId = $RunPrefix + '_' + $cell.Replace(':', '_')
        $namespace = Join-Path $projectRoot ('results\labse_stage03\' + $runId)
        if (Test-Path -LiteralPath $namespace) { throw "Output already exists: $namespace" }
    }
    if (-not $RunInference) {
        Write-Output 'READY: four approved-scope cells; FP32/batch8; checkpoint reference passed.'
        Write-Output 'Preparation only. To encode, rerun with -RunInference -ApprovedBy <owner> -ApprovalReference <approval>.'
        Write-Output ('Output prefix: results/labse_stage03/' + $RunPrefix)
        return
    }
    foreach ($cell in $cells) {
        $runId = $RunPrefix + '_' + $cell.Replace(':', '_')
        Write-Output ('Starting ' + $cell + ' -> ' + $runId)
        & $labsePython -B -m pipeline.labse_stage03 encode $runId --snapshot $snapshot --cell $cell --approved-by $ApprovedBy --approval-reference $ApprovalReference --reference-receipt $referencePath
        if ($LASTEXITCODE -ne 0) { throw "Cell failed: $cell. Existing outputs preserved; no automatic retry or batch reduction." }
    }
} finally {
    Pop-Location
}
