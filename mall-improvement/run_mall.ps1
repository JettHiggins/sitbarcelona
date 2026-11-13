param([ValidateSet('supervised','robust','calibrated')][string]$Candidate='supervised',[switch]$Verify)
$ErrorActionPreference='Stop'
$runtime='C:\Users\mallw\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$deps=Join-Path $PSScriptRoot 'dependencies'
$env:PYTHONPATH="$PSScriptRoot;$deps"
$env:OMP_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
Push-Location $PSScriptRoot
try {
    if ($Candidate -eq 'supervised') {
        if ($Verify) { & $runtime predict_referee_supervised.py --verify }
        else { & $runtime predict_referee_supervised.py }
    } else {
        if ($Verify) { & $runtime predict_robust.py --candidate $Candidate --verify }
        else { & $runtime predict_robust.py --candidate $Candidate }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Prediction failed' }
} finally { Pop-Location }
