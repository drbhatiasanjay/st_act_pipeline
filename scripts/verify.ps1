[CmdletBinding()]
param(
    [ValidateSet("quick", "full")]
    [string]$Mode = "quick"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot

try {
    # Match the maintained production scope. Repository-wide Ruff currently includes
    # legacy diagnostics/tests with known lint debt; changed Python files are covered
    # separately by the existing pre-commit hook.
    & py -3.12 -m ruff check src run_pipeline.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($Mode -eq "full") {
        Write-Host "Full mode reports the repository's existing mypy debt and runs all tests."
        & py -3.12 -m mypy
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & py -3.12 -m pytest -q
    }
    else {
        Write-Host "Quick mode runs the critical flow-conservation regression guard."
        Write-Host "Full mypy and pytest remain opt-in because their current baseline is red/long-running."
        & py -3.12 -m pytest tests/test_tracker.py -q
    }

    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
