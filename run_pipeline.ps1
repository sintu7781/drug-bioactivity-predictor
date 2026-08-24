# ============================================================
# Drug Bioactivity Predictor
# End-to-End Machine Learning Pipeline
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       DRUG BIOACTIVITY PREDICTION PIPELINE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $ProjectRoot

Write-Host "Project root:" -ForegroundColor DarkGray
Write-Host "  $ProjectRoot"
Write-Host ""

# ------------------------------------------------------------
# Python executable
# ------------------------------------------------------------

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host ""
    Write-Host "ERROR: Python virtual environment was not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected:"
    Write-Host "  $Python"
    Write-Host ""
    Write-Host "Create the virtual environment with:"
    Write-Host "  python -m venv .venv"
    Write-Host ""
    Write-Host "Then activate it and install dependencies:"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  python -m pip install -r requirements.txt"
    Write-Host ""
    exit 1
}

Write-Host "Python:" -ForegroundColor DarkGray
& $Python --version

Write-Host ""

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

function Invoke-PipelineStep {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Module
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "$Name" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    & $Python -m $Module

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Pipeline step failed." -ForegroundColor Red
        Write-Host "Module: $Module" -ForegroundColor Red
        Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host ""
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "SUCCESS: $Name completed." -ForegroundColor Green
}

# ============================================================
# 1. Download EGFR activity data
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 1/8 - Download EGFR bioactivity data" `
    -Module "src.download_data"

# ============================================================
# 2. Download / resolve molecule SMILES
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 2/8 - Download molecule structures" `
    -Module "src.download_molecules"

# ============================================================
# 3. Curate dataset
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 3/8 - Curate bioactivity dataset" `
    -Module "src.curate"

# ============================================================
# 4. Build molecular features
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 4/8 - Build molecular features" `
    -Module "src.build_features"

# ============================================================
# 5. Train machine-learning models
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 5/8 - Train machine-learning models" `
    -Module "src.train"

# ============================================================
# 6. Evaluate trained model
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 6/8 - Evaluate model" `
    -Module "src.evaluate"

# ============================================================
# 7. Scaffold-based evaluation
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 7/8 - Run scaffold-based evaluation" `
    -Module "src.scaffold_evaluate"

# ============================================================
# 8. Final model validation
# ============================================================

Invoke-PipelineStep `
    -Name "STEP 8/8 - Run final model validation" `
    -Module "src.final_evaluation"

# ============================================================
# Pipeline completed
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "          PIPELINE COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Generated artifacts:" -ForegroundColor Cyan
Write-Host ""

$Artifacts = @(
    "data\processed\egfr_ic50_curated.csv",
    "data\processed\egfr_features.npz",
    "models\bioactivity_model.joblib",
    "models\model_metadata.json",
    "reports\data_quality.json",
    "reports\model_comparison.csv",
    "reports\scaffold_evaluation.json",
    "reports\model_validation.json",
    "reports\figures\roc_curve.png",
    "reports\figures\precision_recall_curve.png",
    "reports\figures\confusion_matrix.png"
)

foreach ($Artifact in $Artifacts) {

    $ArtifactPath = Join-Path $ProjectRoot $Artifact

    if (Test-Path $ArtifactPath) {
        Write-Host "  [OK] $Artifact" -ForegroundColor Green
    }
    else {
        Write-Host "  [MISSING] $Artifact" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  Run the application:"
Write-Host "    python -m streamlit run app.py"
Write-Host ""
Write-Host "  Run tests:"
Write-Host "    pytest"
Write-Host ""
Write-Host "  Run code quality checks:"
Write-Host "    ruff check ."
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""