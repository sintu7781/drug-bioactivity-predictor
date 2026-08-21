$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=========================================="
Write-Host "Drug Bioactivity Prediction Pipeline"
Write-Host "=========================================="
Write-Host ""

python -m src.download_data

python -m src.download_molecules

python -m src.curate

python -m src.build_features

python -m src.train

python -m src.evaluate

Write-Host ""
Write-Host "=========================================="
Write-Host "Pipeline completed successfully."
Write-Host "=========================================="