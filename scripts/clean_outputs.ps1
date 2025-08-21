# ============================================================
# Clean Outputs and Checkpoints Script (PowerShell Version)
# Removes all training outputs and model checkpoints
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "GAMMA HEDGE CLEANUP SCRIPT" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will remove:" -ForegroundColor Yellow
Write-Host "  - All files in outputs/ directory" -ForegroundColor Yellow
Write-Host "  - All files in checkpoints/ directory" -ForegroundColor Yellow
Write-Host "  - All visualization files" -ForegroundColor Yellow
Write-Host "  - All training logs and results" -ForegroundColor Yellow
Write-Host ""

# Check if we're in the correct directory
if (-not (Test-Path "main.py")) {
    Write-Host "ERROR: Please run this script from the gamma_hedge project root directory" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Ask for confirmation
$confirm = Read-Host "Are you sure you want to continue? (y/N)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Cleanup cancelled." -ForegroundColor Green
    Read-Host "Press Enter to exit"
    exit 0
}

Write-Host ""
Write-Host "Starting cleanup..." -ForegroundColor Green
Write-Host ""

# Create directories if they don't exist
if (-not (Test-Path "outputs")) { New-Item -ItemType Directory -Path "outputs" | Out-Null }
if (-not (Test-Path "checkpoints")) { New-Item -ItemType Directory -Path "checkpoints" | Out-Null }

# Count files before cleanup
Write-Host "Counting files before cleanup..." -ForegroundColor Cyan
$outputFiles = @()
$checkpointFiles = @()

if (Test-Path "outputs") {
    $outputFiles = Get-ChildItem -Path "outputs" -Recurse -File -ErrorAction SilentlyContinue
}
if (Test-Path "checkpoints") {
    $checkpointFiles = Get-ChildItem -Path "checkpoints" -Recurse -File -ErrorAction SilentlyContinue
}

$totalFiles = $outputFiles.Count + $checkpointFiles.Count
Write-Host "Found $totalFiles files to remove." -ForegroundColor Cyan
Write-Host ""

# Clean outputs directory
Write-Host "[1/2] Cleaning outputs directory..." -ForegroundColor Yellow
try {
    if (Test-Path "outputs\*") {
        Remove-Item -Path "outputs\*" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  - Removed all files from outputs/" -ForegroundColor Green
    } else {
        Write-Host "  - outputs/ directory is already empty" -ForegroundColor Green
    }
} catch {
    Write-Host "  - Warning: Some files in outputs/ may be in use" -ForegroundColor Yellow
}

# Clean checkpoints directory
Write-Host "[2/2] Cleaning checkpoints directory..." -ForegroundColor Yellow
try {
    if (Test-Path "checkpoints\*") {
        Remove-Item -Path "checkpoints\*" -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  - Removed all files from checkpoints/" -ForegroundColor Green
    } else {
        Write-Host "  - checkpoints/ directory is already empty" -ForegroundColor Green
    }
} catch {
    Write-Host "  - Warning: Some files in checkpoints/ may be in use" -ForegroundColor Yellow
}

# Clean temporary debug files
Write-Host ""
Write-Host "Cleaning temporary debug files..." -ForegroundColor Yellow
$debugFiles = Get-ChildItem -Path "." -Name "debug_*.py" -ErrorAction SilentlyContinue
if ($debugFiles.Count -gt 0) {
    Remove-Item -Path "debug_*.py" -Force -ErrorAction SilentlyContinue
    Write-Host "  - Removed $($debugFiles.Count) debug script files" -ForegroundColor Green
} else {
    Write-Host "  - No debug files found" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Cleaned directories:" -ForegroundColor Green
Write-Host "  - outputs/           (training results, logs, visualizations)" -ForegroundColor White
Write-Host "  - checkpoints/       (model checkpoints, best/final models)" -ForegroundColor White
Write-Host "  - debug files        (temporary debug scripts)" -ForegroundColor White
Write-Host ""
Write-Host "You can now run fresh training sessions without interference" -ForegroundColor Green
Write-Host "from previous results." -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"