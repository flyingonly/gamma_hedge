@echo off
REM ============================================================
REM Clean Outputs and Checkpoints Script
REM Removes all training outputs and model checkpoints
REM ============================================================

echo.
echo ============================================================
echo GAMMA HEDGE CLEANUP SCRIPT
echo ============================================================
echo.
echo This script will remove:
echo   - All files in outputs/ directory
echo   - All files in checkpoints/ directory
echo   - All visualization files
echo   - All training logs and results
echo.

REM Check if we're in the correct directory
if not exist "main.py" (
    echo ERROR: Please run this script from the gamma_hedge project root directory
    echo Current directory: %cd%
    pause
    exit /b 1
)

REM Ask for confirmation
set /p confirm="Are you sure you want to continue? (y/N): "
if /i not "%confirm%"=="y" (
    echo Cleanup cancelled.
    pause
    exit /b 0
)

echo.
echo Starting cleanup...
echo.

REM Create directories if they don't exist to avoid errors
if not exist "outputs" mkdir outputs
if not exist "checkpoints" mkdir checkpoints

REM Count files before cleanup
echo Counting files before cleanup...
set file_count=0
for /r outputs %%f in (*) do set /a file_count+=1
for /r checkpoints %%f in (*) do set /a file_count+=1
echo Found %file_count% files to remove.
echo.

REM Clean outputs directory
echo [1/2] Cleaning outputs directory...
if exist "outputs\*" (
    del /s /q "outputs\*" >nul 2>&1
    for /d %%d in ("outputs\*") do rd /s /q "%%d" >nul 2>&1
    echo   - Removed all files from outputs/
) else (
    echo   - outputs/ directory is already empty
)

REM Clean checkpoints directory
echo [2/2] Cleaning checkpoints directory...
if exist "checkpoints\*" (
    del /s /q "checkpoints\*" >nul 2>&1
    for /d %%d in ("checkpoints\*") do rd /s /q "%%d" >nul 2>&1
    echo   - Removed all files from checkpoints/
) else (
    echo   - checkpoints/ directory is already empty
)

REM Also clean any temporary debug files in project root
echo.
echo Cleaning temporary debug files...
if exist "debug_*.py" (
    del /q "debug_*.py" >nul 2>&1
    echo   - Removed debug script files
)

echo.
echo ============================================================
echo CLEANUP COMPLETED SUCCESSFULLY
echo ============================================================
echo.
echo Cleaned directories:
echo   - outputs/           (training results, logs, visualizations)
echo   - checkpoints/       (model checkpoints, best/final models)
echo   - debug files        (temporary debug scripts)
echo.
echo You can now run fresh training sessions without interference
echo from previous results.
echo.
pause