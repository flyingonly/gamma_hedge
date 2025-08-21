@echo off
REM ============================================================
REM Quick Clean Script - No confirmation required
REM Silently removes all training outputs and checkpoints
REM ============================================================

REM Check if we're in the correct directory
if not exist "main.py" (
    echo ERROR: Please run this script from the gamma_hedge project root directory
    exit /b 1
)

echo Cleaning outputs and checkpoints...

REM Create directories if they don't exist
if not exist "outputs" mkdir outputs
if not exist "checkpoints" mkdir checkpoints

REM Clean outputs directory silently
if exist "outputs\*" (
    del /s /q "outputs\*" >nul 2>&1
    for /d %%d in ("outputs\*") do rd /s /q "%%d" >nul 2>&1
)

REM Clean checkpoints directory silently
if exist "checkpoints\*" (
    del /s /q "checkpoints\*" >nul 2>&1
    for /d %%d in ("checkpoints\*") do rd /s /q "%%d" >nul 2>&1
)

REM Clean debug files
if exist "debug_*.py" (
    del /q "debug_*.py" >nul 2>&1
)

echo Done! Cleaned outputs/, checkpoints/, and debug files.