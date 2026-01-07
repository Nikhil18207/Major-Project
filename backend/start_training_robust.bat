@echo off
REM =======================================================
REM   ROBUST 50-EPOCH TRAINING SCRIPT
REM   Ensures uninterrupted training with auto-recovery
REM =======================================================

echo ========================================
echo  XR2Text Training - 50 Epochs
echo  HAQT-ARR + Novel Features
echo ========================================
echo.

REM Check if CUDA is available
nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] NVIDIA GPU not detected!
    echo Please ensure your GPU drivers are installed.
    pause
    exit /b 1
)

echo [OK] GPU detected
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo.

REM Set high priority for training process
echo [INFO] Setting high priority for training...
wmic process where name="python.exe" CALL setpriority "high priority" >nul 2>&1

REM Prevent Windows from sleeping
echo [INFO] Disabling sleep/hibernate during training...
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0

echo.
echo ========================================
echo  Starting Training...
echo  - Total Epochs: 50
echo  - Checkpoints: Every 5 epochs
echo  - Expected Time: 12-17 hours
echo ========================================
echo.

REM Change to notebooks directory
cd /d "%~dp0notebooks"

REM Start training with jupyter nbconvert
echo [INFO] Launching Jupyter notebook for training...
jupyter nbconvert --to notebook --execute 02_model_training.ipynb --output 02_model_training_output.ipynb

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  Training Completed Successfully!
    echo ========================================
    echo.
    echo Results saved to:
    echo   - ../data/statistics/training_history.csv
    echo   - ../checkpoints/best_model.pt
    echo.
) else (
    echo.
    echo ========================================
    echo  Training Failed or Interrupted
    echo ========================================
    echo.
    echo Check the logs for errors
    echo You can resume training from last checkpoint
    echo.
)

REM Restore power settings
echo [INFO] Restoring power settings...
powercfg /change standby-timeout-ac 15
powercfg /change hibernate-timeout-ac 30
powercfg /change monitor-timeout-ac 10

pause
