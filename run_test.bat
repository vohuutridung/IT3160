@echo off
cd /d "%~dp0"
echo ======================================================
echo    DANH GIA MODEL CHESS POLICY TREN TEST SET
echo ======================================================

if not exist checkpoints\best_model.pt (
    echo [LOI] Khong tim thay checkpoints\best_model.pt.
    echo Ban can chay run_train.bat de train model truoc!
    pause
    exit /b 1
)

echo Dang danh gia model tren 75.000 mau test...
echo.

python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test-path data/tensors/official-v1/test.h5

pause
