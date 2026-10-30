@echo off
cd /d "%~dp0"
echo ======================================================
echo    STARTING CHESS POLICY RESNET-16x256 TRAINING
echo ======================================================
echo Dang khoi dong huan luyen tren GPU RTX 3050 Ti...
echo.

python scripts/train.py --batch-size 128 --grad-accum-steps 4 --lr 0.0003 --epochs 8 --checkpoint-dir checkpoints

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================
    echo  HUAN LUYEN HOAN TAT!
    echo  Model tot nhat da luu tai: checkpoints/best_model.pt
    echo ======================================================
) else (
    echo.
    echo [LOI] Qua trinh huan luyen gap su co. Vui long kiem tra log o tren.
)
pause
