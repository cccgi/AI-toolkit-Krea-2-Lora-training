@echo off
setlocal
set ENV_NAME=ai-toolkit-perceptual

echo === Step 2: Download face-detection/embedding/upscale models ===
echo.

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ABORT] Could not activate conda env "%ENV_NAME%".
    echo Run setup\00_install_ai_toolkit_perceptual.bat first.
    pause
    exit /b 1
)

python -m pip show onnxruntime spandrel >nul 2>nul
if errorlevel 1 (
    python -m pip install onnxruntime opencv-python-headless pillow spandrel numpy
)

python "%~dp002_download_face_models.py"
if errorlevel 1 (
    echo [ABORT] Face/upscale model download failed. See docs\FACE_MODEL_MIRRORS.md.
    pause
    exit /b 1
)

echo.
echo === Step 2 complete ===
echo All setup steps are done. You can now double-click:
echo   RUN_ME_train_kimberly_style_lora.bat  (rename/copy per subject)
echo or read docs\QUICKSTART.md for the full walkthrough.
pause
