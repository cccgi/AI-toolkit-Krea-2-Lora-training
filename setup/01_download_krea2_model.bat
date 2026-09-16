@echo off
REM ============================================================
REM Step 1: Download the Krea-2 RAW base model + Qwen3-VL-4B text
REM encoder + Qwen-Image VAE from Hugging Face.
REM
REM Requires a Hugging Face account with access granted to
REM krea/Krea-2-Raw (gated repo -- request access at
REM https://huggingface.co/krea/Krea-2-Raw first).
REM ============================================================
setlocal

set ENV_NAME=ai-toolkit-perceptual

echo === Step 1: Download Krea-2 RAW model files ===
echo.

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ABORT] Could not activate conda env "%ENV_NAME%".
    echo Run setup\00_install_ai_toolkit_perceptual.bat first.
    pause
    exit /b 1
)

python -m pip show huggingface_hub >nul 2>nul
if errorlevel 1 (
    python -m pip install huggingface_hub
)

echo.
echo Checking Hugging Face login...
python -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"
if errorlevel 1 (
    echo.
    echo You are not logged into Hugging Face yet.
    echo Do NOT paste your token into this window if it echoes to a log ^-
    echo the hf CLI itself masks it, so it's safe to run interactively:
    echo.
    hf auth login
    if errorlevel 1 (
        echo [ABORT] Hugging Face login failed.
        pause
        exit /b 1
    )
)

echo.
echo Downloading model files (this may take a while: ~13GB transformer + ~9GB text encoder + ~0.25GB VAE)...
python "%~dp0download_krea2_model.py"
if errorlevel 1 (
    echo [ABORT] Model download failed. Common cause: access to
    echo krea/Krea-2-Raw has not been granted to your account yet.
    echo Request access at https://huggingface.co/krea/Krea-2-Raw
    pause
    exit /b 1
)

echo.
echo === Step 1 complete ===
echo Next: run setup\02_download_face_models.bat
pause
