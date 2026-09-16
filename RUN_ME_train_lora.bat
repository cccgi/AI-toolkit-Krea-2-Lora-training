@echo off
REM ============================================================
REM DOUBLE-CLICK THIS FILE TO TRAIN A KREA-2 CHARACTER LORA
REM
REM Before running: edit the 4 values below for your subject, then
REM save and double-click. Everything else (ingest, crop, upscale,
REM mask, caption, smoke test, train, evaluate) is automatic.
REM ============================================================
setlocal

REM ---- EDIT THESE 4 VALUES FOR YOUR SUBJECT -------------------
set SUBJECT_NAME=kimberly
set RAW_PHOTOS_DIR=C:\path\to\your\raw\photos\folder
set TRIGGER=kmb
set CLASS_NOUN=woman
REM --------------------------------------------------------------

REM ---- Advanced (validated defaults; only change if you know why)
set AI_TOOLKIT_DIR=%~dp0..\ai-toolkit-perceptual
set RANK=32
set STEPS=2800
set RESOLUTION=1024
set ENV_NAME=ai-toolkit-perceptual

echo ============================================================
echo  Krea-2 Character LoRA Training Pipeline
echo  Subject:    %SUBJECT_NAME%
echo  Trigger:    %TRIGGER%
echo  Raw photos: %RAW_PHOTOS_DIR%
echo ============================================================
echo.

if not exist "%RAW_PHOTOS_DIR%" (
    echo [ABORT] RAW_PHOTOS_DIR does not exist: %RAW_PHOTOS_DIR%
    echo Edit this .bat file and set it to your actual photo folder.
    pause
    exit /b 1
)

if not exist "%AI_TOOLKIT_DIR%\run.py" (
    echo [ABORT] ai-toolkit-perceptual not found at %AI_TOOLKIT_DIR%
    echo Run setup\00_install_ai_toolkit_perceptual.bat first.
    pause
    exit /b 1
)

call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ABORT] Could not activate conda env "%ENV_NAME%".
    echo Run setup\00_install_ai_toolkit_perceptual.bat first.
    pause
    exit /b 1
)

nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>nul
echo.
echo [Gate] Checking GPU is free before starting...
for /f "tokens=1" %%i in ('nvidia-smi --query-gpu=memory.used --format=csv^,noheader^,nounits') do set GPU_USED=%%i
if %GPU_USED% GTR 2000 (
    echo [ABORT] GPU already has more than 2GB in use. Another job may be
    echo         running. Close it first, then re-run this launcher.
    echo         This script will NOT kill anything for you.
    pause
    exit /b 1
)

python "%~dp0pipeline\run_full_pipeline.py" ^
    --ai_toolkit_dir "%AI_TOOLKIT_DIR%" ^
    --raw_photos_dir "%RAW_PHOTOS_DIR%" ^
    --subject_name "%SUBJECT_NAME%" ^
    --trigger "%TRIGGER%" ^
    --class_noun "%CLASS_NOUN%" ^
    --rank %RANK% ^
    --steps %STEPS% ^
    --resolution %RESOLUTION%

if errorlevel 1 (
    echo.
    echo [FAILED] Pipeline aborted. See the error above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  DONE. Your LoRA and evaluation report are in:
echo  workspace\%SUBJECT_NAME%\output\
echo ============================================================
pause
