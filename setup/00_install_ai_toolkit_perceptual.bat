@echo off
REM ============================================================
REM Step 0: Clone the ai-toolkit-perceptual fork and create its
REM conda environment.
REM
REM This project trains on the "ai-toolkit-perceptual" fork
REM (BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual), not
REM vanilla ostris/ai-toolkit. This fork is what was actually used
REM and verified to work for every Krea-2 and Flux.2 Klein LoRA
REM run in this project's research (see docs/WHY_THIS_FORK.md).
REM ============================================================
setlocal

set REPO_URL=https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual.git
set INSTALL_DIR=%~dp0..\ai-toolkit-perceptual
set ENV_NAME=ai-toolkit-perceptual

echo === Step 0: Clone ai-toolkit-perceptual and create conda env ===
echo.

where conda >nul 2>nul
if errorlevel 1 (
    echo [ABORT] conda was not found on PATH.
    echo Install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

if exist "%INSTALL_DIR%" (
    echo [skip] %INSTALL_DIR% already exists. Delete it first if you want a clean re-clone.
) else (
    echo Cloning %REPO_URL% ...
    git clone --recurse-submodules "%REPO_URL%" "%INSTALL_DIR%"
    if errorlevel 1 (
        echo [ABORT] git clone failed.
        pause
        exit /b 1
    )
)

echo.
echo Creating conda environment "%ENV_NAME%" (Python 3.11) ...
call conda create -n %ENV_NAME% python=3.11 -y
if errorlevel 1 (
    echo [ABORT] conda create failed.
    pause
    exit /b 1
)

echo.
echo Installing PyTorch (CUDA 12.1 build) and ai-toolkit requirements ...
call conda activate %ENV_NAME%
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "%INSTALL_DIR%\requirements.txt"

echo.
echo === Step 0 complete ===
echo ai-toolkit-perceptual is installed at: %INSTALL_DIR%
echo conda environment: %ENV_NAME%
echo.
echo Next: run setup\01_download_krea2_model.bat
pause
