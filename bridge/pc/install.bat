@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   Installing the Clash Royale bot (pc folder)
echo ============================================================
echo.
python --version
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH", then run this file again.
  pause
  exit /b 1
)
echo.
echo --- 1/3 PyTorch for the NVIDIA GPU (about 3 GB, be patient) ---
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo.
echo --- 2/3 The other dependencies ---
python -m pip install -r requirements.txt
echo.
echo --- 3/3 Java 21 (for the crforge twin) ---
java -version 2>&1 | findstr /r /c:"version \"2[1-9]" >nul
if errorlevel 1 (
  winget install -e --id EclipseAdoptium.Temurin.21.JRE
) else (
  echo Java 21 or newer is already installed.
)
echo.
echo ============================================================
echo   Install done.
echo   CLOSE this window, open a NEW terminal in this folder
echo   (right-click in the folder, "Open in Terminal") and type:
echo.
echo       python bot_pc.py check
echo ============================================================
pause
