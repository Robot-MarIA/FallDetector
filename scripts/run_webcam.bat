@echo off
REM Run pose risk detection with webcam
REM Usage: run_webcam.bat [options]

echo ============================================
echo Pose Risk Detection - Webcam Mode
echo ============================================

cd /d "%~dp0.."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Run with visualization by default
python main.py --source webcam --show %*

pause
