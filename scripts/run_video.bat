@echo off
REM Run pose risk detection with video file
REM Usage: run_video.bat <video_path> [options]

echo ============================================
echo Pose Risk Detection - Video Mode
echo ============================================

cd /d "%~dp0.."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check if video path is provided
if "%~1"=="" (
    echo ERROR: Video path required
    echo Usage: run_video.bat ^<video_path^> [options]
    echo Example: run_video.bat C:\Videos\test.mp4 --show
    pause
    exit /b 1
)

REM Run with the provided video
python main.py --source video --path "%~1" --show %2 %3 %4 %5

pause
