@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
for %%D in ("C:\miniconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3") do (
    if not defined PYTHON_EXE if exist "%%~D\envs\rtm_env_50\python.exe" set "PYTHON_EXE=%%~D\envs\rtm_env_50\python.exe"
)
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [ERROR] Python not found.
    pause
    exit /b 1
)

echo ==================================================
echo   Mobile Camera Test Menu (tmp)
echo ==================================================
echo   1. [Android USB ] scrcpy camera (1280x720 @ 30fps)
echo   2. [Android WiFi] scrcpy Wi-Fi camera (auto TCP/IP)
echo   3. [iPhone/Andr ] SRT Ultra-Low-Latency Listener (:9000, USB/WiFi)
echo   4. [iPhone/Andr ] HTTP MJPEG / RTSP URL Stream
echo   5. [Android     ] List phone camera IDs (scrcpy)
echo   6. [Windows     ] Scan Windows cameras (DSHOW + MSMF)
echo ==================================================
set /p CHOICE="Select [1-6]: "

if "%CHOICE%"=="1" (
    "%PYTHON_EXE%" mobile_cam_test.py --mode scrcpy --size 1280x720 --fps 30 --facing back
) else if "%CHOICE%"=="2" (
    set /p WIFI_IP="Phone Wi-Fi IP (Enter for auto-detect via USB): "
    if "%WIFI_IP%"=="" (
        "%PYTHON_EXE%" mobile_cam_test.py --mode scrcpy_wifi --size 1280x720 --fps 30 --facing back
    ) else (
        "%PYTHON_EXE%" mobile_cam_test.py --mode scrcpy_wifi --wifi-ip %WIFI_IP% --size 1280x720 --fps 30 --facing back
    )
) else if "%CHOICE%"=="3" (
    "%PYTHON_EXE%" mobile_cam_test.py --mode srt --port 9000 --size 1280x720
) else if "%CHOICE%"=="4" (
    set /p STREAM_URL="Enter stream URL (e.g. http://192.168.0.10:8081/video): "
    "%PYTHON_EXE%" mobile_cam_test.py --mode url --url "%STREAM_URL%"
) else if "%CHOICE%"=="5" (
    "%PYTHON_EXE%" mobile_cam_test.py --mode scrcpy --list-cameras
) else if "%CHOICE%"=="6" (
    "%PYTHON_EXE%" mobile_cam_test.py --scan
) else (
    echo Invalid selection.
)
pause
