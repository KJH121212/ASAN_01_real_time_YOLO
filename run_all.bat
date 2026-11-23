@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo   Starting ASAN Socket Server + Streamlit Client
echo ==================================================

start "ASAN Socket Server (COCO-23)" cmd /k "cd /d "%~dp0socket_server" && call run_server.bat"
timeout /t 2 /nobreak >nul
start "ASAN Streamlit Client (COCO-23)" cmd /k "cd /d "%~dp0streamlit_client" && call run_client.bat"

exit /b 0
