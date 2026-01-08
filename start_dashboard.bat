@echo off
chcp 65001 >nul
echo ============================================================
echo STARTING FRAUD DETECTION DASHBOARD
echo ============================================================
echo.
cd /d "%~dp0"
python app_simple.py
pause
