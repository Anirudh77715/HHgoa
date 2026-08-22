@echo off
REM One-click local run. Double-click this file.
REM
REM This starts the full app -- API and frontend together on one port -- which
REM is the simplest way to get a working demo. No deploy, no accounts, no
REM configuration. Everything runs on your machine.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   Could not find .venv\Scripts\python.exe
  echo   Create it first:  py -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting Ask HH Goa...
echo.
echo   When it says "Application startup complete", open this in GOOGLE CHROME:
echo.
echo       http://localhost:7860
echo.
echo   Chrome or Edge specifically -- voice input does not work in other
echo   browsers, including in-app preview browsers.
echo.
echo   Press Ctrl+C here to stop.
echo.

REM Give the server a moment to bind, then open Chrome at the right page.
start "" /b cmd /c "timeout /t 4 /nobreak >nul && start chrome http://localhost:7860 || start http://localhost:7860"

.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 7860

pause
