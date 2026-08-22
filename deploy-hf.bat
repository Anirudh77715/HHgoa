@echo off
REM Push this project to a Hugging Face Space so it gets a public URL.
REM
REM Do the two browser steps FIRST (see DEPLOY.md or the checklist below),
REM then double-click this file.
REM
REM   1. Create the Space:  https://huggingface.co/new-space
REM        Owner    : your username
REM        Space name: ask-hh-goa
REM        License  : mit
REM        SDK      : *** Docker ***  ->  Blank
REM        Hardware : CPU basic (free)
REM        Visibility: Public
REM
REM   2. Create an access token: https://huggingface.co/settings/tokens
REM        "Create new token" -> type WRITE -> copy it.
REM
REM When this script runs, git will ask for:
REM        Username : your Hugging Face username
REM        Password : PASTE THE TOKEN (not your account password)
REM
REM Nothing is stored by this script. Git may offer to remember the token in
REM Windows Credential Manager; that is between you and git.

setlocal
cd /d "%~dp0"

echo.
set /p HFUSER=Your Hugging Face username:
if "%HFUSER%"=="" (
  echo   No username entered. Aborting.
  pause
  exit /b 1
)

set HFSPACE=ask-hh-goa
set /p HFSPACE=Space name [ask-hh-goa]:
if "%HFSPACE%"=="" set HFSPACE=ask-hh-goa

set HFURL=https://huggingface.co/spaces/%HFUSER%/%HFSPACE%

echo.
echo   Target: %HFURL%
echo.

REM Re-point the remote every run so a typo on a previous attempt is not sticky.
git remote remove hf 2>nul
git remote add hf "%HFURL%"

echo   Pushing... (username = %HFUSER%, password = your WRITE token)
echo.
git push hf main

if errorlevel 1 (
  echo.
  echo   Push failed. Most common causes:
  echo     - Pasted your account password instead of a WRITE access token
  echo     - The Space does not exist yet, or the name is misspelled
  echo     - The Space was created with an SDK other than Docker
  echo.
  pause
  exit /b 1
)

echo.
echo   Pushed. The Space is building now - this takes about 5-10 minutes
echo   the first time, because it bakes the embedding model into the image.
echo.
echo   Watch the build log here:
echo       %HFURL%
echo.
echo   When it says "Running", your public URL is:
echo       https://%HFUSER%-%HFSPACE%.hf.space
echo.
echo   Check it works by opening:
echo       https://%HFUSER%-%HFSPACE%.hf.space/healthz
echo   You should see:  {"status":"ok","vectors":91,...}
echo.
echo   Then tell Claude the URL and it will wire up Netlify and verify it.
echo.
pause
