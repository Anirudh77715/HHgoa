@echo off
REM Push this project to a Hugging Face Space so it gets a public URL.
REM
REM SECURITY: this script never asks for your token and you should never type
REM one into this window. The ONLY place the token goes is the "Password"
REM prompt that git itself shows. An earlier version of this script asked for a
REM Space name straight after the username, and a token got pasted there --
REM which put a live write credential into a URL and into terminal scrollback.
REM Hence the hf_ guards below.

setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo   BEFORE RUNNING THIS, the Space must already exist.
echo  ============================================================
echo.
echo   1. Open   https://huggingface.co/new-space
echo   2. Space name          : ask-hh-goa
echo   3. License             : mit
echo   4. Select the SDK      : DOCKER    (then choose "Blank")
echo   5. Hardware            : CPU basic - free
echo   6. Visibility          : Public
echo   7. Click "Create Space"
echo.
echo   This script checks the Space exists before pushing, so if you
echo   have not done the above it will stop and tell you.
echo.
pause
echo.

:askuser
set "HFUSER="
set /p HFUSER=Your Hugging Face USERNAME (not a token):
if "!HFUSER!"=="" (
  echo   Nothing entered.
  goto askuser
)
echo !HFUSER! | findstr /b /c:"hf_" >nul
if not errorlevel 1 (
  echo.
  echo   ***  STOP. That looks like an ACCESS TOKEN, not a username.  ***
  echo   A token starting with hf_ is a password. Do not type it here.
  echo   Go to https://huggingface.co/settings/tokens and DELETE that
  echo   token now, then create a new one and run this again.
  echo.
  pause
  exit /b 1
)

set "HFSPACE=ask-hh-goa"
set "REPLY="
set /p REPLY=Space name - just press ENTER for "ask-hh-goa":
if not "!REPLY!"=="" set "HFSPACE=!REPLY!"

echo !HFSPACE! | findstr /b /c:"hf_" >nul
if not errorlevel 1 (
  echo.
  echo   ***  STOP. That is an ACCESS TOKEN, not a Space name.  ***
  echo   Go to https://huggingface.co/settings/tokens and DELETE that
  echo   token now - it has been exposed - then create a new one.
  echo.
  pause
  exit /b 1
)

set "HFURL=https://huggingface.co/spaces/!HFUSER!/!HFSPACE!"

echo.
echo   Checking that !HFURL! exists...
curl -s -o nul -w "%%{http_code}" "https://huggingface.co/api/spaces/!HFUSER!/!HFSPACE!" > "%TEMP%\hfcheck.txt" 2>nul
set /p CODE=<"%TEMP%\hfcheck.txt"
del "%TEMP%\hfcheck.txt" 2>nul

if not "!CODE!"=="200" (
  echo.
  echo   Could not find that Space ^(HTTP !CODE!^).
  echo.
  echo   Create it first at https://huggingface.co/new-space
  echo     - Space name must be exactly:  !HFSPACE!
  echo     - Owner must be:               !HFUSER!
  echo     - SDK must be:                 Docker
  echo     - Visibility:                  Public
  echo.
  pause
  exit /b 1
)

echo   Found it.
echo.
echo  ============================================================
echo   Git will now ask for a Username and a Password.
echo       Username : !HFUSER!
echo       Password : paste your WRITE ACCESS TOKEN
echo                  ^(https://huggingface.co/settings/tokens^)
echo                  NOT your account password - HF rejects those.
echo   The password is invisible while typing. That is normal.
echo  ============================================================
echo.

git remote remove hf 2>nul
git remote add hf "!HFURL!"

git push hf main
set PUSHRC=%errorlevel%

REM Do not leave the remote configured; it keeps the URL out of .git/config
REM and means a future run always re-validates the Space.
git remote remove hf 2>nul

if not "%PUSHRC%"=="0" (
  echo.
  echo   Push failed.
  echo     - "Password authentication ... no longer supported" means you
  echo       typed your account password. Use a WRITE token instead.
  echo     - 403 means the token is Read-only. Create one with type Write.
  echo.
  pause
  exit /b 1
)

echo.
echo   Pushed. The Space is building - 5 to 10 minutes the first time,
echo   because it bakes the embedding model into the image.
echo.
echo   Watch the build:   !HFURL!
echo   Public URL:        https://!HFUSER!-!HFSPACE!.hf.space
echo   Health check:      https://!HFUSER!-!HFSPACE!.hf.space/healthz
echo.
echo   Tell Claude when it says "Running" and it will verify everything.
echo.
pause
