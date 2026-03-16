@echo off
title AlphaLens AI - Equity Scanner
echo.
echo  ============================================
echo   AlphaLens AI - Starting Up...
echo  ============================================
echo.

cd /d "%~dp0"

REM --- Find Python ---
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON=python
    goto :found
)
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON=python3
    goto :found
)
where py >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON=py
    goto :found
)
REM Check common install locations
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
    goto :found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
    goto :found
)
if exist "C:\Python312\python.exe" (
    set PYTHON=C:\Python312\python.exe
    goto :found
)
if exist "C:\Python311\python.exe" (
    set PYTHON=C:\Python311\python.exe
    goto :found
)

echo ERROR: Python not found. Please install Python 3.10+ from python.org
echo Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:found
echo Found Python: %PYTHON%
%PYTHON% --version

echo.
echo [1/3] Installing dependencies...
%PYTHON% -m pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)
echo      Done!

echo.
echo [2/3] Starting API server on http://localhost:8000 ...
start /b %PYTHON% -m uvicorn api.server:app --host 0.0.0.0 --port 8000 >nul 2>&1

echo      Waiting for server...
timeout /t 4 /nobreak >nul

echo.
echo [3/3] Triggering scan of 20 stocks...
curl -s -X POST http://localhost:8000/scan/start >nul 2>&1

echo.
echo  ============================================
echo   AlphaLens AI is LIVE!
echo  ============================================
echo.
echo   API:       http://localhost:8000
echo   Dashboard: Opening in browser now...
echo.
echo  ============================================
echo.

start "" "dashboard\index.html"

echo Waiting for scan to finish...
:poll
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/scan/status 2>nul | findstr /C:"\"running\":false" >nul
if %errorlevel% neq 0 goto poll

echo.
echo  ============================================
echo   Scan complete! Check your browser.
echo   Press any key to stop the server and exit.
echo  ============================================
pause >nul

taskkill /f /im python.exe /fi "WINDOWTITLE eq *uvicorn*" >nul 2>&1
