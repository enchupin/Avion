@echo off
setlocal
set "ROOT=%~dp0"
set "EXE=%ROOT%build-local\src\Release\chocolate-doom.exe"
set "IWAD=%ROOT%iwads\freedoom-0.13.0\freedoom-0.13.0\freedoom2.wad"

rem Prefer the project environment over the Windows Store python alias.
if not defined AVION_SR_PYTHON if exist "%ROOT%..\.avion-runtime\venv\Scripts\python.exe" set "AVION_SR_PYTHON=%ROOT%..\.avion-runtime\venv\Scripts\python.exe"
if not defined AVION_SR_PYTHON if exist "%ROOT%..\.venv\Scripts\python.exe" set "AVION_SR_PYTHON=%ROOT%..\.venv\Scripts\python.exe"

if not exist "%EXE%" set "EXE=%ROOT%runtime\win-x64\chocolate-doom.exe"
if not exist "%EXE%" set "EXE=%ROOT%build-local\src\Debug\chocolate-doom.exe"

if not exist "%EXE%" (
    echo chocolate-doom.exe was not found.
    echo Download or pull the complete kjw branch, including Doom\runtime\win-x64.
    echo Optional rebuild script: Doom\scripts\package-windows.ps1
    pause
    exit /b 1
)

if not exist "%IWAD%" (
    echo freedoom2.wad was not found:
    echo %IWAD%
    pause
    exit /b 1
)

rem Electron waits for the game and uses a writable per-user config directory.
set "WAIT="
if "%AVION_DOOM_WAIT%"=="1" set "WAIT=/wait"
if defined AVION_DOOM_CONFIG_DIR (
    start "" %WAIT% /D "%ROOT%.." "%EXE%" -window -srcompare -srlatency -iwad "%IWAD%" -config "%AVION_DOOM_CONFIG_DIR%\default.cfg" -extraconfig "%AVION_DOOM_CONFIG_DIR%\chocolate-doom.cfg" %*
) else (
    start "" %WAIT% /D "%ROOT%.." "%EXE%" -window -srcompare -srlatency -iwad "%IWAD%" %*
)
exit /b %ERRORLEVEL%
