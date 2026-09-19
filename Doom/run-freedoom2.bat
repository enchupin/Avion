@echo off
set "ROOT=%~dp0"
set "EXE=%ROOT%build-local\src\Release\chocolate-doom.exe"
set "IWAD=%ROOT%iwads\freedoom-0.13.0\freedoom-0.13.0\freedoom2.wad"

if not exist "%EXE%" set "EXE=%ROOT%build-local\src\Debug\chocolate-doom.exe"

if not exist "%EXE%" (
    echo chocolate-doom.exe was not found. Build Doom first.
    pause
    exit /b 1
)

if not exist "%IWAD%" (
    echo freedoom2.wad was not found:
    echo %IWAD%
    pause
    exit /b 1
)

start "" /D "%ROOT%.." "%EXE%" -window -srcompare -srlatency -iwad "%IWAD%" %*
