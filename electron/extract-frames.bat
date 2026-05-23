@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CAPTURE_PATH=%~1"

if "%CAPTURE_PATH%"=="" (
  for /f "usebackq delims=" %%F in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$dir = Join-Path $env:USERPROFILE 'Videos\Avion Captures'; if (-not (Test-Path $dir)) { exit 2 }; $raw = Get-ChildItem -LiteralPath $dir -Filter *.bgra -File; $png = Get-ChildItem -LiteralPath $dir -Directory | Where-Object { $_.Name -like 'avion-capture-*-1fps' }; @($raw + $png) | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"`) do (
    set "CAPTURE_PATH=%%F"
  )
)

if "%CAPTURE_PATH%"=="" (
  echo No .bgra capture file or 1 FPS PNG capture folder was found in "%USERPROFILE%\Videos\Avion Captures".
  echo Run capture first, or drag a .bgra file or 1 FPS folder onto this BAT file.
  pause
  exit /b 1
)

if not exist "%CAPTURE_PATH%" (
  echo Capture file was not found:
  echo "%CAPTURE_PATH%"
  pause
  exit /b 1
)

echo Extracting frames from:
echo "%CAPTURE_PATH%"
echo.

pushd "%SCRIPT_DIR%"
call npm run extract:frames -- -CapturePath "%CAPTURE_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if not "%EXIT_CODE%"=="0" (
  echo Frame extraction failed.
) else (
  echo Frame extraction completed.
)

pause
exit /b %EXIT_CODE%
