$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$sourceDir = Join-Path $root "native\capture-host"
$buildDir = Join-Path $sourceDir "build"

$vswhereCandidates = @(
  (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
  (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
)

$vswhere = $vswhereCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vswhere) {
  throw "vswhere.exe was not found. Install Visual Studio 2022 Build Tools with the Desktop development with C++ workload."
}

$vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsInstall) {
  throw "Visual Studio C++ build tools were not found. Install the Desktop development with C++ workload."
}

$vsDevCmd = Join-Path $vsInstall "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
  throw "VsDevCmd.bat was not found at $vsDevCmd."
}

$cmake = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if (-not (Test-Path $cmake)) {
  $cmakeCommand = Get-Command cmake -ErrorAction SilentlyContinue
  if (-not $cmakeCommand) {
    throw "CMake was not found. Install the CMake component in Visual Studio Build Tools."
  }
  $cmake = $cmakeCommand.Source
}

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$configure = "`"$cmake`" -S `"$sourceDir`" -B `"$buildDir`" -G `"Visual Studio 17 2022`" -A x64"
$build = "`"$cmake`" --build `"$buildDir`" --config Release"
$command = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 && $configure && $build"

& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Built AvionCaptureHost.exe"
