$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$outputDir = Join-Path $root "dist"
$packageDir = Join-Path $outputDir "Avion-win32-x64"
$appDir = Join-Path $packageDir "resources\app"
$appExe = Join-Path $packageDir "Avion.exe"
$nativeHost = Join-Path $appDir "native\AvionCaptureHost.exe"

Push-Location $root
try {
  & npx electron-packager . Avion `
    --platform=win32 `
    --arch=x64 `
    --out=dist `
    --overwrite `
    --no-asar `
    --ignore="node_modules" `
    --ignore="dist" `
    --ignore="native[\\/]capture-host" `
    --ignore="scripts" `
    --ignore="package-lock\.json"

  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
finally {
  Pop-Location
}

if (-not (Test-Path $appExe)) {
  throw "Expected packaged app was not found at $appExe."
}

if (-not (Test-Path $nativeHost)) {
  throw "Expected native host was not found at $nativeHost."
}

$pathsToRemove = @(
  (Join-Path $appDir ".idea"),
  (Join-Path $appDir "scripts"),
  (Join-Path $appDir "native\capture-host"),
  (Join-Path $appDir "node_modules"),
  (Join-Path $appDir "package-lock.json")
)

foreach ($pathToRemove in $pathsToRemove) {
  if (Test-Path $pathToRemove) {
    Remove-Item -LiteralPath $pathToRemove -Recurse -Force
  }
}

# Ship the evaluation worker, architecture, sample images and existing weights.
# Python/PyTorch remain an explicit runtime prerequisite.
$sourceSr = Join-Path (Split-Path -Parent $root) "SR"
$bundledSr = Join-Path $packageDir "resources\SR"
$evaluationFiles = @(
  "benchmark.py",
  "roi_dataset.py",
  "sr_worker.py",
  "IMDN\utils.py",
  "requirements-evaluation.txt",
  "IMDN\model\architecture.py",
  "IMDN\model\block.py",
  "IMDN\checkpoints\IMDN_x2.pth",
  "IMDN\checkpoints\IMDN_x3.pth",
  "IMDN\checkpoints\IMDN_x4.pth"
)
foreach ($relativeFile in $evaluationFiles) {
  $sourceFile = Join-Path $sourceSr $relativeFile
  $destinationFile = Join-Path $bundledSr $relativeFile
  if (-not (Test-Path -LiteralPath $sourceFile)) {
    throw "Missing evaluation dependency: $sourceFile"
  }
  New-Item -ItemType Directory -Path (Split-Path -Parent $destinationFile) -Force | Out-Null
  Copy-Item -LiteralPath $sourceFile -Destination $destinationFile -Force
}
$sampleDirectory = Join-Path $bundledSr "IMDN\Test_Datasets\Set5"
New-Item -ItemType Directory -Path $sampleDirectory -Force | Out-Null
Get-ChildItem -LiteralPath (Join-Path $sourceSr "IMDN\Test_Datasets\Set5") -Filter "*.bmp" | ForEach-Object {
  Copy-Item -LiteralPath $_.FullName -Destination $sampleDirectory -Force
}

Write-Host "Packaged Avion app at $appExe"
Write-Host "Bundled native host at $nativeHost"

# The existing Freedoom build is optional; evaluation of saved ROI data always works.
$sourceDoom = Join-Path (Split-Path -Parent $root) "Doom"
$bundledDoom = Join-Path $packageDir "resources\Doom"
$doomExe = Join-Path $sourceDoom "build-local\src\Release\chocolate-doom.exe"
$doomWad = "iwads\freedoom-0.13.0\freedoom-0.13.0\freedoom2.wad"
if ((Test-Path -LiteralPath $doomExe) -and (Test-Path -LiteralPath (Join-Path $sourceDoom $doomWad))) {
  $doomBin = Join-Path $bundledDoom "build-local\src\Release"
  New-Item -ItemType Directory -Path $doomBin -Force | Out-Null
  Copy-Item -LiteralPath (Join-Path $sourceDoom "COPYING.md") -Destination $bundledDoom -Force
  Get-ChildItem -LiteralPath (Split-Path -Parent $doomExe) -File | Where-Object { $_.Extension -in '.exe', '.dll' } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $doomBin -Force
  }
  $wadTarget = Join-Path $bundledDoom $doomWad
  New-Item -ItemType Directory -Path (Split-Path -Parent $wadTarget) -Force | Out-Null
  Copy-Item -LiteralPath (Join-Path $sourceDoom $doomWad) -Destination $wadTarget -Force
  # Preserve Freedoom's redistribution notice alongside its assets.
  Get-ChildItem -LiteralPath (Split-Path -Parent (Join-Path $sourceDoom $doomWad)) -File | Where-Object { $_.Extension -ne '.wad' } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Split-Path -Parent $wadTarget) -Force
  }
}
