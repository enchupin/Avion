param([string]$OutputDirectory)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$outputDir = if ($OutputDirectory) { [IO.Path]::GetFullPath($OutputDirectory) } else { Join-Path $root "dist" }
$packageDir = Join-Path $outputDir "Avion-win32-x64"
$appDir = Join-Path $packageDir "resources\app"
$appExe = Join-Path $packageDir "Avion.exe"

Push-Location $root
try {
  $packagerCli = Join-Path $root "node_modules\@electron\packager\bin\electron-packager.mjs"
  if (-not (Test-Path -LiteralPath $packagerCli)) {
    throw "Electron packager is missing. Run npm install in electron first."
  }
  & node $packagerCli . Avion `
    --platform=win32 `
    --arch=x64 `
    --out="$outputDir" `
    --overwrite `
    --no-asar `
    --ignore="node_modules" `
    --ignore="dist" `
    --ignore="native" `
    --ignore="scripts" `
    --ignore="tests" `
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

$pathsToRemove = @(
  (Join-Path $appDir ".idea"),
  (Join-Path $appDir "scripts"),
  (Join-Path $appDir "native"),
  (Join-Path $appDir "node_modules"),
  (Join-Path $appDir "package-lock.json")
)

foreach ($pathToRemove in $pathsToRemove) {
  if (Test-Path $pathToRemove) {
    $resolvedTarget = (Resolve-Path -LiteralPath $pathToRemove).Path
    $expectedRoot = [IO.Path]::GetFullPath($appDir).TrimEnd('\') + '\'
    if (-not $resolvedTarget.StartsWith($expectedRoot, [StringComparison]::OrdinalIgnoreCase)) {
      throw "Refusing to remove a path outside the packaged app: $resolvedTarget"
    }
    Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
  }
}

# Ship the live SR worker, environment probe, architecture and weights.
# The environment panel installs Python/PyTorch into writable user data on demand.
$sourceSr = Join-Path (Split-Path -Parent $root) "SR"
$bundledSr = Join-Path $packageDir "resources\SR"
$runtimeFiles = @(
  "roi_dataset.py",
  "sr_worker.py",
  "IMDN\utils.py",
  "requirements-evaluation.txt",
  "check_environment.py",
  "IMDN\model\architecture.py",
  "IMDN\model\block.py",
  "IMDN\checkpoints\IMDN_x2.pth",
  "IMDN\checkpoints\IMDN_x3.pth",
  "IMDN\checkpoints\IMDN_x4.pth"
)
foreach ($relativeFile in $runtimeFiles) {
  $sourceFile = Join-Path $sourceSr $relativeFile
  $destinationFile = Join-Path $bundledSr $relativeFile
  if (-not (Test-Path -LiteralPath $sourceFile)) {
    throw "Missing SR runtime dependency: $sourceFile"
  }
  New-Item -ItemType Directory -Path (Split-Path -Parent $destinationFile) -Force | Out-Null
  Copy-Item -LiteralPath $sourceFile -Destination $destinationFile -Force
}
Write-Host "Packaged Avion app at $appExe"

# Ship the portable runtime, including its provenance and redistribution notices.
$sourceDoom = Join-Path (Split-Path -Parent $root) "Doom"
$bundledDoom = Join-Path $packageDir "resources\Doom"
$doomExe = Join-Path $sourceDoom "runtime\win-x64\chocolate-doom.exe"
$doomWad = "iwads\freedoom-0.13.0\freedoom-0.13.0\freedoom2.wad"
if ((Test-Path -LiteralPath $doomExe) -and (Test-Path -LiteralPath (Join-Path $sourceDoom $doomWad))) {
  $doomBin = Join-Path $bundledDoom "runtime\win-x64"
  New-Item -ItemType Directory -Path $doomBin -Force | Out-Null
  Copy-Item -LiteralPath (Join-Path $sourceDoom "COPYING.md") -Destination $bundledDoom -Force
  Copy-Item -LiteralPath (Join-Path $sourceDoom "AUTHORS") -Destination $bundledDoom -Force
  foreach ($launcher in @('run-freedoom2.bat', 'run-freedoom2-roi.bat')) {
    Copy-Item -LiteralPath (Join-Path $sourceDoom $launcher) -Destination $bundledDoom -Force
  }
  Get-ChildItem -LiteralPath (Split-Path -Parent $doomExe) -File | Where-Object { $_.Extension -in '.exe', '.dll' } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $doomBin -Force
  }
  foreach ($notice in @('licenses', 'build-info.json')) {
    Copy-Item -LiteralPath (Join-Path (Split-Path -Parent $doomExe) $notice) -Destination $doomBin -Recurse -Force
  }
  $wadTarget = Join-Path $bundledDoom $doomWad
  New-Item -ItemType Directory -Path (Split-Path -Parent $wadTarget) -Force | Out-Null
  Copy-Item -LiteralPath (Join-Path $sourceDoom $doomWad) -Destination $wadTarget -Force
  # Preserve Freedoom's redistribution notice alongside its assets.
  Get-ChildItem -LiteralPath (Split-Path -Parent (Join-Path $sourceDoom $doomWad)) -File | Where-Object { $_.Extension -ne '.wad' } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Split-Path -Parent $wadTarget) -Force
  }
}
