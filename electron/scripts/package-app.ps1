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

Write-Host "Packaged Avion app at $appExe"
Write-Host "Bundled native host at $nativeHost"
