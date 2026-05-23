param(
  [Parameter(Mandatory = $true)]
  [Alias("VideoPath")]
  [string] $CapturePath,

  [string] $OutputDir,

  [string] $Pattern = "frame_%06d.png"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $CapturePath)) {
  throw "Capture file was not found: $CapturePath"
}

$resolvedCapture = Resolve-Path $CapturePath
$captureItem = Get-Item $resolvedCapture

if ($captureItem.PSIsContainer) {
  $pngCount = (Get-ChildItem -LiteralPath $captureItem.FullName -Filter *.png -File -ErrorAction SilentlyContinue | Measure-Object).Count
  Write-Host "Frames are already saved in $($captureItem.FullName)"
  Write-Host "PNG frames: $pngCount"
  return
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
  throw "ffmpeg was not found on PATH. Install ffmpeg to extract recorded frames."
}

if (-not $OutputDir) {
  $OutputDir = Join-Path $captureItem.DirectoryName ($captureItem.BaseName + "-frames")
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$outputPattern = Join-Path $OutputDir $Pattern
if ($captureItem.Extension -ieq ".bgra") {
  $metadataPath = Join-Path $captureItem.DirectoryName ($captureItem.BaseName + ".json")
  if (-not (Test-Path $metadataPath)) {
    throw "Lossless capture metadata was not found: $metadataPath"
  }

  $metadata = Get-Content $metadataPath -Raw | ConvertFrom-Json
  $videoSize = "$($metadata.width)x$($metadata.height)"
  & $ffmpeg.Source `
    -hide_banner `
    -y `
    -f rawvideo `
    -pixel_format bgra `
    -video_size $videoSize `
    -i $resolvedCapture `
    -vsync 0 `
    $outputPattern
} else {
  & $ffmpeg.Source -hide_banner -y -i $resolvedCapture -vsync 0 $outputPattern
}

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Write-Host "Extracted frames to $OutputDir"
