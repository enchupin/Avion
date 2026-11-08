param(
    [string]$CMake = 'cmake',
    [string]$VcpkgRoot = (Join-Path $PSScriptRoot '..\.tools\vcpkg')
)

$ErrorActionPreference = 'Stop'
$doomRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoRoot = Split-Path -Parent $doomRoot
$buildDir = Join-Path $doomRoot 'build-portable'
$runtimeDir = Join-Path $doomRoot 'runtime\win-x64'
$toolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
if (-not (Test-Path -LiteralPath $toolchain)) {
    throw 'vcpkg is missing. Clone https://github.com/microsoft/vcpkg.git into Doom/.tools/vcpkg and run bootstrap-vcpkg.bat, or pass -VcpkgRoot.'
}
$toolchain = (Resolve-Path -LiteralPath $toolchain).Path

if (-not (Get-Command $CMake -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $CMake = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' | Select-Object -First 1
    }
    if (-not $CMake -or -not (Test-Path -LiteralPath $CMake)) {
        throw 'CMake was not found. Install the Visual Studio C++ workload with CMake tools, or pass -CMake.'
    }
}

# Static SDL, codec libraries and MSVC runtime: only Windows system DLLs remain.
& $CMake -S $doomRoot -B $buildDir -G 'Visual Studio 17 2022' -A x64 `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" '-DVCPKG_TARGET_TRIPLET=x64-windows-static' '-DVCPKG_MANIFEST_INSTALL=ON' `
    '-DCMAKE_POLICY_DEFAULT_CMP0091=NEW' '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>'
if ($LASTEXITCODE -ne 0) { throw 'Doom configuration failed.' }
& $CMake --build $buildDir --config Release --target chocolate-doom --parallel
if ($LASTEXITCODE -ne 0) { throw 'Doom build failed.' }

$builtExe = Join-Path $buildDir 'src\Release\chocolate-doom.exe'
$licenseDir = Join-Path $runtimeDir 'licenses'
New-Item -ItemType Directory -Path $licenseDir -Force | Out-Null
Copy-Item -LiteralPath $builtExe -Destination $runtimeDir -Force
# Doom's license and authors already ship in Doom/COPYING.md and Doom/AUTHORS.
$libraries = @('sdl2', 'sdl2-mixer', 'sdl2-net', 'libpng', 'libsamplerate', 'libogg', 'libvorbis', 'wavpack', 'zlib')
foreach ($library in $libraries) {
    $copyright = Join-Path $buildDir "vcpkg_installed\x64-windows-static\share\$library\copyright"
    Copy-Item -LiteralPath $copyright -Destination (Join-Path $licenseDir "$library.txt") -Force
}

# Record the actual source bytes, including any uncommitted engine changes.
$sourceHashes = [ordered]@{}
$sourceFiles = Get-ChildItem -LiteralPath (Join-Path $doomRoot 'CMakeLists.txt'), (Join-Path $doomRoot 'vcpkg.json') -File
foreach ($directory in @('src', 'cmake', 'textscreen', 'pcsound', 'opl', 'win32', 'data')) {
    $sourceFiles += Get-ChildItem -LiteralPath (Join-Path $doomRoot $directory) -Recurse -File
}
foreach ($file in ($sourceFiles | Sort-Object FullName)) {
    $relative = $file.FullName.Substring($doomRoot.Length + 1).Replace('\', '/')
    $sourceHashes[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$baseCommit = & git -C $repoRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Could not identify the source repository commit.' }
$info = [ordered]@{
    platform = 'windows-x64'
    configuration = 'Release'
    linkage = 'static (including MSVC runtime)'
    sourceRepository = 'https://github.com/enchupin/Avion'
    baseCommit = $baseCommit
    vcpkgBaseline = (Get-Content -LiteralPath (Join-Path $doomRoot 'vcpkg.json') -Raw | ConvertFrom-Json).'builtin-baseline'
    executableSha256 = (Get-FileHash -LiteralPath $builtExe -Algorithm SHA256).Hash.ToLowerInvariant()
    sourceSha256 = $sourceHashes
}
$info | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $runtimeDir 'build-info.json') -Encoding UTF8
Write-Host "Portable Doom and licenses updated in $runtimeDir"
